"""
GraphRAG 默认切分实现。

默认参数的来源
--------------
取自 Microsoft GraphRAG 的 chunking 配置（graphrag 的
`config/defaults.py` 与 chunking_config）：

    size            1200     目标 token 数
    overlap         100      相邻块重叠 token 数
    encoding_model  cl100k_base
    type            tokens   （另一个选项是 sentence）

**encoding_model 用 cl100k_base 是刻意的**：这是 GraphRAG 被引用最广
的那组默认值，跟着它走意味着"同一篇文档切出来的块与 GraphRAG 一致"，
token 数也对得上 —— 换成 o200k_base 数字会变。中文场景下 o200k_base
更省 token（同样的 size 能装更多内容），所以它也是个参数，想换就换。

算法：在句界处累积，不是按 token 窗口硬切
----------------------------------------
GraphRAG 的 token 策略是先按句界切开、再攒到接近 size 个 token。
这样 chunk 不会在一句话中间断开 —— 断在句子中间会让实体/关系抽取
拿到半句话，抽出来的东西质量明显更差。本实现照此行事。

与 GraphRAG 的两处**刻意差异**
------------------------------
1. **overlap 是句级的，不是精确的 token 数。**
   GraphRAG 按 token 精确回退；这里回退的粒度是整句（丢掉前几直到
   剩余 <= overlap）。因此实际重叠可能略小于配置值。
   要精确到 token 就得在句子中间切开，与上一条的初衷直接冲突 ——
   两者不可兼得，这里选了保句界。

2. **不实现 prepend_metadata**（把文档字段拼在每个 chunk 前面）。
   那会破坏 ChunkSegment 的偏移契约（content 必须等于
   text[start:end]），落库的 start_index / end_index 会指不回原文。
   而且这件事在本流水线里有专门的阶段：阶段 8 chunk_enrichment 的
   职责正是"继承自文档级的元数据"。放那儿做比在切分时塞进去干净。

超长句的处理
------------
整句就超过 size 个 token 时（例如一整段没有任何句末标点），按 token
预算二分出字符切点，切成多段。这一步是必需的 —— 否则一篇没有标点
的文档会变成单个巨型 chunk，抽取提示词直接爆上下文。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.services.document.domain.schemas import ChunkRequest

from .text import (
    ChunkSegment,
    TextDocumentChunker,
    merge_spans,
    shrink_whitespace,
    split_by_separator,
    token_length,
)

__all__ = [
    "GraphRagChunkingConfig",
    "TxtGraphRagChunker",
]

#: GraphRAG 的默认值 —— 单独提出来是为了让"这个数字从哪来"可查。
_GRAPH_RAG_SIZE = 1200
_GRAPH_RAG_OVERLAP = 100
_GRAPH_RAG_ENCODING = "cl100k_base"

#: 句界正则。用 finditer 匹配，命中的部分归到**前一句末尾**。
#:
#:     [。！？；]+      中文句末标点（可连用，例如"？！"）
#:     [.!?;](?=\s)    英文句末标点，要求后面跟空白 —— 避免把
#:                     "3.14" / "e.g." 当成句末
#:     \n+             换行。段落之间常常没有标点，靠它切
#:
#: GraphRAG 自带的边界正则是 `(?<=\.)\s*\n`，只管英文 + 换行。
#: 这里扩到中文标点，否则一篇中文长文会被当成一整句，
#: 超长句兜底会被反复触发，切出来的块反而不贴语义。
_DEFAULT_SENTENCE_BOUNDARY = r"[。！？；]+|[.!?;](?=\s)|\n+"


# ============================================================
# 参数结构
# ============================================================
#
# 每一项都会经 get_config_schema() 变成前端表单，
# 所以 Field 的 title / description / default 是**对外契约**的一部分。


class GraphRagChunkingConfig(BaseModel):
    """GraphRAG 切分的参数结构。"""

    size: int = Field(
        default=_GRAPH_RAG_SIZE,
        ge=1,
        le=32768,
        title="块大小（token）",
        description=(
            "每个 chunk 的目标 token 数，GraphRAG 默认 1200。"
            "调小会提高抽取粒度但显著增加 LLM 调用次数（成本）。"
            "注意是**目标**：整句就超长的段会被打散，但打散点不保证"
            "正好落在 size。"
        ),
    )

    overlap: int = Field(
        default=_GRAPH_RAG_OVERLAP,
        ge=0,
        title="重叠（token）",
        description=(
            "相邻 chunk 的重叠 token 数，GraphRAG 默认 100。"
            "文档的逻辑边界很少与 chunk 边界对齐，重叠能让边界附近的"
            "实体/关系也被抽到。实际重叠是句级的，可能略小于此值。"
        ),
    )

    encoding_model: Literal["cl100k_base", "o200k_base"] = Field(
        default=_GRAPH_RAG_ENCODING,
        title="编码模型",
        description=(
            "cl100k_base = GraphRAG 的历史默认，与它的 token 数一致；"
            "o200k_base = 较新（GPT-4o），中文更省 token。"
            "换编码会改变『1200 token』的实际含义。"
        ),
    )

    sentence_boundary: str = Field(
        default=_DEFAULT_SENTENCE_BOUNDARY,
        title="句界正则",
        description=(
            "用来找句子边界，命中的部分归到前一句末尾。"
            "默认覆盖中文句末标点、英文句末标点（要求后跟空白，"
            "避免切到小数）与换行。"
        ),
    )

    strip_whitespace: bool = Field(
        default=True,
        title="去除首尾空白",
        description="去掉每块首尾的空白字符，只收两端、不动段内缩进。",
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraphRagChunkingConfig":

        if self.overlap >= self.size:
            raise ValueError(
                f"overlap({self.overlap}) 必须小于 size({self.size})，"
                "否则每个块都会把前一块整个吞进来。"
            )

        if not self.sentence_boundary:
            raise ValueError(
                "sentence_boundary 不能为空 —— 空正则等于不做句界切分，"
                "整篇会退化成一个块（或全靠超长句兜底硬切）。"
            )

        # 在写入配置时就编译一次，把非法正则提前到 API 边界，
        # 而不是等流水线跑到阶段 7 才炸。
        import re

        try:
            re.compile(self.sentence_boundary)
        except re.error as exc:
            raise ValueError(
                f"sentence_boundary 不是合法正则："
                f"{self.sentence_boundary!r} — {exc}"
            ) from exc

        return self


# ============================================================
# 切分算法
# ============================================================


def _largest_prefix_within(
        text: str,
        start: int,
        end: int,
        *,
        size: int,
        encoding_name: str,
) -> int:
    """
    二分找最大的 cut ∈ (start, end]，使 text[start:cut] 的 token 数
    <= size。至少返回 start + 1，保证调用方能前进。

    为什么用二分而不是直接按 token 窗口切：tiktoken 的 token 边界可能
    落在**一个多字节字符中间**，decode 一个 token 子区间会得到替换符，
    拿不到可靠的字符偏移。二分则始终在真实 encode 上验证，切点必定
    合法（`best` 只在校验通过时才被赋值）。
    """

    lo, hi = start + 1, end
    best = start + 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if token_length(text[start:mid], encoding_name) <= size:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def _split_oversized(
        text: str,
        start: int,
        end: int,
        *,
        size: int,
        encoding_name: str,
) -> list[tuple[int, int]]:
    """把一个超长句按 token 预算切成若干 span。"""

    spans: list[tuple[int, int]] = []
    cursor = start

    while cursor < end:

        cut = _largest_prefix_within(
            text, cursor, end,
            size=size, encoding_name=encoding_name,
        )

        if cut <= cursor:  # 防御：二分不应返回原地踏步的点
            cut = cursor + 1

        spans.append((cursor, cut))
        cursor = cut

    return spans


# ============================================================
# 实现
# ============================================================


class TxtGraphRagChunker(TextDocumentChunker):
    """GraphRAG 默认切分器（句界 + token 累积）。

    ⚠️ 键必须显式覆盖：中间层 TextDocumentChunker 的 name 是
    TEXT_TXT_RECURSIVE_CHUNKER（为另一个实现准备的）。不覆盖的话
    两个实现会拿到同一个 key，在 PipelineRegistry.processors 那张
    dict 里互相覆盖 —— 而且是静默的。
    """

    name = "TEXT_TXT_GRAPH_CHUNKER"

    config_model = GraphRagChunkingConfig

    async def execute_chunk(
            self,
            request: ChunkRequest,
    ) -> list[ChunkSegment]:
        """按 GraphRAG 的句界累积策略切分。"""

        config = self.config_model.model_validate(request.config)

        text = request.text

        if not text:
            return []

        encoding_name = config.encoding_model

        def length_fn(piece: str) -> int:
            return token_length(piece, encoding_name)

        # 1. 按句界切开（分隔符跟随前一句，可还原原文）
        sentences = split_by_separator(
            text,
            config.sentence_boundary,
            is_regex=True,
            keep_separator=True,
        )

        # 2. 超长句先打散，避免单个巨块
        pieces: list[tuple[int, int]] = []

        for start, end in sentences:

            if length_fn(text[start:end]) > config.size:
                pieces.extend(
                    _split_oversized(
                        text, start, end,
                        size=config.size,
                        encoding_name=encoding_name,
                    )
                )
            else:
                pieces.append((start, end))

        # 3. 句级累积 + 重叠
        spans = merge_spans(
            text,
            pieces,
            length_fn=length_fn,
            chunk_size=config.size,
            chunk_overlap=config.overlap,
        )

        # 4. 收尾并组装
        segments: list[ChunkSegment] = []

        for start, end in spans:

            if config.strip_whitespace:
                start, end = shrink_whitespace(text, start, end)

            if end <= start:
                continue

            piece = text[start:end]

            segments.append(
                ChunkSegment(
                    text=piece,
                    start_index=start,
                    end_index=end,
                    # 本实现按 token 度量，所以 token_count 总是有值,
                    # 不是估算。
                    token_count=length_fn(piece),
                )
            )

        return segments
