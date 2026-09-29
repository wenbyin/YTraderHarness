"""
递归切分实现（具体层）。

继承层级（严格对应 parser / cleaner）
------------------------------------
    DocumentProcessor
      └── DocumentChunker          chunker/base.py   role 基类
            └── TextDocumentChunker chunker/text.py  中间基类（模板）
                  └── TxtRecursiveChunker            具体实现 ← 本文件

本文件只做两件事：
    1. 定义参数结构（RecursiveChunkingConfig），**全部对外暴露**
    2. 实现 execute_chunk()：把正文递归切成裸文本段

实体组装（uuid / content_hash / chunk_index / 计数）由
TextDocumentChunker.chunk() 负责，这里不重复。

关于偏移
--------
算法全程用 **span（字符区间）** 而不是字符串搬运，所以每一段都能
精确指回原文对应的区间，落库的 start_index / end_index 是可验证的
（template 写出的 content 必然等于 request.text[start:end]）。

递归切分的固有行为（不是 bug）
------------------------------
分隔符用尽还是太长的段会**超过 chunk_size** 保留下来 —— 没有任何
办法在不丢内容的前提下把它切得更小。要避免就提供更细的 separators，
或者把 chunk_size 调大。
"""

from __future__ import annotations

import re
from typing import Callable, Literal

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
from ..tokenization import model_token_length

__all__ = [
    "RecursiveChunkingConfig",
    "TxtRecursiveChunker",
]


# ============================================================
# 长度度量
# ============================================================

#: 本实现固定用 o200k_base（GPT-4o 的编码）。
#: 选它而不是 cl100k_base：对中文的压缩明显更好，同样的 chunk_size
#: 下能装更多实际内容。要换编码用 graphrag 那个实现 ——
#: 它把 encoding_model 暴露成了参数。
_TOKEN_ENCODING_NAME = "o200k_base"


def _token_len(text: str) -> int:
    return token_length(text, _TOKEN_ENCODING_NAME)


_LENGTH_FUNCTIONS: dict[str, Callable[[str], int]] = {
    "char": len,
    "token": _token_len,
}


#: 默认分隔符阶梯：中文优先。
#:
#: 递归切分按顺序尝试，越靠前越"粗"。中文没有空格分词，所以句末
#: 标点（。！？；）必须排在空格之前，否则会在句子中间断开。
#: 英文句末标点带上尾随空格，避免切到 "3.14" 这类小数。
_DEFAULT_SEPARATORS: list[str] = [
    "\n\n",                     # 段落
    "\n",                       # 行
    "。", "！", "？", "；",      # 中文句末
    ". ", "! ", "? ", "; ",     # 英文句末
    "，", ",",                   # 子句
    " ",                        # 词
]


# ============================================================
# 参数结构
# ============================================================
#
# 每一项都会经 get_config_schema() 变成前端表单，
# 所以 Field 的 title / description / default 是**对外契约**的一部分。


class RecursiveChunkingConfig(BaseModel):
    """递归切分的完整参数结构。"""

    chunk_size: int = Field(
        default=512,
        ge=1,
        le=32768,
        title="块长度",
        description=(
            "每个 chunk 的目标长度，单位由 length_function 决定。"
            "注意这是**目标**不是硬上限：分隔符用尽仍过长的段会原样保留。"
        ),
    )

    chunk_overlap: int = Field(
        default=64,
        ge=0,
        title="重叠长度",
        description=(
            "相邻 chunk 之间保留的重叠长度，用来避免答案正好被切在"
            "边界上。必须小于 chunk_size。"
        ),
    )

    length_function: Literal["char", "token", "model"] = Field(
        default="char",
        title="长度度量",
        description=(
            "char  = 按字符数（快、无依赖）；"
            "token = 按 o200k_base 分词数；"
            "model = 按**向量模型自己词表**的分词数。"
            "\n\n"
            "**要跟模型输入上限比大小时必须用 model。** o200k_base 与"
            "向量模型的词表不是一回事 —— 实测中文下 bge-small-zh 要比"
            "o200k_base 多吃 16%~33%，所以按 token 设的 1200 在模型看来"
            "是 1400~1600，而它的上限只有 512。超出部分被**静默截断**："
            "不报错、向量照常产出，只是 chunk 的尾巴从未被向量化。"
            "\n\n"
            "选 model 时需同时给 length_tokenizer（该向量模型的路径）。"
        ),
    )

    length_tokenizer: str | None = Field(
        default=None,
        title="向量模型路径",
        description=(
            "length_function='model' 时用哪个模型的词表，取 HF 模型名"
            "或本地目录。其余取值下忽略。"
            "\n\n"
            "**由流水线自动填**（本分支 embedder 的模型路径），不需要"
            "手写 —— 但它会随 chunker_config 一起落库，是 chunk 身份的"
            "一部分，所以必须是稳定的字符串而不是运行时对象。"
        ),
    )

    separators: list[str] = Field(
        # 刻意用 default= 而不是 default_factory=：
        # model_json_schema() 表达不了 factory，会输出 "default": null，
        # 前端表单于是渲染成一个空列表，看不到真实的分隔符阶梯。
        # pydantic v2 对可变默认值是**逐实例深拷贝**的，不存在共享。
        default=list(_DEFAULT_SEPARATORS),
        title="分隔符阶梯",
        description=(
            "按顺序尝试的切分依据，越靠前越粗。"
            "列表用尽还是太长的段会原样保留。"
            "中文场景请把句末标点排在空格之前（默认已如此）。"
        ),
    )

    keep_separator: bool = Field(
        default=True,
        title="保留分隔符",
        description=(
            "开启时分隔符跟随前一段，把各段按序拼回去能还原原文；"
            "关闭时丢弃分隔符。"
        ),
    )

    is_separator_regex: bool = Field(
        default=False,
        title="分隔符按正则解析",
        description=(
            "开启时 separators 里的每一项都是正则表达式；"
            "关闭时按字面量匹配（特殊字符会被转义）。"
        ),
    )

    strip_whitespace: bool = Field(
        default=True,
        title="去除首尾空白",
        description=(
            "去掉每段首尾的空白字符。关闭时切出来的段可能以换行/空格"
            "开头。"
        ),
    )

    min_chunk_size: int = Field(
        default=0,
        ge=0,
        title="最小块长度",
        description=(
            "短于该值的段会**并入前一段**（不丢弃内容）。"
            "0 表示不做这个处理。注意并入后那一段可能超过 chunk_size。"
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "RecursiveChunkingConfig":

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 "
                f"chunk_size({self.chunk_size})，否则每个块都会把"
                "前一块整个吞进来。"
            )

        if self.min_chunk_size and self.min_chunk_size >= self.chunk_size:
            raise ValueError(
                f"min_chunk_size({self.min_chunk_size}) 必须小于 "
                f"chunk_size({self.chunk_size})，否则所有块都会被"
                "反复并成一块。"
            )

        if not self.separators:
            raise ValueError(
                "separators 不能为空 —— 空列表等于完全不切分，"
                "整篇正文会变成单个 chunk。"
            )

        if self.is_separator_regex:
            # 在这里编译一次，把非法正则的错误提前到"写入配置时"，
            # 而不是等流水线跑到阶段 7 才炸。
            for pattern in self.separators:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"separators 里的正则不合法：{pattern!r} — {exc}"
                    ) from exc

        return self


# ============================================================
# 切分算法
# ============================================================


def _split_text(
        text: str,
        separators: list[str],
        *,
        length_fn: Callable[[str], int],
        chunk_size: int,
        keep_separator: bool,
        is_regex: bool,
) -> list[tuple[int, int]]:
    """
    递归切分：返回每一段都 <= chunk_size 的 span（分隔符用尽则例外）。

    拿第一个分隔符切；切出来仍过长的段，换下一个（更细的）分隔符
    继续切。`len(pieces) <= 1` 的分支既是"这个分隔符切不动"的判断，
    也是递归终止的保证 —— 每次递归 separators 都少一个。
    """

    if not text:
        return []

    if not separators:
        return [(0, len(text))]

    separator, rest = separators[0], separators[1:]

    pieces = split_by_separator(
        text,
        separator,
        is_regex=is_regex,
        keep_separator=keep_separator,
    )

    if len(pieces) <= 1:
        return _split_text(
            text, rest,
            length_fn=length_fn, chunk_size=chunk_size,
            keep_separator=keep_separator, is_regex=is_regex,
        )

    spans: list[tuple[int, int]] = []

    for start, end in pieces:

        if length_fn(text[start:end]) <= chunk_size:
            spans.append((start, end))
            continue

        # 这一段还是太长：用剩下的分隔符继续切，偏移折算回本层。
        for sub_start, sub_end in _split_text(
                text[start:end], rest,
                length_fn=length_fn, chunk_size=chunk_size,
                keep_separator=keep_separator, is_regex=is_regex,
        ):
            spans.append((start + sub_start, start + sub_end))

    return spans


def _absorb_small(
        text: str,
        spans: list[tuple[int, int]],
        *,
        length_fn: Callable[[str], int],
        min_chunk_size: int,
) -> list[tuple[int, int]]:
    """
    把过短的块**并入前一块**。

    刻意用"并入"而不是"丢弃"：丢弃会静默少内容，而 chunk 落库之后
    没人看得出原文里少了哪一句。

    开头的那个短块没有前一块可并，原样保留。
    """

    if min_chunk_size <= 0 or not spans:
        return spans

    merged: list[tuple[int, int]] = []

    for span in spans:
        short = length_fn(text[span[0]:span[1]]) < min_chunk_size

        if short and merged:
            previous = merged.pop()
            merged.append((previous[0], span[1]))
        else:
            merged.append(span)

    return merged


# ============================================================
# 实现
# ============================================================


def _length_function_of(
        config: RecursiveChunkingConfig,
) -> Callable[[str], int]:
    """
    按配置选长度函数。

    静态的两种直接查表；"model" 要按 config.length_tokenizer 现场
    构造一个闭包。每次新建闭包不会重复加载 tokenizer ——
    tokenization.tokenizer_for 按名字做了 lru_cache。

    length_tokenizer 缺失时**显式报错**而不是回落到 tiktoken：两者
    差 16%~33%，静默回落等于让人以为按模型上限量过了，而那正是本
    取值要避免的事。
    """

    if config.length_function != "model":
        return _LENGTH_FUNCTIONS[config.length_function]

    if not config.length_tokenizer:
        raise ValueError(
            "length_function='model' 需要 length_tokenizer 指明用哪个"
            "模型的词表，但没有给。该值由流水线按本分支的 embedder "
            "自动填入（见 pipeline/nodes/chunking.py）—— 手工构造配置"
            "时请自己给。"
        )

    model_name = config.length_tokenizer

    return lambda text: model_token_length(text, model_name)


class TxtRecursiveChunker(TextDocumentChunker):
    """递归切分器。

    键（name）继承自 TextDocumentChunker —— 与解析器那边
    get_parser_key() 放在 TextDocumentParser 上同理，
    具体实现不重复声明。
    """

    config_model = RecursiveChunkingConfig

    async def execute_chunk(
            self,
            request: ChunkRequest,
    ) -> list[ChunkSegment]:
        """按配置递归切分正文，返回裸文本段。"""

        config = self.config_model.model_validate(request.config)
        length_fn = _length_function_of(config)

        text = request.text

        if not text:
            return []

        spans = _split_text(
            text,
            list(config.separators),
            length_fn=length_fn,
            chunk_size=config.chunk_size,
            keep_separator=config.keep_separator,
            is_regex=config.is_separator_regex,
        )

        spans = merge_spans(
            text,
            spans,
            length_fn=length_fn,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        spans = _absorb_small(
            text,
            spans,
            length_fn=length_fn,
            min_chunk_size=config.min_chunk_size,
        )

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
                    # 两种 token 度量都要填 token_count，char 才留空。
                    # 复用已解析的 length_fn 而不是再写一遍分支 ——
                    # 原来这里写死 == "token"，加了第三种度量之后新取值
                    # 会静默变成 None（token_count 恒为空）。
                    token_count=(
                        length_fn(piece)
                        if config.length_function != "char"
                        else None
                    ),
                )
            )

        return segments
