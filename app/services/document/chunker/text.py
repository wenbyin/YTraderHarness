"""
文本分块器的中间基类（.txt 家族）。

与 parser / cleaner 的层级一一对应
----------------------------------
    parser/base.py    DocumentParser          role 基类：process() -> parse()
    parser/text.py    TextDocumentParser      中间基类：模板 parse() -> execute_parse()
    parser/txt.py     CommonTxtDocumentParser 具体实现：execute_parse()

    cleaner/base.py   DocumentCleaner         role 基类：process() -> clean()
    cleaner/text.py   TextDocumentCleaner     中间基类：模板 clean() -> execute_clean()
    cleaner/txt.py    TxtDocumentCleaner      具体实现：execute_clean()

    chunker/base.py   DocumentChunker         role 基类：process() -> chunk()
    chunker/text.py   TextDocumentChunker     中间基类：模板 chunk() -> execute_chunk()
    chunker/txt.py    TxtRecursiveChunker     具体实现：execute_chunk()

这一层做的是**组装**，与 cleaner 那边"包装"的分工同理：

    具体实现只回答"在哪儿切"（返回裸文本段 ChunkSegment），
    "把段落包成完整的 DocumentChunk 实体"（uuid / hash / chunk_index /
    字符计数 / 身份 id）这套共性活全在模板方法里。

好处不只是少写重复代码：document_chunk 表上有
    UNIQUE(document_version_id, document_retrieval_definition_id, chunk_index)
所以 chunk_index 必须**在本 retrieval 内从 0 连续**。让每个实现自己
编号，早晚会有人跳号或从 1 开始，然后落库撞唯一约束或者查出来缺块。
放在模板里统一编，实现就算返回了空段也影响不到编号的连续性。

键（name）也放在这一层，与 get_parser_key() 放在 TextDocumentParser
上同理。
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from app.services.document.domain.entities import DocumentChunk
from app.services.document.domain.schemas import ChunkRequest

from .base import DocumentChunker

__all__ = [
    "ChunkSegment",
    "TextDocumentChunker",
    "encoding_for",
    "merge_spans",
    "shrink_whitespace",
    "split_by_separator",
    "token_length",
]


# ============================================================
# 共用的度量与切分工具
# ============================================================
#
# 放中间层而不是各自的实现里：按 token 度量、按分隔符求 span 这两件事
# 是**整个 .txt 家族**共用的，具体实现只是拿它们组合出自己的策略。


# 分词与长度度量已提到中立模块 —— embedder 的稀疏实现也要用同一套，
# 放在这里会让 embedder 反向依赖 chunker 包（两个平级的 role 不该互相
# import）。这里 re-export 保持既有 import 路径不变。
from ..tokenization import encoding_for, token_length  # noqa: F401


def split_by_separator(
        text: str,
        separator: str,
        *,
        is_regex: bool,
        keep_separator: bool,
) -> list[tuple[int, int]]:
    """
    按**一个**分隔符把 text 切成若干 span（偏移相对 text 自身）。

    span 而不是子串：下游需要保留每一段在原文里的位置，
    才能写出可验证的 start_index / end_index。全程只搬区间，
    最后一步才切字符串。

    keep_separator=True 时分隔符跟随**前一段**，因此按序拼接可还原
    原文；开头的分隔符没有"前一段"可跟，会自成一个 span。
    """

    if not text:
        return []

    if separator == "":
        # 空分隔符 ≡ 不再切（保留原样的语义，不是"按字符硬切"）。
        return [(0, len(text))]

    pattern = separator if is_regex else re.escape(separator)
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [(0, len(text))]

    spans: list[tuple[int, int]] = []
    cursor = 0

    for match in matches:
        cut = match.end() if keep_separator else match.start()

        if cut > cursor:
            spans.append((cursor, cut))

        cursor = match.end()

    if cursor < len(text):
        spans.append((cursor, len(text)))

    return spans


def merge_spans(
        text: str,
        spans: list[tuple[int, int]],
        *,
        length_fn: Callable[[str], int],
        chunk_size: int,
        chunk_overlap: int,
) -> list[tuple[int, int]]:
    """
    把细碎的 span 贪心合并成尽量接近 chunk_size 的块。

    两个实现（递归切分、GraphRAG 句级累积）用的是同一套"攒到接近
    chunk_size 就吐一块"的逻辑，区别只在 length_fn（字符 / token）
    和 span 从哪来 —— 所以放在中间层共用，避免两份各自漂移。

    overlap 的做法：每 emit 一块之后，从队首丢掉若干段，直到剩余
    长度 <= chunk_overlap —— 剩下的那部分就构成与上一块的重叠。
    """

    chunks: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    current_len = 0

    for span in spans:

        span_len = length_fn(text[span[0]:span[1]])

        if current and current_len + span_len > chunk_size:
            chunks.append((current[0][0], current[-1][1]))

            while current and current_len > chunk_overlap:
                dropped = current.pop(0)
                current_len -= length_fn(
                    text[dropped[0]:dropped[1]]
                )

        current.append(span)
        current_len += span_len

    if current:
        chunks.append((current[0][0], current[-1][1]))

    return chunks


def shrink_whitespace(
        text: str,
        start: int,
        end: int,
) -> tuple[int, int]:
    """
    把区间两端是空白的部分收掉，返回新区间。

    只收两端，不动中间 —— 段落内部的缩进与空行是有意义的。
    """

    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def _locator_of() -> dict[str, Any]:
    """
    chunk 在源文件里的位置。

    ⚠️ 这三个键在**切分阶段填不了**，值为 None
    ------------------------------------------
    chunker 手上只有正文和字符偏移 —— 页码范围来自解析阶段的版面信息，
    章节路径来自结构识别，两者都不在它手里。

    填它们的是**阶段 8 的 chunk_enrichment**，它的职责原文就是
    「补定位信息（locator：页码、章节路径、表格坐标）」。

    那为什么现在就声明出来？因为数据库里 `locator` 是自由 dict，
    键的有无本身就是前端要处理的分支。先定下形状（永远是这三个键），
    前端按"值可能为 null"写就行，不必再处理"键可能不存在"。
    """

    return {
        "page_start": None,
        "page_end": None,
        "section_path": None,
    }


def _metadata_of(request: Any) -> dict[str, Any]:
    """
    chunk 的结构性字段。

    document_chunk 表里**没有**下列三个字段的列，按约定放 metadata：

        document_id                  所属 Document
        document_type_definition_id  所属文档类型
        parent_chunk_id              父 chunk（小块归大块）

    为什么不加列：它们要么能从 document_version_id 推导、要么目前
    没有任何实现会产出（分层切分还没做）。放 metadata 免迁移；
    等哪个真的需要"按它查询/连接"时，再提成正式列。

    ⚠️ 注意 metadata 里还有上游累积的文档级元数据（`**request.metadata`），
    目前那份是污染的 —— 见 chunking.py 的说明。这里只往里加键，
    不清理旧的。

    ⚠️ parent_chunk_id 目前**恒为 None**：没有任何 chunker 产出父子关系。
    声明它是为了让形状固定，不是为了假装已经有了。
    """

    return {
        # 上游累积的文档级元数据先落一份，阶段 8 的 enricher 在
        # 这个基础上补定位信息与派生字段。
        **dict(request.metadata),
        # 结构性字段放后面 —— 同名时以它们为准。
        "document_id": str(request.document_id),
        "document_type_definition_id": str(
            request.document_type_definition_id
        ),
        "parent_chunk_id": None,
    }


class ChunkSegment(BaseModel):
    """
    一段切好的文本 —— execute_chunk() 的产出。

    刻意**只带文本、偏移、可选的 token 数**，不带 id / hash /
    chunk_index：那些是组装阶段的事（见下面的模板方法），
    切分实现不该关心。

    start_index / end_index 是**相对输入正文**的字符偏移，且必须满足
    `request.text[start_index:end_index] == text` —— 模板据此保证
    落库的偏移能真实定位回原文。半开区间 [start, end)。
    """

    text: str

    start_index: int = Field(
        ge=0,
        description="在输入正文中的起始字符偏移（含）。",
    )

    end_index: int = Field(
        ge=0,
        description="在输入正文中的结束字符偏移（不含）。",
    )

    token_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "该段的 token 数。只有按 token 度量长短的实现才填得出来；"
            "按字符度量时留 None，不要瞎估。"
        ),
    )


class TextDocumentChunker(DocumentChunker, ABC):
    """纯文本分块器的中间基类。"""

    #: 与解析器的 TEXT_TXT_COMMON_PARSER、清洗器的
    #: TEXT_TXT_COMMON_CLEANER 并列的键。
    #: 一旦发布**永不复用** —— 它会冻结进 PipelineConfigSnapshot。
    name = "TEXT_TXT_RECURSIVE_CHUNKER"

    async def chunk(
            self,
            request: ChunkRequest,
    ) -> list[DocumentChunk]:
        """
        对外暴露的 chunk 方法（模板）。

        具体实现只写 execute_chunk()，实体组装在这里统一做。
        """

        segments = await self.execute_chunk(request)

        chunks: list[DocumentChunk] = []

        for segment in segments:

            # 空段直接丢掉：它会占掉一个 chunk_index，落库后表现为
            # "有一块查出来是空的"，而 chunk_index 会出现空洞。
            if not segment.text:
                continue

            chunks.append(
                DocumentChunk(
                    document_chunk_id=uuid4(),
                    document_version_id=(
                        request.document_version_id
                    ),
                    # 身份由实现自己盖 —— 调用方传不进来一个错的 key。
                    # 与 chunker_config 一起决定"这份 chunk 归谁用"：
                    # retrieval 定义里的同两项归一化后相等就用它。
                    chunker_key=self.get_key(),
                    # 这里存的是**调用方传来的那份**，而它必须是归一化过
                    # 的（见 ChunkRequest 的说明）—— 落库值与查询侧算出的
                    # 值必须逐字节一致，否则 retrieval 找不到自己的 chunk。
                    chunker_config=dict(request.config),
                    # document_processor_definition_id 留 None：
                    # 扇出分支的 payload 里没有它，补进去要改
                    # BranchInput，单独一轮做。
                    chunk_index=len(chunks),
                    start_index=segment.start_index,
                    end_index=segment.end_index,
                    content=segment.text,
                    content_hash=hashlib.sha256(
                        segment.text.encode("utf-8")
                    ).hexdigest(),
                    char_count=len(segment.text),
                    token_count=segment.token_count,
                    locator=_locator_of(),
                    metadata=_metadata_of(request),
                )
            )

        return chunks

    @abstractmethod
    async def execute_chunk(
            self,
            request: ChunkRequest,
    ) -> list[ChunkSegment]:
        """
        真正切分正文，返回裸文本段。

        只需保证：段内文本与 request.text 的对应偏移一致
        （template 会把它们原样写进 DocumentChunk 的
        content / start_index / end_index）。
        """
        raise NotImplementedError

