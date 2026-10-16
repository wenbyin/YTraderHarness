"""
chunk 实体 -> 对外契约的**共用映射**。

为什么不放在某一个 service 里
----------------------------
映射只有一个（`DocumentChunk` -> `DocumentChunkSummary`），但读它的入口
有两个，而且分属两个服务：

    VersionChunkService   版本页：某个 (版本, retrieval) 下的 chunk 列表
    DocumentChunkService  单条：只拿 document_chunk_id 取一块

放成某个服务的私有静态方法，另一个服务要用就只能去 import 对方的
**实现类** —— 两个 impl 互相依赖，而它们本来是并列的。所以提到模块级。

`preview_chars` 的语义（两条端点共用，别各写一份）
-------------------------------------------------
只截 `content`，**不动 `char_count`** —— 后者是截断前的真实长度，前端
要靠它判断"这条到底有多大 / 还有多少没看到"。把 `char_count` 也改成
截断后的长度，前端就再也看不出被截过。
"""

from __future__ import annotations

from ..domain.entities.document_chunk import DocumentChunk
from ..domain.schemas.version_chunks import DocumentChunkSummary

__all__ = ["to_chunk_summary"]


def to_chunk_summary(
        chunk: DocumentChunk,
        preview_chars: int | None = None,
) -> DocumentChunkSummary:
    """一块 chunk 的对外形态。`preview_chars` 留空即全文。"""

    content = chunk.content
    truncated = False

    if preview_chars is not None and len(content) > preview_chars:
        content = content[:preview_chars]
        truncated = True

    return DocumentChunkSummary(
        document_chunk_id=chunk.document_chunk_id,
        chunk_index=chunk.chunk_index,
        content=content,
        content_truncated=truncated,
        char_count=chunk.char_count,
        token_count=chunk.token_count,
        start_index=chunk.start_index,
        end_index=chunk.end_index,
        content_hash=chunk.content_hash,
        locator=chunk.locator,
        metadata=dict(chunk.metadata or {}),
        created_at=chunk.created_at,
    )
