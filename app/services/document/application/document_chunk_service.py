"""
单条 Chunk 的查询契约：只拿 `document_chunk_id` 取一块。

为什么需要它（谁在问）
----------------------
"这块 chunk 到底写了什么"有两个读者，**都不在版本页上**：

    检索评测的阶段 trace   逐层追踪金标时，`gold_chunk_ids` 只是一串
                           UUID；看不到正文就答不出"它为什么没被召回"
                          （是语义不匹配，还是正文里根本没有这个说法）
    带 preview_chars 的列表 列表接口可能返回被截断的 content
                          （`content_truncated=true`），要全文只能按 id 回查

契约位置
--------
`document_type_definition.yaml` 的 `GET /document-chunks/{document_chunk_id}`
—— 那里明确写着 "does not require document_id, document_version_id, or
document_processor_definition_id to be supplied by the caller"。
本服务就是它落地的地方。

与 VersionChunkService 的边界
----------------------------
那个服务是**产物视角**：按 (版本, retrieval) 看"这一路切出了什么"，
它的 chunk 归属由 chunker 身份推导，所以必须先有一个 version 与一个
retrieval 才问得出来。

这个服务是**单块视角**：手里已经有 id 了 —— 那个 id 是从别处（评测的
金标、列表行）拿到的。再要求调用方补上版本与 retrieval 是多余的，
而且评测侧**根本没有**这两个 id（`EvalRunCase` 里没有）。

只读，无副作用。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService
from app.services.document.domain.schemas.version_chunks import (
    DocumentChunkSummary,
)

__all__ = ["DocumentChunkService"]


class DocumentChunkService(BaseService):
    """按 chunk id 取单条 Chunk 的只读查询。"""

    @abstractmethod
    async def get_chunk(
            self,
            document_chunk_id: UUID,
            *,
            preview_chars: int | None = None,
    ) -> DocumentChunkSummary:
        """
        按 `document_chunk_id` 取一块 chunk。

        preview_chars:
            只返回 content 的前 N 个字符，并用 `content_truncated` 标记。
            默认 **None = 全文** —— 这个端点的存在意义就是"看不到正文时
            按 id 回查"，默认截断会把它变得没用。要给防护的调用方
            （比如正文可能是整篇几十万字的"不分块"配置）自己显式传。

        Raises:
            NotFoundError: 该 chunk 不存在（含 id 合法但已随重跑被清掉）。
        """
        raise NotImplementedError
