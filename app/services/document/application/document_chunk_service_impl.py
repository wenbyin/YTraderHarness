"""
单条 Chunk 查询的实现。

一次请求一跳查询：chunk id -> document_chunk 行。没有版本校验、没有
retrieval 解析 —— 那些是"产物视角"才需要的（见契约文件的说明）。

⚠️ 取不到就是 404，**不返回空壳**
-------------------------------
不存在（id 打错、或重跑把它清掉了）与"存在但正文为空"是两回事：
前者调用方该去查自己手里的 id 是哪来的，后者是数据本身的问题。
返回空壳会让这两种情况看起来一样。
"""

from __future__ import annotations

from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext

from ..domain.schemas.version_chunks import DocumentChunkSummary
from ..repository.document_chunk import DocumentChunkRepository
from .chunk_mapping import to_chunk_summary
from .document_chunk_service import DocumentChunkService

__all__ = ["DocumentChunkServiceImpl"]


class DocumentChunkServiceImpl(DocumentChunkService):

    def __init__(
            self,
            *,
            chunk_repository: DocumentChunkRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._chunk_repository = chunk_repository

    async def get_chunk(
            self,
            document_chunk_id: UUID,
            *,
            preview_chars: int | None = None,
    ) -> DocumentChunkSummary:

        chunk = await self._chunk_repository.get(document_chunk_id)

        if chunk is None:
            raise NotFoundError(
                f"DocumentChunk not found: {document_chunk_id}"
            )

        return to_chunk_summary(chunk, preview_chars)
