"""
PostgreSQL implementation of DocumentChunkRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from ..domain.entities import DocumentChunk
from ..models.sqlalchemy.document_chunk import (
    DocumentChunk as DocumentChunkModel,
)
from .document_chunk import DocumentChunkRepository

class DocumentChunkRepositoryImpl(DocumentChunkRepository):

    # ============================================================
    # Create
    # ============================================================

    async def add_many(
            self,
            chunks: list[DocumentChunk],
    ) -> None:

        if not chunks:
            return

        self._session.add_all(
            [
                DocumentChunkModel(
                    document_chunk_id=chunk.document_chunk_id,
                    document_version_id=(
                        chunk.document_version_id
                    ),
                    chunker_key=chunk.chunker_key,
                    chunker_config=dict(chunk.chunker_config),
                    document_processor_definition_id=(
                        chunk.document_processor_definition_id
                    ),
                    chunk_index=chunk.chunk_index,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    token_count=chunk.token_count,
                    char_count=chunk.char_count,
                    locator=dict(chunk.locator or {}),
                    metadata_=dict(chunk.metadata),
                )
                for chunk in chunks
            ]
        )

        await self._session.flush()

    # ============================================================
    # Read
    # ============================================================

    async def get(
            self,
            document_chunk_id: UUID,
    ) -> DocumentChunk | None:

        stmt = (
            select(DocumentChunkModel)
            .where(
                DocumentChunkModel.document_chunk_id
                == document_chunk_id
            )
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def list(
            self,
            *,
            document_version_id: UUID | None = None,
            chunker_key: str | None = None,
            chunker_config: dict[str, Any] | None = None,
            processor_definition_id: UUID | None = None,
            chunk_index: int | None = None,
            keyword: str | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentChunk], int]:

        conditions = []

        if document_version_id is not None:
            conditions.append(
                DocumentChunkModel.document_version_id
                == document_version_id
            )

        if chunker_key is not None:
            # 版本 × chunker_key 是唯一约束前缀，表上也有复合索引
            # （ix_document_chunk_version_chunker），所以这个过滤走索引。
            conditions.append(
                DocumentChunkModel.chunker_key == chunker_key
            )

        if chunker_config is not None:
            # jsonb 的 = 是语义相等（键序无关），与写入侧的归一化形式
            # 一致，因此这里能直接比。
            conditions.append(
                DocumentChunkModel.chunker_config
                == dict(chunker_config)
            )

        if processor_definition_id is not None:
            conditions.append(
                DocumentChunkModel.document_processor_definition_id
                == processor_definition_id
            )

        if chunk_index is not None:
            conditions.append(
                DocumentChunkModel.chunk_index == chunk_index
            )

        if keyword:
            conditions.append(
                DocumentChunkModel.content.ilike(
                    self._like_pattern(keyword),
                    escape="\\",
                )
            )

        # --------------------------------------------------------
        # Total count
        # --------------------------------------------------------

        count_stmt = (
            select(func.count())
            .select_from(DocumentChunkModel)
        )

        if conditions:
            count_stmt = count_stmt.where(*conditions)

        total = (
            await self._session.execute(count_stmt)
        ).scalar_one()

        # --------------------------------------------------------
        # Current page
        # --------------------------------------------------------

        page_stmt = select(DocumentChunkModel)

        if conditions:
            page_stmt = page_stmt.where(*conditions)

        page_stmt = (
            page_stmt
            .order_by(
                DocumentChunkModel.document_version_id.asc(),
                # 一个版本下可能有多份切分产出（不同 chunker），
                # 不按 chunker_key 归拢的话它们会按 chunk_index 交错。
                DocumentChunkModel.chunker_key.asc(),
                DocumentChunkModel.chunk_index.asc(),
                DocumentChunkModel.document_chunk_id.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return (
            [self._to_entity(model) for model in models],
            int(total),
        )

    async def list_ids(
            self,
            document_version_id: UUID,
            processor_definition_ids: list[UUID],
    ) -> list[UUID]:

        if not processor_definition_ids:
            return []

        stmt = (
            select(DocumentChunkModel.document_chunk_id)
            .where(
                DocumentChunkModel.document_version_id
                == document_version_id,
                DocumentChunkModel.document_processor_definition_id
                .in_(processor_definition_ids),
            )
            .order_by(
                DocumentChunkModel.chunk_index.asc()
            )
        )

        return list(
            (
                await self._session.execute(stmt)
            ).scalars().all()
        )

    async def list_by_ids(
            self,
            document_chunk_ids: list[UUID],
    ) -> list[DocumentChunk]:

        if not document_chunk_ids:
            # 空列表不能直接丢给 in_()：SQLAlchemy 会退化成
            # "WHERE false" 之类的表达式，虽然结果对但语义含糊；
            # 更重要的是挡住"调用方忘了过滤，传了个空集合进来"
            # 被误当成"不过滤"。
            return []

        stmt = select(DocumentChunkModel).where(
            DocumentChunkModel.document_chunk_id.in_(
                document_chunk_ids
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(model) for model in models]

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _like_pattern(
            value: str,
    ) -> str:
        """转义 LIKE 通配符后构造包含匹配 pattern。"""

        escaped = (
            value
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        return f"%{escaped}%"

    @staticmethod
    def _to_entity(
            model: DocumentChunkModel,
    ) -> DocumentChunk:

        return DocumentChunk(
            document_chunk_id=model.document_chunk_id,
            document_version_id=model.document_version_id,
            chunker_key=model.chunker_key,
            chunker_config=dict(model.chunker_config or {}),
            document_processor_definition_id=None,
            chunk_index=model.chunk_index,
            start_index=model.start_index,
            end_index=model.end_index,
            content=model.content,
            content_hash=model.content_hash,
            token_count=model.token_count,
            char_count=model.char_count,
            locator=dict(model.locator or {}),
            metadata=dict(model.metadata_),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
