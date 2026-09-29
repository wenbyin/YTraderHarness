"""
PostgreSQL implementation of DocumentVersionRetrievalBindingRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import DocumentVersionRetrievalBinding
from ..models.sqlalchemy.document_version_retrieval_binding import (
    DocumentVersionRetrievalBinding as DocumentVersionRetrievalBindingModel,
)
from .document_version_retrieval_binding import (
    DocumentVersionRetrievalBindingRepository,
)

class DocumentVersionRetrievalBindingRepositoryImpl(
    DocumentVersionRetrievalBindingRepository
):
    """
    upsert 依赖表上的
    uq_document_version_retrieval_binding
        (document_version_id, document_retrieval_definition_id)
    约束。重复处理同一版本时更新既有行，不产生重复绑定。
    """

    # ============================================================
    # Create / Update
    # ============================================================

    async def upsert(
            self,
            binding: DocumentVersionRetrievalBinding,
    ) -> DocumentVersionRetrievalBinding:

        stmt = insert(
            DocumentVersionRetrievalBindingModel
        ).values(
            document_version_retrieval_binding_id=(
                binding.document_version_retrieval_binding_id
            ),
            document_version_id=binding.document_version_id,
            document_retrieval_definition_id=(
                binding.document_retrieval_definition_id
            ),
            document_index_binding_id=(
                binding.document_index_binding_id
            ),
            processing_status=binding.processing_status,
            error_message=binding.error_message,
            created_at=binding.created_at,
            updated_at=binding.updated_at,
        )

        stmt = stmt.on_conflict_do_update(
            constraint="uq_document_version_retrieval_binding",
            set_={
                "document_index_binding_id": (
                    stmt.excluded.document_index_binding_id
                ),
                "processing_status": (
                    stmt.excluded.processing_status
                ),
                "error_message": stmt.excluded.error_message,
                "updated_at": stmt.excluded.updated_at,
            },
        ).returning(DocumentVersionRetrievalBindingModel)

        model = (
            await self._session.execute(stmt)
        ).scalar_one()

        return self._to_entity(model)

    # ============================================================
    # Read
    # ============================================================

    async def get(
            self,
            document_version_retrieval_binding_id: UUID,
    ) -> DocumentVersionRetrievalBinding | None:

        stmt = select(DocumentVersionRetrievalBindingModel).where(
            DocumentVersionRetrievalBindingModel
            .document_version_retrieval_binding_id
            == document_version_retrieval_binding_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def get_by_version_and_retrieval(
            self,
            document_version_id: UUID,
            document_retrieval_definition_id: UUID,
    ) -> DocumentVersionRetrievalBinding | None:

        stmt = select(DocumentVersionRetrievalBindingModel).where(
            DocumentVersionRetrievalBindingModel.document_version_id
            == document_version_id,
            DocumentVersionRetrievalBindingModel
            .document_retrieval_definition_id
            == document_retrieval_definition_id,
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def list_by_version(
            self,
            document_version_id: UUID,
    ) -> list[DocumentVersionRetrievalBinding]:

        stmt = (
            select(DocumentVersionRetrievalBindingModel)
            .where(
                DocumentVersionRetrievalBindingModel.document_version_id
                == document_version_id
            )
            .order_by(
                DocumentVersionRetrievalBindingModel.created_at.asc(),
                DocumentVersionRetrievalBindingModel
                .document_retrieval_definition_id.asc(),
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    async def list_by_versions(
            self,
            document_version_ids: list[UUID],
    ) -> list[DocumentVersionRetrievalBinding]:

        if not document_version_ids:
            return []

        stmt = (
            select(DocumentVersionRetrievalBindingModel)
            .where(
                DocumentVersionRetrievalBindingModel.document_version_id
                .in_(document_version_ids)
            )
            .order_by(
                DocumentVersionRetrievalBindingModel.document_version_id
                .asc(),
                DocumentVersionRetrievalBindingModel.created_at.asc(),
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    # ============================================================
    # Delete
    # ============================================================

    async def delete_by_version(
            self,
            document_version_id: UUID,
    ) -> None:

        await self._session.execute(
            delete(DocumentVersionRetrievalBindingModel).where(
                DocumentVersionRetrievalBindingModel.document_version_id
                == document_version_id
            )
        )

        await self._session.flush()

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _to_entity(
            model: DocumentVersionRetrievalBindingModel,
    ) -> DocumentVersionRetrievalBinding:

        return DocumentVersionRetrievalBinding(
            document_version_retrieval_binding_id=(
                model.document_version_retrieval_binding_id
            ),
            document_version_id=model.document_version_id,
            document_retrieval_definition_id=(
                model.document_retrieval_definition_id
            ),
            document_index_binding_id=(
                model.document_index_binding_id
            ),
            processing_status=model.processing_status,
            error_message=model.error_message,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
