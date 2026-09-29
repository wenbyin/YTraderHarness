"""
PostgreSQL implementation of DocumentIndexBindingRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select

from ..domain.entities import DocumentIndexBinding
from ..models.sqlalchemy.document_index_binding import (
    DocumentIndexBinding as DocumentIndexBindingModel,
)
from .document_index_binding import (
    DocumentIndexBindingRepository,
)

class DocumentIndexBindingRepositoryImpl(
    DocumentIndexBindingRepository
):

    # ============================================================
    # Create
    # ============================================================

    async def add_many(
            self,
            bindings: list[DocumentIndexBinding],
    ) -> None:

        if not bindings:
            return

        self._session.add_all([
            DocumentIndexBindingModel(
                document_index_binding_id=(
                    binding.document_index_binding_id
                ),
                document_retrieval_definition_id=(
                    binding.document_retrieval_definition_id
                ),
                backend_type=binding.backend_type,
                index_identifier=binding.index_identifier,
                created_at=binding.created_at,
                updated_at=binding.updated_at,
            )
            for binding in bindings
        ])

        await self._session.flush()

    # ============================================================
    # Read
    # ============================================================

    async def get(
            self,
            document_index_binding_id: UUID,
    ) -> DocumentIndexBinding | None:

        stmt = select(DocumentIndexBindingModel).where(
            DocumentIndexBindingModel.document_index_binding_id
            == document_index_binding_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def get_by_instance(
            self,
            document_retrieval_definition_id: UUID,
            backend_type: str,
            index_identifier: str,
    ) -> DocumentIndexBinding | None:

        stmt = select(DocumentIndexBindingModel).where(
            DocumentIndexBindingModel
            .document_retrieval_definition_id
            == document_retrieval_definition_id,
            DocumentIndexBindingModel.backend_type == backend_type,
            DocumentIndexBindingModel.index_identifier
            == index_identifier,
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def list_by_retrieval_definition(
            self,
            document_retrieval_definition_id: UUID,
    ) -> list[DocumentIndexBinding]:

        stmt = (
            select(DocumentIndexBindingModel)
            .where(
                DocumentIndexBindingModel
                .document_retrieval_definition_id
                == document_retrieval_definition_id
            )
            .order_by(
                DocumentIndexBindingModel.backend_type.asc(),
                DocumentIndexBindingModel.index_identifier.asc(),
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    # ============================================================
    # Update
    # ============================================================

    async def update(
            self,
            binding: DocumentIndexBinding,
    ) -> DocumentIndexBinding:

        stmt = select(DocumentIndexBindingModel).where(
            DocumentIndexBindingModel.document_index_binding_id
            == binding.document_index_binding_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                "DocumentIndexBinding not found: "
                f"{binding.document_index_binding_id}"
            )

        model.backend_type = binding.backend_type
        model.index_identifier = binding.index_identifier
        model.updated_at = binding.updated_at

        await self._session.flush()
        await self._session.refresh(model)

        return self._to_entity(model)

    # ============================================================
    # Delete
    # ============================================================

    async def delete_by_retrieval_definitions(
            self,
            retrieval_definition_ids: list[UUID],
    ) -> None:

        if not retrieval_definition_ids:
            return

        await self._session.execute(
            delete(DocumentIndexBindingModel).where(
                DocumentIndexBindingModel
                .document_retrieval_definition_id
                .in_(retrieval_definition_ids)
            )
        )

        await self._session.flush()

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _to_entity(
            model: DocumentIndexBindingModel,
    ) -> DocumentIndexBinding:

        return DocumentIndexBinding(
            document_index_binding_id=(
                model.document_index_binding_id
            ),
            document_retrieval_definition_id=(
                model.document_retrieval_definition_id
            ),
            backend_type=model.backend_type,
            index_identifier=model.index_identifier,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
