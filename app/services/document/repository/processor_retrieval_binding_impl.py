"""
PostgreSQL implementation of DocumentProcessorRetrievalBindingRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import delete, select

from ..domain.entities import DocumentProcessorRetrievalBinding
from ..models.sqlalchemy.document_processor_definition import (
    DocumentProcessorDefinition as DocumentProcessorDefinitionModel,
)
from ..models.sqlalchemy.document_processor_retrieval_binding import (
    DocumentProcessorRetrievalBinding as BindingModel,
)
from .processor_retrieval_binding import (
    DocumentProcessorRetrievalBindingRepository,
)

class DocumentProcessorRetrievalBindingRepositoryImpl(
    DocumentProcessorRetrievalBindingRepository
):

    # ============================================================
    # Replace
    # ============================================================

    async def replace_for_processor(
            self,
            processor_definition_id: UUID,
            retrieval_definition_ids: list[UUID],
    ) -> None:

        await self.delete_by_processors([processor_definition_id])

        if not retrieval_definition_ids:
            return

        now = datetime.now(timezone.utc)

        self._session.add_all([
            BindingModel(
                document_processor_retrieval_binding_id=uuid4(),
                document_processor_definition_id=(
                    processor_definition_id
                ),
                document_retrieval_definition_id=(
                    retrieval_definition_id
                ),
                created_at=now,
                updated_at=now,
            )
            for retrieval_definition_id in retrieval_definition_ids
        ])

        await self._session.flush()

    # ============================================================
    # Read
    # ============================================================

    async def list_by_processor(
            self,
            processor_definition_id: UUID,
    ) -> list[DocumentProcessorRetrievalBinding]:

        stmt = select(BindingModel).where(
            BindingModel.document_processor_definition_id
            == processor_definition_id
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    async def list_by_processors(
            self,
            processor_definition_ids: list[UUID],
    ) -> list[DocumentProcessorRetrievalBinding]:

        if not processor_definition_ids:
            return []

        stmt = select(BindingModel).where(
            BindingModel.document_processor_definition_id
            .in_(processor_definition_ids)
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    async def list_by_type_definition(
            self,
            document_type_definition_id: UUID,
    ) -> list[DocumentProcessorRetrievalBinding]:

        stmt = (
            select(BindingModel)
            .join(
                DocumentProcessorDefinitionModel,
                BindingModel.document_processor_definition_id
                == DocumentProcessorDefinitionModel
                .document_processor_definition_id,
            )
            .where(
                DocumentProcessorDefinitionModel
                .document_type_definition_id
                == document_type_definition_id
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    # ============================================================
    # Delete
    # ============================================================

    async def delete_by_processors(
            self,
            processor_definition_ids: list[UUID],
    ) -> None:

        if not processor_definition_ids:
            return

        await self._session.execute(
            delete(BindingModel).where(
                BindingModel.document_processor_definition_id
                .in_(processor_definition_ids)
            )
        )

        await self._session.flush()

    async def delete_by_retrievals(
            self,
            retrieval_definition_ids: list[UUID],
    ) -> None:

        if not retrieval_definition_ids:
            return

        await self._session.execute(
            delete(BindingModel).where(
                BindingModel.document_retrieval_definition_id
                .in_(retrieval_definition_ids)
            )
        )

        await self._session.flush()

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _to_entity(model: BindingModel) -> (
            DocumentProcessorRetrievalBinding):

        return DocumentProcessorRetrievalBinding(
            document_processor_retrieval_binding_id=(
                model.document_processor_retrieval_binding_id
            ),
            document_processor_definition_id=(
                model.document_processor_definition_id
            ),
            document_retrieval_definition_id=(
                model.document_retrieval_definition_id
            ),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
