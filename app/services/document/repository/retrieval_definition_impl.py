"""
PostgreSQL implementation of DocumentRetrievalDefinitionRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select

from ..domain.entities import DocumentRetrievalDefinition
from ..domain.schemas.enum import (
    DefinitionStatus,
    RetrievalType,
)
from ..models.sqlalchemy.document_retrieval_definition import (
    DocumentRetrievalDefinition as DocumentRetrievalDefinitionModel,
)
from .retrieval_definition import (
    DocumentRetrievalDefinitionRepository,
)

class DocumentRetrievalDefinitionRepositoryImpl(
    DocumentRetrievalDefinitionRepository
):

    # ============================================================
    # Create
    # ============================================================

    async def add(
            self,
            retrieval: DocumentRetrievalDefinition,
    ) -> DocumentRetrievalDefinition:

        model = DocumentRetrievalDefinitionModel(
            document_retrieval_definition_id=(
                retrieval.retrieval_definition_id
            ),
            document_type_definition_id=(
                retrieval.document_type_definition_id
            ),
            retrieval_type=retrieval.retrieval_type,
            chunker_name=retrieval.chunker_name,
            chunker_config=dict(retrieval.chunker_config),
            retriever_name=retrieval.retriever_name,
            retriever_config=dict(retrieval.retriever_config),
            storage_backend=retrieval.storage_backend,
            storage_config=dict(retrieval.storage_config),
            status=retrieval.status,
            created_by=retrieval.created_by,
            created_at=retrieval.created_at,
            updated_at=retrieval.updated_at,
            deleted_at=retrieval.deleted_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    # ============================================================
    # Read
    # ============================================================

    async def get(
            self,
            retrieval_definition_id: UUID,
    ) -> DocumentRetrievalDefinition | None:

        stmt = select(DocumentRetrievalDefinitionModel).where(
            DocumentRetrievalDefinitionModel
            .document_retrieval_definition_id
            == retrieval_definition_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def get_by_retrieval_type(
            self,
            document_type_definition_id: UUID,
            retrieval_type: RetrievalType,
            *,
            include_deleted: bool = True,
    ) -> DocumentRetrievalDefinition | None:

        stmt = select(DocumentRetrievalDefinitionModel).where(
            DocumentRetrievalDefinitionModel
            .document_type_definition_id
            == document_type_definition_id,
            DocumentRetrievalDefinitionModel.retrieval_type
            == retrieval_type,
        )

        if not include_deleted:
            stmt = stmt.where(
                DocumentRetrievalDefinitionModel.status
                == DefinitionStatus.ACTIVE
            )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    # ============================================================
    # Update
    # ============================================================

    async def update(
            self,
            retrieval: DocumentRetrievalDefinition,
    ) -> DocumentRetrievalDefinition:

        stmt = select(DocumentRetrievalDefinitionModel).where(
            DocumentRetrievalDefinitionModel
            .document_retrieval_definition_id
            == retrieval.retrieval_definition_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                "DocumentRetrievalDefinition not found: "
                f"{retrieval.retrieval_definition_id}"
            )

        model.chunker_name = retrieval.chunker_name
        model.chunker_config = dict(retrieval.chunker_config)
        model.retriever_name = retrieval.retriever_name
        model.retriever_config = dict(retrieval.retriever_config)
        model.storage_backend = retrieval.storage_backend
        model.storage_config = dict(retrieval.storage_config)
        model.status = retrieval.status
        model.deleted_at = retrieval.deleted_at
        model.updated_at = retrieval.updated_at

        await self._session.flush()
        await self._session.refresh(model)

        return self._to_entity(model)

    # ============================================================
    # List
    # ============================================================

    async def list(
            self,
            *,
            document_type_definition_id: UUID | None = None,
            retrieval_type: RetrievalType | None = None,
            status: DefinitionStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentRetrievalDefinition], int]:

        conditions = []

        if document_type_definition_id is not None:
            conditions.append(
                DocumentRetrievalDefinitionModel
                .document_type_definition_id
                == document_type_definition_id
            )

        if retrieval_type is not None:
            conditions.append(
                DocumentRetrievalDefinitionModel.retrieval_type
                == retrieval_type
            )

        if status is not None:
            conditions.append(
                DocumentRetrievalDefinitionModel.status == status
            )

        count_stmt = (
            select(func.count())
            .select_from(DocumentRetrievalDefinitionModel)
        )
        if conditions:
            count_stmt = count_stmt.where(*conditions)
        total = (
            await self._session.execute(count_stmt)
        ).scalar_one()

        page_stmt = select(DocumentRetrievalDefinitionModel)
        if conditions:
            page_stmt = page_stmt.where(*conditions)
        page_stmt = (
            page_stmt
            .order_by(
                DocumentRetrievalDefinitionModel
                .retrieval_type.asc(),
                DocumentRetrievalDefinitionModel
                .document_retrieval_definition_id.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models], int(total)

    async def list_all_by_type_definition(
            self,
            document_type_definition_id: UUID,
            *,
            include_deleted: bool = False,
    ) -> list[DocumentRetrievalDefinition]:

        stmt = select(DocumentRetrievalDefinitionModel).where(
            DocumentRetrievalDefinitionModel
            .document_type_definition_id
            == document_type_definition_id
        )

        if not include_deleted:
            stmt = stmt.where(
                DocumentRetrievalDefinitionModel.status
                == DefinitionStatus.ACTIVE
            )

        stmt = stmt.order_by(
            DocumentRetrievalDefinitionModel.retrieval_type.asc()
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    async def list_by_ids(
            self,
            retrieval_definition_ids: list[UUID],
    ) -> list[DocumentRetrievalDefinition]:

        if not retrieval_definition_ids:
            return []

        stmt = select(DocumentRetrievalDefinitionModel).where(
            DocumentRetrievalDefinitionModel
            .document_retrieval_definition_id
            .in_(retrieval_definition_ids)
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _to_entity(
            model: DocumentRetrievalDefinitionModel,
    ) -> DocumentRetrievalDefinition:

        return DocumentRetrievalDefinition(
            retrieval_definition_id=(
                model.document_retrieval_definition_id
            ),
            document_type_definition_id=(
                model.document_type_definition_id
            ),
            retrieval_type=model.retrieval_type,
            chunker_name=model.chunker_name,
            chunker_config=dict(model.chunker_config),
            retriever_name=model.retriever_name,
            retriever_config=dict(model.retriever_config),
            storage_backend=model.storage_backend,
            storage_config=dict(model.storage_config),
            status=model.status,
            created_by=model.created_by,
            created_at=model.created_at,
            updated_at=model.updated_at,
            deleted_at=model.deleted_at,
        )
