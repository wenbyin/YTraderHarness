"""
PostgreSQL implementation of DocumentProcessorDefinitionRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select

from ..domain.entities import DocumentProcessorDefinition
from ..domain.schemas.enum import DefinitionStatus
from ..models.sqlalchemy.document_processor_definition import (
    DocumentProcessorDefinition as DocumentProcessorDefinitionModel,
)
from .processor_definition import (
    DocumentProcessorDefinitionRepository,
)

class DocumentProcessorDefinitionRepositoryImpl(
    DocumentProcessorDefinitionRepository
):

    # ============================================================
    # Create
    # ============================================================

    async def add(
            self,
            processor: DocumentProcessorDefinition,
    ) -> DocumentProcessorDefinition:

        model = DocumentProcessorDefinitionModel(
            document_processor_definition_id=(
                processor.document_processor_definition_id
            ),
            document_type_definition_id=(
                processor.document_type_definition_id
            ),
            file_extension=processor.file_extension,
            mime_types=list(processor.mime_types),
            loader_name=processor.loader_name,
            loader_config=dict(processor.loader_config),
            processor_name=processor.processor_name,
            processor_config=dict(processor.processor_config),
            status=processor.status,
            created_by=processor.created_by,
            created_at=processor.created_at,
            updated_at=processor.updated_at,
            deleted_at=processor.deleted_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    # ============================================================
    # Read
    # ============================================================

    async def get(
            self,
            processor_definition_id: UUID,
    ) -> DocumentProcessorDefinition | None:

        stmt = select(DocumentProcessorDefinitionModel).where(
            DocumentProcessorDefinitionModel
            .document_processor_definition_id
            == processor_definition_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def get_by_extension(
            self,
            document_type_definition_id: UUID,
            file_extension: str,
            *,
            include_deleted: bool = True,
    ) -> DocumentProcessorDefinition | None:

        stmt = select(DocumentProcessorDefinitionModel).where(
            DocumentProcessorDefinitionModel
            .document_type_definition_id
            == document_type_definition_id,
            DocumentProcessorDefinitionModel.file_extension
            == file_extension,
        )

        if not include_deleted:
            stmt = stmt.where(
                DocumentProcessorDefinitionModel.status
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
            processor: DocumentProcessorDefinition,
    ) -> DocumentProcessorDefinition:

        stmt = select(DocumentProcessorDefinitionModel).where(
            DocumentProcessorDefinitionModel
            .document_processor_definition_id
            == processor.document_processor_definition_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                "DocumentProcessorDefinition not found: "
                f"{processor.document_processor_definition_id}"
            )

        model.file_extension = processor.file_extension
        model.mime_types = list(processor.mime_types)
        model.loader_name = processor.loader_name
        model.loader_config = dict(processor.loader_config)
        model.processor_name = processor.processor_name
        model.processor_config = dict(processor.processor_config)
        model.status = processor.status
        model.deleted_at = processor.deleted_at
        model.updated_at = processor.updated_at

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
            file_extension: str | None = None,
            status: DefinitionStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentProcessorDefinition], int]:

        conditions = []

        if document_type_definition_id is not None:
            conditions.append(
                DocumentProcessorDefinitionModel
                .document_type_definition_id
                == document_type_definition_id
            )

        if file_extension:
            conditions.append(
                DocumentProcessorDefinitionModel.file_extension
                == file_extension
            )

        if status is not None:
            conditions.append(
                DocumentProcessorDefinitionModel.status == status
            )

        count_stmt = (
            select(func.count())
            .select_from(DocumentProcessorDefinitionModel)
        )
        if conditions:
            count_stmt = count_stmt.where(*conditions)
        total = (
            await self._session.execute(count_stmt)
        ).scalar_one()

        page_stmt = select(DocumentProcessorDefinitionModel)
        if conditions:
            page_stmt = page_stmt.where(*conditions)
        page_stmt = (
            page_stmt
            .order_by(
                DocumentProcessorDefinitionModel.file_extension.asc(),
                DocumentProcessorDefinitionModel
                .document_processor_definition_id.asc(),
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
    ) -> list[DocumentProcessorDefinition]:

        stmt = select(DocumentProcessorDefinitionModel).where(
            DocumentProcessorDefinitionModel
            .document_type_definition_id
            == document_type_definition_id
        )

        if not include_deleted:
            stmt = stmt.where(
                DocumentProcessorDefinitionModel.status
                == DefinitionStatus.ACTIVE
            )

        stmt = stmt.order_by(
            DocumentProcessorDefinitionModel.file_extension.asc()
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
            model: DocumentProcessorDefinitionModel,
    ) -> DocumentProcessorDefinition:

        return DocumentProcessorDefinition(
            document_processor_definition_id=(
                model.document_processor_definition_id
            ),
            document_type_definition_id=(
                model.document_type_definition_id
            ),
            file_extension=model.file_extension,
            mime_types=list(model.mime_types),
            loader_name=model.loader_name,
            loader_config=dict(model.loader_config),
            processor_name=model.processor_name,
            processor_config=dict(model.processor_config),
            status=model.status,
            created_by=model.created_by,
            created_at=model.created_at,
            updated_at=model.updated_at,
            deleted_at=model.deleted_at,
        )
