"""
PostgreSQL implementation of DocumentTypeDefinitionRepository.

Responsibilities
----------------
- CRUD operations against the document_type_definition table.
- Query definitions by id / type_key.

Non-responsibilities
--------------------
- Transaction commit / rollback
- Processor pipeline management
- Document / Chunk / Index

Transaction rule
----------------
This repository NEVER calls:

    session.commit()
    session.rollback()

Transaction lifecycle is owned by the application/service layer.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, or_, select

from ..domain.entities import DocumentTypeDefinition
from ..domain.schemas.enum import DefinitionStatus
from ..models.sqlalchemy.document_type_definition import (
    DocumentTypeDefinition as DocumentTypeDefinitionModel,
)
from .type_definition import (
    DocumentTypeDefinitionRepository,
)

class DocumentTypeDefinitionRepositoryImpl(
    DocumentTypeDefinitionRepository
):
    """
    PostgreSQL implementation of
    DocumentTypeDefinitionRepository.
    """

    # ============================================================
    # Create
    # ============================================================

    async def add(
            self,
            definition: DocumentTypeDefinition,
    ) -> DocumentTypeDefinition:
        """
        Create a DocumentTypeDefinition record.

        Only PostgreSQL is touched.
        No transaction commit is performed.
        """

        model = DocumentTypeDefinitionModel(
            document_type_definition_id=(
                definition.document_type_definition_id
            ),
            tenant_id=definition.tenant_id,
            type_key=definition.type_key,
            name=definition.name,
            description=definition.description,
            modality_type=definition.modality_type,
            analysis_mode=definition.analysis_mode,
            allow_update=definition.allow_update,
            version_mode=definition.version_mode,
            fusion_strategy=definition.fusion_strategy,
            fusion_config=dict(definition.fusion_config),
            rerank_strategy=definition.rerank_strategy,
            rerank_config=dict(definition.rerank_config),
            revision=definition.revision,
            metadata_=dict(definition.metadata),
            status=definition.status,
            created_at=definition.created_at,
            updated_at=definition.updated_at,
        )

        self._session.add(model)

        # Make INSERT visible in the current transaction.
        # Do NOT commit here.
        await self._session.flush()

        return self._to_entity(model)

    # ============================================================
    # Read
    # ============================================================

    async def get(
            self,
            document_type_definition_id: UUID,
    ) -> DocumentTypeDefinition | None:
        """
        Get by primary key.

        Both active and logically deleted records may be returned.
        """

        stmt = (
            select(DocumentTypeDefinitionModel)
            .where(
                DocumentTypeDefinitionModel
                .document_type_definition_id
                == document_type_definition_id
            )
        )

        result = await self._session.execute(stmt)

        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def get_many(
            self,
            document_type_definition_ids: list[UUID],
    ) -> list[DocumentTypeDefinition]:

        if not document_type_definition_ids:
            return []

        stmt = select(DocumentTypeDefinitionModel).where(
            DocumentTypeDefinitionModel
            .document_type_definition_id
            .in_(document_type_definition_ids)
        )

        result = await self._session.execute(stmt)

        return [
            self._to_entity(model)
            for model in result.scalars().all()
        ]

    async def get_by_type_key(
            self,
            type_key: str,
    ) -> DocumentTypeDefinition | None:
        """
        Get by unique type_key.

        Used for duplicate checks on create.
        """

        stmt = (
            select(DocumentTypeDefinitionModel)
            .where(
                DocumentTypeDefinitionModel.type_key
                == type_key
            )
        )

        result = await self._session.execute(stmt)

        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    # ============================================================
    # Update
    # ============================================================

    async def update(
            self,
            definition: DocumentTypeDefinition,
    ) -> DocumentTypeDefinition:
        """
        Update an existing DocumentTypeDefinition record.

        只操作 PostgreSQL。
        不 commit / rollback。
        """

        stmt = (
            select(DocumentTypeDefinitionModel)
            .where(
                DocumentTypeDefinitionModel
                .document_type_definition_id
                == definition.document_type_definition_id
            )
        )

        result = await self._session.execute(stmt)

        model = result.scalar_one_or_none()

        if model is None:
            raise ValueError(
                "DocumentTypeDefinition not found: "
                f"{definition.document_type_definition_id}"
            )

        # --------------------------------------------------------
        # Update mutable fields
        # --------------------------------------------------------

        model.name = definition.name
        model.description = definition.description

        # modality_type 创建后不可修改，这里不更新。
        model.analysis_mode = definition.analysis_mode

        model.allow_update = definition.allow_update
        model.version_mode = definition.version_mode
        model.fusion_strategy = definition.fusion_strategy
        model.fusion_config = dict(definition.fusion_config)
        model.rerank_strategy = definition.rerank_strategy
        model.rerank_config = dict(definition.rerank_config)
        model.revision = definition.revision
        model.metadata_ = dict(definition.metadata)
        model.status = definition.status
        model.updated_at = definition.updated_at

        # --------------------------------------------------------
        # Do NOT update:
        #
        # model.document_type_definition_id
        # model.type_key
        # model.tenant_id
        # model.created_at
        #
        # type_key 是业务唯一标识，创建后不可修改。
        # created_at 表示原始创建时间。
        # --------------------------------------------------------

        await self._session.flush()

        return self._to_entity(model)

    # ============================================================
    # List
    # ============================================================

    async def list(
            self,
            *,
            tenant_id: UUID | None = None,
            status: DefinitionStatus | None = None,
            keyword: str | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentTypeDefinition], int]:
        """
        Query one page of DocumentTypeDefinitions.

        排序固定为 created_at DESC, document_type_definition_id DESC。

        第二排序键是必需的：
        created_at 可能重复，缺少稳定 tiebreaker 时
        翻页会出现记录重复或遗漏。
        """

        conditions = []

        if tenant_id is not None:
            conditions.append(
                DocumentTypeDefinitionModel.tenant_id
                == tenant_id
            )

        if status is not None:
            conditions.append(
                DocumentTypeDefinitionModel.status
                == status
            )

        if keyword:
            pattern = self._like_pattern(keyword)

            conditions.append(
                or_(
                    DocumentTypeDefinitionModel.type_key.ilike(
                        pattern,
                        escape="\\",
                    ),
                    DocumentTypeDefinitionModel.name.ilike(
                        pattern,
                        escape="\\",
                    ),
                )
            )

        # --------------------------------------------------------
        # Total count
        # --------------------------------------------------------

        count_stmt = (
            select(func.count())
            .select_from(DocumentTypeDefinitionModel)
        )

        if conditions:
            count_stmt = count_stmt.where(*conditions)

        total = (
            await self._session.execute(count_stmt)
        ).scalar_one()

        # --------------------------------------------------------
        # Current page
        # --------------------------------------------------------

        page_stmt = select(DocumentTypeDefinitionModel)

        if conditions:
            page_stmt = page_stmt.where(*conditions)

        page_stmt = (
            page_stmt
            .order_by(
                DocumentTypeDefinitionModel.created_at.desc(),
                DocumentTypeDefinitionModel
                .document_type_definition_id
                .desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        result = await self._session.execute(page_stmt)

        models = result.scalars().all()

        return (
            [self._to_entity(model) for model in models],
            int(total),
        )

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _like_pattern(
            value: str,
    ) -> str:
        """
        Build a case-insensitive contains pattern.

        转义 LIKE 通配符，避免调用方传入的 % 和 _
        被当作通配符处理。
        """

        escaped = (
            value
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        return f"%{escaped}%"

    @staticmethod
    def _to_entity(
            model: DocumentTypeDefinitionModel,
    ) -> DocumentTypeDefinition:
        """
        Convert SQLAlchemy Model -> Domain Entity.

        No business logic is performed here.
        """

        return DocumentTypeDefinition(
            document_type_definition_id=(
                model.document_type_definition_id
            ),
            tenant_id=model.tenant_id,
            type_key=model.type_key,
            name=model.name,
            description=model.description,
            modality_type=model.modality_type,
            analysis_mode=model.analysis_mode,
            allow_update=model.allow_update,
            version_mode=model.version_mode,
            fusion_strategy=model.fusion_strategy,
            fusion_config=dict(model.fusion_config),
            rerank_strategy=model.rerank_strategy,
            rerank_config=dict(model.rerank_config),
            revision=model.revision,
            metadata=dict(model.metadata_),
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
