"""
PostgreSQL implementation of DocumentRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
Transaction lifecycle is owned by the application/service layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, or_, select

from ..domain.entities import Document
from ..domain.schemas.enum import DocumentStatus
from ..models.sqlalchemy.document import (
    Document as DocumentModel,
)
from .document import DocumentRepository

class DocumentRepositoryImpl(DocumentRepository):

    # ============================================================
    # Create
    # ============================================================

    async def add(
            self,
            document: Document,
    ) -> Document:
        """Create a Document record. No physical file operation."""

        model = DocumentModel(
            document_id=document.document_id,
            document_type_definition_id=(
                document.document_type_definition_id
            ),
            tenant_id=document.tenant_id,
            document_key=document.document_key,
            title=document.title,
            current_document_version_id=(
                document.current_document_version_id
            ),
            status=document.status,
            metadata_=dict(document.metadata),
            created_by=document.created_by,
            created_at=document.created_at,
            updated_at=document.updated_at,
            deleted_at=document.deleted_at,
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
            document_id: UUID,
    ) -> Document | None:

        stmt = (
            select(DocumentModel)
            .where(
                DocumentModel.document_id == document_id
            )
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def get_many(
            self,
            document_ids: list[UUID],
    ) -> list[Document]:

        if not document_ids:
            return []

        stmt = select(DocumentModel).where(
            DocumentModel.document_id.in_(document_ids)
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(model) for model in models]

    async def get_by_key(
            self,
            tenant_id: UUID | None,
            document_type_definition_id: UUID,
            document_key: str,
    ) -> Document | None:

        stmt = (
            select(DocumentModel)
            .where(
                DocumentModel.document_type_definition_id
                == document_type_definition_id,
                DocumentModel.document_key
                == document_key,
            )
        )

        if tenant_id is None:
            stmt = stmt.where(
                DocumentModel.tenant_id.is_(None)
            )
        else:
            stmt = stmt.where(
                DocumentModel.tenant_id == tenant_id
            )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    # ============================================================
    # Update
    # ============================================================

    async def update(
            self,
            document: Document,
    ) -> Document:

        stmt = (
            select(DocumentModel)
            .where(
                DocumentModel.document_id
                == document.document_id
            )
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                "Document not found: "
                f"{document.document_id}"
            )

        model.tenant_id = document.tenant_id
        model.document_key = document.document_key
        model.title = document.title
        model.current_document_version_id = (
            document.current_document_version_id
        )
        model.status = document.status
        model.metadata_ = dict(document.metadata)
        model.updated_at = document.updated_at
        model.deleted_at = document.deleted_at

        # document_id / document_type_definition_id / created_at
        # 不可修改。

        await self._session.flush()

        # flush() 之后 ORM 会标记被修改的属性为 expired，
        # 直接读取会触发同步惰性加载，
        # 在 async 上下文抛 MissingGreenlet。
        # 因此显式 refresh 一次再映射。
        await self._session.refresh(model)

        return self._to_entity(model)

    # ============================================================
    # List
    # ============================================================

    async def list(
            self,
            *,
            tenant_id: UUID | None = None,
            document_type_definition_id: UUID | None = None,
            status: DocumentStatus | None = None,
            keyword: str | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[Document], int]:

        conditions = []

        if tenant_id is not None:
            conditions.append(
                DocumentModel.tenant_id == tenant_id
            )

        if document_type_definition_id is not None:
            conditions.append(
                DocumentModel.document_type_definition_id
                == document_type_definition_id
            )

        if status is not None:
            conditions.append(
                DocumentModel.status == status
            )

        if keyword:
            pattern = self._like_pattern(keyword)
            conditions.append(
                or_(
                    DocumentModel.title.ilike(
                        pattern, escape="\\"),
                    DocumentModel.document_key.ilike(
                        pattern, escape="\\"),
                )
            )

        count_stmt = (
            select(func.count())
            .select_from(DocumentModel)
        )
        if conditions:
            count_stmt = count_stmt.where(*conditions)
        total = (
            await self._session.execute(count_stmt)
        ).scalar_one()

        page_stmt = select(DocumentModel)
        if conditions:
            page_stmt = page_stmt.where(*conditions)
        page_stmt = (
            page_stmt
            .order_by(
                DocumentModel.created_at.desc(),
                DocumentModel.document_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models], int(total)

    @staticmethod
    def _like_pattern(value: str) -> str:
        """转义 LIKE 通配符后构造包含匹配 pattern。"""
        escaped = (
            value
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        return f"%{escaped}%"

    async def soft_delete(
            self,
            document_id: UUID,
    ) -> Document:

        stmt = select(DocumentModel).where(
            DocumentModel.document_id == document_id
        )
        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                f"Document not found: {document_id}"
            )

        if model.status == DocumentStatus.DELETED:
            raise ValueError(
                f"Document already deleted: {document_id}"
            )

        model.status = DocumentStatus.DELETED
        model.deleted_at = datetime.now(timezone.utc)
        model.updated_at = model.deleted_at

        await self._session.flush()
        await self._session.refresh(model)

        return self._to_entity(model)

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _to_entity(
            model: DocumentModel,
    ) -> Document:

        return Document(
            document_id=model.document_id,
            document_type_definition_id=(
                model.document_type_definition_id
            ),
            tenant_id=model.tenant_id,
            document_key=model.document_key,
            title=model.title,
            current_document_version_id=(
                model.current_document_version_id
            ),
            status=model.status,
            metadata=dict(model.metadata_),
            created_by=model.created_by,
            created_at=model.created_at,
            updated_at=model.updated_at,
            deleted_at=model.deleted_at,
        )
