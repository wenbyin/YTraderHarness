"""
PostgreSQL implementation of DocumentVersionRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select, update

from ..domain.entities import DocumentVersion
from ..domain.schemas.enum import VersionStatus
from ..models.sqlalchemy.document import (
    Document as DocumentModel,
)
from ..models.sqlalchemy.document_version import (
    DocumentVersion as DocumentVersionModel,
)
from .document_version import DocumentVersionRepository

class DocumentVersionRepositoryImpl(
    DocumentVersionRepository
):

    # ============================================================
    # Create
    # ============================================================

    async def add(
            self,
            version: DocumentVersion,
    ) -> DocumentVersion:

        model = DocumentVersionModel(
            document_version_id=version.document_version_id,
            document_id=version.document_id,
            file_id=version.file_id,
            version_number=version.version_number,
            version_status=version.version_status,
            processing_status=version.processing_status,
            meta_data=dict(version.meta_data),
            created_by=version.created_by,
            created_at=version.created_at,
            updated_at=version.updated_at,
            completed_at=version.completed_at,
        )

        self._session.add(model)

        await self._session.flush()

        return self._to_entity(model)

    # ============================================================
    # Read
    # ============================================================

    async def get(
            self,
            document_version_id: UUID,
    ) -> DocumentVersion | None:

        stmt = (
            select(DocumentVersionModel)
            .where(
                DocumentVersionModel.document_version_id
                == document_version_id
            )
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def list_active_ids_by_type_definitions(
            self,
            document_type_definition_ids: list[UUID],
    ) -> list[UUID]:

        if not document_type_definition_ids:
            return []

        stmt = (
            select(DocumentVersionModel.document_version_id)
            .join(
                DocumentModel,
                DocumentModel.document_id
                == DocumentVersionModel.document_id,
            )
            .where(
                DocumentModel.document_type_definition_id.in_(
                    document_type_definition_ids
                ),
                DocumentVersionModel.version_status
                == VersionStatus.ACTIVE,
            )
        )

        return list(
            (
                await self._session.execute(stmt)
            ).scalars().all()
        )

    async def get_many(
            self,
            document_version_ids: list[UUID],
    ) -> list[DocumentVersion]:

        if not document_version_ids:
            return []

        stmt = select(DocumentVersionModel).where(
            DocumentVersionModel.document_version_id.in_(
                document_version_ids
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(model) for model in models]

    async def list_by_document(
            self,
            document_id: UUID,
            *,
            status: VersionStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentVersion], int]:

        conditions = [
            DocumentVersionModel.document_id == document_id
        ]

        if status is not None:
            conditions.append(
                DocumentVersionModel.version_status == status
            )

        count_stmt = (
            select(func.count())
            .select_from(DocumentVersionModel)
            .where(*conditions)
        )
        total = (
            await self._session.execute(count_stmt)
        ).scalar_one()

        page_stmt = (
            select(DocumentVersionModel)
            .where(*conditions)
            .order_by(
                DocumentVersionModel.version_number.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models], int(total)

    async def list_all_by_document(
            self,
            document_id: UUID,
    ) -> list[DocumentVersion]:

        stmt = (
            select(DocumentVersionModel)
            .where(
                DocumentVersionModel.document_id == document_id
            )
            .order_by(
                DocumentVersionModel.version_number.asc()
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [self._to_entity(m) for m in models]

    async def max_version_number(
            self,
            document_id: UUID,
    ) -> int:

        stmt = (
            select(
                func.max(
                    DocumentVersionModel.version_number
                )
            )
            .where(
                DocumentVersionModel.document_id
                == document_id
            )
        )

        value = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return int(value or 0)

    # ============================================================
    # Update
    # ============================================================

    async def activate(
            self,
            document_id: UUID,
            version_id: UUID,
            *,
            now: datetime,
    ) -> int:

        # 一条 UPDATE 完成让位，而不是把整份版本列表查出来再逐个 update：
        # 一个文档攒到几十版时，后者是几十次往返。
        #
        # WHERE 里的三个条件是契约的一部分（见 ABC 的说明）：
        # 只动**同文档**下、**不是本版本**、且**当前为 active** 的行。
        # 于是 deleted / failed / pending 都不会被波及。
        supersede = (
            update(DocumentVersionModel)
            .where(
                DocumentVersionModel.document_id == document_id,
                DocumentVersionModel.document_version_id
                != version_id,
                DocumentVersionModel.version_status
                == VersionStatus.ACTIVE,
            )
            .values(
                version_status=VersionStatus.SUPERSEDED,
                updated_at=now,
            )
        )

        superseded = (
            await self._session.execute(supersede)
        ).rowcount or 0

        # 再把本版本置为 active。
        activated = (
            await self._session.execute(
                update(DocumentVersionModel)
                .where(
                    DocumentVersionModel.document_version_id
                    == version_id
                )
                .values(
                    version_status=VersionStatus.ACTIVE,
                    updated_at=now,
                )
            )
        ).rowcount or 0

        # 影响 0 行 = 这个版本不存在。此时上面的让位可能已经生效了 ——
        # 与其把文档留在一个"零个 active"的状态里，不如直接报错让
        # 调用方的事务回滚。
        if activated == 0:
            raise ValueError(
                f"DocumentVersion not found: {version_id}"
            )

        return superseded

    async def update(
            self,
            version: DocumentVersion,
    ) -> DocumentVersion:

        stmt = (
            select(DocumentVersionModel)
            .where(
                DocumentVersionModel.document_version_id
                == version.document_version_id
            )
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                "DocumentVersion not found: "
                f"{version.document_version_id}"
            )

        model.version_status = version.version_status
        model.processing_status = version.processing_status
        model.meta_data = dict(version.meta_data)
        model.completed_at = version.completed_at
        model.updated_at = version.updated_at

        # document_version_id / document_id / file_id /
        # version_number / created_at 不可修改。

        await self._session.flush()

        # 见 document_impl.update 的说明：
        # flush 后属性会被 expired，需要显式 refresh。
        await self._session.refresh(model)

        return self._to_entity(model)

    # ============================================================
    # Mapping
    # ============================================================

    @staticmethod
    def _to_entity(
            model: DocumentVersionModel,
    ) -> DocumentVersion:

        return DocumentVersion(
            document_version_id=model.document_version_id,
            document_id=model.document_id,
            file_id=model.file_id,
            version_number=model.version_number,
            version_status=model.version_status,
            processing_status=model.processing_status,
            meta_data=dict(model.meta_data),
            created_by=model.created_by,
            created_at=model.created_at,
            updated_at=model.updated_at,
            completed_at=model.completed_at,
        )
