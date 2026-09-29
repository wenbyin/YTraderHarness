"""
PostgreSQL implementation of FileRepository.

Responsibilities
----------------
- CRUD operations against the file table.
- Query file metadata.
- Check file existence by checksum.

Non-responsibilities
--------------------
- File IO
- Object storage
- Transaction commit / rollback
- File checksum calculation
- File upload
- File physical deletion

Transaction rule
----------------
This repository NEVER calls:

    session.commit()
    session.rollback()

Transaction lifecycle is owned by the application/service layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import File
from ..domain.schemas.enum import FileOrigin, FileStatus
from ..models.sqlalchemy.file import File as FileModel
from .file import FileRepository

class FileRepositoryImpl(FileRepository):
    """
    PostgreSQL implementation of FileRepository.

    Example:

        async with session.begin():
            file = await repository.create(file)

        # transaction committed by Service layer
    """

    # ============================================================
    # Create
    # ============================================================

    async def add(
            self,
            file: File,
    ) -> File:
        """
        Create a File database record.

        Only PostgreSQL is touched.

        No physical file operation is performed.
        No transaction commit is performed.
        """

        model = FileModel(
            file_id=file.file_id,
            tenant_id=file.tenant_id,
            file_name=file.file_name,
            content_type=file.content_type,
            file_size=file.file_size,
            checksum=file.checksum,
            storage_backend=file.storage_backend,
            storage_key=file.storage_key,
            origin=file.origin,
            status=file.status,
            metadata_=file.metadata,
            uploaded_by=file.uploaded_by,
            created_at=file.created_at,
            updated_at=file.updated_at,
            deleted_at=file.deleted_at,
        )

        self._session.add(model)

        # Make INSERT visible in the current transaction.
        # Do NOT commit here.
        await self._session.flush()

        return self._to_entity(model)

    async def add_artifact_if_absent(
            self,
            file: File,
    ) -> File | None:

        stmt = insert(FileModel).values(
            file_id=file.file_id,
            tenant_id=file.tenant_id,
            file_name=file.file_name,
            content_type=file.content_type,
            file_size=file.file_size,
            checksum=file.checksum,
            storage_backend=file.storage_backend,
            storage_key=file.storage_key,
            origin=file.origin,
            status=file.status,
            metadata_=file.metadata,
            uploaded_by=file.uploaded_by,
            created_at=file.created_at,
            updated_at=file.updated_at,
            deleted_at=file.deleted_at,
        ).on_conflict_do_nothing(
            index_elements=[
                FileModel.tenant_id,
                FileModel.checksum,
                FileModel.file_name,
            ],
            # ⚠️ 这里必须编译成**字面量**，不能写成
            #
            #       FileModel.origin == FileOrigin.PIPELINE_ARTIFACT
            #
            # 后者会被编译成绑定参数（$n），而 Postgres 做 ON CONFLICT
            # 推断时要求 WHERE 子句与部分索引的谓词**在计划期等价** ——
            # 一个绑定参数无法被证明等价于索引里的
            # `(origin)::text = 'pipeline_artifact'::text`，于是推断
            # 失败，报 InvalidColumnReferenceError（"没有匹配 ON
            # CONFLICT 说明的唯一或者排除约束"）。
            #
            # 症状很有迷惑性：索引确实存在、定义也正确（见
            # models/sqlalchemy/file.py 的 uq_file_artifact_content），
            # 但语句就是过不去 —— 问题不在索引，在**写法**。
            #
            # 值取自枚举而不是再写一遍字符串字面量：索引谓词那边写死了
            # 'pipeline_artifact'，两边必须一致，从枚举取值至少保证
            # 这里不会因为枚举改名而漂移。
            index_where=text(
                f"origin = '{FileOrigin.PIPELINE_ARTIFACT.value}'"
            ),
        ).returning(FileModel)

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            return None

        await self._session.refresh(model)

        return self._to_entity(model)

    # ============================================================
    # Get
    # ============================================================

    async def get(
            self,
            file_id: UUID,
    ) -> File | None:
        """
        Get File by primary key.

        Both active and logically deleted records may be returned.

        Whether deleted files are allowed for a specific operation
        is decided by Service layer.
        """

        stmt = (
            select(FileModel)
            .where(
                FileModel.file_id == file_id
            )
        )

        result = await self._session.execute(
            stmt
        )

        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def update(self, file: File) -> File:
        """
            Update an existing File record.

            只操作 PostgreSQL。
            不负责物理文件 IO。
            不 commit / rollback。
            """
        stmt = (
            select(FileModel)
            .where(
                FileModel.file_id == file.file_id
            )
        )

        result = await self._session.execute(
            stmt
        )

        model = result.scalar_one_or_none()

        if model is None:
            raise ValueError(
                f"File not found: {file.file_id}"
            )

        # --------------------------------------------------------
        # Update mutable fields
        # --------------------------------------------------------

        model.tenant_id = file.tenant_id
        model.file_name = file.file_name
        model.content_type = file.content_type
        model.file_size = file.file_size
        model.checksum = file.checksum
        model.storage_backend = file.storage_backend
        model.storage_key = file.storage_key
        model.origin = file.origin
        model.status = file.status
        model.metadata_ = dict(file.metadata)
        model.uploaded_by = file.uploaded_by
        model.updated_at = file.updated_at
        model.deleted_at = file.deleted_at

        # --------------------------------------------------------
        # Do NOT update:
        #
        # model.file_id
        # model.created_at
        #
        # file_id is the primary identity.
        # created_at represents original creation time.
        # --------------------------------------------------------

        await self._session.flush()

        return self._to_entity(model)

    # ============================================================
    # Exists by checksum
    # ============================================================

    async def exists_by_checksum(
            self,
            tenant_id: UUID | None,
            checksum: str,
    ) -> bool:
        """
        Check whether an active File with the specified checksum
        exists for the tenant.

        NOTE:
            The current business rule is:

                upload_file()
                -> always creates a new File

            Therefore this method only provides query capability.
            FileService decides whether duplicate detection is used.
        """

        stmt = (
            select(FileModel.file_id)
            .where(
                FileModel.checksum == checksum,
                FileModel.status == FileStatus.ACTIVE,
            )
            .limit(1)
        )

        if tenant_id is not None:
            stmt = stmt.where(
                FileModel.tenant_id == tenant_id
            )

        result = await self._session.execute(
            stmt
        )

        return (
                result.scalar_one_or_none()
                is not None
        )

    # ============================================================
    # Get by checksum
    # ============================================================

    async def get_by_checksum(
            self,
            tenant_id: UUID | None,
            checksum: str,
            *,
            file_name: str | None = None,
            origin: FileOrigin | None = None,
    ) -> File | None:
        """
        取一个可复用的 active File（最近创建的优先）。

        与 exists_by_checksum 的**语义差异**（刻意的）：

            exists_by_checksum
                tenant_id 为 None 时**不加租户过滤**，
                因此会命中任意租户的同 checksum 文件。
                这是既有行为，保留不动。

            get_by_checksum
                tenant_id 为 None 时只匹配 tenant_id IS NULL。
                因为本方法是给**复用**用的 —— 复用错租户的
                文件是数据隔离问题，不能靠"碰巧没人这么用"。

        取不到或已逻辑删除时返回 None。
        """

        stmt = (
            select(FileModel)
            .where(
                FileModel.checksum == checksum,
                FileModel.status == FileStatus.ACTIVE,
            )
            .order_by(FileModel.created_at.desc())
            .limit(1)
        )

        if tenant_id is None:
            stmt = stmt.where(
                FileModel.tenant_id.is_(None)
            )
        else:
            stmt = stmt.where(
                FileModel.tenant_id == tenant_id
            )

        if file_name is not None:
            stmt = stmt.where(
                FileModel.file_name == file_name
            )

        if origin is not None:
            stmt = stmt.where(FileModel.origin == origin)

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    # ============================================================
    # Get active
    # ============================================================

    async def get_active(
            self,
            file_id: UUID,
    ) -> File | None:
        """
        Get an active File.

        This is a convenience method for Service layer operations
        that require a non-deleted file.
        """

        stmt = (
            select(FileModel)
            .where(
                FileModel.file_id == file_id,
                FileModel.status == FileStatus.ACTIVE,
            )
        )

        result = await self._session.execute(
            stmt
        )

        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    # ============================================================
    # Logical delete
    # ============================================================

    async def soft_delete(
            self,
            file_id: UUID,
    ) -> File:
        """
        Logically delete a File.

        Database only.

        This method does NOT:
            - delete physical file
            - call FileStorage
            - clean storage
            - clean any Document reference

        Physical cleanup is handled separately by Cleanup Job.
        """

        stmt = (
            select(FileModel)
            .where(
                FileModel.file_id == file_id
            )
        )

        result = await self._session.execute(
            stmt
        )

        model = result.scalar_one_or_none()

        if model is None:
            raise ValueError(
                f"File not found: {file_id}"
            )

        if model.status == FileStatus.DELETED:
            raise ValueError(
                f"File already deleted: {file_id}"
            )

        now = datetime.now(
            timezone.utc
        )

        model.status = FileStatus.DELETED
        model.deleted_at = now
        model.updated_at = now

        await self._session.flush()

        return self._to_entity(model)

    # ============================================================
    # List
    # ============================================================

    async def list(
            self,
            *,
            tenant_id: UUID | None = None,
            status: FileStatus | None = None,
            origin: FileOrigin | None = None,
            file_name: str | None = None,
            content_type: str | None = None,
            uploaded_by: UUID | None = None,
            created_from: datetime | None = None,
            created_to: datetime | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[File], int]:
        """
        Query one page of Files.

        所有过滤条件之间是 AND 关系。

        排序固定为：

            created_at DESC, file_id DESC

        file_id 作为第二排序键是必需的：
        created_at 可能重复，缺少稳定 tiebreaker 时
        翻页会出现记录重复或遗漏。

        Only PostgreSQL is touched.
        No physical file operation is performed.
        No transaction commit is performed.
        """

        # --------------------------------------------------------
        # Build filter conditions
        # --------------------------------------------------------

        conditions = []

        if tenant_id is not None:
            conditions.append(
                FileModel.tenant_id == tenant_id
            )

        if status is not None:
            conditions.append(
                FileModel.status == status
            )

        if origin is not None:
            conditions.append(FileModel.origin == origin)

        if file_name:
            conditions.append(
                FileModel.file_name.ilike(
                    self._like_pattern(file_name),
                    escape="\\",
                )
            )

        if content_type:
            conditions.append(
                FileModel.content_type.startswith(
                    content_type
                )
            )

        if uploaded_by is not None:
            conditions.append(
                FileModel.uploaded_by == uploaded_by
            )

        if created_from is not None:
            conditions.append(
                FileModel.created_at >= created_from
            )

        if created_to is not None:
            conditions.append(
                FileModel.created_at <= created_to
            )

        # --------------------------------------------------------
        # Total count
        # --------------------------------------------------------

        count_stmt = (
            select(func.count())
            .select_from(FileModel)
        )

        if conditions:
            count_stmt = count_stmt.where(
                *conditions
            )

        total = (
            await self._session.execute(count_stmt)
        ).scalar_one()

        # --------------------------------------------------------
        # Current page
        # --------------------------------------------------------

        page_stmt = select(FileModel)

        if conditions:
            page_stmt = page_stmt.where(
                *conditions
            )

        page_stmt = (
            page_stmt
            .order_by(
                FileModel.created_at.desc(),
                FileModel.file_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        result = await self._session.execute(
            page_stmt
        )

        models = result.scalars().all()

        return (
            [
                self._to_entity(model)
                for model in models
            ],
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
            model: FileModel,
    ) -> File:
        """
        Convert SQLAlchemy Model -> Domain Entity.

        No business logic is performed here.
        """

        return File(
            file_id=model.file_id,
            tenant_id=model.tenant_id,
            file_name=model.file_name,
            content_type=model.content_type,
            file_size=model.file_size,
            checksum=model.checksum,
            storage_backend=model.storage_backend,
            storage_key=model.storage_key,
            origin=model.origin,
            status=model.status,
            metadata=dict(model.metadata_),
            uploaded_by=model.uploaded_by,
            created_at=model.created_at,
            updated_at=model.updated_at,
            deleted_at=model.deleted_at,
        )
