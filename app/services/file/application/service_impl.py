from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.common.service.base import ServiceContext

from ..domain.entities import File
from ..domain.schemas.enum import FileOrigin, FileStatus
from ..domain.schemas.file import (
    DeleteFileRequest,
    DeleteFileResponse,
    FileContentResponse,
    FileListItem,
    FileResponse,
    GetFileContentRequest,
    GetFileRequest,
    GetFileResponse,
    ListFilesRequest,
    ListFilesResponse,
    UploadFileRequest,
)
from ..repository.file import FileRepository
from ..storage.base import FileStorage
from .service import FileService


class FileServiceImpl(FileService):
    """
    FileService 默认实现。

    职责：
        - 管理 File 元数据
        - 调用 FileStorage 保存物理文件
        - 保证 File 与物理文件之间的基本一致性

    不负责：
        - Document
        - DocumentTypeDefinition
        - Chunk
        - Index
        - RAG
    """

    def __init__(
            self,
            repository: FileRepository,
            storage: FileStorage,
            *,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._repository = repository
        self._storage = storage

    async def upload_file(
            self,
            request: UploadFileRequest,
    ) -> FileResponse:
        """
        上传一个新的文件资产。

        流程：

            1. 校验请求
            2. 读取文件
            3. 计算 checksum
            4. 创建 file_id
            5. 保存物理文件
            6. 创建 PostgreSQL File 记录
            7. 返回文件 metadata

        注意：
            每次显式调用 upload_file 都创建新的 File。
            不根据文件名判断是否为同一个 File。

        checksum 主要用于：
            - 文件完整性校验
            - 后续去重能力
            - 数据一致性校验

        当前版本不自动根据 checksum 去重。
        """

        # ====================================================
        # 1. Validate
        # ====================================================

        self._validate_upload_request(request)

        # ====================================================
        # 2. Persist
        # ====================================================
        #
        # 注意：每次显式调用 upload_file 都创建新的 File，
        # **不**按 checksum 去重 —— 那是 store_artifact 的行为。
        # 用户上传的是独立资产，两个用户传同一份文件应当各有记录。

        return await self._persist(
            file_id=uuid4(),
            file_name=request.file_name,
            content=request.content,
            content_type=request.content_type,
            tenant_id=request.tenant_id,
            metadata=request.metadata,
            uploaded_by=request.uploaded_by,
        )

    async def store_artifact(
            self,
            *,
            kind: str,
            content: bytes,
            content_type: str | None = None,
            tenant_id: UUID | None = None,
            metadata: dict | None = None,
    ) -> FileResponse:
        """
        写入一个流水线中间产物。

        与 upload_file 的区别（刻意的）：

            origin
                写为 pipeline_artifact。file 表因此能区分
                产物与用户资产，GET /files 默认只返回后者。

            file_name
                填 kind（parsed_text / chunks / vectors ...）。
                它是内部标识，不是给用户看的展示名。

            去重
                按 (tenant_id, checksum, kind, origin) 复用既有 active File。
                kind 必须参与 —— 否则相同内容的不同产物会被合并，
                后者的 kind 会变成前者的。
                reprocess 同一文件、多次运行共享同一段正文时
                不会重复落盘 —— 这是产物与用户资产最大的不同。

            uploaded_by
                恒为 None。产物没有上传者。

        不做面向用户的上传校验（空文件名等）—— kind 由代码给定，
        空内容则直接视为异常。
        """

        if not kind.strip():
            raise ValueError("artifact kind cannot be empty")

        if not content:
            raise ValueError(
                f"artifact content cannot be empty (kind={kind!r})"
            )

        checksum = hashlib.sha256(content).hexdigest()

        # file_name=kind 是关键：只按 checksum 去重会把
        # 相同内容的不同产物（parsed_text vs clean_text）
        # 合并成一条，ref 里的 kind 就撒谎了。
        existing = await self._repository.get_by_checksum(
            tenant_id,
            checksum,
            file_name=kind,
            origin=FileOrigin.PIPELINE_ARTIFACT,
        )

        if existing is not None:
            # 内容寻址命中：同租户下已有完全相同的内容，
            # 直接复用它的 file_id 与 storage_key。
            return self._to_file_response(existing)

        created = await self._persist(
            file_id=uuid4(),
            file_name=kind,
            content=content,
            content_type=(
                content_type
                or "application/octet-stream"
            ),
            tenant_id=tenant_id,
            metadata=dict(metadata or {}),
            uploaded_by=None,
            origin=FileOrigin.PIPELINE_ARTIFACT,
            dedupe=True,
        )

        if created is not None:
            return created

        # 走到这里说明并发冲突：另一个分支在我们查询之后、
        # 我们插入之前写入了同内容的产物。它现在必然查得到。
        winner = await self._repository.get_by_checksum(
            tenant_id,
            checksum,
            file_name=kind,
            origin=FileOrigin.PIPELINE_ARTIFACT,
        )

        if winner is None:
            # 唯一索引报了冲突，但按同样的条件却查不到 ——
            # 说明索引与查询条件不一致，是代码 bug 而非并发。
            raise RuntimeError(
                "Artifact insert conflicted on "
                "uq_file_artifact_content, but the existing row "
                f"is not findable (kind={kind!r}, "
                f"checksum={checksum[:12]}...). "
                "The unique index and the lookup predicate "
                "have diverged."
            )

        return self._to_file_response(winner)

    async def _persist(
            self,
            *,
            file_id: UUID,
            file_name: str,
            content: bytes,
            content_type: str | None,
            tenant_id: UUID | None,
            metadata: dict,
            uploaded_by: UUID | None,
            origin: FileOrigin = FileOrigin.USER_UPLOAD,
            dedupe: bool = False,
    ) -> FileResponse | None:
        """
        写物理文件 + 建 file 行 + 返回响应。

        upload_file 与 store_artifact 共用本方法 —— 中间的
        **补偿逻辑**（物理文件已写入但 DB 失败时要把它删掉）
        很容易漏，不该复制两份。任一处漏了都会留下孤儿文件。

        dedupe
            True 时走 add_artifact_if_absent（并发安全），
            冲突时删掉刚写的物理文件并返回 None。

        Returns:
            正常写入 -> FileResponse；dedupe 冲突 -> None。

        Raises:
            RuntimeError: 存储层返回的大小 / checksum 与内容不符。
        """

        checksum = hashlib.sha256(content).hexdigest()
        file_size = len(content)

        stored_file = None

        try:
            stored_file = await self._storage.put(
                file_id=file_id,
                file_name=file_name,
                content=content,
                content_type=content_type,
            )

            if stored_file.file_size != file_size:
                raise RuntimeError(
                    "Stored file size does not match "
                    "uploaded file size"
                )

            if stored_file.checksum != checksum:
                raise RuntimeError(
                    "Stored file checksum does not match "
                    "uploaded file checksum"
                )

            now = datetime.now(timezone.utc)

            entity = File(
                file_id=file_id,
                tenant_id=tenant_id,
                file_name=file_name,
                content_type=content_type,
                file_size=file_size,
                checksum=checksum,
                storage_backend=stored_file.storage_backend,
                storage_key=stored_file.storage_key,
                origin=origin,
                status=FileStatus.ACTIVE,
                metadata=metadata,
                uploaded_by=uploaded_by,
                created_at=now,
                updated_at=now,
                deleted_at=None,
            )

            created_file = (
                await self._repository.add_artifact_if_absent(
                    entity
                )
                if dedupe
                else await self._repository.add(entity)
            )

            if created_file is None:
                # 并发下另一个分支抢先写入了同内容的产物。
                # 刚写的物理文件没有对应的 DB 行，删掉它，
                # 由调用方去复用赢家那条。
                await self._storage.delete(
                    stored_file.storage_key
                )
                return None

            return self._to_file_response(created_file)

        except Exception:
            # 补偿：物理文件写成功但后续失败时删掉它，
            # 避免 "Storage 有文件 + DB 没记录" 的孤儿。
            if stored_file is not None:
                try:
                    await self._storage.delete(
                        stored_file.storage_key
                    )
                except Exception:
                    # 不覆盖原始异常。真实环境这里应记
                    # orphan_file_cleanup_failed 指标。
                    pass

            raise
        # ============================================================
        # Get Metadata
        # ============================================================

    async def get_file(
            self,
            request: GetFileRequest,
    ) -> GetFileResponse:
        """
        Get File metadata.

        This method only queries PostgreSQL.
        It does NOT access physical storage.

        Raises:
            ValueError:
                File does not exist.
        """

        self._validate_file_id(
            request.file_id
        )

        file_entity = await self._repository.get(
            request.file_id
        )

        if file_entity is None:
            raise ValueError(
                f"File not found: "
                f"{request.file_id}"
            )

        return self._to_file_response(
            file_entity
        )

        # ============================================================
        # Get File Content
        # ============================================================

    async def get_file_content(
            self,
            request: GetFileContentRequest,
    ) -> FileContentResponse:
        """
        Read physical file content.

        Flow:

            file_id
              ↓
            PostgreSQL
              ↓
            storage_key
              ↓
            FileStorage.get()
              ↓
            bytes
        """

        self._validate_file_id(
            request.file_id
        )

        file_entity = await self._repository.get(
            request.file_id
        )

        if file_entity is None:
            raise ValueError(
                f"File not found: "
                f"{request.file_id}"
            )

        if (
                file_entity.status
                == FileStatus.DELETED
        ):
            raise ValueError(
                f"File has been deleted: "
                f"{request.file_id}"
            )

        content = await self._storage.get(
            file_entity.storage_key
        )

        return FileContentResponse(
            file_id=file_entity.file_id,
            file_name=file_entity.file_name,
            content_type=(
                file_entity.content_type
            ),
            content=content,
        )

    # ============================================================
    # Logical Delete
    # ============================================================

    async def delete_file(
            self,
            request: DeleteFileRequest,
    ) -> DeleteFileResponse:
        """
        Logically delete a File.

        Important
        ---------
        This method does NOT physically delete the file.

        It only changes PostgreSQL state:

            active
              ↓
            deleted

        Physical cleanup is performed later by a dedicated
        Cleanup Job / Cron process.
        """
        self._validate_file_id(request.file_id)

        file_entity = await self._repository.get(request.file_id)

        if file_entity is None:
            raise ValueError(
                f"File not found: "
                f"{request.file_id}"
            )

        if file_entity.status == FileStatus.DELETED:
            raise ValueError(
                f"File already deleted: "
                f"{request.file_id}"
            )

        deleted_at = datetime.now(timezone.utc)
        file_entity.status = FileStatus.DELETED
        file_entity.deleted_at = deleted_at
        file_entity.updated_at = deleted_at
        updated_file = await self._repository.update(
            file_entity
        )

        return DeleteFileResponse(
            file_id=updated_file.file_id,
            status=updated_file.status.value,
            deleted_at=updated_file.deleted_at
        )

    # ============================================================
    # List
    # ============================================================

    async def list_files(
            self,
            request: ListFilesRequest,
    ) -> ListFilesResponse:
        """
        List Files with pagination.

        This method only queries PostgreSQL.
        It does NOT access physical storage.

        status 与 include_deleted 的优先级：

            status 显式指定       -> 只按该 status 过滤
            include_deleted=True  -> 不限制 status
            否则                  -> 只返回 active

        Raises:
            ValueError:
                created_from 晚于 created_to。
        """

        # ====================================================
        # 1. Validate
        # ====================================================

        if (
                request.created_from is not None
                and request.created_to is not None
                and request.created_from
                > request.created_to
        ):
            raise ValueError(
                "created_from must not be later "
                "than created_to"
            )

        # ====================================================
        # 2. Resolve status filter
        # ====================================================

        status = self._resolve_list_status(
            request
        )

        # ====================================================
        # 3. Query one page
        # ====================================================

        offset = (
            request.page - 1
        ) * request.page_size

        file_entities, total = (
            await self._repository.list(
                tenant_id=request.tenant_id,
                status=status,
                origin=request.origin,
                file_name=request.file_name,
                content_type=request.content_type,
                uploaded_by=request.uploaded_by,
                created_from=request.created_from,
                created_to=request.created_to,
                offset=offset,
                limit=request.page_size,
            )
        )

        # ====================================================
        # 4. Build response
        # ====================================================

        total_pages = (
            (total + request.page_size - 1)
            // request.page_size
        )

        return ListFilesResponse(
            items=[
                self._to_file_list_item(
                    file_entity
                )
                for file_entity in file_entities
            ],
            page=request.page,
            page_size=request.page_size,
            total=total,
            total_pages=total_pages,
        )

    # ============================================================
    # Validation
    # ============================================================

    @staticmethod
    def _resolve_list_status(
            request: ListFilesRequest,
    ) -> FileStatus | None:
        """
        Resolve the effective status filter for listing.

        返回 None 表示不按 status 过滤。
        """

        if request.status is not None:
            return request.status

        if request.include_deleted:
            return None

        return FileStatus.ACTIVE

    @staticmethod
    def _validate_upload_request(
            request: UploadFileRequest,
    ) -> None:
        """
        Validate upload request before touching storage.
        """

        if not request.file_name.strip():
            raise ValueError(
                "file_name cannot be empty"
            )

        if request.content is None:
            raise ValueError(
                "file content cannot be None"
            )

        if len(request.content) == 0:
            raise ValueError(
                "file content cannot be empty"
            )

        # 注意：这里原本还有一段 `request.file_size` 的校验，
        # 但 UploadFileRequest 上没有该字段 —— 那段是死代码，
        # 因为本方法此前从未被调用（upload_file 用的是内联校验）。
        # 校验实际大小已由 _persist 内的
        # `stored_file.file_size != file_size` 覆盖。

    @staticmethod
    def _validate_file_id(
            file_id: UUID,
    ) -> None:
        """
        Basic File ID validation.

        Pydantic already guarantees UUID type,
        so this method mainly exists as a single
        service-level validation boundary.
        """

        if not isinstance(
                file_id,
                UUID,
        ):
            raise ValueError(
                "file_id must be a UUID"
            )

    # ============================================================
    # Entity -> DTO
    # ============================================================

    @staticmethod
    def _to_file_list_item(
            file_entity: File,
    ) -> FileListItem:
        """
        Convert Domain Entity to list item DTO.

        刻意不输出 storage_key / storage_backend：

        storage_key 是存储层实现细节，
        列表调用方不需要，也不应该凭它绕过
        FileService 直接访问物理文件。
        """

        return FileListItem(
            file_id=file_entity.file_id,

            tenant_id=file_entity.tenant_id,

            file_name=file_entity.file_name,

            content_type=file_entity.content_type,

            file_size=file_entity.file_size,

            checksum=file_entity.checksum,

            status=(
                file_entity.status.value
            ),

            uploaded_by=file_entity.uploaded_by,

            created_at=file_entity.created_at,

            updated_at=file_entity.updated_at,

            deleted_at=file_entity.deleted_at,
        )

    @staticmethod
    def _to_file_response(
            file_entity: File,
    ) -> FileResponse:
        """
        Convert Domain Entity to API response DTO.

        This mapping intentionally keeps API schema
        independent from SQLAlchemy models.
        """

        return FileResponse(
            file_id=file_entity.file_id,

            tenant_id=file_entity.tenant_id,

            file_name=file_entity.file_name,

            content_type=file_entity.content_type,

            file_size=file_entity.file_size,

            checksum=file_entity.checksum,

            storage_backend=(
                file_entity.storage_backend
            ),

            storage_key=(
                file_entity.storage_key
            ),

            status=(
                file_entity.status.value
            ),

            metadata=dict(
                file_entity.metadata
            ),

            created_at=file_entity.created_at,

            updated_at=file_entity.updated_at,

            deleted_at=file_entity.deleted_at,
        )
