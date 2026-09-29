"""
流水线中间产物的读写。

产物即 File
-----------
产物**不是**独立的存储概念，它就是 File 模块里的一条 File ——
同一张表、同一套 FileStorage、同一个 checksum 索引。

这样做的好处是只有一套存储抽象和一套生命周期，而且将来给
File 加引用计数 / 超时清理时，产物自动被覆盖。

产物与用户资产靠 `file.origin` 区分：

    user_upload        用户上传的资产，GET /files 默认只返回它
    pipeline_artifact  流水线产物，需显式 ?origin=pipeline_artifact

产物的 file_name 填 kind（parsed_text / chunks / ...），
它是内部标识，不是展示名。

仍待办（需要 file 模块的引用计数 / 超时清理）：
产物目前会永久留在 file 表里，没有回收机制。

为什么正文不进图状态
--------------------
state 里只放 ArtifactRef（file_id + checksum + 计数 + preview），
正文通过本模块存取。ArtifactRef 里的 preview 上限 512 字符 ——
它让 state 的大小与文档大小**解耦**：不论正文是 1 KB 还是
100 MB，state 都是同一个量级。

事务
----
每次写入/读取自成一段短事务，模式同 SqlAlchemyRunRecorder。
节点不需要感知 session。

不变式：**产物必须在节点 return 之前提交**。否则持久化的
checkpoint 可能引用一个尚未落盘的产物，续跑时以极难调试的
方式失败。
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.common.repository.factory import RepositoryFactory
from app.common.service.factory import ServiceFactory
from app.services.file.application.service import FileService
from app.services.file.domain.schemas.file import (
    GetFileContentRequest,
)
from app.services.file.repository.file import FileRepository
from app.services.file.storage.base import FileStorage

__all__ = [
    "ArtifactStore",
    "FileArtifactStore",
    "sha256_of_bytes",
    "sha256_of_text",
]


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@runtime_checkable
class ArtifactStore(Protocol):
    """
    产物读写端口。

    节点拿到的是这个 Protocol，因此换成对象存储实现时不改节点代码。
    各方法返回的 dict 就是可直接放进 state 的 ArtifactRef。
    """

    async def put_text(
            self,
            *,
            kind: str,
            text: str,
            tenant_id: UUID | None = None,
            content_type: str | None = None,
            token_count: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    async def put_bytes(
            self,
            *,
            kind: str,
            content: bytes,
            tenant_id: UUID | None = None,
            content_type: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    async def get_text(self, file_id: UUID) -> str | None:
        ...

    async def get_bytes(self, file_id: UUID) -> bytes | None:
        ...

    async def reference(
            self,
            file_id: UUID,
            *,
            preview: str = "",
    ) -> dict[str, Any]:
        """为**已存在**的 File 构造 ref，不写新文件。"""
        ...


class FileArtifactStore:
    """
    走 FileService 的实现。

    自持 session_factory（和 SqlAlchemyRunRecorder 一样）——
    产物写入跑在流水线的后台任务里，不持有任何请求事务。
    """

    #: ArtifactRef.preview 的长度上限，与
    #: infra/graph/instrumentation.py 的护栏保持一致。
    PREVIEW_LIMIT = 512

    def __init__(
            self,
            *,
            session_factory: async_sessionmaker[AsyncSession],
            storage: FileStorage,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage

        # 产物读写不持有请求事务（见类 docstring），因此两个工厂都处在
        # session_factory 模式，每次操作由 scope() 现开一段短事务。
        #
        # storage 放进公共池子：FileServiceImpl 的构造参数叫 storage，
        # 注解是 FileStorage，工厂会按类型把它匹配上 ——
        # 与 lifespan 里那一条是同一个来源。
        self._repositories = RepositoryFactory(
            session_factory=session_factory
        )
        self._services = ServiceFactory(
            self._repositories,
            pool={"file_storage": storage},
        )

    # ========================================================
    # Write
    # ========================================================

    async def put_text(
            self,
            *,
            kind: str,
            text: str,
            tenant_id: UUID | None = None,
            content_type: str | None = None,
            token_count: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        payload = text.encode("utf-8")

        file_response = await self._store(
            kind=kind,
            content=payload,
            content_type=content_type or "text/plain",
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return self._to_ref(
            file_response,
            char_count=len(text),
            token_count=token_count,
            preview=text[: self.PREVIEW_LIMIT],
        )

    async def put_bytes(
            self,
            *,
            kind: str,
            content: bytes,
            tenant_id: UUID | None = None,
            content_type: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        file_response = await self._store(
            kind=kind,
            content=content,
            content_type=content_type,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return self._to_ref(
            file_response,
            char_count=None,
            token_count=None,
            preview="",
        )

    # ========================================================
    # Read
    # ========================================================

    async def reference(
            self,
            file_id: UUID,
            *,
            preview: str = "",
    ) -> dict[str, Any]:
        """
        为**已存在**的 File 构造 ArtifactRef，不写任何新文件。

        用途：ingestion 阶段。源文件本来就是 file 表里的一条
        File（用户上传的那个），产物改走 File 之后就没必要再
        复制一份 —— 直接引用即可。之前那样做等于一份 432 KB
        的文档白存 432 KB。

        preview 由调用方给 —— 本方法不读内容（读整份文件只为
        截 512 字符不划算，而且要读的话下游 parsing 本来就会读）。
        """

        # 只读：不开显式事务。
        async with self._repositories.scope(begin=False) as repos:
            entity = await repos.create(FileRepository).get(file_id)

        if entity is None:
            raise ValueError(f"File not found: {file_id}")

        return {
            "file_id": str(entity.file_id),
            "checksum": entity.checksum,
            "kind": entity.file_name,
            "mime_type": entity.content_type,
            "byte_size": entity.file_size,
            "storage_key": entity.storage_key,
            "char_count": None,
            "token_count": None,
            "preview": preview[: self.PREVIEW_LIMIT],
        }

    async def get_text(
            self,
            file_id: UUID,
    ) -> str | None:

        content = await self.get_bytes(file_id)

        if content is None:
            return None

        return content.decode("utf-8")

    async def get_bytes(
            self,
            file_id: UUID,
    ) -> bytes | None:
        """
        按 file_id 取回内容。

        走 FileService.get_file_content —— 它会先查库拿
        storage_key 再读存储。多一次主键查询，换来的是
        storage_key 不作为身份出现在 state 里（它是实现细节，
        而且换存储后端时会变）。
        """

        # 只读：不开显式事务。
        async with self._services.scope(
            FileService, begin=False
        ) as service:

            try:
                response = await service.get_file_content(
                    GetFileContentRequest(file_id=file_id)
                )

            except ValueError:
                # File 不存在 / 已逻辑删除。返回 None 让调用方
                # 决定是抛错还是走别的分支。
                return None

        return response.content

    # ========================================================
    # Internals
    # ========================================================

    async def _store(
            self,
            *,
            kind: str,
            content: bytes,
            content_type: str | None,
            tenant_id: UUID | None,
            metadata: dict[str, Any] | None,
    ):
        """
        一段短事务完成写入。

        去重在 store_artifact 内部完成（按 tenant + checksum
        复用既有 File），因此 reprocess 同一文件时不会重复落盘。
        """

        async with self._services.scope(FileService) as service:
            return await service.store_artifact(
                kind=kind,
                content=content,
                content_type=content_type,
                tenant_id=tenant_id,
                metadata=metadata,
            )

    @staticmethod
    def _to_ref(
            file_response: Any,
            *,
            char_count: int | None,
            token_count: int | None,
            preview: str,
    ) -> dict[str, Any]:
        """
        构造 ArtifactRef。

        preview 是**唯一**进入 state 的内容，必须截断 ——
        它让 state 大小与文档大小解耦。
        assert_checkpoint_safe 会在有人塞进完整正文时报错。
        """

        return {
            "file_id": str(file_response.file_id),
            "checksum": file_response.checksum,
            "kind": file_response.file_name,
            "mime_type": file_response.content_type,
            "byte_size": file_response.file_size,
            "char_count": char_count,
            "token_count": token_count,
            "preview": preview,
        }
