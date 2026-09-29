from abc import abstractmethod

from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import File
from ..domain.schemas.enum import FileOrigin, FileStatus


class FileRepository(BaseRepository):

    @abstractmethod
    async def add(
            self,
            file: File,
    ) -> File:
        """
        保存文件并返回 storage_key。
        """
        raise NotImplementedError

    @abstractmethod
    async def add_artifact_if_absent(
            self,
            file: File,
    ) -> File | None:
        """
        插入一行产物；若已存在同 (tenant, checksum, file_name) 的
        产物则**不插入**并返回 None。

        存在的意义是并发安全。产物去重是"先查后插"，而 Send 扇出
        的多个分支并发执行 —— 实测两个分支同时查到"不存在"、
        各插一行，同 checksum 的产物相差 2ms 写了两份。

        靠 uq_file_artifact_content 这个**偏唯一索引**兜底
        （只约束 origin='pipeline_artifact'，用户上传不受影响）。

        用 ON CONFLICT DO NOTHING 而不是捕获 IntegrityError：
        后者会让当前事务进入 aborted 状态，后续 SELECT 直接报错，
        拿不到"谁赢了"。DO NOTHING 不报错，事务可继续。

        Returns:
            插入成功 -> 新实体；冲突 -> None。
        """
        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            file: File
    ) -> File:
        """
        逻辑删除数据库文件 storage_key。
        """
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            file_id: UUID,
    ) -> File | None:
        raise NotImplementedError

    @abstractmethod
    async def get_active(
            self,
            file_id: UUID,
    ) -> File | None:
        """
        只返回未逻辑删除的 File。

        供需要“必须是有效文件”的 Service 操作使用。
        """
        raise NotImplementedError

    @abstractmethod
    async def soft_delete(
            self,
            file_id: UUID,
    ) -> File:
        """
        逻辑删除一个 File。

        只修改 PostgreSQL。
        不删除物理文件，不清理 storage。
        """
        raise NotImplementedError

    @abstractmethod
    async def exists_by_checksum(
            self,
            tenant_id: UUID | None,
            checksum: str,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def get_by_checksum(
            self,
            tenant_id: UUID | None,
            checksum: str,
            *,
            file_name: str | None = None,
            origin: FileOrigin | None = None,
    ) -> File | None:
        """
        按 (tenant_id, checksum) 取一个可复用的 active File。

        与 exists_by_checksum 的区别：本方法返回**实体**，
        调用方能拿到 file_id 与 storage_key 直接复用。

        用途是内容寻址去重 —— 流水线的中间产物会反复产生
        相同内容（reprocess 同一份文件、多次运行共享同一段
        正文），命中就复用既有 File，不重复落盘。

        file_name
            可选。产物去重时必须传 kind —— 见下方说明。

        origin
            可选。产物去重时必须传 pipeline_artifact ——
            否则可能复用到用户上传的同内容文件，
            那条 File 是用户资产，不该被当成产物引用。

        取不到或已逻辑删除时返回 None。

        为什么 file_name 要参与匹配
        ---------------------------
        只按 (tenant, checksum) 去重会**跨类型合并**：流水线里
        parsed_text 与 clean_text 在某次运行中内容恰好相同
        （占位实现下必然相同），后写的那个就会复用前者的 File，
        于是 ref 里的 kind 变成前者的 —— 它开始撒谎。

        kind 是产物在流水线模型里的身份，相同内容的不同产物
        仍是不同产物，因此去重键是 (tenant, checksum, kind)。
        """
        raise NotImplementedError

    @abstractmethod
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
        分页查询 File。

        返回：

            (当前页的实体列表, 满足条件的总数)

        总数与当前页数据在同一次数据库往返中获取，
        保证两者基于同一个快照。

        注意：
            Repository 只负责过滤、排序、分页。

            是否排除 deleted、如何解释 status 的优先级，
            由 Service 层决定后通过 status 参数传入。

            status=None 表示不按 status 过滤。
        """
        raise NotImplementedError
