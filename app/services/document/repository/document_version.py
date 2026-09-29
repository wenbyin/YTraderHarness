from abc import abstractmethod

from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentVersion


class DocumentVersionRepository(BaseRepository):
    """
    DocumentVersion 持久化抽象。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()。
    """

    @abstractmethod
    async def add(
            self,
            version: DocumentVersion,
    ) -> DocumentVersion:
        """新增一个 DocumentVersion。只操作 PostgreSQL，不 commit。"""
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            document_version_id: UUID,
    ) -> DocumentVersion | None:
        raise NotImplementedError

    @abstractmethod
    async def list_active_ids_by_type_definitions(
            self,
            document_type_definition_ids: list[UUID],
    ) -> list[UUID]:
        """
        取这些文档类型下**当前生效版本**的 id。

        供查询期做"只检索当前激活版本"的过滤：索引里同时存着旧版本与
        新版本的 chunk（reprocess 只增不删），不把生效版本下推到索引层
        的话，top_k 会被陈旧版本吃掉。

        "当前生效"以 DocumentVersion.version_status == ACTIVE 为准 ——
        它由 activate() 维护（激活某版本时把同文档下其余 ACTIVE 的
        置为 SUPERSEDED），所以一个文档至多有一个。

        入参为空时返回空列表。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_many(
            self,
            document_version_ids: list[UUID],
    ) -> list[DocumentVersion]:
        """
        按主键批量取版本。

        查询期回表用：Chunk 只带 document_version_id，
        要拿到 document_id 得经过版本这一跳。
        返回顺序不保证与入参一致。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_by_document(
            self,
            document_id: UUID,
            *,
            status: VersionStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentVersion], int]:
        """
        分页查询某个 Document 下的版本。

        参数与 GET /documents/{id}/versions 的 query 参数对应。
        按 version_number 升序。

        返回 (当前页实体列表, 总命中数)。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_all_by_document(
            self,
            document_id: UUID,
    ) -> list[DocumentVersion]:
        """
        取出某个 Document 下的全部版本，不分页。

        供详情页一次性返回 versions 使用。
        """
        raise NotImplementedError

    @abstractmethod
    async def activate(
            self,
            document_id: UUID,
            version_id: UUID,
            *,
            now: datetime,
    ) -> int:
        """
        把 version_id 置为 active，同文档下**其他仍为 active** 的版本
        让位（superseded）。返回被让位的版本数。

        谁调用它
        --------
        所有"把某个版本设为文档当前版本"的地方：

            finalize 节点      流水线跑完、且全部成功时
            update_document    用户显式切换当前版本时

        两处都必须走这里，否则会出现"文档指向某版本、但该版本的
        version_status 仍是 pending"这种自相矛盾的状态 —— 而前端正是
        靠 version_status 决定给用户看哪一版的。

        为什么只动 active 的那些
        ------------------------
        其余状态原样保留，各有理由：

            deleted     已逻辑删除。改成 superseded 等于**把它复活**。
            failed      承载着"这版为什么不能用"，盖掉就看不出它失败过。
            pending     还没跑完。它跑完时自己会来争 active，现在抢先
                        标掉只是徒增抖动。
            superseded  已经让位过了，重复写没有意义。

        只动 active 的结果正好是那条不变式：**一个文档下至多一个 active**。

        只操作 PostgreSQL，不 commit。
        """
        raise NotImplementedError

    @abstractmethod
    async def max_version_number(
            self,
            document_id: UUID,
    ) -> int:
        """
        该 Document 当前最大版本号。

        没有版本时返回 0。
        """
        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            version: DocumentVersion,
    ) -> DocumentVersion:
        """更新一个 DocumentVersion。只操作 PostgreSQL，不 commit。"""
        raise NotImplementedError
