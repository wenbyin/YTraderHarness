from abc import abstractmethod

from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import Document
from ..domain.schemas.enum import DocumentStatus


class DocumentRepository(BaseRepository):
    """
    逻辑 Document 持久化抽象。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()，
    事务生命周期由 Service 层拥有。
    """

    @abstractmethod
    async def add(
            self,
            document: Document,
    ) -> Document:
        """新增一个 Document。只操作 PostgreSQL，不 commit。"""
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            document_id: UUID,
    ) -> Document | None:
        """
        按主键查询。

        已逻辑删除的记录也会返回，
        是否允许使用由 Service 层决定。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_many(
            self,
            document_ids: list[UUID],
    ) -> list[Document]:
        """
        按主键批量取 Document。

        查询期回表用：要拿 Document.document_type_definition_id
        才能给结果标注它属于哪个业务类型。
        返回顺序不保证与入参一致。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_by_key(
            self,
            tenant_id: UUID | None,
            document_type_definition_id: UUID,
            document_key: str,
    ) -> Document | None:
        """
        按业务唯一标识查询。

        用于创建时校验 document_key 是否重复。
        """
        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            document: Document,
    ) -> Document:
        """更新一个 Document。只操作 PostgreSQL，不 commit。"""
        raise NotImplementedError

    @abstractmethod
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
        """
        分页查询 Document 列表。

        参数与 GET /documents 的 query 参数一一对应。

        keyword 对 title / document_key 做不区分大小写包含匹配。
        排序固定为 created_at 倒序，第二排序键 document_id 保证翻页稳定。

        返回 (当前页实体列表, 总命中数)。
        """
        raise NotImplementedError

    @abstractmethod
    async def soft_delete(
        self,
        document_id: UUID,
    ) -> Document:
        """
        逻辑删除 Document。

        只改 status + deleted_at，不物理删除，
        也不触碰该 Document 下的版本与 Chunk。

        Raises:
            ValueError: Document 不存在或已删除。
        """
        raise NotImplementedError
