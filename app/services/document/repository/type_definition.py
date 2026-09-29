from abc import abstractmethod

from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentTypeDefinition
from ..domain.schemas.enum import DefinitionStatus


class DocumentTypeDefinitionRepository(BaseRepository):
    """
    DocumentTypeDefinition 持久化抽象。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()，
    事务生命周期由 Service 层拥有。
    """

    @abstractmethod
    async def add(
            self,
            definition: DocumentTypeDefinition,
    ) -> DocumentTypeDefinition:
        """
        新增一条 DocumentTypeDefinition。

        只操作 PostgreSQL。
        不 commit。
        """
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            document_type_definition_id: UUID,
    ) -> DocumentTypeDefinition | None:
        """
        按主键查询。

        已逻辑删除的记录也会返回，
        是否允许使用由 Service 层决定。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_many(
            self,
            document_type_definition_ids: list[UUID],
    ) -> list[DocumentTypeDefinition]:
        """
        按主键批量查询。

        查询期一次要校验最多 50 个 DTD，逐个 get 是 50 次往返。
        返回顺序不保证与入参一致 —— 调用方自建映射。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_by_type_key(
            self,
            type_key: str,
    ) -> DocumentTypeDefinition | None:
        """
        按 type_key 查询。

        type_key 唯一，用于创建时的重名检查。
        """
        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            definition: DocumentTypeDefinition,
    ) -> DocumentTypeDefinition:
        """
        更新一条 DocumentTypeDefinition。

        只操作 PostgreSQL。
        不 commit。
        """
        raise NotImplementedError

    @abstractmethod
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
        分页查询。

        返回 (当前页实体列表, 满足条件的总数)。

        keyword 同时匹配 type_key 与 name。

        status=None 表示不按 status 过滤。
        """
        raise NotImplementedError
