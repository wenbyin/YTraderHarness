from abc import abstractmethod

from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentRetrievalDefinition
from ..domain.schemas.enum import (
    DefinitionStatus,
    RetrievalType,
)


class DocumentRetrievalDefinitionRepository(BaseRepository):
    """
    DocumentRetrievalDefinition 持久化抽象。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()。
    """

    @abstractmethod
    async def add(
            self,
            retrieval: DocumentRetrievalDefinition,
    ) -> DocumentRetrievalDefinition:
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            retrieval_definition_id: UUID,
    ) -> DocumentRetrievalDefinition | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_retrieval_type(
            self,
            document_type_definition_id: UUID,
            retrieval_type: RetrievalType,
            *,
            include_deleted: bool = True,
    ) -> DocumentRetrievalDefinition | None:
        """
        按自然键 (type, retrieval_type) 查询。

        自然键唯一约束包含已软删的行，
        因此级联保存时必须带上 include_deleted
        才能复用（复活）旧行。
        """
        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            retrieval: DocumentRetrievalDefinition,
    ) -> DocumentRetrievalDefinition:
        raise NotImplementedError

    @abstractmethod
    async def list(
            self,
            *,
            document_type_definition_id: UUID | None = None,
            retrieval_type: RetrievalType | None = None,
            status: DefinitionStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentRetrievalDefinition], int]:
        """分页查询。status=None 表示不过滤。"""
        raise NotImplementedError

    @abstractmethod
    async def list_all_by_type_definition(
            self,
            document_type_definition_id: UUID,
            *,
            include_deleted: bool = False,
    ) -> list[DocumentRetrievalDefinition]:
        """取出某个类型定义下的全部 retrieval，不分页。"""
        raise NotImplementedError

    @abstractmethod
    async def list_by_ids(
            self,
            retrieval_definition_ids: list[UUID],
    ) -> list[DocumentRetrievalDefinition]:
        raise NotImplementedError
