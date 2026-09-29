from abc import abstractmethod

from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentProcessorDefinition
from ..domain.schemas.enum import DefinitionStatus


class DocumentProcessorDefinitionRepository(BaseRepository):
    """
    文件前处理定义持久化抽象。

    该表描述 [后缀名 → loader → processor]，
    不涉及 chunker / retrieval / index。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()。
    """

    @abstractmethod
    async def add(
            self,
            processor: DocumentProcessorDefinition,
    ) -> DocumentProcessorDefinition:
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            processor_definition_id: UUID,
    ) -> DocumentProcessorDefinition | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_extension(
            self,
            document_type_definition_id: UUID,
            file_extension: str,
            *,
            include_deleted: bool = True,
    ) -> DocumentProcessorDefinition | None:
        """
        按自然键 (type, file_extension) 查询。

        自然键唯一约束包含已软删的行，
        因此级联保存时必须带上 include_deleted
        才能复用（复活）旧行。
        """
        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            processor: DocumentProcessorDefinition,
    ) -> DocumentProcessorDefinition:
        raise NotImplementedError

    @abstractmethod
    async def list(
            self,
            *,
            document_type_definition_id: UUID | None = None,
            file_extension: str | None = None,
            status: DefinitionStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentProcessorDefinition], int]:
        """分页查询。status=None 表示不过滤。"""
        raise NotImplementedError

    @abstractmethod
    async def list_all_by_type_definition(
            self,
            document_type_definition_id: UUID,
            *,
            include_deleted: bool = False,
    ) -> list[DocumentProcessorDefinition]:
        """取出某个类型定义下的全部前处理定义，不分页。"""
        raise NotImplementedError
