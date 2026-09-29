from abc import abstractmethod

from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentProcessorRetrievalBinding


class DocumentProcessorRetrievalBindingRepository(BaseRepository):
    """
    N:N 绑定关系持久化抽象。

    这是纯配置关系，没有独立的生命周期状态：
    级联保存时整体替换，不做软删除。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()。
    """

    @abstractmethod
    async def replace_for_processor(
            self,
            processor_definition_id: UUID,
            retrieval_definition_ids: list[UUID],
    ) -> None:
        """
        整体替换某个前处理定义的绑定集合。

        空列表表示解绑全部。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_by_processor(
            self,
            processor_definition_id: UUID,
    ) -> list[DocumentProcessorRetrievalBinding]:
        raise NotImplementedError

    @abstractmethod
    async def list_by_type_definition(
            self,
            document_type_definition_id: UUID,
    ) -> list[DocumentProcessorRetrievalBinding]:
        """
        取出某个类型定义下的全部绑定。

        需要 join processor 表来限定所属类型。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_by_processors(
            self,
            processor_definition_ids: list[UUID],
    ) -> list[DocumentProcessorRetrievalBinding]:
        """
        批量取出多个前处理定义的绑定。

        用于给列表接口填充 retrieval_types 派生字段。
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_by_processors(
            self,
            processor_definition_ids: list[UUID],
    ) -> None:
        """
        删除这些前处理定义的全部绑定。

        级联保存中被软删的子项走这里，
        避免留下悬空引用。
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_by_retrievals(
            self,
            retrieval_definition_ids: list[UUID],
    ) -> None:
        raise NotImplementedError
