from abc import abstractmethod

from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentIndexBinding


class DocumentIndexBindingRepository(BaseRepository):
    """
    DocumentIndexBinding 持久化抽象。

    该表只描述“某个 RetrievalDefinition 实际落到哪个 Index 实例”，
    是纯实例关系，**不承载版本级执行状态**。

    版本 × Retrieval 的执行状态在
    DocumentVersionRetrievalBinding 上，
    因为同一个 Index 实例会被多个版本复用，
    状态挂在实例上会互相覆盖。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()。
    """

    @abstractmethod
    async def add_many(
            self,
            bindings: list[DocumentIndexBinding],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            document_index_binding_id: UUID,
    ) -> DocumentIndexBinding | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_instance(
            self,
            document_retrieval_definition_id: UUID,
            backend_type: str,
            index_identifier: str,
    ) -> DocumentIndexBinding | None:
        """
        按唯一键
        (retrieval_definition_id, backend_type, index_identifier)
        查询，用于写入前复用已有实例关系。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_by_retrieval_definition(
            self,
            document_retrieval_definition_id: UUID,
    ) -> list[DocumentIndexBinding]:
        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            binding: DocumentIndexBinding,
    ) -> DocumentIndexBinding:
        """更新 backend_type / index_identifier。"""
        raise NotImplementedError

    @abstractmethod
    async def delete_by_retrieval_definitions(
            self,
            retrieval_definition_ids: list[UUID],
    ) -> None:
        """
        删除这些 RetrievalDefinition 的全部实例关系。

        供类型定义级联删除时清理。
        """
        raise NotImplementedError
