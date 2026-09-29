"""
Repository contract for DocumentVersionRetrievalBinding.

Transaction rule
----------------
实现不得调用 session.commit() / session.rollback()，
事务边界由调用方（组合根或 pipeline 的 node_span）决定。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentVersionRetrievalBinding
from ..domain.schemas.enum import ProcessingStatus


class DocumentVersionRetrievalBindingRepository(BaseRepository):
    """
    DocumentVersionRetrievalBinding 的持久化契约。

    写入以 upsert 为主：同一 (document_version_id,
    document_retrieval_definition_id) 在重复处理时应更新而非插入，
    这与表上的 uq_document_version_retrieval_binding 约束一致。
    """

    @abstractmethod
    async def upsert(
            self,
            binding: DocumentVersionRetrievalBinding,
    ) -> DocumentVersionRetrievalBinding:
        """
        按 (document_version_id, document_retrieval_definition_id)
        插入或更新一行。
        """
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            document_version_retrieval_binding_id: UUID,
    ) -> DocumentVersionRetrievalBinding | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_version_and_retrieval(
            self,
            document_version_id: UUID,
            document_retrieval_definition_id: UUID,
    ) -> DocumentVersionRetrievalBinding | None:
        raise NotImplementedError

    @abstractmethod
    async def list_by_version(
            self,
            document_version_id: UUID,
    ) -> list[DocumentVersionRetrievalBinding]:
        """
        列出某个版本的全部 retrieval 绑定。

        用于聚合 DocumentVersion.processing_status：
            全部 completed 才是 completed；
            任一 failed 即 failed；
            任一 processing 即 processing；
            否则 pending。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_by_versions(
            self,
            document_version_ids: list[UUID],
    ) -> list[DocumentVersionRetrievalBinding]:
        raise NotImplementedError

    @abstractmethod
    async def delete_by_version(
            self,
            document_version_id: UUID,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def aggregate_status(
            bindings: list[DocumentVersionRetrievalBinding],
    ) -> ProcessingStatus:
        """
        由若干 Retrieval 绑定聚合出版本级的 processing_status。

        规则（见 ProcessingStatus 的 docstring）：
            空集合                          -> pending
            任一 failed                     -> failed
            任一 processing                 -> processing
            全部 completed                  -> completed
            否则                            -> pending

        这是纯函数，不碰数据库，因此放在 ABC 上供各实现复用。
        """

        if not bindings:
            return ProcessingStatus.PENDING

        statuses = {b.processing_status for b in bindings}

        if ProcessingStatus.FAILED in statuses:
            return ProcessingStatus.FAILED

        if ProcessingStatus.PROCESSING in statuses:
            return ProcessingStatus.PROCESSING

        if statuses == {ProcessingStatus.COMPLETED}:
            return ProcessingStatus.COMPLETED

        return ProcessingStatus.PENDING
