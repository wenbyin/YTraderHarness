from abc import abstractmethod

from typing import Any
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentChunk


class DocumentChunkRepository(BaseRepository):
    """
    DocumentChunk 持久化抽象。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()。
    """

    @abstractmethod
    async def add_many(
            self,
            chunks: list[DocumentChunk],
    ) -> None:
        """
        批量写入 Chunk。

        供后续接入真正的切分流水线时使用。
        当前版本没有调用方。
        """
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            document_chunk_id: UUID,
    ) -> DocumentChunk | None:
        raise NotImplementedError

    @abstractmethod
    async def list(
            self,
            *,
            document_version_id: UUID | None = None,
            chunker_key: str | None = None,
            chunker_config: dict[str, Any] | None = None,
            processor_definition_id: UUID | None = None,
            chunk_index: int | None = None,
            keyword: str | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentChunk], int]:
        """
        分页查询 Chunk。

        返回 (当前页实体列表, 满足条件的总数)。

        这里按 **chunker 身份** 过滤而不是按 retrieval
        ---------------------------------------------
        document_chunk 上存的就是 (chunker_key, chunker_config)，没有
        retrieval 列 —— 切分产出只取决于 chunker，见实体的说明。
        调用方（服务层）负责先把 retrieval 解析成身份：
        `chunker/identity.py` 的 chunker_identity() 给出同样的两项。

        仓储不自己做这个解析 —— 它没有注册表，也拿不到 chunker 实现的
        config_model，自己拼一份归一化逻辑必然与写入侧漂移。

        排序固定为 document_version_id, chunker_key, chunk_index。
        带上 chunker_key 是必要的：一个版本下可能有多份切分产出，
        只按 chunk_index 排会让它们交错在一起。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_ids(
            self,
            document_version_id: UUID,
            processor_definition_ids: list[UUID],
    ) -> list[UUID]:
        """
        取出指定版本下、由指定 Pipeline 产出的 Chunk ID。

        用于 DocumentDetail.canonical_chunk_ids。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_by_ids(
            self,
            document_chunk_ids: list[UUID],
    ) -> list[DocumentChunk]:
        """
        按主键批量取 Chunk。

        查询期回表用：索引召回只回 chunk_id / score / metadata，
        正文与归属（版本 / 文档 / 类型）只有 PostgreSQL 有。

        返回顺序**不保证**与入参一致 —— 调用方按 chunk_id 自建
        映射对齐。顺序是融合阶段的职责，不是这里的。
        入参为空时返回空列表，不要退化成全表扫描。
        """
        raise NotImplementedError
