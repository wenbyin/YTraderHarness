"""
处理结果页的查询契约：按 版本 × retrieval 查 Chunk。

为什么单独一个服务
------------------
它读的是**业务表**（document_chunk / document_version_retrieval_binding /
document_retrieval_definition），需要数据库事务 —— 与只读进程内静态数据的
ProcessorCapabilityService 不同。

但它也不该塞进 DocumentRuntimeService（665 行，职责是文档/版本的生命周期）
或 PipelineQueryService（执行视角：运行记录与节点行）。这是**产物视角**：
这个版本切出了什么、每一路切了多少、内容长什么样。三个读者、三种变更频率，
合在一起只会互相拖累。

只读，无副作用。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas.version_chunks import (
    ChunkProjectionListResponse,
    DocumentChunkListResponse,
    DocumentVersionRetrievalListResponse,
)

__all__ = ["VersionChunkService"]


class VersionChunkService(BaseService):
    """版本下的 retrieval（tab）与 chunk 的只读查询。"""

    @abstractmethod
    async def list_retrievals(
            self,
            document_version_id: UUID,
    ) -> DocumentVersionRetrievalListResponse:
        """
        某个版本实际参与处理的 retrieval 列表 —— 前端据此生成 N 个 tab。

        数据来源是版本级的绑定记录（DocumentVersionRetrievalBinding），
        因此**只返回这个版本真正跑过的** retrieval，而不是该类型下当前
        配置的全部 retrieval。

        Raises:
            NotFoundError: 该版本不存在。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_chunks(
            self,
            document_version_id: UUID,
            document_retrieval_definition_id: UUID,
            *,
            preview_chars: int | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> DocumentChunkListResponse:
        """
        某个 tab（= 某版本下的某一路 retrieval）的 chunk 列表。

        排序固定为 chunk_index 升序：它是该 retrieval 内从 0 连续的编号，
        因此页与页之间不会漂移，翻页不会重复或漏块。

        preview_chars:
            只返回每条 content 的前 N 个字符，并用 content_truncated 标记。
            选中「不分块」那个 chunker 时整篇正文就是一个 chunk，
            单条可能几十万字 —— 这是给那种情况的防护。

        Raises:
            NotFoundError: 该版本不存在，或该 retrieval 没有参与这个版本。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_projections(
            self,
            document_chunk_ids: list[UUID],
            *,
            document_retrieval_definition_id: UUID | None = None,
            include_vector: bool = False,
    ) -> ChunkProjectionListResponse:
        """
        按 chunk_id 批量取**投影** —— 这些 chunk 当时写给索引的是什么。

        与 list_chunks 的区别：那个给的是**原件**（PG 里的 chunk），
        这个给的是**投影**（带向量、只含索引需要的字段的那份产物）。

        一条 chunk 会有**多路**投影
        --------------------------
        chunk 是版本级的事实，不归任何一路 retrieval（见 DocumentChunk
        的说明），所以同一块会被多路各自写进不同索引、形态还不同。
        因此响应的每一项里是一个 projections 列表，而不是单个投影。

        传 document_retrieval_definition_id 可以只看其中一路。

        数据来源是 embedding 节点写出的产物，不是索引本身 ——
        所以索引挂着也能看，代价是它反映的是"当时发出去的"而不是
        "索引里现在存着的"。

        结果与入参**一一对应且同序**，取不到的那些用
        unavailable_reason 说明原因（不抛错、也不悄悄丢掉）——
        前端点了一条却什么都拿不到时，需要知道为什么。

        include_vector:
            true 时额外返回完整向量。稠密向量 512 个 float，稀疏向量
            动辄几百项，列表默认只给摘要。
        """
        raise NotImplementedError
