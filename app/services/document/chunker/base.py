"""
分块能力契约（阶段 7，role=chunk）。

职责：
    把清洗后的正文按 RetrievalDefinition 的配置切成若干块。

不负责：
    归一化 / 清洗 / 元数据抽取（阶段 4-6，那是 processor 一族）
    向量化（阶段 9，Embedder）
    Index / Storage

分块与其它 role 的关键差别：它**改变基数**
------------------------------------------
一份正文进，N 个 chunk 出。因此 process() 返回的是
ProcessorResult.chunks，而不是 text —— text 表达不了列表。

产物是 **domain entity**，不是 CanonicalChunk
--------------------------------------------
    DocumentChunk   带 document_retrieval_definition_id / chunk_index /
                    content_hash，可以直接落 document_chunk 表
    CanonicalChunk  写给索引后端的投影，由阶段 10 现算

两者不要混：这是 chunking.py 的 TODO 里专门强调过的一点。

分块是**扇出**阶段：同一个 DocumentVersion 按每个 RetrievalDefinition
各切一份，因此 document_retrieval_definition_id 是 ChunkRequest 的
必填项（见 processing.py 的说明）。
"""

from __future__ import annotations

from abc import abstractmethod

from app.services.document.domain.entities import DocumentChunk
from app.services.document.domain.schemas import (
    ChunkRequest,
    ProcessorResult,
)
from app.services.document.domain.schemas.enum import ProcessorRole

# ⚠️ DocumentProcessor 目前住在 parser/ 包里，与 cleaner/base.py 同样的
# 历史错位。继承关系本身是对的，只是 import 路径难看。
from app.services.document.parser.processor import (
    TextDocumentProcessor,
)

__all__ = ["DocumentChunker"]


class DocumentChunker(TextDocumentProcessor):
    """
    分块实现契约。

    与 DocumentParser / DocumentCleaner 平级：都继承
    DocumentProcessor，彼此互不继承。
    """

    role = ProcessorRole.CHUNK

    def get_process_type(self) -> str:
        return "PROCESSOR_CHUNKER"

    async def process(
            self,
            request: ChunkRequest,
    ) -> ProcessorResult:
        """统一入口：节点只认 process()。"""

        chunks = await self.chunk(request)

        return ProcessorResult(
            document_id=request.document_id,
            document_version_id=request.document_version_id,
            # 分块是唯一改变基数的 role，产物走 chunks 通道。
            chunks=chunks,
            # metadata 是累积的 —— 这一阶段不增不减，原样带回上游那份。
            metadata=dict(request.metadata),
        )

    @abstractmethod
    async def chunk(
            self,
            request: ChunkRequest,
    ) -> list[DocumentChunk]:
        """
        切分正文，返回**完整的** DocumentChunk 实体。

        身份字段（document_chunk_id / chunk_index / content_hash）
        由实现侧统一填，调用方拿到即可落库。
        """
        raise NotImplementedError
