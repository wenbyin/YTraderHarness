"""
Chunk 增强能力契约（阶段 8，role=chunk_enrich）。

职责：
    给切好的 chunk 补上**它自己正文里没有、但它属于谁**的信息。

    - 继承自文档级的元数据（标题、来源、业务标签、公司/主体标识）
    - 定位信息（页码、章节路径、表格坐标）
    - 为检索服务的派生字段

    典型场景：一份按公司维度组织的长文档，中间的纯数据 chunk（例如
    一长串股东户数）正文里根本不会重复公司名 —— 查询时"公司名"这个
    条件在 chunk 层面就是不可见的。增强阶段把文档级的公司标识继承
    下来，查询侧才能用它做**过滤**（而不是靠相似度碰运气）。

为什么并入 DocumentProcessor 体系
--------------------------------
原本它是个独立的 ABC（policy/enricher.py），结果：

    - CAPABILITY_PACKAGES 里没有 policy 包
    - found_capabilities() 只反射 DocumentProcessor 的子类

两条加起来的后果是它**从来没有被注册过** —— registry.enrichers 那张
表一直是空的，registry.enricher(name) 必然抛 CapabilityNotFound。

chunker / embedder 早就并进来了（见 registry.py 里那两段说明），
enricher 是同一条路上的第三个。ProcessorRole.CHUNK_ENRICH 这个枚举值
也一直存在，现在终于有实现挂上去。

为什么必须用 process() 而不是直接暴露 enrich()
--------------------------------------------
节点只认 process()（与 parser / cleaner / chunker / embedder 一致），
enrich() 是实现侧要写的、面向领域的方法。这层包装与
embedder/base.py 的 embed() 完全同构。

不得改动 content 与 chunk_index
-------------------------------
两者撑起了 document_chunk 表上的
UNIQUE(document_version_id, chunker_key, chunker_config, chunk_index)
与 content_hash 的一致性。改了它们，唯一约束会判重、content_hash 会
对不上正文，而且**不报错**。

所以增强只能加在 metadata / locator 上。想把上下文注入**正文**去
改善向量召回的话，那是**向量化那一步**的事（拼给 embedder 的文本），
不是这里。

顺序即正确性
------------
enrich() 返回的列表必须与输入**等长且等序** —— 调用方按 chunk_index
与产物里的条目对齐。错位不会报错，只会让 metadata 张冠李戴。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from ..domain.schemas import (
    EnrichRequest,
    ProcessorResult,
)
from ..domain.schemas.enum import ProcessorRole

# ⚠️ DocumentProcessor 目前住在 parser/ 包里，与 cleaner/base.py、
# chunker/base.py 同样的历史错位。继承关系本身是对的。
from ..parser.processor import TextDocumentProcessor

__all__ = ["DocumentChunkEnricher"]


class DocumentChunkEnricher(TextDocumentProcessor):
    """
    Chunk 增强实现契约。

    与 DocumentParser / DocumentCleaner / DocumentChunker /
    DocumentEmbedder 平级：都继承 DocumentProcessor，彼此互不继承。
    """

    role = ProcessorRole.CHUNK_ENRICH

    def get_process_type(self) -> str:
        return "PROCESSOR_ENRICHER"

    async def process(
            self,
            request: EnrichRequest,
    ) -> ProcessorResult:
        """
        统一入口：节点只认 process()。

        embedder 把向量装进 ProcessorResult.vectors，这里同理把增强
        后的 chunk 装进 .chunks —— 产物形状与 chunker 一致，下游
        （embedding / indexing）不用认识"这一阶段做过什么"。
        """

        enriched = await self.enrich(
            list(request.chunks),
            config=request.config,
            # 取 document_metadata 而不是 metadata —— 后者的语义是
            # "阶段间累积的元数据袋"，拿它当文档事实用会让增强静默地
            # 什么都不加（见 EnrichRequest.document_metadata 的说明）。
            document_metadata=dict(request.document_metadata),
        )

        # 顺序即正确性：错位不会报错，只会让 metadata 挂到别的 chunk 上。
        if len(enriched) != len(request.chunks):
            raise ValueError(
                f"{self.get_key()}: enrich() 返回 {len(enriched)} 条，"
                f"但输入是 {len(request.chunks)} 条。"
                "两者必须一一对应 —— 错位会让元数据张冠李戴，"
                "而且不会报错。"
            )

        return ProcessorResult(
            document_id=request.document_id,
            document_version_id=request.document_version_id,
            chunks=enriched,
            metadata=dict(request.metadata),
        )

    @abstractmethod
    async def enrich(
            self,
            chunks: list,
            *,
            config: dict[str, Any],
            document_metadata: dict[str, Any],
    ) -> list:
        """
        就地增强一批 chunk 并返回。

        Args:
            chunks:
                待增强的 DocumentChunk。**不要改它们的 content 与
                chunk_index**（见模块 docstring）。

            config:
                该路 retrieval 的配置（当前是 chunker_config 那份，
                阶段 8 还没有自己的配置字段）。

            document_metadata:
                文档级元数据 —— 标题、来源、业务标识等要继承下去的东西。

        Returns:
            与入参**等长且等序**的列表。原地修改后返回同一个列表也可以，
            但调用方按位置对齐，所以顺序不能变。

        可选产出 `chunk.embed_text`：
            喂给阶段 9 向量化的文本（`前缀 + 空格 + 正文`）。不产出也不会
            报错 —— 阶段 9 会退回 content —— 但那等于这条分支的向量看不见
            "这块属于谁"，正是这个字段要解决的问题。所以要么老老实实产，
            要么让"没产"在观测里看得见（阶段 9 会记 embed_text_missing）。
        """
        raise NotImplementedError
