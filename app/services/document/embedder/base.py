"""
向量化能力契约（阶段 9，role=embed）。

职责：
    list[str] → list[list[float]]

不负责：
    Chunk（那是 chunker）
    Index（那是 IndexBackend）
    Retriever —— 前者把文本变成向量（写入期），后者用向量召回（查询期）。
    本模块只管前者。

顺序即正确性
------------
embed() 返回的向量必须与输入文本**等长且顺序一致**。
顺序错位**不会报错**，只会让检索质量"轻微变差" —— 这类问题极难
发现。因此实现侧必须校验长度（见 ollama.py），调用方则按 index
把向量写回各自的 chunk。

维度必须"调用前"就能拿到
------------------------
dimensions 是向量空间的事实，Milvus collection 的 schema 需要它 ——
也就是**建 collection 之前**就要能拿到，不能等第一次 embed 之后
再回填。因此它是类属性而不是"首次调用后才知道"的运行时值。
未知维度的实现留 None，由调用方在首次 embed 后回填。

与 embedder/ollama.py 的分工
----------------------------
本文件只声明契约（role、入口、维度），不含任何后端细节。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from app.services.document.domain.schemas import (
    EmbedRequest,
    GraphFragment,
    ProcessorResult,
)
from app.services.document.domain.schemas.enum import (
    EmbeddingOutput,
    ProcessorRole,
)

# ⚠️ DocumentProcessor 目前住在 parser/ 包里，与 cleaner/base.py、
# chunker/base.py 同样的历史错位。继承关系本身是对的。
from app.services.document.parser.processor import (
    TextDocumentProcessor,
)

__all__ = ["DocumentEmbedder"]


class DocumentEmbedder(TextDocumentProcessor):
    """
    向量化实现契约。

    与 DocumentParser / DocumentCleaner / DocumentChunker 平级：
    都继承 DocumentProcessor，彼此互不继承。
    """

    role = ProcessorRole.EMBED

    #: 产出哪种形态的向量。决定 embed() 的返回结构，也决定它能不能
    #: 服务某一类 retrieval —— 见 EmbeddingOutput 的说明。
    output_kind: ClassVar[EmbeddingOutput] = EmbeddingOutput.DENSE

    #: 向量维度。建 Milvus collection 的 schema 需要它，因此在**任何
    #: 一次调用之前**就要能拿到（理由见模块 docstring）。
    #: 未知的实现留 None，由调用方在首次 embed 后回填。
    #:
    #: 对稀疏实现而言它是**哈希空间大小**：稀疏向量的"维度"是词表规模，
    #: 取值必须与写入索引时一致，且一旦变小，已有向量全部作废。
    dimensions: ClassVar[int | None] = None

    #: 查询侧要加的指令前缀，None 表示查询与文档两侧用法完全一致。
    #:
    #: 有些检索模型对**查询**与**文档**要求不同的输入构造方式（典型是
    #: BGE 一族的指令前缀）。两者不一致**不会报错** —— 维度对得上、
    #: 检索照常返回结果，只是召回率明显下降。所以这个差异必须由实现
    #: 自己声明，而不是让调用方去猜。
    query_instruction: ClassVar[str | None] = None

    #: 度量本模型输入长度时该用**哪份词表**（HF 模型名或本地目录）。
    #: None = 本实现没有可用的词表，调用方不该按模型上限量长度。
    #:
    #: 为什么由实现声明而不是调用方去猜：模型对单次输入的 token 上限
    #: 是在**它自己词表**里算的。拿 tiktoken 之类的通用编码去比会系统
    #: 性低估（实测中文下 bge 要比 o200k_base 多吃 16%~33%），后果是
    #: 模型**静默截断** —— 不报错、向量照常产出，只是 chunk 的尾巴从未
    #: 被向量化。所以"跟模型上限比大小"这件事必须问实现要词表。
    tokenizer_id: ClassVar[str | None] = None

    def get_process_type(self) -> str:
        return "PROCESSOR_EMBEDDER"

    async def embed_query(
            self,
            text: str,
            *,
            config: dict[str, Any],
    ) -> list[float] | dict[int, float]:
        """
        把**查询文本**向量化，供查询期召回使用。

        与 embed() 的区别只有一处：按 query_instruction 加前缀。
        分批 / 归一化 / 保序校验全部复用 embed()，这里不重复实现。

        为什么单独一个方法而不是让调用方自己拼前缀：
        前缀是**模型的身份**，不是调用方该知道的事。放在实现里，
        换模型时前缀跟着换，调用方一行不用动。

        返回单条向量（不是列表）—— 查询侧一次只向量化一个字符串，
        让调用方去取 [0] 只会多一次可能搞错的索引。
        """

        prepared = (
            f"{self.query_instruction}{text}"
            if self.query_instruction
            else text
        )

        vectors = await self.embed([prepared], config=config)

        if len(vectors) != 1:
            raise ValueError(
                f"{self.get_key()}: embed_query 期望 1 个向量，"
                f"实际得到 {len(vectors)} 个。"
            )

        return vectors[0]

    async def process(
            self,
            request: EmbedRequest,
    ) -> ProcessorResult:
        """
        统一入口：节点只认 process()。

        embed 的输入与产物都是列表，因此这里把向量列表装进
        ProcessorResult.vectors（与 chunker 装 chunks 同理）。
        具体是 list[list[float]] 还是 list[dict[int, float]]，
        由 output_kind 决定。
        """

        vectors = await self.embed(
            request.texts,
            config=request.config,
        )

        return ProcessorResult(
            document_id=request.document_id,
            document_version_id=request.document_version_id,
            vectors=vectors,
            # metadata 是累积的 —— 这一阶段不增不减，原样带回上游那份。
            metadata=dict(request.metadata),
        )

    @abstractmethod
    async def embed(
            self,
            texts: list[str],
            *,
            config: dict[str, Any],
    ) -> (
        list[list[float]]
        | list[dict[int, float]]
        | list[GraphFragment]
    ):
        """
        加工一批文本，产出与之一一对应的表示。

        Returns:
            与 texts **等长且顺序一致**的列表。具体形态由 output_kind
            决定：

                DENSE   list[list[float]]        定长浮点数组
                SPARSE  list[dict[int, float]]   维度索引 -> 权重
                GRAPH   list[GraphFragment]      实体提及 + 关系的局部图

            稀疏实现**不要**把稀疏向量展开成定长稠密列表再返回 ——
            词表空间 20 万，展开后每条 chunk 都是几十万个浮点，
            内存与序列化都会被拖垮。形态本来就是稀疏的，就按稀疏返回。

        Raises:
            ValueError: 返回长度与输入不一致时（实现侧必须自检，
                不能把这个判断留给调用方）。
        """
        raise NotImplementedError
