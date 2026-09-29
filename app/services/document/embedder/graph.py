"""
图抽取实现（实体 + 关系）—— **实现先空置**。

它是什么
--------
把每一段文本抽成一张**局部图**：这段里提到了哪些实体、它们之间有什么
关系，结果是 GraphFragment（见 domain/schemas/processing.py）：

    {
      "entities":  [{"name": "贵州茅台", "type": "公司", "description": "..."}],
      "relations": [{"source": "贵州茅台", "target": "白酒",
                     "type": "属于", "description": "...", "weight": 1.0}]
    }

供 retrieval_type=graph 那一路上游使用 —— 它的产物最终进图库（Neo4j
之类），而不是向量索引。

⚠️ 它不产出向量，但它归在 embedder 这一族
-----------------------------------------
机制上是对的：阶段 9 本来就是"按这个 retrieval 需要什么，挑一个能力来
加工这批 chunk"，dense/sparse 产出向量、graph 产出图，只是产出物的三种
形态（见 EmbeddingOutput）。

但**名字**是从"向量化"那个角度起的，现在名不副实：一个产出图结构的
东西叫 Embedder、方法叫 embed()、还要回答 dimensions（对它毫无意义，
下面留 None）。

要不要把这一族改名（例如 ChunkRepresentation / produce()），是个独立
决定 —— 牵动 embedder/ 下的三个实现、注册表方法名、以及节点。这里
先按现状接入，不改名。

⚠️ 为什么空置还把它接进节点
----------------------------
因为不接的话，retrieval_type=graph 会**静默透传** —— 流水线照常成功、
版本照常转 active，但图谱是空的，而且没有任何地方说"这里没做"。

接上之后，配了 graph retrieval 的版本会**明确失败**并说明未实现。
这与本项目的既定原则一致：让"能力还没实现"在第一次运行时立刻暴露，
而不是悄悄跳过某个阶段假装成功。

想暂时恢复静默透传：把 embedding.py 里 _REQUIRED_OUTPUT 的 graph 那一行
注释掉即可（那里有说明）。
"""

from __future__ import annotations

from typing import Any

from app.services.document.domain.schemas import GraphFragment
from app.services.document.domain.schemas.enum import EmbeddingOutput

from .base import DocumentEmbedder

__all__ = ["GraphExtractionEmbedder"]


class GraphExtractionEmbedder(DocumentEmbedder):
    """
    实体 / 关系抽取器（**尚未实现**）。

    实现落地时大致要定这几件事（现在**刻意不写进 config_model** ——
    为一个还没实现的东西先摆一堆填不上的字段，只会让前端渲染出一张
    改不了任何行为的表单）：

        entity_types       要抽哪些类型的实体（公司 / 人 / 指标 …）
                           不收窄的话抽取会发散，同一段文本出来的实体
                           类型五花八门，后面归并也没法做
        relation_types     要抽哪些关系
        max_gleaning       抽完再自查几轮补漏（GraphRAG 默认 1 轮）
                           —— 直接决定 LLM 调用次数与成本
        model / prompt     用哪个模型、走哪套提示词

    与 chunker 的关系
    -----------------
    抽取的输入是**chunk 的正文**，所以上游 chunker 的粒度直接决定抽取
    质量：块太大，一次抽不干净还会漏；块太小，跨句的关系会断。
    GraphRAG 的默认（1200 token）就是为这件事调的 ——
    TEXT_TXT_GRAPH_CHUNKER 那套参数与它是配套的。
    """

    output_kind = EmbeddingOutput.GRAPH

    #: 对图抽取没有意义 —— 它不产出向量，没有"维度"这回事。
    #: 留 None 而不是编一个数字：索引侧对该形态不做维度校验
    #: （见 embedding.py 的 _validate）。
    dimensions = None

    name = "TEXT_GRAPH_EMBEDDER"

    #: 暂无参数。参数随实现一起落地，理由见类 docstring。
    config_model = None

    async def embed(
            self,
            texts: list[str],
            *,
            config: dict[str, Any],
    ) -> list[GraphFragment]:
        """
        对每段文本抽出一张局部图。

        **尚未实现。** 这里显式抛错而不是返回空图 —— 返回空图会让
        "抽取没做"伪装成"这段文本里确实没有实体"，流水线照常成功、
        版本照常转 active，而图谱是空的且无人知晓。
        """

        raise NotImplementedError(
            f"{self.get_key()}: 实体/关系抽取尚未实现。"
            "配了 retrieval_type=graph 的版本会在此失败 —— "
            "这是刻意的：静默产出一张空图会让「没做」看起来像「做完了」。"
            "实现落地前请先不要启用 graph retrieval。"
        )

