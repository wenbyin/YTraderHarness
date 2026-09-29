"""
阶段 9：Embedding（扇出）。

归属：DocumentRetrievalDefinition。

哪些 retrieval 要跑这一步
--------------------------
由 retrieval_type 决定，且**决定用哪种能力**：

    dense_vector   -> 稠密向量   Milvus FLOAT_VECTOR / ES dense_vector
    sparse_vector  -> 稀疏向量   Milvus SPARSE_FLOAT_VECTOR / ES sparse_vector
    graph          -> 局部图     实体提及 + 关系 -> 图库（Neo4j 之类）
    document / keyword -> 什么都不用产，透传

三种产物**不可互换**：拿稠密模型去服务 sparse_vector 不会报错 ——
维度对得上、检索照常返回结果，只是排序没有意义。所以这里按形态精确
匹配能力（registry.embedder_for），而不是"有 embedder 就用"。

⚠️ 注意 graph 那一路产出的**不是向量**。本阶段叫 embedding 是历史叫法，
它真正的职责是"按 retrieval 需要什么，加工这一批 chunk"。

顺序即正确性
------------
embed() 返回的向量必须与输入 chunk **严格等序**。
顺序错位**不会报错**，只会让检索质量"轻微变差" —— 这类问题极难发现。
因此本节点做三件事：

    1. 取出 chunk 后**先按 chunk_index 排序**再抽正文。
       上游写出来时本来就有序，但不能依赖"它应该有序"。
    2. 拿到向量后校验条数。
    3. 校验维度/维度索引范围。

产物的形状
----------
写出的是一份**自描述**的对象，而不是裸的向量数组 —— 索引阶段需要知道
"这是什么形态、多少维、哪条 chunk 对应哪个向量"：

    {
      "kind": "dense",
      "embedder": "TEXT_BGE_SMALL_ZH_EMBEDDER",
      "dimensions": 512,
      "items": [
        {"document_chunk_id": "...", "chunk_index": 0, "vector": [...]},
        ...
      ]
    }

稀疏时 items 里是 "sparse": {"13323": 0.38, ...} —— JSON 的键只能是
字符串，所以维度索引会被序列化成字符串，读的一方要转回 int。

图抽取时是 "graph": {"entities": [...], "relations": [...]} —— 每段文本
一张**局部**图，不保证跨段连得上（归并是后续阶段的事）。
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span

from app.services.document.domain.schemas.enum import (
    RETRIEVAL_EMBEDDING_KIND,
    EmbeddingOutput,
    ProcessorRole,
)

from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..state import BranchInput
from ._branch import branch_span_kwargs, ref_summary
from ._common import file_id_of, tenant_of

_SPEC = next(s for s in NODE_SPECS if s.node_id == "embedding")

#: retrieval_type -> 本阶段要产出什么。
#:
#: 不在表里的类型**什么都不用产**，走透传（仍要占住 vector_refs 通道的
#: 位置，否则下一阶段的路由会以为这个分支在上游就失败了）。
#:
#: ⚠️ graph 那一行会让配了 graph retrieval 的版本**失败**，因为
#: GraphExtractionEmbedder 还没实现（它显式抛 NotImplementedError）。
#:
#: 这是刻意的：不接的话 graph retrieval 会**静默透传** —— 流水线照常
#: 成功、版本照常转 active，但图谱是空的，且没有任何地方说"这里没做"。
#:
#: 想暂时恢复静默透传：把 graph 这一行注释掉即可。
#: retrieval_type -> 需要产出哪一种向量。
#:
#: 表本身在 domain/schemas/enum.py 的 RETRIEVAL_EMBEDDING_KIND ——
#: 查询期读的是**同一份**。原来两边各手写一张，正是"改了一处忘了
#: 另一处"的温床（不一致时维度照样对得上、检索照样返回，只是排序
#: 没有意义），所以收敛成了一处。
_REQUIRED_OUTPUT = RETRIEVAL_EMBEDDING_KIND

__all__ = ["embedding_node"]


async def embedding_node(
        state: BranchInput,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """向量化本分支的 chunk。返回 {"vector_refs": {rid: ArtifactRef}}。"""

    context = runtime.context
    branch = state["branch"]
    rid = branch["document_retrieval_definition_id"]
    retrieval_type = branch["retrieval_type"]

    async with node_span(
            runtime,
            **branch_span_kwargs(_SPEC, state),
            input_summary={
                "retrieval_type": retrieval_type,
                "upstream_ref": ref_summary(state.get("upstream_ref")),
            },
    ) as span:

        upstream = state.get("upstream_ref")

        # ----------------------------------------------------
        # 1. 不需要向量的类型：透传
        # ----------------------------------------------------

        required = _REQUIRED_OUTPUT.get(retrieval_type)

        if required is None:
            span.set_metrics({
                "skipped": True,
                "reason": "该 retrieval 类型不需要向量",
            })
            return {"vector_refs": {rid: upstream}}

        if upstream is None:
            raise ValueError(
                f"{rid}: embedding 拿不到 upstream_ref —— "
                "上游路由屏障没有把本分支的 chunk 产物注入进来。"
            )

        # ----------------------------------------------------
        # 2. 取 chunk
        # ----------------------------------------------------

        raw = await context.artifact_store.get_text(
            file_id_of(upstream)
        )

        chunks = json.loads(raw) if raw else []

        if not chunks:
            # 没有 chunk 就没有向量。写一份空产物而不是跳过，
            # 让下游拿到的形状保持一致。
            ref = await context.artifact_store.put_text(
                kind="vectors",
                text=json.dumps({
                    "kind": required.value,
                    "embedder": None,
                    "dimensions": None,
                    "items": [],
                }),
                tenant_id=tenant_of(state),
                content_type="application/json",
            )
            span.set_metrics({
                "vector_count": 0,
                "note": "上游没有 chunk",
            })
            return {"vector_refs": {rid: ref}}

        # 顺序即正确性：先按 chunk_index 排序再抽正文。
        chunks.sort(key=lambda chunk: chunk["chunk_index"])

        texts = [chunk["content"] for chunk in chunks]

        # ----------------------------------------------------
        # 3. 按形态取 embedder
        # ----------------------------------------------------

        embedder = context.registry.embedder_for(required)

        if embedder is None:
            raise RuntimeError(
                f"{rid}: retrieval_type={retrieval_type!r} 需要 "
                f"{required.value} 向量，但该形态没有已注册的 embedder。"
                "应当显式失败，而不是产出一份没有向量的结果。"
            )

        # ----------------------------------------------------
        # 4. 执行
        # ----------------------------------------------------
        #
        # ⚠️ 配置来源：读的是 branch["retriever_config"]。
        #
        # 这个位置是**将就**的：retriever_config 本该只放查询期召回参数
        # （top_k / score_threshold 之类），把 embedding 参数放进去是
        # 两回事混一个字典。
        #
        # 先这么接的理由：写入期与查询期必须用**同一个** embedding 模型
        # 与参数，而查询期的配置就是 retriever 那份 —— 放一起至少保证
        # 两边读的是同一处。
        #
        # 未知键会被 config_model 忽略（extra=ignore），所以现在塞着
        # retriever 的参数不会报错，只是不生效。
        # 等配置表有了专门的 embedder_config 字段，改这里一行即可。
        # ----------------------------------------------------

        vectors = await embedder.embed(
            texts,
            config=dict(branch.get("retriever_config") or {}),
        )

        # ----------------------------------------------------
        # 5. 校验
        # ----------------------------------------------------

        if len(vectors) != len(texts):
            raise ValueError(
                f"{rid}: embedder 返回 {len(vectors)} 个向量，但输入是 "
                f"{len(texts)} 条 chunk —— 两者必须一一对应。"
                "错位不会报错，只会让检索质量轻微变差，极难发现。"
            )

        expected_dim = type(embedder).dimensions

        _validate(rid, required, vectors, expected_dim)

        # ----------------------------------------------------
        # 6. 写出
        # ----------------------------------------------------

        # 每条 item 都**带上 chunk 自己的字段**（content / hash / 计数 /
        # metadata），再挂上本阶段产出的表示。
        #
        # 为什么要把正文也带进来：下一阶段（indexing）只拿得到这一个
        # ref（扇出屏障只注入 upstream_ref），而 CanonicalChunk 的
        # content 是必填 —— 不在这里带上的话，索引阶段要么去查库
        # （在扇出节点里再开一次事务），要么拿到一份没有正文的 chunk。
        #
        # 代价是正文多存一份（产物是 File，不是 state）。对 1200 token
        # 的块，正文约占这份产物的 20%，可接受。
        if required is EmbeddingOutput.DENSE:
            items = [
                {**chunk, "vector": vector}
                for chunk, vector in zip(chunks, vectors)
            ]

        elif required is EmbeddingOutput.GRAPH:
            # 图抽取：每段文本一张局部图（实体提及 + 关系）。
            # 只做序列化 —— 归并成全局图是后续阶段的事，
            # 本阶段产出的东西**不保证**跨段连得上。
            items = [
                {
                    **chunk,
                    "graph": fragment.model_dump(mode="json"),
                }
                for chunk, fragment in zip(chunks, vectors)
            ]

        else:
            # JSON 的键只能是字符串 —— 维度索引序列化成字符串，
            # 读的一方负责转回 int。
            items = [
                {
                    **chunk,
                    "sparse": {
                        str(dim): weight
                        for dim, weight in vector.items()
                    },
                }
                for chunk, vector in zip(chunks, vectors)
            ]

        ref = await context.artifact_store.put_text(
            kind="vectors",
            text=json.dumps({
                "kind": required.value,
                "embedder": embedder.get_key(),
                "dimensions": expected_dim,
                "items": items,
            }),
            tenant_id=tenant_of(state),
            content_type="application/json",
        )

        span.set_metrics({
            "vector_count": len(items),
            "embedding_dimensions": expected_dim,
            "embedding_kind": required.value,
            "embedder_key": embedder.get_key(),
            "skipped": False,
        })

        span.set_output_summary({
            "ref": {"checksum": ref["checksum"]},
            "vector_ref": {
                "file_id": ref["file_id"],
            },
            "embedder": embedder.get_key(),
            "kind": required.value,
        })

        return {"vector_refs": {rid: ref}}


def _validate(
        rid: str,
        kind: EmbeddingOutput,
        vectors: list[Any],
        expected_dim: int | None,
) -> None:
    """
    维度校验。写进索引之前拦住，否则索引侧的 schema 会对不上。

    两种形态的校验不一样：
        稠密  每条向量的**长度**必须等于声明维度
        稀疏  每个**维度索引**必须落在 [0, 声明维度) 内
    """

    if expected_dim is None:
        return

    if kind is EmbeddingOutput.DENSE:

        wrong = [
            index for index, vector in enumerate(vectors)
            if len(vector) != expected_dim
        ]

        if wrong:
            raise ValueError(
                f"{rid}: 期望每条向量 {expected_dim} 维，但第 {wrong[:5]} "
                "条的维度不对 —— 索引的 schema 是按声明维度建的，"
                "写入前必须拦住。"
            )

        return

    # 稀疏：维度索引必须落在声明的哈希/词表空间内。
    for index, vector in enumerate(vectors):
        for dim in vector:
            if not isinstance(dim, int) or not 0 <= dim < expected_dim:
                raise ValueError(
                    f"{rid}: 第 {index} 条向量出现了越界的维度索引 "
                    f"{dim!r}（声明维度 {expected_dim}）—— "
                    "索引侧无法存放，必须拦住。"
                )
