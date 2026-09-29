"""
阶段 10：Indexing（扇出）。

归属：DocumentRetrievalDefinition + DocumentIndexBinding。

把本分支的 chunk（以及向量，若需要）写进 IndexBinding 指向的
具体索引实例。

本节点是扇出的**出口**：它把本分支的最终结果写进父状态的
retrieval_results 通道（reducer: add），供 finalize 汇总。

上游给的是什么，取决于 retrieval_type
--------------------------------------
扇出屏障只注入一个 upstream_ref（见 graph.py 的 BARRIER_NODES），
它的内容随本路 retrieval 而变：

    dense / sparse / graph   向量产物 —— 每条 item 是
                             「chunk 的全部字段 + vector/sparse/graph」
                             （embedding 节点刻意把 chunk 字段一起带上了，
                             否则这里拿不到正文，见 embedding.py 的说明）
    document / keyword       embedding 阶段**透传**，所以这里是
                             chunk 产物本身（一个 chunk 数组）

因此读取时要按 retrieval_type 分辨，不能假设只有一种形状。

容器名按 retrieval_type 推导，不来自数据库
------------------------------------------
每个 retrieval_type 对应**一个固定的物理容器**：

    document         PostgreSQL 的 document_chunk 表（就是它的容器）
    keyword          ES 的 document_instance_keyword
    sparse_vector    ES 的 document_instance_sparse_vector
    dense_vector     Milvus 的 document_instance_dense_vector
    graph            Neo4j 的 document_instance_graph（还没接入）

容器名硬编码在后端（`_EXTERNAL_INDEX`），不开放给用户配置 —— 理由与
命名规则见那段常量上方的说明。

为什么**不读** branch["index_identifier"]
----------------------------------------
那个值来自 `document_index_binding` 表，而那张表**没有任何写入路径**
（`add_many` / `update` / `get_by_instance` 全项目零调用方，也没有对应
API），因此它恒为 None。读它的后果是永远走"没有外部索引实例"那条路，
ES / Milvus 的入库代码永远跑不到 —— 而且不报错。

后端类型现在取自 retrieval 定义的 storage_backend；按既定方向它最终应当
来自 **version 快照里的 backend_type**，那样 retrieval_type → backend_type
的绑定关系才可放开配置。快照补上那段之前，这两个值是同一个。

document（PostgreSQL）没有外部索引实例
---------------------------------------
它的 chunk 在阶段 7 就已经落进 document_chunk 表了 —— 那张表**就是**
它的索引。本节点不写任何外部系统，只如实报告数量。

失败处理
--------
这里抛错**不会**让整个运行失败 —— LangGraph 的 Send 分支相互
独立，一个分支抛错只让该分支失败，其余照常跑完，
最终运行状态由 runner 依据节点行推导为 partial。

这正是 partial 存在的意义：ES 挂了但 Milvus 成了，
文档在那部分索引里是**真实可检索**的，不该笼统报 failed。
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span

from app.services.document.domain.schemas import (
    CanonicalChunk,
    IndexUpsertRequest,
)
from app.services.document.domain.schemas.enum import StorageBackend

from ..catalog import NODE_SPECS
from ..context import DocumentPipelineContext
from ..containers import (
    EXTERNAL_INDEX,
    PG_RETRIEVAL_TYPES,
)
from ..registry import CapabilityNotFound
from ..state import BranchInput, RetrievalResult
from ._branch import branch_span_kwargs, ref_summary
from ._common import file_id_of

_SPEC = next(s for s in NODE_SPECS if s.node_id == "indexing")

#: 上游是**向量产物**（每条 item 带 vector/sparse/graph）的 retrieval。
#: 其余（document / keyword）上游是 chunk 数组本身。
_VECTOR_PAYLOADS: dict[str, str] = {
    "dense_vector": "vector",
    "sparse_vector": "sparse",
    "graph": "graph",
}


# ============================================================
# 容器名：每 retrieval_type 一个，硬编码在后端
# ============================================================
#
# 定义与命名理由已提到 pipeline/containers.py —— 查询期要用
# **完全相同**的名字去召回，两份定义迟早漂移，而漂移的表现是
# "写得进去、搜不出来"（索引照样建、照样写，只是永远命中不了）。
#
# 这里保留这两个别名只是为了不动本文件下面的引用点。

_EXTERNAL_INDEX = EXTERNAL_INDEX
_PG_RETRIEVAL_TYPES = PG_RETRIEVAL_TYPES

__all__ = ["indexing_node"]


async def indexing_node(
        state: BranchInput,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    写入索引。

    Returns:
        {"retrieval_results": [RetrievalResult]}
    """

    context = runtime.context
    branch = state["branch"]
    rid = branch["document_retrieval_definition_id"]
    retrieval_type = branch["retrieval_type"]
    # 容器名按 retrieval_type 推导，不读 branch["index_identifier"]
    # （它恒为 None，理由见文件头）。这里先算出来只为埋点，
    # PG 那一路会是 None。
    container = _EXTERNAL_INDEX.get(retrieval_type)

    async with node_span(
            runtime,
            **branch_span_kwargs(_SPEC, state),
            input_summary={
                "retrieval_type": retrieval_type,
                "storage_backend": branch["storage_backend"],
                "container": container,
                "upstream_ref": ref_summary(state.get("upstream_ref")),
            },
    ) as span:

        payload = await _load_upstream(state, context)

        chunks = _to_canonical_chunks(
            payload,
            retrieval_type,
            document_id=state["document_id"],
            document_version_id=state["document_version_id"],
        )

        # ----------------------------------------------------
        # 容器名按 retrieval_type 推导（见文件头的说明）
        # ----------------------------------------------------
        #
        # 刻意**不读** branch["index_identifier"]：那个值来自
        # document_index_binding 表，而那张表没有任何写入路径
        # （add_many / update / get_by_instance 全项目零调用方），
        # 所以它恒为 None —— 读它等于永远走"没有外部索引"那条路，
        # ES/Milvus 的代码永远跑不到。
        # ----------------------------------------------------

        # ----------------------------------------------------
        # 情况一：容器就是 PostgreSQL 那张表
        # ----------------------------------------------------
        #
        # chunk 在阶段 7 已经落进 document_chunk，那张表就是它的索引。
        # 没有外部系统要写 —— 但仍要如实报告数量与 token 数，
        # 否则 finalize 聚合出的这一路结果会是空的。
        # ----------------------------------------------------

        if retrieval_type in _PG_RETRIEVAL_TYPES:
            result = _result(
                branch=branch,
                rid=rid,
                chunks=chunks,
                dimensions=None,
            )

            span.set_metrics({
                "upserted": 0,
                "container": None,
                "note": "PostgreSQL retrieval：chunk 已由 chunking 落库，"
                        "容器就是 document_chunk 表",
            })
            span.set_output_summary({"result": dict(result)})

            return {"retrieval_results": [result]}

        # ----------------------------------------------------
        # 情况二：还没接入的类型，明确失败
        # ----------------------------------------------------
        #
        # 现在只有 graph 会走到这里（它的抽取与入库都还没做）。
        # 不把它当"没有外部索引"悄悄放过 —— 那会让一个没实现的类型
        # 看起来跑成功了。
        # ----------------------------------------------------

        if container is None:
            raise NotImplementedError(
                f"{rid}: retrieval_type={retrieval_type!r} 还没有对应的"
                "容器与入库实现。请先实现它，或者不要启用这一路 retrieval。"
            )

        # ----------------------------------------------------
        # 情况三：写外部索引
        # ----------------------------------------------------

        # ⚠️ 后端类型现在取自 retrieval 定义的 storage_backend。
        #
        # 按既定方向，它最终应当来自 **version 快照里的 backend_type**
        # （见 document.yaml 的 PipelineConfigSnapshot）—— 那样
        # retrieval_type -> backend_type 的绑定关系才可放开配置，
        # 而运行期只读冻结值。快照补上那段之前，这两个值是同一个。
        backend_type = StorageBackend(branch["storage_backend"])

        # 拿不到后端会抛 CapabilityNotFound，两种原因：
        #
        #   1. 数据源没跑 —— ES/Milvus 都是 required: false，连不上就不注册
        #   2. **这一路的 storage_backend 配错了** —— 指向了一个不存在
        #      IndexBackend 的后端（例如 dense_vector 配成 postgresql）
        #
        # 两种都要让这一路**明确失败**，而不是假装写成功了。
        # 原始异常只说"注册表里没有 X"，会把人引去查注册表，
        # 而问题多半在检索定义上 —— 所以在这里补上上下文。
        try:
            backend = context.registry.index_backend(backend_type)

        except CapabilityNotFound as exc:
            raise CapabilityNotFound(
                f"{rid}: retrieval_type={retrieval_type!r} 的 "
                f"storage_backend 是 {backend_type.value!r}，"
                "但没有对应的 IndexBackend 实现。\n"
                "已注册的后端："
                f"{sorted(b.value for b in context.registry.index_backends)}"
                "（数据源没跑时该后端不会注册）。\n"
                "注意 PostgreSQL **不是**独立的 IndexBackend —— "
                "它只服务 retrieval_type='document'，容器就是 "
                "document_chunk 表，而且那张表没有向量列。\n"
                "一条 dense_vector / keyword / sparse_vector 的 retrieval "
                "必须指向 elasticsearch / milvus 之一。"
            ) from exc

        await backend.upsert(
            IndexUpsertRequest(
                document_id=state["document_id"],
                document_version_id=state["document_version_id"],
                document_type_definition_id=state[
                    "document_type_definition_id"
                ],
                index_identifier=container,
                # 后端的连接信息来自注入的客户端（datasource 层管），
                # 这里传的 storage_config 是留给后端私有配置的扩展位，
                # 当前两个实现都没读它。
                index_config=dict(branch.get("storage_config") or {}),
                chunks=chunks,
            )
        )

        dimensions = _dimensions_of(chunks)

        result = _result(
            branch=branch,
            rid=rid,
            chunks=chunks,
            dimensions=dimensions,
        )

        span.set_metrics({
            "upserted": len(chunks),
            "container": container,
            "storage_backend": backend_type.value,
            "embedding_dimensions": dimensions,
        })

        span.set_output_summary({"result": dict(result)})

        return {"retrieval_results": [result]}


# ============================================================
# helpers
# ============================================================


async def _load_upstream(
        state: BranchInput,
        context: DocumentPipelineContext,
) -> Any:
    """读回上游产物并解析 JSON。"""

    upstream = state.get("upstream_ref")

    if upstream is None:
        raise ValueError(
            "indexing 拿不到 upstream_ref —— "
            "上游路由屏障没有把本分支的产物注入进来。"
        )

    raw = await context.artifact_store.get_text(
        file_id_of(upstream)
    )

    return json.loads(raw) if raw else []


def _to_canonical_chunks(
        payload: Any,
        retrieval_type: str,
        document_id: str,
        document_version_id: str,
) -> list[CanonicalChunk]:
    """
    上游产物 -> CanonicalChunk 列表。

    两种形状（见模块 docstring）在这里归一：向量产物是
    {"kind": ..., "items": [...]}，chunk 产物是裸数组。

    身份 id 从**节点 state** 取而不是从产物里取：chunk 产物没有存
    document_id / document_version_id（chunking 时它们就在请求里，
    没必要在每个 chunk 上重复一遍），而 CanonicalChunk 要求它们。
    节点本来就握着这两个值，直接补上比把它们塞进产物干净。

    刻意**不**把 CanonicalChunk 的这两个字段改成可选：它们是
    document_chunk 唯一约束的一部分，"这个 chunk 属于哪个版本"
    不该可空。
    """

    if isinstance(payload, dict):
        items = payload.get("items") or []
    else:
        items = payload or []

    chunks: list[CanonicalChunk] = []

    for item in items:

        # 向量产物里的 sparse 键是字符串（JSON 限制），而
        # CanonicalChunk 声明的是 dict[int, float] —— pydantic 会转，
        # 但这里显式转一次，免得读代码的人以为它本来就是 int。
        sparse = item.get("sparse")
        if sparse is not None:
            sparse = {int(dim): float(w) for dim, w in sparse.items()}

        content = item.get("content") or ""

        chunks.append(
            CanonicalChunk(
                document_chunk_id=item["document_chunk_id"],
                document_id=document_id,
                document_version_id=document_version_id,
                chunk_index=item["chunk_index"],
                content=content,
                content_hash=item.get("content_hash") or "",
                token_count=item.get("token_count"),
                char_count=item.get("char_count") or len(content),
                locator=item.get("locator") or {},
                metadata=item.get("metadata") or {},
                vector=item.get("vector"),
                sparse_vector=sparse,
                context_text=_context_text_of(item),
            )
        )

    return chunks


#: 组装 context_text 的字段，**顺序即权重语义**（靠前的更该被搜到）。
#:
#: 取的是阶段 8 继承下来的那几个 —— 它们描述"这条 chunk 属于谁"，
#: 正是正文里往往不重复、但查询会提到的东西。
_CONTEXT_KEYS: tuple[str, ...] = (
    "company_name",
    "company_code",
    "title",
    "section",
)


def _context_text_of(item: dict[str, Any]) -> str | None:
    """
    从 chunk 的 metadata 里拼出一小段"它属于谁"的上下文，供 ES 的
    multi_match 与高亮使用。

    为什么不是整个 metadata 序列化进去：里面混着 locator、页码、
    document_id 这类**不该参与相关性排序**的东西 —— 把 document_id
    的 UUID 塞进检索字段，只会让 BM25 在无意义的 token 上打分。

    为什么是拼成一段文本而不是保留结构化：ES 那个字段是要被**分词后
    匹配**的（cjk 分析器），结构在那里没有意义；拼成一段能让"东方财富"
    和 300059 都成为可匹配的词项。
    """

    metadata = item.get("metadata") or {}

    parts = [
        str(metadata[key]).strip()
        for key in _CONTEXT_KEYS
        if metadata.get(key) not in (None, "")
    ]

    # 去重且保序：company_name 与 title 经常是同一个串（"东方财富"），
    # 重复写进去会虚增该词项的频次，让 BM25 打分失真。
    seen: list[str] = []

    for part in parts:
        if part not in seen:
            seen.append(part)

    return " ".join(seen) or None


def _dimensions_of(
        chunks: list[CanonicalChunk],
) -> int | None:
    """这一批向量的维度。没有向量（keyword/document）时返回 None。"""

    for chunk in chunks:
        if chunk.vector:
            return len(chunk.vector)

    return None


def _result(
        *,
        branch: dict[str, Any],
        rid: str,
        chunks: list[CanonicalChunk],
        dimensions: int | None,
) -> RetrievalResult:
    """本分支的最终结果，交回父状态供 finalize 汇总。"""

    return RetrievalResult(
        document_retrieval_definition_id=rid,
        document_index_binding_id=branch[
            "document_index_binding_id"
        ],
        status="completed",
        chunk_count=len(chunks),
        token_count=sum(
            chunk.token_count or 0 for chunk in chunks
        ),
        embedding_dimensions=dimensions,
        error_type=None,
        error_message=None,
    )
