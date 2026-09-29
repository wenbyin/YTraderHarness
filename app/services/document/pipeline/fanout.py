"""
扇出路由：把主状态转成一批 Send。

为什么需要一个"路由屏障"节点夹在每两个扇出阶段之间
--------------------------------------------------
实测得到的两条 Send 语义（见 state.py 的说明）：

    1. Send 的 payload 取代 state，分支节点看不到父状态
    2. Send 只作用于那**一个**节点；该节点的下游边看到的
       是**父状态**，不是分支状态

第 2 条意味着 `chunking → chunk_enrichment` 这样的普通边根本
传不了分支数据 —— chunk_enrichment 会拿到父状态，
既看不到本分支的 chunk，也不知道自己属于哪个 retrieval。

所以形状必须是：

    metadata_extraction  --Send×N-->  chunking
    chunking  --edge-->  _after_chunking  --Send×N-->  chunk_enrichment
    chunk_enrichment --edge--> _after_enrichment --Send×N--> embedding
    embedding --edge--> _after_embedding --Send×N--> indexing
    indexing --edge--> finalize_version

其中 `_after_*` 是**纯路由屏障**：不埋点、不在 NODE_SPECS 里、
不出现在可观测量里。它唯一的价值是「等所有分支跑完再发下一批」——
LangGraph 在 super-step 边界上天然有这个屏障。

每个路由函数都做同一件事：
    遍历 branches，按 retrieval_definition_id 从父状态取出
    该分支上一阶段的产物，连同分支配置一起塞进 Send payload。

空列表的处理
------------
路由函数返回空列表时，LangGraph **不报错**，而是让运行静默结束
（实测）。那样 finalize 永远不会执行，版本状态就永远停在
processing。因此每个路由函数在分支为空时都显式发一个
finalize —— 这不是防御性冗余，是必需的。
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from .state import (
    ArtifactRef,
    BranchInput,
    DocumentPipelineState,
    RetrievalBranch,
)

__all__ = [
    "ROUTES",
    "route_to_chunking",
    "route_to_chunk_enrichment",
    "route_to_embedding",
    "route_to_indexing",
]


def _build_payload(
        branch: RetrievalBranch,
        state: DocumentPipelineState,
        upstream: ArtifactRef | None,
) -> BranchInput:
    """
    为**一个**分支构造 Send payload。

    这里必须把分支节点需要的**所有**东西都放进去 ——
    它看不到父状态。漏一个字段的表现是分支节点 KeyError，
    而报错不会提示是因为 Send 的语义。
    """

    return BranchInput(
        branch=branch,
        document_id=state["document_id"],
        document_version_id=state["document_version_id"],
        document_type_definition_id=(
            state["document_type_definition_id"]
        ),
        clean_ref=state["clean_ref"],  # type: ignore[typeddict-item]
        extract_ref=state.get("extract_ref"),
        upstream_ref=upstream,
        retrieval_results=[],
    )


#: 产物是**版本级**的通道 —— 键是 chunker 身份，不是 retrieval。
#:
#: 切分的产出只取决于 (正文, chunker, 配置)，共用 chunker 的几路
#: 共用同一份，所以它按身份归集。其余三个通道的产物确实是
#: per-retrieval 的（向量形态、目标容器都不同），仍按 retrieval 归集。
_VERSION_LEVEL_CHANNELS = frozenset({"chunk_refs"})


def _pool_key(
        branch: RetrievalBranch,
        channel: str,
) -> str:
    """本分支去哪个键下取（或被归集到哪个键）上游产物。"""

    if channel in _VERSION_LEVEL_CHANNELS:
        return branch["chunker_digest"]

    return branch["document_retrieval_definition_id"]


def _chunking_plan(
        branches: list[RetrievalBranch],
) -> list[RetrievalBranch]:
    """
    切分计划：**每个 chunker 身份一个分支**。

    切分的产出只取决于 (正文, chunker, 配置)，与 retrieval 无关，所以
    共用同一个 chunker 的几路只该切一次。不收敛的话它们会各切一遍，
    产出的 chunk 落库时撞 UNIQUE(version, chunker_key, chunker_config,
    chunk_index) —— 第二路直接失败。

    取的每个身份的第一路作为代表：切分用不到 retrieval 侧的任何字段，
    用哪一路当代表都一样。
    """

    seen: dict[str, RetrievalBranch] = {}

    for branch in branches:
        seen.setdefault(branch["chunker_digest"], branch)

    return list(seen.values())


def _route(
        stage: str,
        state: DocumentPipelineState,
        upstream_channel: str | None,
) -> list[Send]:
    """
    通用路由：给每个分支发一个 Send。

    upstream_channel 是父状态里存本分支上游产物的 dict 通道名
    （chunk_refs / enrich_refs / vector_refs）。第一阶段为 None。
    """

    branches: list[RetrievalBranch] = list(
        state.get("branches") or []
    )

    if not branches:
        # 没有任何 retrieval 参与。仍然要收敛到 finalize，
        # 否则运行静默结束、版本状态永远停在 processing。
        return [Send("finalize_version", {})]

    pool: dict[str, ArtifactRef] = (
        state.get(upstream_channel) or {}  # type: ignore[literal-required]
        if upstream_channel
        else {}
    )

    # 切分按身份收敛；其余阶段仍按 retrieval。
    planned = (
        _chunking_plan(branches)
        if upstream_channel is None
        else branches
    )

    sends: list[Send] = []

    for branch in planned:

        upstream = (
            None
            if upstream_channel is None
            else pool.get(
                _pool_key(branch, upstream_channel)
            )
        )

        if upstream_channel and upstream is None:
            # 上游阶段没为这个分支产出东西，说明该分支在上游
            # 就已经失败/被跳过。直接让它进 finalize，
            # 由 finalize 依据 retrieval_results 判定 partial。
            continue

        sends.append(
            Send(
                stage,
                _build_payload(branch, state, upstream),
            )
        )

    if not sends:
        return [Send("finalize_version", {})]

    return sends


# ============================================================
# 各阶段的入口路由
# ============================================================


def route_to_chunking(
        state: DocumentPipelineState,
) -> list[Send]:
    """阶段 6 之后：对每个 retrieval 发起切分。"""
    return _route("chunking", state, None)


def route_to_chunk_enrichment(
        state: DocumentPipelineState,
) -> list[Send]:
    """阶段 7 之后：带上各自切出的 chunk，发起增强。"""
    return _route(
        "chunk_enrichment", state, "chunk_refs"
    )


def route_to_embedding(
        state: DocumentPipelineState,
) -> list[Send]:
    """阶段 8 之后：带上各自增强后的 chunk，发起向量化。"""
    return _route("embedding", state, "enrich_refs")


def route_to_indexing(
        state: DocumentPipelineState,
) -> list[Send]:
    """阶段 9 之后：带上各自算出的向量，发起写入索引。"""
    return _route("indexing", state, "vector_refs")


#: 屏障节点名 -> 路由函数。graph.py 据此装配条件边。
#:
#: 屏障节点名以下划线开头，且**不在** NODE_SPECS 里 ——
#: 它们是图机制，不是流水线阶段，不应出现在观测数据中。
ROUTES: dict[str, Any] = {
    "metadata_extraction": route_to_chunking,
    "_after_chunking": route_to_chunk_enrichment,
    "_after_chunk_enrichment": route_to_embedding,
    "_after_embedding": route_to_indexing,
}
