"""
图的装配。

拓扑
----
    START
      -> ingestion -> format_detection -> parsing
      -> normalization -> cleaning -> metadata_extraction
      --Send×N--> chunking                            ┐
      -> _after_chunking --Send×N--> chunk_enrichment  │
      -> _after_chunk_enrichment --Send×N--> embedding │ 扇出
      -> _after_embedding --Send×N--> indexing         ┘
      -> finalize_version -> END

`_after_*` 是**纯路由屏障**：不埋点、不在 NODE_SPECS 里、
不出现在任何观测数据中。它们存在的唯一理由是 LangGraph 的
Send 只作用于单个节点、其下游边看到的是父状态（详见 fanout.py），
因此每两个扇出阶段之间需要一次"等齐再发下一批"。

空分支
------
任一路由函数在分支为空时会改发 finalize（见 fanout.py）——
LangGraph 对空 Send 列表不报错，而是让运行静默结束，
那样 finalize 永远不执行、版本状态永远停在 processing。

重试
----
只有**瞬时性**失败值得重试（索引抖动、连接超时）。
NotImplementedError / ValueError 这类是配置或实现问题，
重试只会把同样的错误重复三遍。因此重试策略是一张**判据**
（见 is_retryable），而不是"失败就重试"。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pymilvus.exceptions import MilvusException

from .catalog import (
    EDGES,
    FANOUT_NODE_IDS,
    NODE_SPECS,
)
from .context import DocumentPipelineContext
from .fanout import ROUTES
from .nodes import (
    chunk_enrichment_node,
    chunking_node,
    cleaning_node,
    embedding_node,
    finalize_version_node,
    format_detection_node,
    indexing_node,
    ingestion_node,
    metadata_extraction_node,
    normalization_node,
    parsing_node,
)
from .state import DocumentPipelineState

__all__ = ["BARRIER_NODES", "build_document_pipeline_graph"]


#: 纯路由屏障。以下划线开头，刻意与 NODE_SPECS 里的阶段区分开。
BARRIER_NODES: dict[str, str] = {
    "_after_chunking": "chunk_refs",
    "_after_chunk_enrichment": "enrich_refs",
    "_after_embedding": "vector_refs",
}


#: 阶段 node_id -> 节点函数。
NODE_FUNCTIONS: dict[str, Any] = {
    "ingestion": ingestion_node,
    "format_detection": format_detection_node,
    "parsing": parsing_node,
    "normalization": normalization_node,
    "cleaning": cleaning_node,
    "metadata_extraction": metadata_extraction_node,
    "chunking": chunking_node,
    "chunk_enrichment": chunk_enrichment_node,
    "embedding": embedding_node,
    "indexing": indexing_node,
    "finalize_version": finalize_version_node,
}


#: 值得重试的异常。其余（配置错误、能力缺失、参数非法）
#: 重试没有意义，只会把同一个错误重复三遍并拖长运行时间。
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def is_retryable(error: BaseException) -> bool:
    """
    这条异常值不值得重试。

    除了上面那三个内置的网络类异常，还要认 Milvus 的服务端抖动 ——
    本机这个实例的 delegator 会反复在"可服务/不可服务"之间跳，写入正好
    撞进窗口时返回：

        code=503 channel distribution is not serviceable:
        channel not available

    pymilvus 内部重试约 9 秒后放弃并抛出，而它不属于上面三类，于是**整条
    写入链直接失败**。这条错误重试一次往往就好了，是这里唯一真正的缺口。

    ⚠️ 不能把 MilvusException 整类收进来：同一个类也盖着"过滤表达式解析
    失败"这类重试毫无意义的错误，那正是上面那句注释要排除的东西。

    ⚠️ 也不能用 `MilvusException.retriable`。那本该是服务端给的判定，实测
    在本机这个版本里**根本没填** —— 连 101 collection not loaded 这种板上
    钉钉的瞬时错误都是 False。唯一被正确填充的是 `is_input_error`：参数
    非法、表达式解析失败那类才是 True，与要排除的类别正好对上。所以判据
    是"是 Milvus 错误、且服务端没把它标成输入错误"。
    """

    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True

    if isinstance(error, MilvusException):
        return not error.is_input_error

    return False


def build_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=3,
        retry_on=is_retryable,
    )


async def _barrier(
        state: DocumentPipelineState,
        runtime: Any,
) -> dict[str, Any]:
    """
    路由屏障：什么都不做，只声明"下一批 Send 从这里发出"。

    它必须是个 async 函数而不是 lambda —— LangGraph 按可调用
    对象的签名决定是否注入 runtime，lambda 的签名会让它拿不到。
    """
    return {}


def build_document_pipeline_graph() -> StateGraph:
    """
    构造（未编译的）主图。

    返回 StateGraph 而不是编译好的图，是为了让组合根决定
    用哪个 checkpointer 编译 —— 生产用 Postgres，冒烟测试
    用 InMemorySaver，图本身不该关心。
    """

    graph = StateGraph(
        DocumentPipelineState,
        context_schema=DocumentPipelineContext,
    )

    retry = build_retry_policy()

    # --------------------------------------------------------
    # 阶段节点
    # --------------------------------------------------------

    for spec in NODE_SPECS:
        graph.add_node(
            spec.node_id,
            NODE_FUNCTIONS[spec.node_id],
            retry_policy=retry,
        )

    # --------------------------------------------------------
    # 路由屏障
    # --------------------------------------------------------

    for barrier_name in BARRIER_NODES:
        graph.add_node(barrier_name, _barrier)

    # --------------------------------------------------------
    # 线性边
    # --------------------------------------------------------

    graph.add_edge(START, "ingestion")

    for source, target in EDGES:
        # EDGES 里的扇出链（chunking -> chunk_enrichment 之类）是
        # 给拓扑指纹和前端画图用的**文档**，不是可装配的边。
        #
        # 真装配会直接坏掉：Send 只作用于单个节点，其下游边看到的是
        # **父状态**，所以 chunk_enrichment 会拿到父状态而不是本分支的
        # chunk，报 KeyError: 'branch'。扇出必须靠
        # 条件边 + 屏障节点，见 fanout.py。
        if source in FANOUT_NODE_IDS or target in FANOUT_NODE_IDS:
            continue

        graph.add_edge(source, target)

    # --------------------------------------------------------
    # 扇出：每个阶段前挂一个条件边
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "metadata_extraction",
        ROUTES["metadata_extraction"],
        ["chunking", "finalize_version"],
    )
    graph.add_conditional_edges(
        "_after_chunking",
        ROUTES["_after_chunking"],
        ["chunk_enrichment", "finalize_version"],
    )
    graph.add_conditional_edges(
        "_after_chunk_enrichment",
        ROUTES["_after_chunk_enrichment"],
        ["embedding", "finalize_version"],
    )
    graph.add_conditional_edges(
        "_after_embedding",
        ROUTES["_after_embedding"],
        ["indexing", "finalize_version"],
    )

    # --------------------------------------------------------
    # 屏障：扇出阶段跑完后汇入，等齐再发下一批
    # --------------------------------------------------------

    graph.add_edge("chunking", "_after_chunking")
    graph.add_edge("chunk_enrichment", "_after_chunk_enrichment")
    graph.add_edge("embedding", "_after_embedding")

    # --------------------------------------------------------
    # 扇入
    # --------------------------------------------------------

    graph.add_edge("indexing", "finalize_version")
    graph.add_edge("finalize_version", END)

    return graph
