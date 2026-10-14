"""
通用 LangGraph 运行时。

分层规则
--------
本包是**共享内核**，不是被驱动的适配器：

    - 只依赖 langgraph / langchain-core / psycopg / sqlalchemy / pydantic
    - 禁止 import app.services 下的任何模块（会反向依赖业务模块）
    - 允许 import app.common 下的共享内核

业务模块可以 import 本包，方向是单向的：

    app/services/document/pipeline/  --->  app/infra/graph/

这是为了让第二个图（检索图 / Agent 图）能原样复用
checkpointer 连接池、callback 处理器与 node_span，
而不必把同一套东西在 document 下再抄一遍。

与 app/infra/datasource/ 的关系
-------------------------------
两者都是共享内核，但职责不同、互不依赖：

    datasource/    连接与连接池（Postgres / ES / Milvus）
    graph/         LangGraph 编排的横切设施（checkpoint、埋点、观测）

graph/ 不自己管理数据库连接，checkpointer 的连接池由
GraphCheckpointerProvider 独立持有，与 datasource 的
SQLAlchemy 池完全分离 —— 两者事务边界不同，绝不能共用。

包含
----
    ids.py              标识与指纹的生成规则
    recorder.py         观测写入端口（Protocol）与快照数据结构
    instrumentation.py  node_span 埋点上下文与 state 体积护栏
    callbacks.py        LangGraph callback 兜底写入
    checkpointer.py     Postgres checkpointer 生命周期管理

callbacks 与 checkpointer 各自只在 runner / main 里用一次，
因此不从这里转发，按需直接 import 其模块。
"""

from .ids import (
    branch_key,
    branch_scope_id,
    new_session_thread_id,
    new_thread_id,
    normalize_edges,
    to_node_slug,
    topology_fingerprint,
)
from .cancellation import (
    RunCancellationRegistry,
    get_run_cancellation_registry,
    reset_run_cancellation_registry,
)
from .dispatcher_impl import (
    InProcessGraphDispatcher,
    build_graph_dispatcher,
    get_graph_dispatcher,
    reset_graph_dispatcher,
)
from .thread_lock import (
    InProcessThreadRunLock,
    ThreadLockHolder,
    get_thread_run_lock,
    reset_thread_run_lock,
)
from .dispatcher import (
    DispatchEnvelope,
    DispatchHandle,
    GraphDispatcher,
)
from .registry import (
    GraphDefinition,
    GraphRegistry,
    NodeSpec,
    RunContext,
    build_topology_snapshot,
)
from .runner import GraphRunner, RunRequest

from .instrumentation import (
    DEFAULT_MAX_STATE_BYTES,
    MAX_PREVIEW_CHARS,
    ExecutionIds,
    NodeSpan,
    assert_checkpoint_safe,
    execution_ids,
    node_span,
    truncate_preview,
)
from .recorder import (
    GraphRunRecorder,
    GraphRunContext,
    NodeRunSnapshot,
    NodeRunStatus,
    NodeRunUpdate,
    RunOutcome,
    RunSnapshot,
    RunStatus,
    RunSummary,
)

__all__ = [
    "DEFAULT_MAX_STATE_BYTES",
    "MAX_PREVIEW_CHARS",
    "DispatchEnvelope",
    "DispatchHandle",
    "ExecutionIds",
    "GraphDefinition",
    "GraphDispatcher",
    "GraphRegistry",
    "GraphRunContext",
    "GraphRunRecorder",
    "GraphRunner",
    "InProcessThreadRunLock",
    "RunCancellationRegistry",
    "NodeRunSnapshot",
    "NodeRunStatus",
    "NodeRunUpdate",
    "NodeSpan",
    "NodeSpec",
    "RunContext",
    "RunOutcome",
    "RunRequest",
    "RunSnapshot",
    "RunStatus",
    "RunSummary",
    "ThreadLockHolder",
    "assert_checkpoint_safe",
    "branch_key",
    "branch_scope_id",
    "build_topology_snapshot",
    "execution_ids",
    "get_thread_run_lock",
    "new_session_thread_id",
    "new_thread_id",
    "node_span",
    "InProcessGraphDispatcher",
    "build_graph_dispatcher",
    "get_graph_dispatcher",
    "get_run_cancellation_registry",
    "reset_graph_dispatcher",
    "normalize_edges",
    "reset_run_cancellation_registry",
    "reset_thread_run_lock",
    "to_node_slug",
    "topology_fingerprint",
    "truncate_preview",
]
