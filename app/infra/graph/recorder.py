"""
观测写入端口与快照数据结构。

本模块只定义「写什么」，不定义「写到哪」——
持久化实现在业务模块（app/services/document/pipeline/observability/
recorder_impl.py），依赖方向保持 infra 不反向依赖 services。

事务约定
--------
GraphRunRecorder 的每个方法**各自负责自己那段短事务**。

这样做的理由：
    - 节点失败永远回滚不掉观测数据（观测写入不与业务写入同事务）
    - 观测写入不会在图运行的全程持有锁
    - node_span 不需要感知 session，只需要一个 recorder

实现方因此在内部自行 open/commit 短会话，
调用方（node_span / callback / runner）不传 session。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

__all__ = [
    "GraphRunContext",
    "GraphRunRecorder",
    "NodeRunSnapshot",
    "NodeRunStatus",
    "NodeRunUpdate",
    "RunOutcome",
    "RunSnapshot",
    "RunStatus",
    "RunSummary",
]


# ============================================================
# Status vocabulary
# ============================================================


class RunStatus(str, Enum):
    """
    一次图运行的执行状态。

    这是**观测层**的状态，与业务的 ProcessingStatus
    （pending/processing/completed/failed）是两套：

        partial
            至少一路扇出分支成功、且至少一路失败。
            例如 ES 挂了但 Milvus 成了 —— 文档是部分可检索的。

        业务状态里没有 partial，因为它要映射成一个
        明确的"能用/不能用"。运行状态映射回业务状态时
        partial -> failed。

    运行状态永远不直接写入 DocumentVersion.processing_status。
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class NodeRunStatus(str, Enum):
    """单个节点单次尝试的执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================
# Snapshots
# ============================================================


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """
    创建一次运行所需的全部信息。

    topology_snapshot 是**运行开始时冻结**的图拓扑。它让一次
    2026-09 的历史运行在今天图已经改过之后，仍能按当时的
    拓扑正确渲染 —— 这是"节点兼容性"的主机制。

    主体用 subject_* 三件套表达，infra **不认识**具体含义：

        subject_type  主体类别，例如 "document_version"
        subject_id    主体 ID，用于索引与按主体查历史
        subject_ref   主体上下文的冻结快照（JSONB）

    document 摄入流水线这样填：

        subject_type="document_version"
        subject_id=document_version_id
        subject_ref={"document_id": ..., "document_type_definition_id":
                     ..., "file_id": ..., "tenant_id": ...}

    为什么不做成 infra 直接持有 document_version_id 这类字段：
    那会让 infra 只能服务一种业务。第二个图（检索图 / Agent 图）
    要复用时，就得先接受一堆它不认识的词汇，或者把这里改一遍。

    代价：失去了到业务表的 FK 级联删除，主体行被删时运行记录
    需要显式清理。
    """

    pipeline_run_id: UUID
    thread_id: str

    graph_name: str
    graph_version: str
    topology_fingerprint: str
    topology_snapshot: dict[str, Any]

    # ---- 通用主体标识 ----
    subject_type: str
    subject_id: UUID
    subject_ref: dict[str, Any] = field(
        default_factory=dict
    )

    trigger_source: str = "manual"

    triggered_by: UUID | None = None

    parent_pipeline_run_id: UUID | None = None

    root_pipeline_run_id: UUID | None = None

    run_config: dict[str, Any] = field(
        default_factory=dict
    )

    resumed_from_checkpoint_id: str | None = None


@dataclass(frozen=True, slots=True)
class NodeRunUpdate:
    """
    一次节点行的 upsert 载荷。

    唯一键是
        (pipeline_run_id, node_id, branch_key, attempt)

    两个写入方（node_span 与 callback）都写这张表，
    重复事件因此天然幂等 —— 后到的只是覆盖同样的字段。

    node_label / node_group / sequence 在**写入时冻结**，
    不从当前代码里的目录反查。这样即便目录改了，
    历史行仍带着当时的展示信息，前端有兜底可渲染。
    """

    pipeline_run_id: UUID
    node_id: str
    node_label: str
    node_group: str
    sequence: int
    attempt: int = 1

    branch_key: str = ""

    #: 本分支所属的主体 ID（扇出时用）。
    #: 含义由业务定义 —— document 摄入流水线填 retrieval definition id，
    #: 别的图可以填任何东西。infra 只拿它做索引与展示关联。
    branch_scope_id: UUID | None = None

    langgraph_task_id: str | None = None

    langgraph_run_id: str | None = None

    parent_run_id: str | None = None

    status: NodeRunStatus = NodeRunStatus.RUNNING

    started_at: datetime | None = None

    finished_at: datetime | None = None

    duration_ms: int | None = None

    input_summary: dict[str, Any] | None = None

    output_summary: dict[str, Any] | None = None

    metrics: dict[str, Any] | None = None

    error_type: str | None = None

    error_message: str | None = None

    error_detail: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NodeRunSnapshot:
    """节点行的读模型。"""

    pipeline_node_run_id: UUID
    pipeline_run_id: UUID
    node_id: str
    node_label: str
    node_group: str
    sequence: int
    attempt: int
    branch_key: str
    status: NodeRunStatus

    branch_scope_id: UUID | None = None
    langgraph_task_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """
    一次运行的节点行汇总。

    runner 用它推导运行终态（completed / partial / failed）——
    这个判断必须看**全部**节点行的结果，而不是只看图有没有抛异常。
    一路扇出成功、一路失败时图是抛异常的，但那是 partial 不是 failed。
    """

    node_count: int = 0
    failed_node_count: int = 0

    #: 失败节点，按 sequence 排序，用于 error_node_id。
    failed_node_ids: tuple[str, ...] = ()

    #: 扇出节点中成功的分支数，用于区分 partial 与 failed。
    completed_node_count: int = 0


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """运行终结时写回 run 行的结果。"""

    status: RunStatus
    finished_at: datetime

    duration_ms: int | None = None

    error_type: str | None = None

    error_node_id: str | None = None

    error_message: str | None = None

    node_count: int = 0

    failed_node_count: int = 0


# ============================================================
# Port
# ============================================================


@runtime_checkable
class GraphRunRecorder(Protocol):
    """
    图运行的观测写入端口。

    每个方法自成一段已提交的短事务，调用方不传 session。
    实现方若抛错，不应吞掉 —— 但调用方（node_span）
    会把观测写入的异常与业务异常区别对待，见 instrumentation.py。
    """

    async def create_run(
            self,
            snapshot: RunSnapshot,
    ) -> None:
        """插入 run 行。必须早于图开始运行。"""
        ...

    async def mark_run_running(
            self,
            pipeline_run_id: UUID,
            *,
            started_at: datetime,
    ) -> None:
        """把 run 行置为 running 并记录开始时间。"""
        ...

    async def finish_run(
            self,
            pipeline_run_id: UUID,
            outcome: RunOutcome,
    ) -> None:
        """写回运行的终态、耗时、错误归属与节点计数。"""
        ...

    async def upsert_node_run(
            self,
            update: NodeRunUpdate,
    ) -> None:
        """
        按唯一键 upsert 一行节点记录。

        两个写入方共用本方法，因此实现必须是
        INSERT ... ON CONFLICT DO UPDATE 语义。
        """
        ...

    async def summarize(
            self,
            pipeline_run_id: UUID,
    ) -> RunSummary:
        """
        汇总该次运行的节点行。

        runner 在图跑完（或抛错）后调用，用真实落到库里的节点结果
        推导终态 —— 而不是靠"图有没有抛异常"来猜。
        """
        ...

    async def mark_orphans_failed(
            self,
            *,
            older_than: datetime,
    ) -> int:
        """
        把卡在 running 的 run / node 行标记为 failed。

        进程中途被杀时，行会永远停在 running，UI 会一直转圈。
        启动时调用一次即可收敛。

        返回被修正的行数，供启动日志使用。
        """
        ...


@runtime_checkable
class GraphRunContext(Protocol):
    """
    LangGraph 的 Runtime.context 需要满足的形状。

    业务模块用自己的 dataclass 实现它并作为图的
    context_schema；infra 只依赖这个结构，不依赖具体类型。

    刻意只要两个字段：node_span 因此完全不需要接触
    session_factory —— 事务由 recorder 内部自理。

    callback 的节点过滤不走这里：那份名单由 runner 在构造
    PersistingCallbackHandler 时直接传，因为它是图级的静态信息，
    不是每次运行的依赖。
    """

    pipeline_run_id: UUID

    recorder: GraphRunRecorder
