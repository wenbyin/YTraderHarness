"""
运行事件 v2 的**统一契约**：一个信封、一个词表、每种事件一个 payload。

本模块是 `docs/event-stream-v2-design.md` 的阶段 0 产物。它**只定义规范**，
不改变现有生产 / 消费路径：

    - 节点侧的 `TraceEmitter` 已直接构建本模块的 `Event`，投影层不再转换
    - 旧的 `RunEvent` / `TraceEvent` 与 v1 写线程已全部删除
    - 生产者、publisher、reader 一行不动
    - 本模块先被测试和后续阶段引用

为什么现在不直接替换旧模型
--------------------------
阶段 0 要先把“规范长什么样”钉下来，并让测试跑通。真正切换生产者在阶段 2、
切换 chat 消费端在阶段 3。现在替换会让一次改动同时承担契约、路由和前端三件
事，出了问题分不清是哪一层。

与 `trace.py` 的关系
--------------------
`SpanKind` / `SpanStatus` / `SpanLink` / `TraceError` 以及 trace 独有的
payload 模型继续从 `trace.py` 复用 —— 它们已经是严格校验的 Pydantic 模型。
本模块只新增：

    - 联合 `EventType`（两份词表的并集，同名去重）
    - `Event` 信封
    - 六个同名事件的合并 payload
    - chat 独有事件的 payload
    - chat / trace 两个放行集合

⚠️ 本模块**不导出** `EventType` 到包级 `app.common.event`
------------------------------------------------------
包级 `EventType` 目前指向 `trace.py` 的那份。两份同名会在调用方制造
“到底 import 到哪一个”的歧义。阶段 4 删除旧模型时再统一导出。
现阶段请这样引用：

    from app.common.event.spec import Event, EventType
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .trace import (
    ApprovalCancelledPayload,
    ApprovalCompletedPayload,
    ApprovalFailedPayload,
    ApprovalStartedPayload,
    ContextStageCompletedPayload,
    ContextStageFailedPayload,
    ContextStageStartedPayload,
    FanoutBranchCreatedPayload,
    FanoutCancelledPayload,
    FanoutCompletedPayload,
    FanoutFailedPayload,
    FanoutStartedPayload,
    GraphTransitionPayload,
    HitlApprovalRequestedPayload,
    HitlDecisionPayload,
    HitlResumedPayload,
    ModelCompletedPayload,
    ModelFailedPayload,
    ModelFallbackPayload,
    ModelFirstTokenPayload,
    ModelRetryScheduledPayload,
    ModelStartedPayload,
    NodeFailedPayload,
    NodeInterruptedPayload,
    NodeResumedPayload,
    NodeSkippedPayload,
    RunAcceptedPayload,
    RunCancelledPayload,
    RunClientDisconnectedPayload,
    RunFirstOutputPayload,
    RunQueuedPayload,
    SpanKind,
    SpanLink,
    SpanStatus,
    ToolFailedPayload,
    ToolIdempotencyDecisionPayload,
    ToolInterruptedPayload,
    ToolResumedPayload,
    ToolRetryScheduledPayload,
    ToolStartedPayload,
    TraceError,
)

__all__ = [
    "ALL_EVENT_TYPES",
    "CHAT_EVENT_TYPES",
    "EVENT_PAYLOAD_MODELS",
    "EVENT_SCHEMA_VERSION",
    "TRACE_EVENT_TYPES",
    "TRACE_ONLY_EVENT_TYPES",
    "Event",
    "EventError",
    "EventType",
    "SpanKind",
    "SpanLink",
    "SpanStatus",
]


#: 新版持久化契约的版本号。消费端不匹配时应**显式失败**，不静默降级。
EVENT_SCHEMA_VERSION: Final[str] = "2.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_event_id() -> str:
    return uuid.uuid4().hex


# ============================================================
# 词表
# ============================================================


class EventType(str, Enum):
    """
    所有运行事件的**联合词表**。

    ⚠️ 值就是线上的字符串，改一个字就是破坏契约。
    同名事件（run.started / run.completed / run.failed / node.started /
    node.completed / tool.completed）只保留一个成员。
    """

    # ---- run ----
    RUN_ACCEPTED = "run.accepted"
    RUN_QUEUED = "run.queued"
    RUN_STARTED = "run.started"
    RUN_FIRST_OUTPUT = "run.first_output"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_CLIENT_DISCONNECTED = "run.client_disconnected"

    # ---- node ----
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    NODE_SKIPPED = "node.skipped"
    NODE_INTERRUPTED = "node.interrupted"
    NODE_RESUMED = "node.resumed"

    # ---- graph / context ----
    GRAPH_TRANSITION = "graph.transition"

    CONTEXT_BUILD_STARTED = "context.build.started"
    CONTEXT_BUILD_COMPLETED = "context.build.completed"
    CONTEXT_BUILD_FAILED = "context.build.failed"
    CONTEXT_PROFILE_LOAD_STARTED = "context.profile_load.started"
    CONTEXT_PROFILE_LOAD_COMPLETED = "context.profile_load.completed"
    CONTEXT_PROFILE_LOAD_FAILED = "context.profile_load.failed"
    CONTEXT_SUMMARY_LOAD_STARTED = "context.summary_load.started"
    CONTEXT_SUMMARY_LOAD_COMPLETED = "context.summary_load.completed"
    CONTEXT_SUMMARY_LOAD_FAILED = "context.summary_load.failed"
    CONTEXT_HISTORY_SELECT_STARTED = "context.history_select.started"
    CONTEXT_HISTORY_SELECT_COMPLETED = "context.history_select.completed"
    CONTEXT_HISTORY_SELECT_FAILED = "context.history_select.failed"
    CONTEXT_MEMORY_RETRIEVE_STARTED = "context.memory_retrieve.started"
    CONTEXT_MEMORY_RETRIEVE_COMPLETED = "context.memory_retrieve.completed"
    CONTEXT_MEMORY_RETRIEVE_FAILED = "context.memory_retrieve.failed"
    CONTEXT_KNOWLEDGE_RETRIEVE_STARTED = "context.knowledge_retrieve.started"
    CONTEXT_KNOWLEDGE_RETRIEVE_COMPLETED = "context.knowledge_retrieve.completed"
    CONTEXT_KNOWLEDGE_RETRIEVE_FAILED = "context.knowledge_retrieve.failed"
    CONTEXT_PROMPT_ASSEMBLE_STARTED = "context.prompt_assemble.started"
    CONTEXT_PROMPT_ASSEMBLE_COMPLETED = "context.prompt_assemble.completed"
    CONTEXT_PROMPT_ASSEMBLE_FAILED = "context.prompt_assemble.failed"

    # ---- model ----
    MODEL_STARTED = "model.started"
    MODEL_FIRST_TOKEN = "model.first_token"
    MODEL_COMPLETED = "model.completed"
    MODEL_FAILED = "model.failed"
    MODEL_RETRY_SCHEDULED = "model.retry_scheduled"
    MODEL_FALLBACK = "model.fallback"

    # ---- tool ----
    TOOL_CALLED = "tool.called"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    TOOL_INTERRUPTED = "tool.interrupted"
    TOOL_RESUMED = "tool.resumed"
    TOOL_RETRY_SCHEDULED = "tool.retry_scheduled"
    TOOL_IDEMPOTENCY_DECISION = "tool.idempotency_decision"

    # ---- message ----
    MESSAGE_STARTED = "message.started"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_COMPLETED = "message.completed"
    MESSAGE_REASONING_DELTA = "message.reasoning.delta"
    MESSAGE_REASONING_COMPLETED = "message.reasoning.completed"

    # ---- fanout ----
    FANOUT_STARTED = "fanout.started"
    FANOUT_BRANCH_CREATED = "fanout.branch_created"
    FANOUT_COMPLETED = "fanout.completed"
    FANOUT_FAILED = "fanout.failed"
    FANOUT_CANCELLED = "fanout.cancelled"

    # ---- approval / hitl ----
    APPROVAL_STARTED = "approval.started"
    APPROVAL_COMPLETED = "approval.completed"
    APPROVAL_FAILED = "approval.failed"
    APPROVAL_CANCELLED = "approval.cancelled"
    HITL_APPROVAL_REQUESTED = "hitl.approval_requested"
    HITL_DECISION = "hitl.decision"
    HITL_RESUMED = "hitl.resumed"
    HITL_REQUESTED = "hitl.requested"
    HITL_RESOLVED = "hitl.resolved"

    # ---- plan / task / subagent ----
    PLAN_PUBLISHED = "plan.published"
    TASK_UPDATED = "task.updated"
    PLAN_CLEARED = "plan.cleared"
    SUBAGENT_CREATED = "subagent.created"
    SUBAGENT_CLAIMED = "subagent.claimed"
    SUBAGENT_TASK_UPDATED = "subagent.task.updated"
    SUBAGENT_COMPLETED = "subagent.completed"


#: 事件错误。复用 trace 校验过的结构，避免两份同义模型漂移。
EventError = TraceError


# ============================================================
# payload 模型
# ============================================================
#
# 每个事件一个严格模型（hitl.requested 例外：它的载荷是审批拦截器的
# 不透明字典，见下面的说明）。同名事件只有**一个**模型，两个 sink 看到
# 完全相同的 data。


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- 六个同名事件 ----


class RunStartedData(_Payload):
    agent_type: str
    graph_name: str


class RunCompletedData(_Payload):
    finish_reason: str | None = None


class RunFailedData(_Payload):
    pass


class NodeStartedData(_Payload):
    node_id: str
    node_type: str | None = None
    node_execution_index: int = Field(default=0, ge=0)
    label: str | None = None
    group: str | None = None
    sequence: int | None = None
    agent_type: str | None = None
    model_role: str | None = None


class NodeCompletedData(_Payload):
    node_id: str
    node_type: str | None = None
    message_count: int | None = Field(default=None, ge=0)
    tool_calls: list[str] = Field(default_factory=list)
    model_provider: str | None = None
    model_name: str | None = None


class ToolCompletedData(_Payload):
    tool_call_id: str | None = None
    tool_name: str | None = None
    output: Any = None
    result_size_bytes: int | None = Field(default=None, ge=0)

    #: 用户展示描述符（renderer / name / summary / props）。
    #: 没有 presenter 的工具**根本不会产这条事件**，所以这里要么是
    #: 一个完整描述符，要么（门控未启用的旧路径）整个键缺席。
    display: dict[str, Any] | None = None


# ---- chat 独有 ----


class MessageStartedData(_Payload):
    role: str | None = None
    visible: bool = True


class MessageDeltaData(_Payload):
    delta: str


class MessageCompletedData(_Payload):
    content: str = ""
    tool_calls: list[str] = Field(default_factory=list)
    finish_reason: str | None = None
    role: str | None = None
    visible: bool = True


class MessageReasoningDeltaData(_Payload):
    delta: str


class MessageReasoningCompletedData(_Payload):
    content: str
    role: str | None = None


class ToolCalledData(_Payload):
    tool_call_id: str | None = None
    tool_name: str | None = None
    input: Any = None

    #: 用户展示描述符（见 ToolCompletedData.display）。
    #: summary 在 called 时只能用 {input.*}，{output.*} 解析成空串；
    #: completed 会再带一次用结果重新渲染的 summary。
    display: dict[str, Any] | None = None


class PlanPublishedData(_Payload):
    plan_id: str
    revision: int
    publish_id: str | None = None
    goal: str
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    current_task_id: str | None = None
    task_count: int = Field(default=0, ge=0)


class TaskUpdatedData(_Payload):
    plan_id: str
    revision: int
    task: dict[str, Any]
    current_task_id: str | None = None


class PlanClearedData(_Payload):
    plan_id: str
    revision: int
    reason: str


class SubagentCreatedData(_Payload):
    plan_id: str
    revision: int
    worker_id: str
    worker_index: int = Field(ge=0)
    task_id: str
    task_count: int = Field(default=0, ge=0)
    pending_count: int = Field(default=0, ge=0)
    context_id: str = ""


class SubagentClaimedData(_Payload):
    plan_id: str
    revision: int
    worker_id: str
    task: dict[str, Any]
    task_id: str
    current_task_id: str | None = None


class SubagentTaskUpdatedData(_Payload):
    plan_id: str
    revision: int
    worker_id: str
    task: dict[str, Any]
    task_id: str
    previous_state: str = ""
    current_task_id: str | None = None


class SubagentCompletedData(_Payload):
    plan_id: str
    revision: int
    worker_id: str
    task: dict[str, Any]
    task_id: str
    status: str


class HitlRequestedData(BaseModel):
    """
    `hitl.requested` 的载荷。

    ⚠️ 这是**唯一**允许额外字段的 payload
    ------------------------------------
    载荷来自审批拦截器的 `_payload()`（`ask` / `tool_key` / `arguments` /
    `options` …），派发层把它当**不透明字典**搬运 —— 换一种需要人工介入的
    处理器时形状会变，而后端不应为此改模型。前端也按可选字段读。
    """

    model_config = ConfigDict(extra="allow")

    interrupt_id: str
    ask: str | None = None
    tool_key: str | None = None
    arguments: dict[str, Any] | None = None
    approver_role: str | None = None
    options: list[str] | None = None
    run_id: str | None = None
    wait_seconds: int | float | None = None
    type: str | None = None
    questions: list[Any] | None = None
    metadata: dict[str, Any] | None = None
    tool_call_id: str | None = None


class HitlResolvedData(_Payload):
    interrupt_id: str
    data: dict[str, Any] = Field(default_factory=dict)


# ---- 全量映射 ----


EVENT_PAYLOAD_MODELS: dict[EventType, type[BaseModel]] = {
    # run
    EventType.RUN_ACCEPTED: RunAcceptedPayload,
    EventType.RUN_QUEUED: RunQueuedPayload,
    EventType.RUN_STARTED: RunStartedData,
    EventType.RUN_FIRST_OUTPUT: RunFirstOutputPayload,
    EventType.RUN_COMPLETED: RunCompletedData,
    EventType.RUN_FAILED: RunFailedData,
    EventType.RUN_CANCELLED: RunCancelledPayload,
    EventType.RUN_CLIENT_DISCONNECTED: RunClientDisconnectedPayload,

    # node
    EventType.NODE_STARTED: NodeStartedData,
    EventType.NODE_COMPLETED: NodeCompletedData,
    EventType.NODE_FAILED: NodeFailedPayload,
    EventType.NODE_SKIPPED: NodeSkippedPayload,
    EventType.NODE_INTERRUPTED: NodeInterruptedPayload,
    EventType.NODE_RESUMED: NodeResumedPayload,

    # graph / context
    EventType.GRAPH_TRANSITION: GraphTransitionPayload,
    EventType.CONTEXT_BUILD_STARTED: ContextStageStartedPayload,
    EventType.CONTEXT_BUILD_COMPLETED: ContextStageCompletedPayload,
    EventType.CONTEXT_BUILD_FAILED: ContextStageFailedPayload,
    EventType.CONTEXT_PROFILE_LOAD_STARTED: ContextStageStartedPayload,
    EventType.CONTEXT_PROFILE_LOAD_COMPLETED: ContextStageCompletedPayload,
    EventType.CONTEXT_PROFILE_LOAD_FAILED: ContextStageFailedPayload,
    EventType.CONTEXT_SUMMARY_LOAD_STARTED: ContextStageStartedPayload,
    EventType.CONTEXT_SUMMARY_LOAD_COMPLETED: ContextStageCompletedPayload,
    EventType.CONTEXT_SUMMARY_LOAD_FAILED: ContextStageFailedPayload,
    EventType.CONTEXT_HISTORY_SELECT_STARTED: ContextStageStartedPayload,
    EventType.CONTEXT_HISTORY_SELECT_COMPLETED: ContextStageCompletedPayload,
    EventType.CONTEXT_HISTORY_SELECT_FAILED: ContextStageFailedPayload,
    EventType.CONTEXT_MEMORY_RETRIEVE_STARTED: ContextStageStartedPayload,
    EventType.CONTEXT_MEMORY_RETRIEVE_COMPLETED: ContextStageCompletedPayload,
    EventType.CONTEXT_MEMORY_RETRIEVE_FAILED: ContextStageFailedPayload,
    EventType.CONTEXT_KNOWLEDGE_RETRIEVE_STARTED: ContextStageStartedPayload,
    EventType.CONTEXT_KNOWLEDGE_RETRIEVE_COMPLETED: ContextStageCompletedPayload,
    EventType.CONTEXT_KNOWLEDGE_RETRIEVE_FAILED: ContextStageFailedPayload,
    EventType.CONTEXT_PROMPT_ASSEMBLE_STARTED: ContextStageStartedPayload,
    EventType.CONTEXT_PROMPT_ASSEMBLE_COMPLETED: ContextStageCompletedPayload,
    EventType.CONTEXT_PROMPT_ASSEMBLE_FAILED: ContextStageFailedPayload,

    # model
    EventType.MODEL_STARTED: ModelStartedPayload,
    EventType.MODEL_FIRST_TOKEN: ModelFirstTokenPayload,
    EventType.MODEL_COMPLETED: ModelCompletedPayload,
    EventType.MODEL_FAILED: ModelFailedPayload,
    EventType.MODEL_RETRY_SCHEDULED: ModelRetryScheduledPayload,
    EventType.MODEL_FALLBACK: ModelFallbackPayload,

    # tool
    EventType.TOOL_CALLED: ToolCalledData,
    EventType.TOOL_STARTED: ToolStartedPayload,
    EventType.TOOL_COMPLETED: ToolCompletedData,
    EventType.TOOL_FAILED: ToolFailedPayload,
    EventType.TOOL_INTERRUPTED: ToolInterruptedPayload,
    EventType.TOOL_RESUMED: ToolResumedPayload,
    EventType.TOOL_RETRY_SCHEDULED: ToolRetryScheduledPayload,
    EventType.TOOL_IDEMPOTENCY_DECISION: ToolIdempotencyDecisionPayload,

    # message
    EventType.MESSAGE_STARTED: MessageStartedData,
    EventType.MESSAGE_DELTA: MessageDeltaData,
    EventType.MESSAGE_COMPLETED: MessageCompletedData,
    EventType.MESSAGE_REASONING_DELTA: MessageReasoningDeltaData,
    EventType.MESSAGE_REASONING_COMPLETED: MessageReasoningCompletedData,

    # fanout
    EventType.FANOUT_STARTED: FanoutStartedPayload,
    EventType.FANOUT_BRANCH_CREATED: FanoutBranchCreatedPayload,
    EventType.FANOUT_COMPLETED: FanoutCompletedPayload,
    EventType.FANOUT_FAILED: FanoutFailedPayload,
    EventType.FANOUT_CANCELLED: FanoutCancelledPayload,

    # approval / hitl
    EventType.APPROVAL_STARTED: ApprovalStartedPayload,
    EventType.APPROVAL_COMPLETED: ApprovalCompletedPayload,
    EventType.APPROVAL_FAILED: ApprovalFailedPayload,
    EventType.APPROVAL_CANCELLED: ApprovalCancelledPayload,
    EventType.HITL_APPROVAL_REQUESTED: HitlApprovalRequestedPayload,
    EventType.HITL_DECISION: HitlDecisionPayload,
    EventType.HITL_RESUMED: HitlResumedPayload,
    EventType.HITL_REQUESTED: HitlRequestedData,
    EventType.HITL_RESOLVED: HitlResolvedData,

    # plan / task / subagent
    EventType.PLAN_PUBLISHED: PlanPublishedData,
    EventType.TASK_UPDATED: TaskUpdatedData,
    EventType.PLAN_CLEARED: PlanClearedData,
    EventType.SUBAGENT_CREATED: SubagentCreatedData,
    EventType.SUBAGENT_CLAIMED: SubagentClaimedData,
    EventType.SUBAGENT_TASK_UPDATED: SubagentTaskUpdatedData,
    EventType.SUBAGENT_COMPLETED: SubagentCompletedData,
}


# ============================================================
# 放行集合
# ============================================================
#
# 放行白名单是**契约**，写在代码里；配置只调优，不改变“哪些事件属于
# 哪条流”这个事实。


CHAT_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.RUN_STARTED,
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
    EventType.RUN_CANCELLED,
    EventType.NODE_STARTED,
    EventType.NODE_COMPLETED,
    EventType.MESSAGE_STARTED,
    EventType.MESSAGE_DELTA,
    EventType.MESSAGE_COMPLETED,
    EventType.MESSAGE_REASONING_DELTA,
    EventType.MESSAGE_REASONING_COMPLETED,
    EventType.TOOL_CALLED,
    EventType.TOOL_COMPLETED,
    EventType.PLAN_PUBLISHED,
    EventType.TASK_UPDATED,
    EventType.PLAN_CLEARED,
    EventType.SUBAGENT_CREATED,
    EventType.SUBAGENT_CLAIMED,
    EventType.SUBAGENT_TASK_UPDATED,
    EventType.SUBAGENT_COMPLETED,
    EventType.HITL_REQUESTED,
    EventType.HITL_RESOLVED,
})


TRACE_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.RUN_ACCEPTED,
    EventType.RUN_QUEUED,
    EventType.RUN_STARTED,
    EventType.RUN_FIRST_OUTPUT,
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
    EventType.RUN_CANCELLED,
    EventType.RUN_CLIENT_DISCONNECTED,
    EventType.NODE_STARTED,
    EventType.NODE_COMPLETED,
    EventType.NODE_FAILED,
    EventType.NODE_SKIPPED,
    EventType.NODE_INTERRUPTED,
    EventType.NODE_RESUMED,
    EventType.GRAPH_TRANSITION,
    EventType.CONTEXT_BUILD_STARTED,
    EventType.CONTEXT_BUILD_COMPLETED,
    EventType.CONTEXT_BUILD_FAILED,
    EventType.CONTEXT_PROFILE_LOAD_STARTED,
    EventType.CONTEXT_PROFILE_LOAD_COMPLETED,
    EventType.CONTEXT_PROFILE_LOAD_FAILED,
    EventType.CONTEXT_SUMMARY_LOAD_STARTED,
    EventType.CONTEXT_SUMMARY_LOAD_COMPLETED,
    EventType.CONTEXT_SUMMARY_LOAD_FAILED,
    EventType.CONTEXT_HISTORY_SELECT_STARTED,
    EventType.CONTEXT_HISTORY_SELECT_COMPLETED,
    EventType.CONTEXT_HISTORY_SELECT_FAILED,
    EventType.CONTEXT_MEMORY_RETRIEVE_STARTED,
    EventType.CONTEXT_MEMORY_RETRIEVE_COMPLETED,
    EventType.CONTEXT_MEMORY_RETRIEVE_FAILED,
    EventType.CONTEXT_KNOWLEDGE_RETRIEVE_STARTED,
    EventType.CONTEXT_KNOWLEDGE_RETRIEVE_COMPLETED,
    EventType.CONTEXT_KNOWLEDGE_RETRIEVE_FAILED,
    EventType.CONTEXT_PROMPT_ASSEMBLE_STARTED,
    EventType.CONTEXT_PROMPT_ASSEMBLE_COMPLETED,
    EventType.CONTEXT_PROMPT_ASSEMBLE_FAILED,
    EventType.MODEL_STARTED,
    EventType.MODEL_FIRST_TOKEN,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_FAILED,
    EventType.MODEL_RETRY_SCHEDULED,
    EventType.MODEL_FALLBACK,
    EventType.TOOL_STARTED,
    EventType.TOOL_COMPLETED,
    EventType.TOOL_FAILED,
    EventType.TOOL_INTERRUPTED,
    EventType.TOOL_RESUMED,
    EventType.TOOL_RETRY_SCHEDULED,
    EventType.TOOL_IDEMPOTENCY_DECISION,
    EventType.FANOUT_STARTED,
    EventType.FANOUT_BRANCH_CREATED,
    EventType.FANOUT_COMPLETED,
    EventType.FANOUT_FAILED,
    EventType.FANOUT_CANCELLED,
    EventType.APPROVAL_STARTED,
    EventType.APPROVAL_COMPLETED,
    EventType.APPROVAL_FAILED,
    EventType.APPROVAL_CANCELLED,
    EventType.HITL_APPROVAL_REQUESTED,
    EventType.HITL_DECISION,
    EventType.HITL_RESUMED,
})


ALL_EVENT_TYPES: frozenset[EventType] = frozenset(EventType)

#: 只属于 trace 的事件。它们没有 trace_id 时不该被构造。
TRACE_ONLY_EVENT_TYPES: frozenset[EventType] = (
    TRACE_EVENT_TYPES - CHAT_EVENT_TYPES
)


# ============================================================
# Event
# ============================================================


class Event(BaseModel):
    """
    一条运行事件。

    信封字段**平铺**，事件自己的内容统一进 `data`。这个形状同时服务
    chat 与 trace：两个 sink 写出的记录字段集合完全相同，只有键不同。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = EVENT_SCHEMA_VERSION

    #: 每条唯一。trace 消费端按它去重。
    event_id: str = Field(default_factory=_new_event_id, min_length=1)

    type: EventType

    #: 产生时刻（UTC）。在构造时取，不等到写入时补 —— 排队 / 背压会把
    #: 写入推后，用写入时刻会让时间戳全挤在一起。
    occurred_at: datetime = Field(default_factory=_utc_now)

    #: 本次运行 id。必填：运行是一等公民。
    run_id: str = Field(min_length=1)

    #: 前端传入的聚合键，可空。trace sink 只收非空。
    trace_id: str | None = None

    #: 业务会话 id（不是 langgraph 的 thread_id）。
    session_id: str | None = None

    #: 节点名。message.* / node.* 使用；run.* 这类没有。
    node: str | None = None

    #: trace span 语义。chat 事件可以为空。
    span_id: str | None = None
    parent_span_id: str | None = None
    span_kind: SpanKind | None = None
    span_name: str | None = None

    #: 重试序号与显式关系。trace 的 retry / resume 靠它还原因果线。
    attempt: int = Field(default=1, ge=1)
    links: list[SpanLink] = Field(default_factory=list)

    #: 扇出分支归属。不是 fanout 的事件留空。
    fanout_id: str | None = None
    branch_id: str | None = None
    branch_index: int | None = Field(default=None, ge=0)

    #: 状态 / 耗时 / 错误提升到信封，run 事件与 span 共用同一套语义。
    status: SpanStatus | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    error: EventError | None = None

    #: 事件自己的业务载荷。构造时按 `type` 严格校验。
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _require_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at 必须带时区（UTC）")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _validate_payload(self) -> "Event":
        model = EVENT_PAYLOAD_MODELS[self.type]
        validated = model.model_validate(self.data)

        # `exclude_none=True`：可选字段没填就不写进载荷，保持与生产者
        # 发送内容一致；必填字段与显式 False / 0 不受影响。
        self.data = validated.model_dump(mode="json", exclude_none=True)

        if self.type in TRACE_ONLY_EVENT_TYPES and not self.trace_id:
            raise ValueError(
                f"{self.type.value} 是 trace 专属事件，必须带 trace_id"
            )

        if self.status is SpanStatus.FAILED and self.error is None:
            raise ValueError(
                f"{self.type.value} 的 status=failed，必须带 error"
            )

        return self

    def to_fields(self) -> dict[str, str]:
        """
        扁平化成 Redis Stream 的字段。

        ⚠️ Redis Stream 的字段值只能是标量。`data` / `error` 必须是 JSON
        字符串；空值统一写空字符串，保证两个 sink 的记录字段集合完全相同。
        """

        return {
            "schema_version": self.schema_version,
            "event": self.type.value,
            "event_id": self.event_id,
            "occurred_at": self.occurred_at.isoformat(),
            "run_id": self.run_id,
            "trace_id": self.trace_id or "",
            "session_id": self.session_id or "",
            "node": self.node or "",
            "span_id": self.span_id or "",
            "parent_span_id": self.parent_span_id or "",
            "span_kind": self.span_kind.value if self.span_kind else "",
            "span_name": self.span_name or "",
            "attempt": str(self.attempt),
            "links": json.dumps(
                [link.model_dump(mode="json") for link in self.links],
                ensure_ascii=False,
            ),
            "fanout_id": self.fanout_id or "",
            "branch_id": self.branch_id or "",
            "branch_index": (
                "" if self.branch_index is None else str(self.branch_index)
            ),
            "status": self.status.value if self.status else "",
            "duration_ms": (
                "" if self.duration_ms is None else str(self.duration_ms)
            ),
            "error": (
                ""
                if self.error is None
                else json.dumps(
                    self.error.model_dump(mode="json", exclude_none=True),
                    ensure_ascii=False,
                )
            ),
            "data": json.dumps(
                self.data,
                ensure_ascii=False,
                default=str,
            ),
        }
