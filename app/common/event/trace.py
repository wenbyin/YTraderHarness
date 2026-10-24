"""
Trace 事件的**词汇与校验**（纯契约，不碰 Redis / LangGraph）。

来源：`design_agent_eval/agent_eval_codegen_prompt.md` 第一阶段。

与本仓库既有 `RunEvent` 的关系
------------------------------
两条流、两套词汇，刻意不合并：

    RunEvent      聊天窗事件（含逐 token delta）   topic=agent_run
    TraceEvent    执行观测事件（无 delta）          key=trace:{trace_id}:events

共同的实现只有"怎么把一条事件写进 Redis"（复用 `RedisEventPublisher`
的队列与 sink），事件模型本身各自独立。

⚠️ 与提示词的一处刻意偏离
--------------------------
提示词写的是 `trace_id = run_id`；本项目的产品要求是**前端传入的
trace_id 为准**，它可以跨 run / 跨 session。因此这里额外保留 `run_id`
字段：`trace_id` 是聚合键，`run_id` 才是"这一次运行"。其余字段与提示词
逐字对齐。

严格校验
--------
`payload` 不是自由字典：每个 `event_type` 都映射到一个 `extra="forbid"`
的 Pydantic 模型（见 `_PAYLOAD_MODELS`）。多一个字段、少一个必填字段、
枚举值写错，都会在构造事件的瞬间抛 `ValidationError` —— 不允许"悄悄
通过"。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "DECISION_VALUES",
    "EventType",
    "SpanKind",
    "SpanLink",
    "SpanStatus",
    "TraceError",
    "TraceEvent",
    "trace_stream_key",
]


# ============================================================
# 枚举
# ============================================================


class EventType(str, Enum):
    """第一阶段唯一允许的事件名。值就是线上字符串。"""

    # A. Run 生命周期
    RUN_ACCEPTED = "run.accepted"
    RUN_QUEUED = "run.queued"
    RUN_STARTED = "run.started"
    RUN_FIRST_OUTPUT = "run.first_output"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_CLIENT_DISCONNECTED = "run.client_disconnected"

    # B. Graph Node
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    NODE_SKIPPED = "node.skipped"
    NODE_INTERRUPTED = "node.interrupted"
    NODE_RESUMED = "node.resumed"

    # C. Graph 控制流
    GRAPH_TRANSITION = "graph.transition"

    # D. ContextBuilder
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

    # E. ModelRuntime
    MODEL_STARTED = "model.started"
    MODEL_FIRST_TOKEN = "model.first_token"
    MODEL_COMPLETED = "model.completed"
    MODEL_FAILED = "model.failed"
    MODEL_RETRY_SCHEDULED = "model.retry_scheduled"
    MODEL_FALLBACK = "model.fallback"

    # F. Tool
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    TOOL_INTERRUPTED = "tool.interrupted"
    TOOL_RESUMED = "tool.resumed"
    TOOL_RETRY_SCHEDULED = "tool.retry_scheduled"
    TOOL_IDEMPOTENCY_DECISION = "tool.idempotency_decision"

    # G. Fanout
    FANOUT_STARTED = "fanout.started"
    FANOUT_BRANCH_CREATED = "fanout.branch_created"
    FANOUT_COMPLETED = "fanout.completed"
    FANOUT_FAILED = "fanout.failed"
    FANOUT_CANCELLED = "fanout.cancelled"

    # H. HITL / approval
    APPROVAL_STARTED = "approval.started"
    APPROVAL_COMPLETED = "approval.completed"
    APPROVAL_FAILED = "approval.failed"
    APPROVAL_CANCELLED = "approval.cancelled"
    HITL_APPROVAL_REQUESTED = "hitl.approval_requested"
    HITL_DECISION = "hitl.decision"
    HITL_RESUMED = "hitl.resumed"

class SpanKind(str, Enum):
    RUN = "run"
    NODE = "node"
    CONTEXT = "context"
    CONTEXT_STAGE = "context_stage"
    MODEL = "model"
    TOOL = "tool"
    FANOUT = "fanout"
    APPROVAL = "approval"


class SpanStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class LinkRelation(str, Enum):
    GRAPH_TRANSITION = "graph_transition"
    RETRY_OF = "retry_of"
    RESUMED_FROM = "resumed_from"
    FALLBACK_FROM = "fallback_from"


#: 幂等决策允许的值。前 5 个是现有代码真实会产生的；后 2 个在第一阶段
#: 不产生（现有实现没有对应分支），保留枚举是为了协议完整。
DECISION_VALUES: tuple[str, ...] = (
    "execute",
    "cache_hit",
    "already_succeeded",
    "already_running",
    "blocked_unknown",
    "quota_rejected",
    "deduplicated",
)


# ============================================================
# 组成部分
# ============================================================


class SpanLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: LinkRelation
    span_id: str = Field(min_length=1)


class TraceError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    code: str | None = None
    message: str = ""
    retryable: bool | None = None


# ============================================================
# payload 模型（按 event_type 严格校验）
# ============================================================


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunAcceptedPayload(_Payload):
    agent_type: str


class RunQueuedPayload(_Payload):
    queue_name: str


class RunStartedPayload(_Payload):
    agent_type: str
    graph_name: str


class RunFirstOutputPayload(_Payload):
    output_channel: str


class RunCompletedPayload(_Payload):
    finish_reason: str | None = None


class RunFailedPayload(_Payload):
    pass


class RunCancelledPayload(_Payload):
    reason: str | None = None


class RunClientDisconnectedPayload(_Payload):
    disconnect_reason: str | None = None


class NodeStartedPayload(_Payload):
    node_name: str
    node_type: str | None = None
    node_execution_index: int = Field(ge=0)


class NodeCompletedPayload(_Payload):
    node_name: str
    node_type: str | None = None


class NodeFailedPayload(_Payload):
    node_name: str
    node_type: str | None = None


class NodeSkippedPayload(_Payload):
    node_name: str
    reason: str | None = None


class NodeInterruptedPayload(_Payload):
    node_name: str
    interrupt_id: str
    interrupt_type: str | None = None


class NodeResumedPayload(_Payload):
    node_name: str
    interrupt_id: str


class GraphTransitionPayload(_Payload):
    from_span_id: str = Field(min_length=1)
    to_span_id: str = Field(min_length=1)
    from_node: str
    to_node: str
    transition_kind: Literal["normal", "conditional", "handoff", "command"]
    route_key: str | None = None


class ContextStageStartedPayload(_Payload):
    pass


class ContextStageCompletedPayload(_Payload):
    item_count: int | None = Field(default=None, ge=0)
    selected_count: int | None = Field(default=None, ge=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)


class ContextStageFailedPayload(_Payload):
    pass


class ModelStartedPayload(_Payload):
    model_role: str
    provider: str
    model_name: str
    streaming: bool


class ModelFirstTokenPayload(_Payload):
    ttft_ms: int = Field(ge=0)


class ModelCompletedPayload(_Payload):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    response_id: str | None = None


class ModelFailedPayload(_Payload):
    pass


class ModelRetryScheduledPayload(_Payload):
    failed_span_id: str = Field(min_length=1)
    next_attempt: int = Field(ge=1)
    delay_ms: int = Field(ge=0)
    reason: str | None = None


class ModelFallbackPayload(_Payload):
    failed_span_id: str = Field(min_length=1)
    requested_model: str
    fallback_model: str
    reason: str | None = None


class ToolStartedPayload(_Payload):
    tool_name: str
    tool_call_id: str


class ToolCompletedPayload(_Payload):
    tool_name: str
    tool_call_id: str
    result_size_bytes: int | None = Field(default=None, ge=0)


class ToolFailedPayload(_Payload):
    tool_name: str
    tool_call_id: str


class ToolInterruptedPayload(_Payload):
    tool_name: str
    tool_call_id: str
    interrupt_id: str


class ToolResumedPayload(_Payload):
    tool_name: str
    tool_call_id: str
    interrupt_id: str


class ToolRetryScheduledPayload(_Payload):
    failed_span_id: str = Field(min_length=1)
    next_attempt: int = Field(ge=1)
    delay_ms: int = Field(ge=0)
    reason: str | None = None


class ToolIdempotencyDecisionPayload(_Payload):
    tool_name: str
    tool_call_id: str
    scope: str | None = None
    decision: Literal[
        "execute",
        "cache_hit",
        "already_succeeded",
        "already_running",
        "blocked_unknown",
        "quota_rejected",
        "deduplicated",
    ]
    existing_execution_status: str | None = None


class FanoutStartedPayload(_Payload):
    fanout_type: str
    dispatch_mode: Literal["parallel", "sequential"]
    expected_branch_count: int | None = Field(default=None, ge=0)


class FanoutBranchCreatedPayload(_Payload):
    branch_id: str
    branch_span_id: str
    branch_index: int = Field(ge=0)
    target_kind: str
    target_name: str


class FanoutCompletedPayload(_Payload):
    created_branch_count: int = Field(ge=0)
    completed_branch_count: int = Field(ge=0)
    failed_branch_count: int = Field(ge=0)
    cancelled_branch_count: int = Field(ge=0)


class FanoutFailedPayload(_Payload):
    pass


class FanoutCancelledPayload(_Payload):
    reason: str | None = None


class ApprovalStartedPayload(_Payload):
    approval_id: str
    action_name: str


class ApprovalCompletedPayload(_Payload):
    approval_id: str
    decision: Literal["approved", "rejected", "expired"]


class ApprovalFailedPayload(_Payload):
    approval_id: str


class ApprovalCancelledPayload(_Payload):
    approval_id: str
    reason: str | None = None


class HitlApprovalRequestedPayload(_Payload):
    approval_id: str
    action_name: str


class HitlDecisionPayload(_Payload):
    approval_id: str
    decision: Literal["approved", "rejected", "expired"]
    actor_type: str | None = None


class HitlResumedPayload(_Payload):
    approval_id: str
    resumed_span_id: str




#: event_type -> 专属 payload 模型。
_PAYLOAD_MODELS: dict[EventType, type[_Payload]] = {
    EventType.RUN_ACCEPTED: RunAcceptedPayload,
    EventType.RUN_QUEUED: RunQueuedPayload,
    EventType.RUN_STARTED: RunStartedPayload,
    EventType.RUN_FIRST_OUTPUT: RunFirstOutputPayload,
    EventType.RUN_COMPLETED: RunCompletedPayload,
    EventType.RUN_FAILED: RunFailedPayload,
    EventType.RUN_CANCELLED: RunCancelledPayload,
    EventType.RUN_CLIENT_DISCONNECTED: RunClientDisconnectedPayload,
    EventType.NODE_STARTED: NodeStartedPayload,
    EventType.NODE_COMPLETED: NodeCompletedPayload,
    EventType.NODE_FAILED: NodeFailedPayload,
    EventType.NODE_SKIPPED: NodeSkippedPayload,
    EventType.NODE_INTERRUPTED: NodeInterruptedPayload,
    EventType.NODE_RESUMED: NodeResumedPayload,
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
    EventType.MODEL_STARTED: ModelStartedPayload,
    EventType.MODEL_FIRST_TOKEN: ModelFirstTokenPayload,
    EventType.MODEL_COMPLETED: ModelCompletedPayload,
    EventType.MODEL_FAILED: ModelFailedPayload,
    EventType.MODEL_RETRY_SCHEDULED: ModelRetryScheduledPayload,
    EventType.MODEL_FALLBACK: ModelFallbackPayload,
    EventType.TOOL_STARTED: ToolStartedPayload,
    EventType.TOOL_COMPLETED: ToolCompletedPayload,
    EventType.TOOL_FAILED: ToolFailedPayload,
    EventType.TOOL_INTERRUPTED: ToolInterruptedPayload,
    EventType.TOOL_RESUMED: ToolResumedPayload,
    EventType.TOOL_RETRY_SCHEDULED: ToolRetryScheduledPayload,
    EventType.TOOL_IDEMPOTENCY_DECISION: ToolIdempotencyDecisionPayload,
    EventType.FANOUT_STARTED: FanoutStartedPayload,
    EventType.FANOUT_BRANCH_CREATED: FanoutBranchCreatedPayload,
    EventType.FANOUT_COMPLETED: FanoutCompletedPayload,
    EventType.FANOUT_FAILED: FanoutFailedPayload,
    EventType.FANOUT_CANCELLED: FanoutCancelledPayload,
    EventType.APPROVAL_STARTED: ApprovalStartedPayload,
    EventType.APPROVAL_COMPLETED: ApprovalCompletedPayload,
    EventType.APPROVAL_FAILED: ApprovalFailedPayload,
    EventType.APPROVAL_CANCELLED: ApprovalCancelledPayload,
    EventType.HITL_APPROVAL_REQUESTED: HitlApprovalRequestedPayload,
    EventType.HITL_DECISION: HitlDecisionPayload,
    EventType.HITL_RESUMED: HitlResumedPayload,
}

#: 必须带 `error` 的事件类型。
_FAILED_EVENTS: frozenset[EventType] = frozenset({
    EventType.RUN_FAILED,
    EventType.NODE_FAILED,
    EventType.CONTEXT_BUILD_FAILED,
    EventType.CONTEXT_PROFILE_LOAD_FAILED,
    EventType.CONTEXT_SUMMARY_LOAD_FAILED,
    EventType.CONTEXT_HISTORY_SELECT_FAILED,
    EventType.CONTEXT_MEMORY_RETRIEVE_FAILED,
    EventType.CONTEXT_KNOWLEDGE_RETRIEVE_FAILED,
    EventType.CONTEXT_PROMPT_ASSEMBLE_FAILED,
    EventType.MODEL_FAILED,
    EventType.TOOL_FAILED,
    EventType.FANOUT_FAILED,
    EventType.APPROVAL_FAILED,
})


# ============================================================
# TraceEvent
# ============================================================


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceEvent(BaseModel):
    """一条 Trace 事件。字段语义见提示词第 2 节。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"

    event_id: str = Field(min_length=1)
    event_type: EventType
    occurred_at: datetime = Field(default_factory=_utc_now)

    #: 聚合键（前端传入）。同一 trace 可跨 run / session。
    trace_id: str = Field(min_length=1)

    #: 本项目的补充字段（见模块 docstring 的偏离说明）。
    run_id: str = Field(min_length=1)

    #: 会话 ID（业务 session_id），不是 langgraph 内部 thread_id。
    thread_id: str | None = None

    span_id: str = Field(min_length=1)
    parent_span_id: str | None = None
    span_kind: SpanKind
    span_name: str = Field(min_length=1)

    status: SpanStatus | None = None
    attempt: int = Field(default=1, ge=1)
    duration_ms: int | None = Field(default=None, ge=0)

    links: list[SpanLink] = Field(default_factory=list)

    fanout_id: str | None = None
    branch_id: str | None = None
    branch_index: int | None = Field(default=None, ge=0)

    payload: dict[str, Any] = Field(default_factory=dict)
    error: TraceError | None = None

    @field_validator("occurred_at")
    @classmethod
    def _require_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at 必须带时区（UTC）")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _validate_payload(self) -> "TraceEvent":
        model = _PAYLOAD_MODELS[self.event_type]
        validated = model.model_validate(self.payload)
        self.payload = validated.model_dump(exclude_none=True)

        if self.event_type in _FAILED_EVENTS and self.error is None:
            raise ValueError(f"{self.event_type.value} 必须带 error")

        return self

    # ----------------------------------------------------------
    # Redis Stream 形状
    # ----------------------------------------------------------

    @property
    def topic(self) -> str:
        """固定 topic。"""

        return "trace"

    @property
    def topic_id(self) -> str:
        """键的右半段：`<trace_id>:events`（见提示词第 6 节）。"""

        return f"{self.trace_id}:events"

    def to_fields(self) -> dict[str, str]:
        """
        扁平化成 Redis Stream 的字段。

        常用字段平铺（便于 XRANGE 后直接看），完整事件再序列化进 `data`。
        Redis Stream 的字段值只能是标量，所以这里全部转字符串。
        """

        return {
            "event": self.event_type.value,
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id or "",
            "span_kind": self.span_kind.value,
            "span_name": self.span_name,
            "occurred_at": self.occurred_at.isoformat(),
            "data": json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                default=str,
            ),
        }


def trace_stream_key(trace_id: str) -> str:
    """`trace:{trace_id}:events`。"""

    return f"trace:{trace_id}:events"
