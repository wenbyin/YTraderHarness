"""
Trace 事件发射器（图内 / 自有 runtime 共用）。

两条发射路径
------------
    TraceEmitter      由 LangGraph 节点 / 自有 runtime 使用：
                      把事件写进当前 LangGraph 的 **custom stream**，
                      由投影层校验后再落 Redis。
    TracePublisher    由 Run 外层（dispatch / service）使用：
                      直接走既有 Redis 发布器。

为什么分开
----------
提示词第 5 节要求：图内事件走 Custom Stream；Run 外层事件发生在图调用
边界之外时，由 Run Orchestrator 用同一事件模型直接发布。把两者混在一个
对象里会让"谁在什么上下文里能 await"变得含糊。

线程 / 并发安全
---------------
`TraceEmitter` **不保存任何跨事件的业务状态**（没有 current_span_id 之类）。
每次 `start()` 都产生一个独立的 `TraceSpan` 句柄，句柄自带 span_id。
并行分支因此各拿各的句柄，不会串线。

发射失败绝不影响主流程
----------------------
`writer` 是 LangGraph 的 stream writer（同步、入队即返回）。这里的
`emit()` 仍然整体包 try/except：事件是旁路，序列化失败或 writer 抛错
都不该让一次 Agent 运行失败。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from .trace import (
    EventType,
    SpanKind,
    SpanLink,
    SpanStatus,
    TraceError,
    TraceEvent,
)

__all__ = [
    "TraceEmitter",
    "TraceSpan",
    "error_from_exception",
    "stable_span_id",
    "tool_span_id",
]

logger = logging.getLogger(__name__)


#: writer 的形状：langgraph 的 stream writer 接受一个可 JSON 序列化的值。
Writer = Callable[[Any], None]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


def stable_span_id(kind: str, *parts: Any) -> str:
    """
    确定性 Span ID（重启 / resume / 重放后仍是同一个）。

    只给**控制流 / 工具**这类需要在多个层之间对齐的 span 用；普通的
    节点、模型、Context 阶段仍然用随机 id（一次执行一个 span）。
    """

    payload = ":".join([kind, *(str(part) for part in parts)])

    return uuid.uuid5(uuid.NAMESPACE_URL, payload).hex


def tool_span_id(trace_id: str, tool_call_id: str) -> str:
    """
    工具 Span 的稳定 id。

    投影层（从协议事件推 tool.*）与工具执行链（幂等 / 重试埋点）必须
    算出**同一个值**，否则同一次调用会被拆成两个 Span。
    """

    return stable_span_id("tool", trace_id, tool_call_id)


def error_from_exception(
        exc: BaseException,
        *,
        code: str | None = None,
        retryable: bool | None = None,
) -> TraceError:
    """
    把异常压成结构化错误。

    ⚠️ 只保留类型名与一条截断后的消息，不带 traceback、不带完整 Prompt /
    工具参数。消息长度在这里就截断，避免敏感内容随事件进 Redis。
    """

    exc_code = code
    if exc_code is None:
        candidate = getattr(exc, "code", None)
        if isinstance(candidate, str) and candidate:
            exc_code = candidate

    message = str(exc)
    if len(message) > 500:
        message = message[:497] + "..."

    return TraceError(
        type=type(exc).__name__,
        code=exc_code,
        message=message,
        retryable=retryable,
    )


class TraceSpan:
    """
    一次 Span 的句柄。

    开始事件在 `TraceEmitter.start()` 里已经发出；这里只负责结束。
    同一个 span_id 覆盖开始 / 结束（以及 failed / cancelled）事件。
    """

    __slots__ = (
        "span_id",
        "span_kind",
        "span_name",
        "parent_span_id",
        "attempt",
        "fanout_id",
        "branch_id",
        "branch_index",
        "links",
        "_emitter",
        "_end_type",
        "_failed_type",
        "_started_at",
    )

    def __init__(
            self,
            *,
            emitter: "TraceEmitter",
            span_id: str,
            span_kind: SpanKind,
            span_name: str,
            parent_span_id: str | None,
            attempt: int,
            fanout_id: str | None,
            branch_id: str | None,
            branch_index: int | None,
            links: Sequence[SpanLink],
            end_type: EventType,
            failed_type: EventType | None,
            started_at: datetime,
    ) -> None:

        self._emitter = emitter
        self.span_id = span_id
        self.span_kind = span_kind
        self.span_name = span_name
        self.parent_span_id = parent_span_id
        self.attempt = attempt
        self.fanout_id = fanout_id
        self.branch_id = branch_id
        self.branch_index = branch_index
        self.links = tuple(links)
        self._end_type = end_type
        self._failed_type = failed_type
        self._started_at = started_at

    # ----------------------------------------------------------

    def complete(
            self,
            payload: Mapping[str, Any] | None = None,
            *,
            status: SpanStatus = SpanStatus.COMPLETED,
            duration_ms: int | None = None,
            occurred_at: datetime | None = None,
    ) -> TraceEvent | None:
        """正常结束。"""

        return self._end(
            self._end_type,
            payload=payload,
            status=status,
            duration_ms=duration_ms,
            occurred_at=occurred_at,
        )

    def fail(
            self,
            error: BaseException | TraceError,
            *,
            payload: Mapping[str, Any] | None = None,
            duration_ms: int | None = None,
    ) -> TraceEvent | None:
        """失败结束。`failed_type` 缺省时退回普通结束事件 + status=failed。"""

        trace_error = (
            error
            if isinstance(error, TraceError)
            else error_from_exception(error)
        )

        event_type = self._failed_type or self._end_type

        return self._emitter._emit(
            event_type,
            span_id=self.span_id,
            span_kind=self.span_kind,
            span_name=self.span_name,
            parent_span_id=self.parent_span_id,
            status=SpanStatus.FAILED,
            attempt=self.attempt,
            duration_ms=self._elapsed(duration_ms),
            links=self.links,
            fanout_id=self.fanout_id,
            branch_id=self.branch_id,
            branch_index=self.branch_index,
            payload=payload or {},
            error=trace_error,
        )

    def cancel(
            self,
            *,
            payload: Mapping[str, Any] | None = None,
            status: SpanStatus = SpanStatus.CANCELLED,
            duration_ms: int | None = None,
    ) -> TraceEvent | None:
        """取消 / 中断收尾（仍用普通结束事件，status 表明真相）。"""

        return self._end(
            self._end_type,
            payload=payload,
            status=status,
            duration_ms=duration_ms,
        )

    # ----------------------------------------------------------

    def _end(
            self,
            event_type: EventType,
            *,
            payload: Mapping[str, Any] | None,
            status: SpanStatus,
            duration_ms: int | None,
            occurred_at: datetime | None = None,
    ) -> TraceEvent | None:

        return self._emitter._emit(
            event_type,
            span_id=self.span_id,
            span_kind=self.span_kind,
            span_name=self.span_name,
            parent_span_id=self.parent_span_id,
            status=status,
            attempt=self.attempt,
            duration_ms=self._elapsed(duration_ms),
            links=self.links,
            fanout_id=self.fanout_id,
            branch_id=self.branch_id,
            branch_index=self.branch_index,
            payload=payload or {},
            occurred_at=occurred_at,
        )

    def _elapsed(self, explicit: int | None) -> int:

        if explicit is not None:
            return explicit

        return max(
            0,
            int((_now() - self._started_at).total_seconds() * 1000),
        )


class TraceEmitter:
    """
    构造并校验 TraceEvent，然后写进 custom stream。

    Parameters
    ----------
    trace_id
        前端传入的聚合键。为空 = trace 未开启：所有方法立即返回 None，
        不构造事件、不调用 writer。
    run_id
        本次运行 id（根 span_id 也用它）。
    thread_id
        业务会话 id（`agent_session.session_id`）。
    writer
        LangGraph 的 custom stream writer。节点里用 `get_stream_writer()`；
        自有 runtime 由调用方把同一个 writer 传下来。
    """

    __slots__ = ("trace_id", "run_id", "thread_id", "_writer")

    def __init__(
            self,
            *,
            trace_id: str | None,
            run_id: str,
            thread_id: str | None = None,
            writer: Writer | None = None,
    ) -> None:

        self.trace_id = trace_id or None
        self.run_id = str(run_id)
        self.thread_id = None if thread_id is None else str(thread_id)
        self._writer = writer

    # ----------------------------------------------------------
    # 状态
    # ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.trace_id and self._writer)

    def new_span_id(self) -> str:
        return _new_id()

    def new_event_id(self) -> str:
        return _new_id()

    # ----------------------------------------------------------
    # 发射
    # ----------------------------------------------------------

    def start(
            self,
            event_type: EventType,
            *,
            span_kind: SpanKind,
            span_name: str,
            end_type: EventType,
            failed_type: EventType | None = None,
            parent_span_id: str | None = None,
            span_id: str | None = None,
            attempt: int = 1,
            links: Sequence[SpanLink] = (),
            fanout_id: str | None = None,
            branch_id: str | None = None,
            branch_index: int | None = None,
            payload: Mapping[str, Any] | None = None,
    ) -> TraceSpan | None:
        """发开始事件并返回句柄。trace 未开启时返回 None。"""

        if not self.enabled:
            return None

        started_at = _now()
        span = TraceSpan(
            emitter=self,
            span_id=span_id or _new_id(),
            span_kind=span_kind,
            span_name=span_name,
            parent_span_id=parent_span_id,
            attempt=attempt,
            fanout_id=fanout_id,
            branch_id=branch_id,
            branch_index=branch_index,
            links=links,
            end_type=end_type,
            failed_type=failed_type,
            started_at=started_at,
        )

        event = self._emit(
            event_type,
            span_id=span.span_id,
            span_kind=span_kind,
            span_name=span_name,
            parent_span_id=parent_span_id,
            status=SpanStatus.RUNNING,
            attempt=attempt,
            duration_ms=None,
            links=links,
            fanout_id=fanout_id,
            branch_id=branch_id,
            branch_index=branch_index,
            payload=payload or {},
            occurred_at=started_at,
        )

        return span if event is not None else None

    def instant(
            self,
            event_type: EventType,
            *,
            span_kind: SpanKind,
            span_name: str,
            span_id: str,
            parent_span_id: str | None = None,
            status: SpanStatus | None = None,
            attempt: int = 1,
            duration_ms: int | None = None,
            links: Sequence[SpanLink] = (),
            fanout_id: str | None = None,
            branch_id: str | None = None,
            branch_index: int | None = None,
            payload: Mapping[str, Any] | None = None,
            error: TraceError | None = None,
            occurred_at: datetime | None = None,
            event_id: str | None = None,
    ) -> TraceEvent | None:
        """发一条瞬时事件（first_token / transition / retry_scheduled 等）。"""

        return self._emit(
            event_type,
            span_id=span_id,
            span_kind=span_kind,
            span_name=span_name,
            parent_span_id=parent_span_id,
            status=status,
            attempt=attempt,
            duration_ms=duration_ms,
            links=links,
            fanout_id=fanout_id,
            branch_id=branch_id,
            branch_index=branch_index,
            payload=payload,
            error=error,
            occurred_at=occurred_at,
            event_id=event_id,
        )

    # ----------------------------------------------------------

    def _emit(
            self,
            event_type: EventType,
            *,
            span_id: str,
            span_kind: SpanKind,
            span_name: str,
            parent_span_id: str | None,
            status: SpanStatus | None,
            attempt: int,
            duration_ms: int | None,
            links: Sequence[SpanLink],
            fanout_id: str | None,
            branch_id: str | None,
            branch_index: int | None,
            payload: Mapping[str, Any] | None,
            error: TraceError | None = None,
            occurred_at: datetime | None = None,
            event_id: str | None = None,
    ) -> TraceEvent | None:

        if not self.enabled:
            return None

        try:
            event = TraceEvent(
                event_id=event_id or _new_id(),
                event_type=event_type,
                occurred_at=occurred_at or _now(),
                trace_id=self.trace_id,
                run_id=self.run_id,
                thread_id=self.thread_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_kind=span_kind,
                span_name=span_name,
                status=status,
                attempt=attempt,
                duration_ms=duration_ms,
                links=list(links),
                fanout_id=fanout_id,
                branch_id=branch_id,
                branch_index=branch_index,
                payload=dict(payload or {}),
                error=error,
            )

            self._writer(event.model_dump(mode="json"))

            return event

        except Exception as exc:  # noqa: BLE001 —— 观测旁路绝不外抛
            logger.warning(
                "Trace 事件发射失败（%s / %s），已丢弃：%s: %s",
                event_type.value,
                span_name,
                type(exc).__name__,
                exc,
            )
            return None
