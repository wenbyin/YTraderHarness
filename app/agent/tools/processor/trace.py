"""
工具执行链的 Trace 埋点工具。

工具处理器是**自有 runtime**，按约定用执行链传下来的 writer
（`ToolRuntime.stream_writer`），不自己调 `get_stream_writer()`。

Span 对齐
--------
工具 Span 的 id 由 `tool_span_id(trace_id, tool_call_id)` 确定性生成；
投影层（从协议 `tools` 事件推 tool.*）用的是同一个函数 —— 于是
idempotency / retry 这些只在处理器里才知道的事件，能挂到同一个
Tool Span 上。
"""

from __future__ import annotations

from typing import Any

from app.common.event import EventType, SpanKind, TraceEmitter
from app.common.event.trace_emitter import tool_span_id

__all__ = [
    "emit_tool_idempotency",
    "emit_tool_retry",
    "tool_trace",
]


def tool_trace(
        runtime: Any,
) -> tuple[TraceEmitter | None, str | None, str | None, str | None]:
    """
    从 ToolRuntime 取 trace 句柄。

    Returns
    -------
    (emitter, span_id, parent_span_id, tool_call_id)
        trace 未开启时 emitter 是 disabled、span_id 为 None。
    """

    if runtime is None:
        return None, None, None, None

    context = getattr(runtime, "context", None)

    trace_id = getattr(context, "trace_id", None)
    run_id = getattr(context, "pipeline_run_id", None)
    session_id = getattr(context, "session_id", None)

    writer = getattr(runtime, "stream_writer", None)
    tool_call_id = getattr(runtime, "tool_call_id", None)

    info = getattr(runtime, "execution_info", None)
    task_id = getattr(info, "task_id", None)

    parent_span_id = task_id or (
        str(run_id) if run_id is not None else None
    )

    emitter = TraceEmitter(
        trace_id=trace_id,
        run_id=str(run_id) if run_id is not None else "",
        thread_id=None if session_id is None else str(session_id),
        writer=writer,
    )

    span_id = (
        tool_span_id(str(trace_id), str(tool_call_id))
        if trace_id is not None and tool_call_id is not None
        else None
    )

    return (
        emitter if emitter.enabled else None,
        span_id,
        parent_span_id,
        None if tool_call_id is None else str(tool_call_id),
    )


def emit_tool_idempotency(
        runtime: Any,
        *,
        tool_name: str,
        decision: str,
        scope: str | None = None,
        existing_execution_status: str | None = None,
) -> None:
    """`tool.idempotency_decision`（瞬时事件）。"""

    emitter, span_id, parent_span_id, call_id = tool_trace(runtime)

    if emitter is None or span_id is None or call_id is None:
        return

    emitter.instant(
        EventType.TOOL_IDEMPOTENCY_DECISION,
        span_kind=SpanKind.TOOL,
        span_name=tool_name,
        span_id=span_id,
        parent_span_id=parent_span_id,
        payload={
            "tool_name": tool_name,
            "tool_call_id": call_id,
            "scope": scope,
            "decision": decision,
            "existing_execution_status": existing_execution_status,
        },
    )


def emit_tool_retry(
        runtime: Any,
        *,
        tool_name: str,
        next_attempt: int,
        delay_ms: int,
        reason: str | None,
) -> None:
    """`tool.retry_scheduled`（瞬时事件）。"""

    emitter, span_id, parent_span_id, _ = tool_trace(runtime)

    if emitter is None or span_id is None:
        return

    emitter.instant(
        EventType.TOOL_RETRY_SCHEDULED,
        span_kind=SpanKind.TOOL,
        span_name=tool_name,
        span_id=span_id,
        parent_span_id=parent_span_id,
        payload={
            "failed_span_id": span_id,
            "next_attempt": next_attempt,
            "delay_ms": delay_ms,
            "reason": reason,
        },
    )
