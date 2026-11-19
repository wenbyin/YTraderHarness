"""Trace 契约：枚举、payload 严格校验、Redis 键与信封。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.event.run_event_projector import RunEventProjector
from app.common.event import (
    EventType,
    SpanKind,
    SpanStatus,
    TraceError,
    trace_stream_key,
)
from app.common.event.spec import Event as CanonicalEvent
from app.common.event.spec import EventType as CanonicalEventType


def _canonical_event(event_type: EventType, data: dict, **overrides):
    """规范 Event —— 投影层现在只收这个形状（节点侧 TraceEmitter 直接产它）。"""

    payload = {
        "event_id": "e-1",
        "type": CanonicalEventType(event_type.value),
        "trace_id": "trace-1",
        "run_id": "run-1",
        "session_id": "session-1",
        "span_id": "span-1",
        "span_kind": SpanKind.MODEL,
        "span_name": "model.invoke",
        "data": data,
    }
    payload.update(overrides)

    return CanonicalEvent(**payload)


def test_event_type_vocabulary_is_fixed() -> None:
    # 抽样钉住词表；新增/改名必须先改提示词契约。
    assert EventType.RUN_ACCEPTED.value == "run.accepted"
    assert EventType.NODE_INTERRUPTED.value == "node.interrupted"
    assert EventType.GRAPH_TRANSITION.value == "graph.transition"
    assert EventType.CONTEXT_HISTORY_SELECT_STARTED.value == (
        "context.history_select.started"
    )
    assert EventType.MODEL_FIRST_TOKEN.value == "model.first_token"
    assert EventType.TOOL_IDEMPOTENCY_DECISION.value == (
        "tool.idempotency_decision"
    )
    assert EventType.FANOUT_BRANCH_CREATED.value == "fanout.branch_created"
    assert EventType.HITL_RESUMED.value == "hitl.resumed"


def test_payload_is_strictly_validated() -> None:
    with pytest.raises(ValidationError):
        _canonical_event(
            EventType.MODEL_STARTED,
            {
                "model_role": "reasoning",
                "provider": "p",
                "model_name": "m",
                "streaming": True,
                "not_allowed": 1,
            },
        )

    with pytest.raises(ValidationError):
        _canonical_event(
            EventType.MODEL_STARTED,
            {"model_role": "reasoning", "provider": "p"},
        )


def test_failed_event_requires_error() -> None:
    with pytest.raises(ValidationError):
        _canonical_event(
            EventType.MODEL_FAILED,
            {},
            status=SpanStatus.FAILED,
        )

    event = _canonical_event(
        EventType.MODEL_FAILED,
        {},
        status=SpanStatus.FAILED,
        error=TraceError(type="ModelTimeoutError", message="timeout"),
    )
    assert event.error is not None


def test_redis_key_and_envelope() -> None:
    event = _canonical_event(
        EventType.MODEL_COMPLETED,
        {"output_tokens": 3},
        status=SpanStatus.COMPLETED,
    )

    assert trace_stream_key("trace-1") == "trace:trace-1:events"

    fields = event.to_fields()

    assert fields["trace_id"] == "trace-1"
    assert fields["run_id"] == "run-1"
    assert fields["span_id"] == "span-1"
    assert fields["event"] == "model.completed"
    assert fields["schema_version"] == "2.0"

    assert json.loads(fields["data"]) == {"output_tokens": 3}


@pytest.mark.asyncio
async def test_unknown_custom_payload_is_ignored() -> None:
    projector = RunEventProjector(
        scope=(),
        run_id="run-1",
        trace_id="trace-1",
    )

    # 不抛、也不发布：_on_custom 是"非法就丢这一条"。
    await projector._on_custom({"hello": "world"}, 0)
    await projector._on_custom(None, 0)


@pytest.mark.asyncio
async def test_fanout_bookkeeping_and_tool_span_alignment(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.agent.event import run_event_projector as pm
    from app.common.event import TraceEmitter

    published: list[Any] = []

    async def fake_emit(event: Any) -> dict[str, bool]:
        if event.trace_id:
            published.append(SimpleNamespace(
                event_type=EventType(event.type.value),
                trace_id=event.trace_id,
                run_id=event.run_id,
                thread_id=event.session_id,
                span_id=event.span_id,
                parent_span_id=event.parent_span_id,
                span_kind=event.span_kind,
                span_name=event.span_name,
                status=event.status,
                duration_ms=event.duration_ms,
                payload=dict(event.data),
                error=event.error,
                event_id=event.event_id,
                fanout_id=event.fanout_id,
                branch_id=event.branch_id,
                branch_index=event.branch_index,
                attempt=event.attempt,
                links=list(event.links),
            ))

        return {"chat": True, "trace": bool(event.trace_id)}

    monkeypatch.setattr(pm, "emit_event", fake_emit)

    projector = RunEventProjector(
        scope=(),
        run_id="run-1",
        trace_id="trace-1",
        session_id="session-1",
    )

    emitter = TraceEmitter(
        trace_id="trace-1",
        run_id="run-1",
        thread_id="session-1",
        writer=lambda _: None,
    )

    emitter.instant(
        EventType.FANOUT_STARTED,
        span_kind=SpanKind.FANOUT,
        span_name="tool_fanout",
        span_id="fanout-1",
        parent_span_id="agent-task",
        payload={
            "fanout_type": "tool_calls",
            "dispatch_mode": "parallel",
            "expected_branch_count": 2,
        },
    )

    for index, scope in enumerate(("scope-1", "scope-2")):
        emitter.instant(
            EventType.FANOUT_BRANCH_CREATED,
            span_kind=SpanKind.FANOUT,
            span_name="tool_fanout",
            span_id="fanout-1",
            parent_span_id="agent-task",
            payload={
                "branch_id": scope,
                "branch_span_id": f"tool-span-{index + 1}",
                "branch_index": index,
                "target_kind": "tool",
                "target_name": "search",
            },
        )

    # event writer 是空的：直接拿 emitter 构造的事件喂给投影层。
    # （这里复用不到 emitter 的内部队列，所以重新构造等价事件。）
    #
    # ⚠️ 顺序必须与真实运行一致：先 fanout.started 建好记账状态，
    #    再 branch_created 才会走到 `branches.add(...)` 那条路。
    started_event = _canonical_event(
        EventType.FANOUT_STARTED,
        {
            "fanout_type": "tool_calls",
            "dispatch_mode": "parallel",
            "expected_branch_count": 2,
        },
        event_id="fanout-started",
        span_id="fanout-1",
        span_kind=SpanKind.FANOUT,
        span_name="tool_fanout",
    )
    await projector._on_custom(started_event.model_dump(mode="json"), 0)

    for scope in ("scope-1", "scope-2"):
        index = 0 if scope == "scope-1" else 1
        event = _canonical_event(
            EventType.FANOUT_BRANCH_CREATED,
            {
                "branch_id": scope,
                "branch_span_id": f"tool-span-{index + 1}",
                "branch_index": index,
                "target_kind": "tool",
                "target_name": "search",
            },
            event_id=f"branch-{index}",
            span_id="fanout-1",
            span_kind=SpanKind.FANOUT,
            span_name="tool_fanout",
        )
        await projector._on_custom(event.model_dump(mode="json"), 0)

    # 两个 tools 分支：start -> result。
    for index, scope in enumerate(("scope-1", "scope-2"), start=1):
        task_id = f"task-{index}"
        call_id = f"call-{index}"

        await projector._on_tasks(
            {
                "id": task_id,
                "name": "tools",
                "input": {
                    "call": {"id": call_id, "name": "search"},
                    "call_scope": scope,
                },
            },
            index,
        )

        await projector._on_tool(
            {
                "event": "tool-started",
                "tool_call_id": call_id,
                "tool_name": "search",
                "input": {},
            },
            index,
        )

        await projector._on_tool(
            {
                "event": "tool-finished",
                "tool_call_id": call_id,
                "output": {"ok": True},
            },
            index + 1,
        )

        await projector._on_tasks(
            {"id": task_id, "name": "tools", "result": {}},
            index + 1,
        )

    types = [event.event_type for event in published]

    assert EventType.FANOUT_COMPLETED in types

    tool_started = next(
        event for event in published
        if event.event_type is EventType.TOOL_STARTED
    )
    # route 预先算的 branch_span_id 与执行链的确定性 id 一致；
    # 父是 tools 分支节点（task_id）。
    assert tool_started.span_id in {"tool-span-1", "tool-span-2"}
    assert tool_started.parent_span_id in {"task-1", "task-2"}
