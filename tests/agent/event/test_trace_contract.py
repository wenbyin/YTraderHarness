"""Trace 契约：枚举、payload 严格校验、Redis 键与信封。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.agent.event.run_event_projector import RunEventProjector
from app.common.event import (
    EventType,
    SpanKind,
    SpanStatus,
    TraceError,
    TraceEvent,
    stream_key,
)


def _event(event_type: EventType, payload: dict, **overrides):
    data = {
        "event_id": "e-1",
        "event_type": event_type,
        "trace_id": "trace-1",
        "run_id": "run-1",
        "thread_id": "session-1",
        "span_id": "span-1",
        "span_kind": SpanKind.MODEL,
        "span_name": "model.invoke",
        "payload": payload,
    }
    data.update(overrides)
    return TraceEvent(**data)


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
        _event(
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
        _event(
            EventType.MODEL_STARTED,
            {"model_role": "reasoning", "provider": "p"},
        )


def test_failed_event_requires_error() -> None:
    with pytest.raises(ValidationError):
        _event(EventType.MODEL_FAILED, {})

    event = _event(
        EventType.MODEL_FAILED,
        {},
        error=TraceError(type="ModelTimeoutError", message="timeout"),
    )
    assert event.error is not None


def test_redis_key_and_envelope() -> None:
    event = _event(
        EventType.MODEL_COMPLETED,
        {"output_tokens": 3},
        status=SpanStatus.COMPLETED,
    )

    assert event.topic == "trace"
    assert event.topic_id == "trace-1:events"
    assert stream_key(event.topic, event.topic_id) == "trace:trace-1:events"

    fields = event.to_fields()

    assert fields["trace_id"] == "trace-1"
    assert fields["run_id"] == "run-1"
    assert fields["span_id"] == "span-1"
    assert fields["event"] == "model.completed"

    decoded = json.loads(fields["data"])
    assert decoded["payload"]["output_tokens"] == 3
    assert decoded["schema_version"] == "1.0"


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
async def test_redis_publisher_writes_the_trace_key() -> None:
    from app.agent.event.publisher import RedisEventPublisher
    from app.agent.event.config import StreamConfig

    class _Sink:
        def __init__(self) -> None:
            self.writes: list[tuple[str, dict]] = []

        def ping(self) -> bool:
            return True

        def write(self, key, fields, *, maxlen, ttl_seconds) -> None:
            self.writes.append((key, dict(fields)))

    sink = _Sink()
    publisher = RedisEventPublisher(
        sink,
        config=StreamConfig(enabled=True, maxlen=100, ttl_seconds=60),
        queue_size=10,
    )
    publisher.start()

    try:
        event = _event(
            EventType.RUN_COMPLETED,
            {},
            status=SpanStatus.COMPLETED,
        )
        assert await publisher.publish(event) is True
        await publisher.drain(timeout=2)

    finally:
        await publisher.aclose(timeout=2)

    assert sink.writes
    key, fields = sink.writes[0]

    assert key == "trace:trace-1:events"
    assert fields["trace_id"] == "trace-1"
    assert fields["event_id"] == "e-1"


@pytest.mark.asyncio
async def test_fanout_bookkeeping_and_tool_span_alignment(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.agent.event import run_event_projector as pm
    from app.common.event import TraceEmitter

    published: list[TraceEvent] = []

    async def fake_publish(self: Any, event: TraceEvent) -> bool:
        published.append(event)
        return True

    async def fake_run_event(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(
        pm.RedisTracePublisher, "publish", fake_publish
    )
    monkeypatch.setattr(pm, "publish_run_event", fake_run_event)

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
    for scope in ("scope-1", "scope-2"):
        index = 0 if scope == "scope-1" else 1
        event = _event(
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

    started_event = _event(
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
