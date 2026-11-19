"""EventRouter：同名事件双投、独有事件单投、trace_id 缺失与异常兜底。"""

from __future__ import annotations

import pytest

from app.agent.event.router import (
    EventRouter,
    SinkPolicy,
    chat_policy,
    trace_policy,
)
from app.common.event.spec import Event, EventType


class Recorder:
    """记录 enqueue 调用；返回值可配。"""

    def __init__(self, result: bool = True) -> None:
        self.calls: list[tuple[Event, str]] = []
        self.result = result

    def __call__(self, event: Event, key: str) -> bool:
        self.calls.append((event, key))
        return self.result


class Boom:
    def __call__(self, event: Event, key: str) -> bool:
        raise RuntimeError("boom")


def _event(
        event_type: EventType,
        data: dict | None = None,
        *,
        trace_id: str | None = "trace-1",
) -> Event:
    return Event(
        type=event_type,
        run_id="run-1",
        trace_id=trace_id,
        data=data or {},
    )


def _router() -> tuple[EventRouter, Recorder, Recorder]:
    chat = Recorder()
    trace = Recorder()

    router = EventRouter([
        (chat_policy(), chat),
        (trace_policy(), trace),
    ])

    return router, chat, trace


# ============================================================
# 放行
# ============================================================


def test_shared_event_goes_to_both_sinks() -> None:
    router, chat, trace = _router()

    event = _event(EventType.NODE_STARTED, {"node_id": "agent"})

    results = router.emit(event)

    assert results == {"chat": True, "trace": True}
    assert [e for e, _ in chat.calls] == [event]
    assert [e for e, _ in trace.calls] == [event]
    assert chat.calls[0][1] == "agent_run:run-1"
    assert trace.calls[0][1] == "trace:trace-1:events"


def test_chat_only_event_skips_trace() -> None:
    router, chat, trace = _router()

    event = _event(EventType.MESSAGE_DELTA, {"delta": "你"})

    results = router.emit(event)

    assert results == {"chat": True}
    assert len(chat.calls) == 1
    assert trace.calls == []


def test_trace_only_event_skips_chat() -> None:
    router, chat, trace = _router()

    event = _event(
        EventType.MODEL_STARTED,
        {
            "model_role": "reasoning",
            "provider": "fake",
            "model_name": "fake-model",
            "streaming": True,
        },
    )

    results = router.emit(event)

    assert results == {"trace": True}
    assert chat.calls == []
    assert len(trace.calls) == 1


def test_shared_event_without_trace_id_goes_to_chat_only() -> None:
    router, chat, trace = _router()

    event = _event(
        EventType.NODE_STARTED,
        {"node_id": "agent"},
        trace_id=None,
    )

    results = router.emit(event)

    assert results == {"chat": True}
    assert len(chat.calls) == 1
    assert trace.calls == []


def test_trace_only_event_without_trace_id_is_rejected_before_routing() -> None:
    # 契约层先拦：trace-only 事件不能没有 trace_id。
    with pytest.raises(Exception):
        _event(
            EventType.MODEL_STARTED,
            {
                "model_role": "reasoning",
                "provider": "fake",
                "model_name": "fake-model",
                "streaming": True,
            },
            trace_id=None,
        )


# ============================================================
# 键与策略
# ============================================================


def test_chat_key_prefix_is_configurable() -> None:
    recorder = Recorder()

    router = EventRouter([
        (chat_policy(key_prefix="custom_run"), recorder),
    ])

    router.emit(_event(EventType.MESSAGE_DELTA, {"delta": "x"}))

    assert recorder.calls[0][1] == "custom_run:run-1"


def test_policies_are_exposed_in_order() -> None:
    router, _, _ = _router()

    assert [policy.name for policy in router.policies] == ["chat", "trace"]


def test_duplicate_sink_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        EventRouter([
            (SinkPolicy("chat", frozenset(), lambda event: None), Recorder()),
            (SinkPolicy("chat", frozenset(), lambda event: None), Recorder()),
        ])


# ============================================================
# 异常兜底
# ============================================================


def test_failing_sink_does_not_break_other_sink() -> None:
    chat = Recorder()
    trace = Recorder()

    router = EventRouter([
        (chat_policy(), chat),
        (trace_policy(), Boom()),
    ])

    results = router.emit(
        _event(EventType.NODE_STARTED, {"node_id": "agent"})
    )

    assert results == {"chat": True, "trace": False}
    assert len(chat.calls) == 1
