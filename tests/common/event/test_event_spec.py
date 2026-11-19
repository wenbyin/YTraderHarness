"""运行事件 v2 契约：联合词表、信封、payload 校验、持久化形状。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.common.event.spec import (
    ALL_EVENT_TYPES,
    CHAT_EVENT_TYPES,
    EVENT_PAYLOAD_MODELS,
    EVENT_SCHEMA_VERSION,
    TRACE_EVENT_TYPES,
    TRACE_ONLY_EVENT_TYPES,
    Event,
    EventType,
    SpanKind,
    SpanStatus,
)


#: `to_fields()` 的固定字段集合。两个 sink 写出的记录必须与它完全一致。
RECORD_FIELDS = {
    "schema_version",
    "event",
    "event_id",
    "occurred_at",
    "run_id",
    "trace_id",
    "session_id",
    "node",
    "span_id",
    "parent_span_id",
    "span_kind",
    "span_name",
    "attempt",
    "links",
    "fanout_id",
    "branch_id",
    "branch_index",
    "status",
    "duration_ms",
    "error",
    "data",
}


def _event(
        event_type: EventType,
        data: dict | None = None,
        **overrides,
) -> Event:
    base = {
        "type": event_type,
        "run_id": "run-1",
        "trace_id": "trace-1",
        "data": data or {},
    }
    base.update(overrides)
    return Event(**base)


# ============================================================
# 词表与映射
# ============================================================


def test_every_event_type_has_a_payload_model() -> None:
    assert ALL_EVENT_TYPES == frozenset(EVENT_PAYLOAD_MODELS)


def test_every_event_type_belongs_to_at_least_one_sink() -> None:
    assert ALL_EVENT_TYPES == (CHAT_EVENT_TYPES | TRACE_EVENT_TYPES)


def test_trace_only_is_the_set_difference() -> None:
    assert TRACE_ONLY_EVENT_TYPES == (TRACE_EVENT_TYPES - CHAT_EVENT_TYPES)


def test_six_shared_events_are_in_both_sinks() -> None:
    shared = {
        EventType.RUN_STARTED,
        EventType.RUN_COMPLETED,
        EventType.RUN_FAILED,
        EventType.NODE_STARTED,
        EventType.NODE_COMPLETED,
        EventType.TOOL_COMPLETED,
    }

    for event_type in shared:
        assert event_type in CHAT_EVENT_TYPES
        assert event_type in TRACE_EVENT_TYPES


def test_run_cancelled_is_in_both_sinks() -> None:
    # v2 起 cancelled 不再折叠成 run.failed，chat 也要能收到。
    assert EventType.RUN_CANCELLED in CHAT_EVENT_TYPES
    assert EventType.RUN_CANCELLED in TRACE_EVENT_TYPES


def test_message_delta_is_chat_only() -> None:
    assert EventType.MESSAGE_DELTA in CHAT_EVENT_TYPES
    assert EventType.MESSAGE_DELTA not in TRACE_EVENT_TYPES


def test_model_started_is_trace_only() -> None:
    assert EventType.MODEL_STARTED in TRACE_EVENT_TYPES
    assert EventType.MODEL_STARTED not in CHAT_EVENT_TYPES


# ============================================================
# 六个同名事件
# ============================================================


def test_run_started_merges_both_payloads() -> None:
    event = _event(
        EventType.RUN_STARTED,
        {"agent_type": "assistant", "graph_name": "agent_react"},
        status=SpanStatus.RUNNING,
        span_id="run-1",
        span_kind=SpanKind.RUN,
        span_name="run",
    )

    assert event.data == {
        "agent_type": "assistant",
        "graph_name": "agent_react",
    }
    assert event.to_fields()["status"] == "running"


def test_run_failed_requires_error() -> None:
    with pytest.raises(ValidationError):
        _event(EventType.RUN_FAILED, {}, status=SpanStatus.FAILED)

    event = _event(
        EventType.RUN_FAILED,
        {},
        status=SpanStatus.FAILED,
        error={"type": "RuntimeError", "message": "boom"},
    )

    assert json.loads(event.to_fields()["error"]) == {
        "type": "RuntimeError",
        "message": "boom",
    }


def test_node_started_uses_node_id_and_accepts_domain_facts() -> None:
    event = _event(
        EventType.NODE_STARTED,
        {
            "node_id": "agent",
            "node_type": "agent",
            "node_execution_index": 0,
            "agent_type": "assistant",
            "model_role": "reasoning",
        },
    )

    assert event.data["node_id"] == "agent"
    assert event.data["model_role"] == "reasoning"


def test_node_started_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _event(
            EventType.NODE_STARTED,
            {"node_id": "agent", "not_allowed": 1},
        )


def test_tool_completed_merges_output_and_result_size() -> None:
    event = _event(
        EventType.TOOL_COMPLETED,
        {
            "tool_call_id": "call-1",
            "tool_name": "search",
            "output": {"items": []},
            "result_size_bytes": 11,
        },
        status=SpanStatus.COMPLETED,
        duration_ms=12,
    )

    assert event.data["output"] == {"items": []}
    assert event.data["result_size_bytes"] == 11
    assert event.to_fields()["duration_ms"] == "12"


# ============================================================
# chat 独有 / trace 独有
# ============================================================


def test_hitl_requested_keeps_opaque_extras() -> None:
    event = _event(
        EventType.HITL_REQUESTED,
        {
            "interrupt_id": "i-1",
            "options": ["approve", "reject"],
            "processor_private_field": "kept",
        },
        trace_id=None,
    )

    assert event.data["options"] == ["approve", "reject"]
    assert event.data["processor_private_field"] == "kept"


def test_trace_only_event_requires_trace_id() -> None:
    with pytest.raises(ValidationError):
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


def test_shared_event_may_have_no_trace_id() -> None:
    event = _event(
        EventType.NODE_STARTED,
        {"node_id": "agent"},
        trace_id=None,
    )

    assert event.trace_id is None


# ============================================================
# 持久化形状
# ============================================================


def test_to_fields_has_stable_schema() -> None:
    event = _event(
        EventType.MESSAGE_DELTA,
        {"delta": "你"},
        trace_id=None,
    )

    fields = event.to_fields()

    assert set(fields) == RECORD_FIELDS
    assert fields["schema_version"] == EVENT_SCHEMA_VERSION
    assert fields["event"] == "message.delta"
    assert fields["trace_id"] == ""
    assert fields["status"] == ""
    assert fields["duration_ms"] == ""
    assert fields["error"] == ""
    assert json.loads(fields["data"]) == {"delta": "你"}


def test_record_has_no_routing_fields() -> None:
    event = _event(EventType.MESSAGE_DELTA, {"delta": "x"})

    fields = event.to_fields()

    assert "topic" not in fields
    assert "topic_id" not in fields


def test_event_id_is_unique_by_default() -> None:
    first = _event(EventType.MESSAGE_DELTA, {"delta": "a"})
    second = _event(EventType.MESSAGE_DELTA, {"delta": "b"})

    assert first.event_id != second.event_id


def test_occurred_at_requires_timezone() -> None:
    from datetime import datetime

    with pytest.raises(ValidationError):
        _event(
            EventType.MESSAGE_DELTA,
            {"delta": "x"},
            occurred_at=datetime(2026, 11, 17, 8, 0, 0),
        )
