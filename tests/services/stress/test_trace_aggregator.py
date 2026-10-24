"""TraceAggregate 的纯函数测试（不碰 Redis）。"""

from __future__ import annotations

from app.common.event import EventType
from app.services.stress.application.trace_aggregator import TraceAggregate

TRACE_ID = "stress:run-1:0"


def _event(
        event_type: EventType,
        *,
        event_id: str,
        span_name: str = "span",
        duration_ms: int | None = None,
        payload: dict | None = None,
        run_id: str = "run-1",
        trace_id: str = TRACE_ID,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type.value,
        "trace_id": trace_id,
        "run_id": run_id,
        "span_name": span_name,
        "duration_ms": duration_ms,
        "payload": payload or {},
    }


def test_trace_aggregate_counts_and_distributions() -> None:
    aggregate = TraceAggregate()

    events = [
        _event(EventType.RUN_STARTED, event_id="e1", payload={"agent_type": "react"}),
        _event(EventType.CONTEXT_BUILD_STARTED, event_id="e2"),
        _event(EventType.NODE_STARTED, event_id="e3", payload={"node_name": "agent", "node_execution_index": 0}),
        _event(
            EventType.NODE_COMPLETED,
            event_id="e4",
            span_name="agent",
            duration_ms=100,
            payload={"node_name": "agent"},
        ),
        _event(
            EventType.MODEL_STARTED,
            event_id="e5",
            payload={"model_role": "reasoning", "provider": "mock", "model_name": "mock", "streaming": True},
        ),
        _event(
            EventType.MODEL_COMPLETED,
            event_id="e6",
            duration_ms=5000,
            payload={"input_tokens": 100, "output_tokens": 50, "finish_reason": "stop"},
        ),
        _event(EventType.MODEL_FAILED, event_id="e7"),
        _event(EventType.MODEL_RETRY_SCHEDULED, event_id="e8", payload={"failed_span_id": "x", "next_attempt": 2, "delay_ms": 10}),
        _event(
            EventType.TOOL_STARTED,
            event_id="e9",
            payload={"tool_name": "mock.echo", "tool_call_id": "c1"},
        ),
        _event(
            EventType.TOOL_COMPLETED,
            event_id="e10",
            duration_ms=200,
            payload={"tool_name": "mock.echo", "tool_call_id": "c1"},
        ),
        _event(EventType.TOOL_FAILED, event_id="e11", payload={"tool_name": "mock.echo", "tool_call_id": "c2"}),
        _event(EventType.FANOUT_STARTED, event_id="e12", payload={"fanout_type": "tools", "dispatch_mode": "parallel"}),
        _event(EventType.FANOUT_BRANCH_CREATED, event_id="e13", payload={"branch_id": "b1", "branch_span_id": "s1", "branch_index": 0, "target_kind": "tool", "target_name": "mock.echo"}),
        _event(EventType.FANOUT_COMPLETED, event_id="e14", payload={"created_branch_count": 1, "completed_branch_count": 1, "failed_branch_count": 1, "cancelled_branch_count": 0}),
    ]

    aggregate.merge_events(events, expected_trace_id=TRACE_ID)
    snapshot = aggregate.snapshot()

    counts = snapshot["trace_counts"]
    assert counts["run_span_count"] == 1
    assert counts["node_span_count"] == 1
    assert counts["model_call_count"] == 1
    assert counts["tool_call_count"] == 1
    assert counts["context_build_count"] == 1
    assert counts["fanout_count"] == 1
    assert counts["model_retry_count"] == 1
    assert counts["model_error_count"] == 1
    assert counts["tool_error_count"] == 1

    model = snapshot["model_metrics"]
    assert model["call_count"] == 1
    assert model["retry_count"] == 1
    assert model["error_count"] == 1
    assert model["duration_ms"]["count"] == 1
    assert model["input_tokens_per_turn"]["count"] == 1
    assert model["output_tokens_per_turn"]["count"] == 1

    assert snapshot["model_input_tokens_sum"] == 100
    assert snapshot["model_output_tokens_sum"] == 50

    assert snapshot["node_metrics"]["agent"]["count"] == 1
    assert snapshot["tool_metrics"]["mock.echo"]["count"] == 1

    fanout = snapshot["fanout_metrics"]
    assert fanout["count"] == 1
    assert fanout["branch_count"] == 1
    assert fanout["failed_branch_count"] == 1

    assert snapshot["events_seen"] == len(events)
    assert snapshot["events_deduplicated"] == 0
    assert snapshot["events_unmatched"] == 0


def test_trace_aggregate_deduplicates_event_ids() -> None:
    aggregate = TraceAggregate()
    event = _event(EventType.RUN_STARTED, event_id="same")

    aggregate.merge_events([event, event])

    assert aggregate.events_seen == 2
    assert aggregate.events_deduplicated == 1
    assert aggregate.snapshot()["trace_counts"]["run_span_count"] == 1


def test_trace_aggregate_marks_unmatched_trace() -> None:
    aggregate = TraceAggregate()
    event = _event(
        EventType.RUN_STARTED,
        event_id="e1",
        trace_id="other-trace",
    )

    aggregate.merge_events([event], expected_trace_id=TRACE_ID)

    assert aggregate.events_unmatched == 1
    assert aggregate.counts["run_span_count"] == 0
