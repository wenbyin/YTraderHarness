"""SSE 帧：v2 的 data / error / links 都从 JSON 字符串解成对象。"""

from __future__ import annotations

import json

from app.agent.event.reader import StreamEntry
from app.agent.event.sse import sse_frame


def _payload(frame: str) -> dict:
    return json.loads(frame.split("data: ", 1)[1].strip())


def test_v2_nested_fields_are_unfolded() -> None:
    entry = StreamEntry(
        id="1-0",
        fields={
            "schema_version": "2.0",
            "event": "node.completed",
            "event_id": "e-1",
            "occurred_at": "2026-11-17T08:00:00+00:00",
            "run_id": "run-1",
            "trace_id": "",
            "session_id": "",
            "node": "agent",
            "span_id": "task-1",
            "parent_span_id": "run-1",
            "span_kind": "node",
            "span_name": "agent",
            "attempt": "1",
            "links": '[{"relation": "retry_of", "span_id": "s-1"}]',
            "fanout_id": "",
            "branch_id": "",
            "branch_index": "",
            "status": "failed",
            "duration_ms": "12",
            "error": '{"type": "NodeError", "message": "boom"}',
            "data": '{"node_id": "agent"}',
        },
    )

    payload = _payload(sse_frame(entry))

    assert payload["event"] == "node.completed"
    assert payload["run_id"] == "run-1"
    assert payload["data"] == {"node_id": "agent"}
    assert payload["error"] == {
        "type": "NodeError",
        "message": "boom",
    }
    assert payload["links"] == [
        {"relation": "retry_of", "span_id": "s-1"}
    ]


def test_broken_json_stays_a_string() -> None:
    entry = StreamEntry(
        id="1-0",
        fields={
            "event": "message.delta",
            "data": "not-json",
            "error": "{broken",
            "links": "oops",
        },
    )

    payload = _payload(sse_frame(entry))

    assert payload["data"] == "not-json"
    assert payload["error"] == "{broken"
    assert payload["links"] == "oops"
