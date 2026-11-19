"""tool.completed 的单一来源：custom stream 投影 + 回调去重。"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.event import run_event_projector as pm
from app.agent.event.run_event_projector import RunEventProjector
from app.common.event.spec import EventType, SpanStatus


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    events: list[Any] = []

    async def _capture(event: Any) -> dict[str, bool]:
        events.append(event)
        return {
            "chat": True,
            "trace": bool(event.trace_id),
        }

    monkeypatch.setattr(pm, "emit_event", _capture)

    return events


def _projector() -> RunEventProjector:
    return RunEventProjector(
        (),
        run_id="run-1",
        trace_id="trace-1",
        session_id="session-1",
    )


async def test_tool_completed_from_custom_stream_is_one_canonical_event(
        captured: list[Any],
) -> None:
    projector = _projector()

    await projector._on_tool_stream_event(
        {
            "event": "tool.completed",
            "tool_call_id": "call-1",
            "tool_name": "search",
            "output": {"items": []},
        },
        timestamp=2000,
    )

    assert len(captured) == 1
    event = captured[0]

    assert event.type is EventType.TOOL_COMPLETED
    assert event.status is SpanStatus.COMPLETED
    assert event.trace_id == "trace-1"
    assert event.span_kind is not None
    assert event.span_id
    assert event.data["tool_call_id"] == "call-1"
    assert event.data["tool_name"] == "search"
    assert event.data["output"] == {"items": []}
    assert event.data["result_size_bytes"] is not None


async def test_tool_failure_goes_to_envelope_error(
        captured: list[Any],
) -> None:
    projector = _projector()

    await projector._on_tool_stream_event(
        {
            "event": "tool.completed",
            "tool_call_id": "call-2",
            "tool_name": "search",
            "error": "boom",
        },
        timestamp=2000,
    )

    event = captured[0]

    assert event.status is SpanStatus.FAILED
    assert event.error is not None
    assert event.error.message == "boom"
    assert "error" not in event.data


async def test_custom_tool_completion_suppresses_callback_trace_finish(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[Any] = []

    async def _fake_emit(event: Any) -> dict[str, bool]:
        published.append(event)
        return {"chat": True, "trace": bool(event.trace_id)}

    monkeypatch.setattr(pm, "emit_event", _fake_emit)

    projector = _projector()
    projector._trace_tool_tasks["call-1"] = {
        "task_id": "task-1",
        "call_scope": "call-1",
        "branch_span_id": None,
        "fanout_id": None,
        "branch_index": None,
        "branch_id": None,
    }

    await projector._on_tool(
        {
            "event": "tool-finished",
            "tool_call_id": "call-1",
            "tool_name": "search",
        },
        timestamp=2000,
    )

    # custom stream 那条会负责收尾；回调这边一条都不能再发。
    assert published == []


async def test_non_custom_tool_still_gets_callback_trace_finish(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[Any] = []

    async def _fake_emit(event: Any) -> dict[str, bool]:
        published.append(event)
        return {"chat": True, "trace": bool(event.trace_id)}

    monkeypatch.setattr(pm, "emit_event", _fake_emit)

    projector = _projector()

    await projector._on_tool(
        {
            "event": "tool-finished",
            "tool_call_id": "call-9",
            "tool_name": "search",
        },
        timestamp=2000,
    )

    assert len(published) == 1
