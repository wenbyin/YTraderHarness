"""子代理事件的 custom stream 信封与投影。"""

from __future__ import annotations

import pytest

from app.agent.event import subagent_events
from app.agent.event.run_event_projector import RunEventProjector
from app.common.event import RunEventType, SUBAGENT_STREAM_KIND

__all__: list[str] = []


def _task(tid: str, state: str = "TASK_STATE_WORKING") -> dict:
    return {"id": tid, "status": {"state": state}}


def test_created_envelope(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        subagent_events, "write_custom", captured.append
    )

    subagent_events.publish_subagent_created(
        plan_id="p1",
        revision=1,
        worker_id="run:worker:0",
        worker_index=0,
        task_id="p1:t1",
        task_count=5,
        pending_count=4,
        context_id="c1",
    )

    assert captured[0]["kind"] == SUBAGENT_STREAM_KIND
    assert captured[0]["event"] == RunEventType.SUBAGENT_CREATED.value
    assert captured[0]["worker_id"] == "run:worker:0"
    assert captured[0]["worker_index"] == 0
    assert captured[0]["task_id"] == "p1:t1"
    assert captured[0]["pending_count"] == 4


def test_claimed_and_progress_envelopes(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        subagent_events, "write_custom", captured.append
    )

    task = _task("p1:t1")

    subagent_events.publish_subagent_claimed(
        plan_id="p1",
        revision=1,
        worker_id="run:worker:0",
        task=task,
        current_task_id="p1:t1",
    )
    subagent_events.publish_subagent_task_updated(
        plan_id="p1",
        revision=1,
        worker_id="run:worker:0",
        task=_task("p1:t1", "TASK_STATE_COMPLETED"),
        previous_state="TASK_STATE_WORKING",
    )
    subagent_events.publish_subagent_completed(
        plan_id="p1",
        revision=1,
        worker_id="run:worker:0",
        task=_task("p1:t1", "TASK_STATE_COMPLETED"),
    )

    assert [item["event"] for item in captured] == [
        RunEventType.SUBAGENT_CLAIMED.value,
        RunEventType.SUBAGENT_TASK_UPDATED.value,
        RunEventType.SUBAGENT_COMPLETED.value,
    ]
    assert captured[0]["task"] == task
    assert captured[0]["worker_id"] == "run:worker:0"
    assert captured[1]["previous_state"] == "TASK_STATE_WORKING"
    assert captured[2]["status"] == "TASK_STATE_COMPLETED"


@pytest.mark.asyncio
async def test_projector_maps_subagent_stream_events() -> None:
    class _FakeProjector:
        def __init__(self) -> None:
            self.events: list[tuple] = []

        async def _emit(self, event_type, *, node, data) -> None:
            self.events.append((event_type, node, data))

    fake = _FakeProjector()

    await RunEventProjector._on_subagent_stream_event(
        fake,
        {
            "kind": SUBAGENT_STREAM_KIND,
            "event": RunEventType.SUBAGENT_CLAIMED.value,
            "node": "subagent_dispatch",
            "plan_id": "p1",
            "worker_id": "run:worker:0",
            "task": {"id": "t1"},
            "current_task_id": "t1",
        },
    )

    event_type, node, data = fake.events[0]

    assert event_type is RunEventType.SUBAGENT_CLAIMED
    assert node == "subagent_dispatch"
    assert data["worker_id"] == "run:worker:0"
    assert data["task"] == {"id": "t1"}
    assert "kind" not in data
    assert "event" not in data


def test_event_values_are_stable_contract() -> None:
    """事件名是字面契约；改一个字就是破坏订阅端。"""

    assert RunEventType.SUBAGENT_CREATED.value == "subagent.created"
    assert RunEventType.SUBAGENT_CLAIMED.value == "subagent.claimed"
    assert (
        RunEventType.SUBAGENT_TASK_UPDATED.value
        == "subagent.task.updated"
    )
    assert RunEventType.SUBAGENT_COMPLETED.value == "subagent.completed"
    assert SUBAGENT_STREAM_KIND == "subagent"
