"""任务板事件的 custom stream 信封。"""

from __future__ import annotations

import pytest

from app.agent.event import task_events
from app.agent.event.run_event_projector import RunEventProjector
from app.common.event import RunEventType, TASK_STREAM_KIND

__all__: list[str] = []


def test_plan_published_envelope(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        task_events, "write_custom", captured.append
    )

    task = {
        "id": "plan-1:t1",
        "status": {"state": "TASK_STATE_SUBMITTED"},
    }

    # 计划发布时还没有 claim：current 必须为空，由首次 task.updated /
    # subagent.claimed 再指过去。
    task_events.publish_plan_published(
        plan_id="plan-1",
        revision=1,
        goal="测试",
        tasks=[task],
        current_task_id=None,
        publish_id="plan-1:1",
    )

    assert captured[0]["kind"] == TASK_STREAM_KIND
    assert captured[0]["event"] == RunEventType.PLAN_PUBLISHED.value
    assert captured[0]["tasks"] == [task]
    assert captured[0]["current_task_id"] is None
    assert captured[0]["publish_id"] == "plan-1:1"


def test_task_updated_and_cleared_envelopes(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        task_events, "write_custom", captured.append
    )

    task = {
        "id": "plan-1:t1",
        "status": {"state": "TASK_STATE_COMPLETED"},
    }

    task_events.publish_task_updated(
        plan_id="plan-1",
        revision=1,
        task=task,
        current_task_id=None,
    )
    task_events.publish_plan_cleared(
        plan_id="plan-1",
        revision=1,
        reason="completed",
    )

    assert [item["event"] for item in captured] == [
        RunEventType.TASK_UPDATED.value,
        RunEventType.PLAN_CLEARED.value,
    ]
    assert captured[0]["task"] == task
    assert captured[1]["reason"] == "completed"


@pytest.mark.asyncio
async def test_projector_maps_task_stream_events() -> None:
    class _FakeProjector:
        def __init__(self) -> None:
            self.events: list[tuple] = []

        async def _emit(self, event_type, *, node, data) -> None:
            self.events.append((event_type, node, data))

    fake = _FakeProjector()

    await RunEventProjector._on_task_stream_event(
        fake,
        {
            "kind": TASK_STREAM_KIND,
            "event": RunEventType.TASK_UPDATED.value,
            "node": "task_listener",
            "plan_id": "p1",
            "task": {"id": "t1"},
            "current_task_id": "t1",
        },
    )

    event_type, node, data = fake.events[0]

    assert event_type is RunEventType.TASK_UPDATED
    assert node == "task_listener"
    assert data["task"] == {"id": "t1"}
    assert "kind" not in data
    assert "event" not in data
