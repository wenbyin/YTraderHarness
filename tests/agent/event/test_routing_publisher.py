"""RouterEventPublisher：规范事件按 sink 放行集合双投 / 单投。"""

from __future__ import annotations

import pytest

from app.agent.event.config import SinkConfig
from app.agent.event.routing import RouterEventPublisher
from app.common.event.spec import Event, EventType


class RecordingSink:
    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, str]]] = []
        self.pings = 0

    def ping(self) -> bool:
        self.pings += 1
        return True

    def write(self, key, fields, *, maxlen, ttl_seconds) -> None:
        self.writes.append((key, dict(fields)))


def _configs(
        *,
        chat_enabled: bool = True,
        trace_enabled: bool = True,
) -> dict[str, SinkConfig]:

    return {
        "chat": SinkConfig(
            enabled=chat_enabled,
            key_prefix="agent_run",
            maxlen=100,
            ttl_seconds=60,
            write_queue_size=10,
            write_workers=1,
        ),
        "trace": SinkConfig(
            enabled=trace_enabled,
            key_prefix="trace",
            maxlen=100,
            ttl_seconds=60,
            write_queue_size=10,
            write_workers=1,
        ),
    }


async def _started_publisher(
        sink: RecordingSink,
) -> RouterEventPublisher:

    publisher = RouterEventPublisher(
        sink,
        configs=_configs(),
    )
    publisher.start()
    return publisher


@pytest.mark.asyncio
async def test_canonical_shared_event_goes_to_both_with_same_shape() -> None:
    sink = RecordingSink()
    publisher = await _started_publisher(sink)

    try:
        event = Event(
            type=EventType.NODE_STARTED,
            run_id="run-1",
            trace_id="trace-1",
            data={"node_id": "agent"},
        )

        results = await publisher.emit(event)
        await publisher.drain(timeout=2)

    finally:
        await publisher.aclose(timeout=2)

    assert results == {"chat": True, "trace": True}
    assert {key for key, _ in sink.writes} == {
        "agent_run:run-1",
        "trace:trace-1:events",
    }

    shapes = [set(fields) for _, fields in sink.writes]
    assert shapes[0] == shapes[1]


@pytest.mark.asyncio
async def test_chat_only_event_skips_trace() -> None:
    sink = RecordingSink()
    publisher = await _started_publisher(sink)

    try:
        event = Event(
            type=EventType.MESSAGE_DELTA,
            run_id="run-1",
            data={"delta": "你"},
        )

        assert await publisher.emit(event) == {"chat": True}
        await publisher.drain(timeout=2)

    finally:
        await publisher.aclose(timeout=2)

    assert [key for key, _ in sink.writes] == ["agent_run:run-1"]


@pytest.mark.asyncio
async def test_trace_only_event_skips_chat() -> None:
    sink = RecordingSink()
    publisher = await _started_publisher(sink)

    try:
        event = Event(
            type=EventType.MODEL_STARTED,
            run_id="run-1",
            trace_id="trace-1",
            data={
                "model_role": "reasoning",
                "provider": "fake",
                "model_name": "fake-model",
                "streaming": True,
            },
        )

        assert await publisher.emit(event) == {"trace": True}
        await publisher.drain(timeout=2)

    finally:
        await publisher.aclose(timeout=2)

    assert [key for key, _ in sink.writes] == ["trace:trace-1:events"]
