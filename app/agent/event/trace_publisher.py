"""
Trace 事件落地：`trace:{trace_id}:events`。

职责边界
--------
    图内事件     TraceEmitter（custom stream） -> 投影层 -> 本模块
    Run 外层事件 dispatch / service        -> 本模块

本模块**只做一件事**：把一条已校验的 `TraceEvent` 交给既有的 Redis
发布器。队列、写线程、连接池、`enabled` 开关全部复用
`app/agent/event/publisher.py`，不另起一套连接生命周期。

键的拼法
--------
既有的 `RedisEventPublisher.publish` 用 `stream_key(event.topic,
event.topic_id)` 拼键。`TraceEvent.topic` 固定是 `"trace"`，
`topic_id` 是 `<trace_id>:events`，于是键正好是：

    trace:{trace_id}:events

与提示词第 6 节逐字一致。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.common.event.trace import (
    EventType,
    SpanKind,
    SpanLink,
    SpanStatus,
    TraceError,
    TraceEvent,
)

from .publisher import EventPublisher, get_event_publisher

__all__ = [
    "RedisTracePublisher",
    "publish_trace_event",
]


logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RedisTracePublisher:
    """
    把 TraceEvent 写进 `trace:{trace_id}:events`。

    `publisher` 默认惰性取进程级单例 —— 组合根启动顺序无关：即使本对象
    先被构造、发布器后装配，`publish()` 那一刻拿到的仍是最终的那个。
    """

    __slots__ = ("_publisher",)

    def __init__(self, publisher: EventPublisher | None = None) -> None:
        self._publisher = publisher

    @property
    def publisher(self) -> EventPublisher:
        return (
            self._publisher
            if self._publisher is not None
            else get_event_publisher()
        )

    @property
    def enabled(self) -> bool:
        return self.publisher.enabled

    async def publish(self, event: TraceEvent) -> bool:
        """投递一条已构造好的事件。**从不抛异常。**"""

        try:
            return await self.publisher.publish(event)

        except Exception as exc:  # noqa: BLE001 —— 旁路绝不外抛
            logger.warning(
                "Trace 事件投递失败（%s / %s）：%s: %s",
                event.event_type.value,
                event.span_name,
                type(exc).__name__,
                exc,
            )
            return False

    async def emit(
            self,
            event_type: EventType,
            *,
            trace_id: str | None,
            run_id: str,
            thread_id: str | None = None,
            span_id: str | None = None,
            parent_span_id: str | None = None,
            span_kind: SpanKind = SpanKind.RUN,
            span_name: str = "run",
            status: SpanStatus | None = None,
            attempt: int = 1,
            duration_ms: int | None = None,
            links: Sequence[SpanLink] = (),
            fanout_id: str | None = None,
            branch_id: str | None = None,
            branch_index: int | None = None,
            payload: Mapping[str, Any] | None = None,
            error: TraceError | None = None,
            event_id: str | None = None,
            occurred_at: datetime | None = None,
    ) -> bool:
        """
        构造 + 校验 + 投递一条事件。

        `trace_id` 为空 = trace 未开启：直接返回 False，不构造事件，
        也不碰发布器。
        """

        if not trace_id:
            return False

        try:
            event = TraceEvent(
                event_id=event_id or uuid.uuid4().hex,
                event_type=event_type,
                occurred_at=occurred_at or _now(),
                trace_id=str(trace_id),
                run_id=str(run_id),
                thread_id=None if thread_id is None else str(thread_id),
                span_id=span_id or str(run_id),
                parent_span_id=parent_span_id,
                span_kind=span_kind,
                span_name=span_name,
                status=status,
                attempt=attempt,
                duration_ms=duration_ms,
                links=list(links),
                fanout_id=fanout_id,
                branch_id=branch_id,
                branch_index=branch_index,
                payload=dict(payload or {}),
                error=error,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Trace 事件构造失败（%s / %s），已丢弃：%s: %s",
                event_type.value,
                span_name,
                type(exc).__name__,
                exc,
            )
            return False

        return await self.publish(event)


#: 进程级入口。无状态，构造一次即可。
_publisher = RedisTracePublisher()


async def publish_trace_event(
        event_type: EventType,
        *,
        trace_id: str | None,
        run_id: str,
        **kwargs: Any,
) -> bool:
    """模块级便捷入口（Run 外层事件用）。"""

    return await _publisher.emit(
        event_type,
        trace_id=trace_id,
        run_id=run_id,
        **kwargs,
    )
