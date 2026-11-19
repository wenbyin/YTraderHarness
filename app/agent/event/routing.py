"""
v2 写入端：chat / trace 两个 sink + 一个路由入口。

唯一入口是 `emit(Event)`：一个事件构建一次，由 `EventRouter` 按
`CHAT_EVENT_TYPES` / `TRACE_EVENT_TYPES` 决定进哪条流、写到哪个键。
调用方不再区分 run 事件与 trace 事件。

节点侧的 `TraceEmitter` 也直接构建规范 `Event`，因此本模块不再做
任何旧模型转换。
"""

from __future__ import annotations

import asyncio
import logging

from app.common.event.spec import Event

from .config import SinkConfig, load_sink_configs
from .publisher import EventSink
from .router import EventRouter, chat_policy, trace_policy
from .sink import SinkRuntime

__all__ = [
    "RouterEventPublisher",
]


logger = logging.getLogger(__name__)


# ============================================================
# 发布器
# ============================================================


class RouterEventPublisher:
    """
    两个 `SinkRuntime` + 一个 `EventRouter`。

    唯一入口 `emit(Event)`：按放行集合分发，每个 sink 自己算键、入队。
    """

    __slots__ = (
        "_sink",
        "_configs",
        "_chat_prefix",
        "_trace_prefix",
        "_chat",
        "_trace",
        "_router",
    )

    def __init__(
            self,
            sink: EventSink,
            *,
            configs: dict[str, SinkConfig] | None = None,
    ) -> None:

        self._sink = sink

        self._configs = configs or load_sink_configs()

        chat_config = self._configs["chat"]
        trace_config = self._configs["trace"]

        self._chat_prefix = chat_config.key_prefix
        self._trace_prefix = trace_config.key_prefix

        self._chat = SinkRuntime(
            name="chat",
            sink=sink,
            config=chat_config,
            serializer=_chat_serializer(chat_config),
        )

        self._trace = SinkRuntime(
            name="trace",
            sink=sink,
            config=trace_config,
            serializer=lambda event: event.to_fields(),
        )

        self._router = EventRouter([
            (chat_policy(self._chat_prefix), self._chat.enqueue),
            (trace_policy(self._trace_prefix), self._trace.enqueue),
        ])

    # --------------------------------------------------------
    # 状态
    # --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._chat.enabled or self._trace.enabled

    @property
    def failures(self) -> int:
        return self._chat.failures + self._trace.failures

    @property
    def rejected(self) -> int:
        return self._chat.rejected + self._trace.rejected

    @property
    def pending(self) -> int:
        return self._chat.pending + self._trace.pending

    def ping(self) -> bool:
        return self._sink.ping()

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------

    def start(self) -> None:
        if self._chat.enabled:
            self._chat.start()

        if self._trace.enabled:
            self._trace.start()

    async def aclose(self, timeout: float = 10.0) -> None:
        await asyncio.gather(
            self._chat.aclose(timeout),
            self._trace.aclose(timeout),
        )

    async def drain(self, timeout: float = 5.0) -> None:
        await asyncio.gather(
            self._chat.drain(timeout),
            self._trace.drain(timeout),
        )

    # --------------------------------------------------------
    # 发布
    # --------------------------------------------------------

    async def emit(self, event: Event) -> dict[str, bool]:
        """新规范入口：按放行集合分发，返回每个 sink 的接收结果。"""

        return self.route(event)

    def route(self, event: Event) -> dict[str, bool]:
        return self._router.emit(event)


# ============================================================
# 小工具
# ============================================================


def _chat_serializer(config: SinkConfig):
    """sink 配置 -> 记录序列化函数。记录形状只有 v2 一种。"""

    return lambda event: event.to_fields()
