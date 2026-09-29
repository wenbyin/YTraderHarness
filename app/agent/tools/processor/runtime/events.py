"""
工具生命周期事件与订阅总线。

用途：把"某个工具开始跑了 / 跑完了 / 失败了"实时推给 Agent Server
或前端界面（SSE）。这是**旁路**——收不到事件不影响调用本身，
因此投递失败一律不往上抛。

投递模型：发布是同步的，订阅是有界的
------------------------------------
    publish()    同步、非阻塞。它在一个 ASYNC 观察者里被调用，
                 而观察者本身是 fire-and-forget 的 —— 在这里 await
                 一个慢消费者等于把工具调用拖住。
    subscribe()  异步上下文管理器，每次订阅拿一个**有界队列**。

队列满了怎么办：**丢最旧的**
---------------------------
不丢新事件（那么界面会停在一个过时的状态上，且永远追不上），
也不阻塞（那会拖住调用）。丢最旧的让界面始终看到**最近**的一段，
这在"看当前发生了什么"这个用途下是对的取舍。

丢弃**必须留痕**：静默丢事件会让界面少显示几条而没人知道，
于是"为什么这里少了一次调用"变成悬案。日志按 2 的幂节流
（第 1、2、4、8… 次丢弃各报一条），既不会刷屏，
也不会把一次长期卡住的订阅者完全藏起来。

⚠️ publish 必须在订阅者所在的事件循环里调用
------------------------------------------
asyncio.Queue 不是线程安全的，跨循环 put 会静默出问题。
当前所有处理器都在同一个循环里跑，因此满足；将来若有线程池
里的执行路径，要改成 call_soon_threadsafe。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

__all__ = [
    "ToolEvent",
    "ToolEventBus",
    "ToolEventType",
    "get_tool_event_bus",
]

logger = logging.getLogger(__name__)


class ToolEventType(str, Enum):
    """
    一次工具调用走到哪一步了。

    DENIED 与 FAILED 是两件事，刻意分开：DENIED 表示**工具根本没跑**
    （被拦截器拒了，例如审批不过），FAILED 表示跑了但失败了。
    界面上"被拦下"与"执行出错"要给人看的解释完全不同。
    """

    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    DENIED = "DENIED"


@dataclass(frozen=True)
class ToolEvent:
    """
    一条工具生命周期事件。

    刻意**不带参数与结果正文** —— 那里面有业务数据（甚至个人
    信息），而这条事件会流向浏览器。要看内容请用日志或调用记录，
    不要把事件总线当成数据通道。

    真需要时由 config 显式打开（见 observer/events.py 的
    include_arguments），并且只在可信的订阅者场景下开。
    """

    type: ToolEventType

    tool_key: str

    occurred_at: datetime

    tool_id: UUID | None = None

    #: 链上第几条记录发出来的（排查"是哪个处理器"用）。
    sequence: int | None = None

    #: 工具自身执行的耗时（毫秒）。ASYNC 观察者算不出这个值 ——
    #: 它只能看墙钟，而墙钟包含它自己之前的那些处理器的开销。
    #: 因此这个数由**执行侧**填，填不了就是 None。
    duration_ms: float | None = None

    #: 失败时的异常类型名与消息（不带 traceback）。
    error_type: str | None = None
    error_message: str | None = None

    #: 处理器想附带的少量结构化信息（例如 cache 的 hit/miss）。
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """
        转成可直接 JSON 序列化的字典（SSE 的 data 段）。

        时间用 ISO 8601 带时区 —— 前端 `new Date(...)` 能直接吃，
        而 Unix 时间戳在不同语言里单位不同（秒 / 毫秒）容易搞错。
        """

        return {
            "type": self.type.value,
            "tool_key": self.tool_key,
            "tool_id": str(self.tool_id) if self.tool_id else None,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at.isoformat(),
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "extra": dict(self.extra),
        }


class ToolEventBus:
    """
    进程内的事件总线：一对多、非阻塞、有界。

    订阅者拿到的队列由自己消费。**不消费的订阅者会被丢事件**，
    但不会拖住任何人 —— 见模块 docstring。
    """

    #: 默认每个订阅者的队列长度。
    #:
    #: 100 是个折中：一次突发里足够装下几十个工具的起止事件，
    #: 又不至于让一个卡住的订阅者攒下几百 MB。
    DEFAULT_QUEUE_SIZE = 100

    def __init__(self, *, queue_size: int | None = None) -> None:

        self._queue_size = queue_size or self.DEFAULT_QUEUE_SIZE

        self._subscribers: list[
            tuple[asyncio.Queue[ToolEvent], list[int]]
        ] = []

        # 丢弃计数放这里而不是闭包里：订阅者取消时要能报出总数。
        self._dropped = 0

    # ========================================================
    # 发布
    # ========================================================

    def publish(self, event: ToolEvent) -> int:
        """
        发布一条事件，返回**成功投递到的订阅者数**。

        非阻塞。队列满时丢掉该队列里最旧的一条，再把新的放进去。
        """

        delivered = 0

        for queue, dropped in list(self._subscribers):

            try:
                queue.put_nowait(event)
                delivered += 1
                continue

            except asyncio.QueueFull:
                pass

            # 丢最旧的一条，给新事件腾位置。
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:      # pragma: no cover - 竞态兜底
                pass

            try:
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:       # pragma: no cover - 竞态兜底
                pass

            dropped[0] += 1
            self._dropped += 1

            # 按 2 的幂节流：1、2、4、8… 次各报一条。
            # 一次都不报会让"界面少了几条"无从解释；每次都报会把
            # 一个卡住的订阅者变成日志洪水。
            if dropped[0] & (dropped[0] - 1) == 0:
                logger.warning(
                    "事件订阅者消费不过来，已丢弃 %d 条（队列上限 %d）。"
                    "该订阅者收到的是**最近**的事件，早期事件已丢。",
                    dropped[0],
                    self._queue_size,
                )

        return delivered

    # ========================================================
    # 订阅
    # ========================================================

    @asynccontextmanager
    async def subscribe(
            self,
            *,
            queue_size: int | None = None,
    ) -> AsyncIterator[asyncio.Queue[ToolEvent]]:
        """
        订阅一段时间的事件。

            async with bus.subscribe() as queue:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event.to_payload())}\\n\\n"

        退出上下文即退订 —— 用 `async with` 而不是手写 add/remove，
        是为了让 SSE 连接断开（客户端关页面、网络断）时那条订阅
        一定会被摘掉。漏摘的订阅者会一直吃事件并占着内存，
        而且症状是"推给一个没人看的队列"，从外部完全看不出来。
        """

        queue: asyncio.Queue[ToolEvent] = asyncio.Queue(
            maxsize=queue_size or self._queue_size
        )

        # 丢弃计数用单元素列表而不是 int：publish 里要就地累加。
        dropped = [0]

        self._subscribers.append((queue, dropped))

        try:
            yield queue

        finally:
            self._subscribers = [
                item
                for item in self._subscribers
                if item[0] is not queue
            ]

            if dropped[0]:
                logger.info(
                    "事件订阅结束，期间丢弃了 %d 条。",
                    dropped[0],
                )

    # ========================================================

    @property
    def subscriber_count(self) -> int:
        """当前订阅者数（探活用）。"""

        return len(self._subscribers)

    @property
    def dropped(self) -> int:
        """累计丢弃数（探活用）。"""

        return self._dropped


#: 进程级单例：事件本来就要汇到一个地方再分发。
_bus: ToolEventBus | None = None


def get_tool_event_bus() -> ToolEventBus:
    """取进程级事件总线。"""

    global _bus

    if _bus is None:
        _bus = ToolEventBus()

    return _bus
