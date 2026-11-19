"""
事件路由：**一个 `emit`，多个 sink 各自决定放行与落键**。

这是 `docs/event-stream-v2-design.md` 的阶段 1 骨架。它只做纯路由：

    - 判断事件类型是否在 sink 的白名单里
    - 计算这条事件在该 sink 里的 Redis 键
    - 把 `(event, key)` 交给 sink 的 enqueue 回调

**不碰线程、不碰 Redis、不 await**。真正的队列与写线程在阶段 2 接到
enqueue 上。这样路由规则可以先独立测透，再谈 IO。

为什么 sink 是策略而不是 if-else
-------------------------------
“哪些事件属于 chat、哪些属于 trace”是**契约**，不是部署配置。写成
`SinkPolicy.accepts` 之后：

    - 加一个事件类型时，编译器 / 测试会提醒两条流各自的归属
    - 配置只能调优队列与 TTL，改不了“message.delta 进不进 trace”
    - 同一个事件被两个 sink 接收时，两边拿到的是同一个 `Event` 对象，
      `event_id` 相同，便于跨流对账

⚠️ `emit` 从不抛异常
------------------
事件是旁路。任何一个 sink 的键计算或入队失败，只影响那一条 sink，
不能把调用方（模型 token 路径 / 图节点）拖下水。失败按 sink 记一次
warning，返回 False。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.common.event.event import stream_key
from app.common.event.spec import (
    CHAT_EVENT_TYPES,
    TRACE_EVENT_TYPES,
    Event,
    EventType,
)

__all__ = [
    "Enqueue",
    "EventRouter",
    "SinkPolicy",
    "chat_policy",
    "trace_policy",
]


logger = logging.getLogger(__name__)


class Enqueue(Protocol):
    """sink 的入队回调。返回是否已接收；**不得抛异常**（router 会兜底）。"""

    def __call__(self, event: Event, key: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class SinkPolicy:
    """
    一条流的**契约**：叫什么、放行哪些事件、键怎么算。

    队列大小 / TTL / 写线程数不在这里 —— 它们是运行期调优，属于 sink
    的实现（阶段 2 的 `SinkRuntime`）。
    """

    name: str

    #: 放行白名单。不在集合里的事件不进这条流。
    accepts: frozenset[EventType]

    #: 事件 -> Redis 键。返回 None 表示“这条事件进不了这个 sink”
    #: （例如没有 trace_id 的事件进 trace）。
    key_of: Callable[[Event], str | None]


class EventRouter:
    """
    把一个 `Event` 分发给所有匹配的 sink。

    构造参数是 `(policy, enqueue)` 对。enqueue 由调用方提供 —— 阶段 1
    用记录器，阶段 2 换成真实的 `SinkRuntime.enqueue`。
    """

    __slots__ = ("_sinks",)

    def __init__(
            self,
            sinks: Sequence[tuple[SinkPolicy, Enqueue]],
    ) -> None:

        names = [policy.name for policy, _ in sinks]

        if len(names) != len(set(names)):
            raise ValueError(f"sink 名称重复：{names}")

        self._sinks = tuple(sinks)

    @property
    def policies(self) -> tuple[SinkPolicy, ...]:
        return tuple(policy for policy, _ in self._sinks)

    def emit(self, event: Event) -> dict[str, bool]:
        """
        分发一条事件，返回 `{sink 名: 是否已接收}`。

        没有匹配的 sink 时返回空字典 —— 那不是错误：trace-only 事件在
        没有 trace 的运行里本来就不该有目的地。
        """

        results: dict[str, bool] = {}

        for policy, enqueue in self._sinks:

            if event.type not in policy.accepts:
                continue

            try:
                key = policy.key_of(event)

                if key is None:
                    continue

                results[policy.name] = bool(enqueue(event, key))

            except Exception as exc:  # noqa: BLE001 —— 旁路绝不外抛
                logger.warning(
                    "事件路由失败（sink=%s, event=%s）：%s: %s。"
                    "这条事件只在这个 sink 丢失，不影响运行。",
                    policy.name,
                    getattr(event, "type", "?"),
                    type(exc).__name__,
                    exc,
                )
                results[policy.name] = False

        return results


# ============================================================
# 默认策略
# ============================================================


def chat_policy(key_prefix: str = "agent_run") -> SinkPolicy:
    """
    chat 流：按 `run_id` 聚合，键是 `<key_prefix>:<run_id>`。

    `key_prefix` 由配置注入；默认值与 `event_stream.yaml` 的历史默认一致。
    """

    def _key_of(event: Event) -> str:
        return stream_key(key_prefix, event.run_id)

    return SinkPolicy(
        name="chat",
        accepts=CHAT_EVENT_TYPES,
        key_of=_key_of,
    )


def trace_policy(key_prefix: str = "trace") -> SinkPolicy:
    """
    trace 流：按 `trace_id` 聚合，键是 `<key_prefix>:<trace_id>:events`。

    ⚠️ 没有 trace_id 时返回 None —— 不构造一个空 trace_id 的键。
    那种键与“没开 trace”长得一模一样，排查时会把“没开”误读成“丢了”。

    ⚠️ 默认前缀必须是 "trace"：stress 的 `TraceAggregator` 按
    `trace:<trace_id>:events` 直接读键。配置加载器会拦掉其它值。
    """

    def _key_of(event: Event) -> str | None:
        if not event.trace_id:
            return None

        return f"{key_prefix}:{event.trace_id}:events"

    return SinkPolicy(
        name="trace",
        accepts=TRACE_EVENT_TYPES,
        key_of=_key_of,
    )
