"""
事件流的**共享词汇**：一条记录长什么样、哪些事件算终止。

⚠️ 订阅机制**不在这里**了
------------------------
改造前这里是每个订阅者各跑一个 `XREAD BLOCK` 循环的实现，于是每个订阅者
常驻占用一条 Redis 连接（池上限 10 —— 第 11 个并发订阅就让全进程的 Redis
访问一起失败）。现在一个 run 只有一条连接、服务它全部订阅者，那套机制在
`run_stream.py`。

本模块只留三样**两边都要用**的东西：

    StreamEntry   Redis 给的一条记录解成 str 之后的形状
    decode        bytes -> str（含"坏字节不该让整条订阅挂掉"的兜底）
    is_terminal   这条记录表示"运行终止了"吗

为什么 `is_terminal` 的判据来自契约层
------------------------------------
`app.common.event` 的 TERMINAL_EVENT_TYPES 才是"哪些事件算终态"的出处。
放这一层是因为它是**契约**：将来若加 run.cancelled，改那一处即可。

⚠️ 它与 `STREAM_CLOSING_EVENT_TYPES` **不是一回事**
-----------------------------------------------
后者是"订阅该收尾了"（多了 hitl.requested）。运行**暂停**等审批并不等于
运行终止 —— DB 转 waiting_input，thread 锁仍被该 run 持有，人类批准后同一
个 run 还会接着往同一条流上写。所以本模块的 `is_terminal` 保持只管终止，
"什么时候关流"那套判断在 run_stream.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.common.event import TERMINAL_EVENT_TYPES

__all__ = [
    "StreamEntry",
    "TERMINAL_VALUES",
    "decode",
    "is_terminal",
]


#: 终止事件名的**值**集合。启动时算一次，免得每条记录都遍历枚举。
TERMINAL_VALUES: frozenset[str] = frozenset(
    item.value for item in TERMINAL_EVENT_TYPES
)


def decode(value: Any) -> str:
    """
    把 Redis 给的东西解成 str。

    `redis.yaml` 里 `decode_responses: false`（那是刻意的默认：bytes 是
    唯一没有信息损失的形态），因此 XREAD / XRANGE 回来的 **id、字段名、
    字段值全是 bytes**，三个都要解。

    `errors="replace"`：一个坏字节不该让整条订阅挂掉。我们写进去的
    一定是合法 UTF-8，所以这条只是防御 —— 但防御的成本是零，而
    "某一条记录让整个 SSE 连接断开"的排查成本很高。
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


@dataclass(frozen=True, slots=True)
class StreamEntry:
    """一条流记录：Redis 的 entry id 加上扁平字段。"""

    id: str

    fields: dict[str, str]

    @property
    def event_type(self) -> str | None:
        """`event` 字段的值（信封字段之一）。"""

        return self.fields.get("event")


def is_terminal(entry: StreamEntry) -> bool:
    """
    这条记录表示运行**终止**了吗 —— 订阅端据此把连接正常关掉。

    判据来自契约层（`app.common.event` 的 TERMINAL_EVENT_TYPES），
    不是写死在这里的字符串。

    ⚠️ 它**不**包含 hitl.requested —— 那是"暂停"，不是"终止"。关流规则
    见 run_stream.py 的模块 docstring。
    """

    return entry.event_type in TERMINAL_VALUES
