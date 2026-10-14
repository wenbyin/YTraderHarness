"""
事件流测试用的替身：一条**内存里的 Redis Stream**。

为什么不是"脚本化返回值"
--------------------
本仓库既有的写法（app/agent/event/event_smoke.py 的 FakeRedis）是把
`xread` 的返回值按顺序排好队、每次弹一个。那对"单个订阅者顺序消费"够用，
但验不了本次改造真正要验的东西 —— **补发与尾随的边界**：

    新订阅者挂上来时，循环要发一条 XRANGE 把 `(start, cursor]` 补给他，
    然后接着 XREAD。这两条命令的**范围必须严丝合缝**地接上，多了会重复
    投递、少了会静默丢一条。

脚本化的返回值验不了这个 —— 那个范围是算出来的，不是排出来的。所以这里
实现的是一条真的能按 id 范围查询的内存流。id 的比较按 `<毫秒>-<序号>`
两个整数来（与 Redis 一致），不是字符串比较 —— 字符串比较下 "10-0" 会
小于 "9-0"，而那种错误正好会把"补发边界"验反。
"""

from __future__ import annotations

import asyncio
from typing import Any

__all__ = [
    "FakeStreamRedis",
    "parse_id",
    "xrange_bounds",
]


def parse_id(value: bytes | str) -> tuple[int, int]:
    """`"1759412273324-0"` -> `(1759412273324, 0)`。"""

    text = value.decode() if isinstance(value, bytes) else value

    ms, _, seq = text.partition("-")

    return int(ms), int(seq or 0)


def xrange_bounds(
        lower: bytes | str,
        upper: bytes | str,
) -> tuple[tuple[int, int], bool, tuple[int, int], bool]:
    """
    把 XRANGE 的起止参数解成 `(值, 是否排除)`。

    XRANGE 的语法里：
        `-` / `+`       最小 / 最大
        `(` 前缀        排除该值
        默认            包含

    ⚠️ 这条语义与 XREAD 的游标**相反**（XREAD 的游标天生是排他的）——
    两边的转换正是 run_stream.py 里 `_catch_up` 要小心的地方，所以替身
    把两边都实现对，免得测试把实现里的错误当作正确。
    """

    def one(raw: bytes | str) -> tuple[tuple[int, int] | None, bool]:

        text = raw.decode() if isinstance(raw, bytes) else raw

        if text == "-":
            return None, False

        if text == "+":
            return None, False

        if text.startswith("("):
            return parse_id(text[1:]), True

        return parse_id(text), False

    lower_value, lower_exclusive = one(lower)
    upper_value, upper_exclusive = one(upper)

    return (
        lower_value or (0, 0),
        lower_exclusive,
        upper_value or (1 << 62, 0),
        upper_exclusive,
    )


class FakeStreamRedis:
    """
    够用的内存 Redis Stream：`xadd` / `xrange` / `xrevrange` / `xread`。

    只实现事件流读端用到的那几条，语义按 Redis 来。多出来的能力（DB、
    过期、MAXLEN）刻意不做 —— 替身做得越全，测试越容易依赖它的细节。
    """

    def __init__(self) -> None:

        #: `(id, fields)`，按 id 升序（append 时维护）。
        self.entries: list[tuple[str, dict[bytes, bytes]]] = []

        #: 每条命令各调了几次 —— "一个 run 只占一条连接"那类断言靠它。
        self.calls: dict[str, int] = {
            "xadd": 0,
            "xrange": 0,
            "xrevrange": 0,
            "xread": 0,
        }

        #: xread 的 block 参数都记下来，用来验"阻塞时长没超过 socket_timeout"
        #: 这类约束。
        self.blocks: list[int | None] = []

        #: 注入读失败：前 N 次 xread 抛这个异常。
        self.xread_error: BaseException | None = None
        self.xread_errors_left = 0

    # --------------------------------------------------------

    def add(self, entry_id: str, event: str, **fields: str) -> str:
        """写一条（等价于 XADD）。字段值按 `decode_responses=false` 存 bytes。"""

        payload: dict[bytes, bytes] = {
            b"event": event.encode(),
        }

        for name, value in fields.items():
            payload[name.encode()] = str(value).encode()

        self.entries.append((entry_id, payload))

        return entry_id

    @property
    def ids(self) -> list[str]:
        return [entry_id for entry_id, _ in self.entries]

    def last_id(self) -> str | None:
        return self.entries[-1][0] if self.entries else None

    # --------------------------------------------------------

    async def xrange(
            self,
            key: str,
            lower: Any,
            upper: Any,
            count: int | None = None,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:

        # ⚠️ 这个 `sleep(0)` 不是摆设 —— 去掉它整套测试会**挂死**
        # ----------------------------------------------------
        # 一条真的 Redis 命令一定会 await 套接字，因此每执行一条都会把控制权
        # 交还事件循环。替身里如果把整个方法写成同步的（只是 `async def`
        # 但内部没有任何 await），`await fake.xread(...)` 就**不会挂起** ——
        # 于是广播循环的 `while True` 从头到尾都在**同一个回调**里跑，事件
        # 循环再也拿不回控制权，测试文件里的任何 await 都永远醒不过来。
        await asyncio.sleep(0)

        self.calls["xrange"] += 1

        lo, lo_excl, hi, hi_excl = xrange_bounds(lower, upper)

        picked = [
            (entry_id, fields)
            for entry_id, fields in self.entries
            if (lo < parse_id(entry_id) if lo_excl else lo <= parse_id(entry_id))
            and (parse_id(entry_id) < hi if hi_excl else parse_id(entry_id) <= hi)
        ]

        if count is not None:
            picked = picked[:count]

        return [
            (entry_id.encode(), fields) for entry_id, fields in picked
        ]

    async def xrevrange(
            self,
            key: str,
            upper: Any,
            lower: Any,
            count: int | None = None,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:

        self.calls["xrevrange"] += 1

        forward = await self.xrange(key, lower, upper)

        forward.reverse()

        if count is not None:
            forward = forward[:count]

        return forward

    async def xread(
            self,
            streams: dict[str, Any],
            block: int | None = None,
            count: int | None = None,
    ) -> list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]] | None:

        # 见 xrange 上那段说明：不 await 就会把事件循环锁死。
        await asyncio.sleep(0)

        self.calls["xread"] += 1

        self.blocks.append(block)

        if self.xread_errors_left > 0 and self.xread_error is not None:
            self.xread_errors_left -= 1
            raise self.xread_error

        key, cursor = next(iter(streams.items()))

        # XREAD 的游标是**排他**的：传进去的 id 本身不返回。
        # 替身必须照做 —— 实现里"补发到 cursor（含）"与"尾随从 cursor
        # 之后"正是靠这一条才能接上，替身要是把它做成包含，两边就都错了。
        if cursor in ("0", "0-0"):
            after = (0, 0)
            exclusive = False
        else:
            after = parse_id(cursor)
            exclusive = True

        picked = [
            (entry_id, fields)
            for entry_id, fields in self.entries
            if (parse_id(entry_id) > after if exclusive else parse_id(entry_id) >= after)
        ]

        if count is not None:
            picked = picked[:count]

        if not picked:
            # 真 Redis 会在这里阻塞 block 毫秒再返回 nil。替身不真的等 ——
            # 等待由测试控制（配一个很小的 idle_probe_after_reads，循环
            # 自己会转得很快），这样测试不必 sleep。
            return None

        return [
            (
                key.encode(),
                [(entry_id.encode(), fields) for entry_id, fields in picked],
            )
        ]
