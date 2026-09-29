"""
工具结果的进程内缓存。

为什么是"进程内"而不是共享缓存
------------------------------
项目里没有 Redis（依赖里没有，grep 也零命中），而为了一个缓存去
引一个外部依赖、再管一套连接池与失效策略，是另一个量级的改动。
进程内缓存解决的是**同一个进程里重复调用同一工具**这个最常见的
场景（一次 Agent 会话里反复查同一个东西），代价是：

    多实例部署时各存各的 → 命中率低于理论值，但**不会错**
    进程重启清零         → 只是冷启动，不影响正确性

真心要跨实例共享时，把本模块换成一个 Redis 实现即可 —— 处理器
只依赖下面五个方法，换实现不用动它们。

⚠️ TTL 用的是 monotonic 时钟
---------------------------
`time.monotonic()` 而不是 `time.time()`：后者会被系统时间调整
（NTP 校时、手工改钟）影响，表现是"缓存条目突然全部过期"或
"永不过期"。这与 asyncio 内部计时用 monotonic 是同一个理由。

⚠️ 键是调用方算好的字符串
------------------------
本模块不关心键怎么来的（工具键 + 参数摘要，见 interceptor/cache.py）。
缓存只做"给定键，存/取"这一件事，让策略留在需要策略的那一层。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CacheStats",
    "InProcessToolCache",
    "get_tool_cache",
]


@dataclass
class _Entry:
    """一条缓存。value 原样存 —— 深拷贝由调用方决定。"""

    value: Any
    expires_at: float


@dataclass
class CacheStats:
    """缓存的运行计数，给观测用（不是契约，字段可增减）。"""

    entries: int = 0
    capacity: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """命中率。一次都没查过时是 0.0（不是除零）。"""

        lookups = self.hits + self.misses

        return self.hits / lookups if lookups else 0.0


class InProcessToolCache:
    """
    带 TTL 与容量上限的进程内缓存。

    容量到了先清过期条目，仍然超了才按 **LRU** 淘汰 ——
    get 命中把条目挪到末尾，淘汰时从头部拿。
    """

    def __init__(self, *, max_entries: int = 1024) -> None:

        if max_entries < 1:
            raise ValueError("max_entries 至少为 1。")

        self._max_entries = max_entries

        # OrderedDict 而不是普通 dict：LRU 需要 move_to_end()，
        # 那是 OrderedDict 的方法。普通 dict 虽然也保序，但没有
        # "把某个键挪到末尾"这种操作。
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        # 保线程交错下的字典完整性。**不**解决跨进程共享 ——
        # 那是上面 docstring 说的另一件事。
        self._lock = threading.Lock()

    # ========================================================
    # 读写
    # ========================================================

    def get(self, key: str) -> tuple[bool, Any]:
        """
        取一条。返回 (是否命中, 值)。

        用元组而不是"返回值或 None"：缓存里存的**可能就是 None**
        （一个返回 null 的工具），那样调用方分不清"没缓存"与
        "缓存了 null"，于是每次都会穿透到底层重算。
        """

        now = time.monotonic()

        with self._lock:

            entry = self._entries.get(key)

            if entry is None:
                self._misses += 1
                return False, None

            if entry.expires_at <= now:
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                return False, None

            # LRU：命中的条目挪到队尾，淘汰时从队首拿。
            self._entries.move_to_end(key)
            self._hits += 1

            return True, entry.value

    def put(
            self,
            key: str,
            value: Any,
            *,
            ttl_seconds: float,
    ) -> None:
        """
        存一条。ttl_seconds <= 0 表示**不缓存**（直接返回）。

        让"不缓存"走这条分支而不是由调用方自己判断：
        ttl=0 是个很容易配出来的值（想想"先关掉缓存"），
        如果它变成"永不过期"或者"存一条立刻过期的垃圾"，
        两种都不好查。
        """

        if ttl_seconds <= 0:
            return

        with self._lock:

            self._entries[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + ttl_seconds,
            )
            self._entries.move_to_end(key)

            self._evict_locked()

    def invalidate(self, key: str) -> bool:
        """删一条。返回它原本在不在。"""

        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> int:
        """清空。返回清掉的条数。"""

        with self._lock:

            count = len(self._entries)
            self._entries.clear()

            return count

    def stats(self) -> CacheStats:
        """当前计数快照。"""

        with self._lock:

            return CacheStats(
                entries=len(self._entries),
                capacity=self._max_entries,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
            )

    # ========================================================
    # internals
    # ========================================================

    def _evict_locked(self) -> None:
        """调用方必须已持锁。先清过期的，再 LRU 淘汰。"""

        if len(self._entries) <= self._max_entries:
            return

        now = time.monotonic()

        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]

        for key in expired:
            del self._entries[key]
            self._expirations += 1

        while len(self._entries) > self._max_entries:
            # 队首是最久没被访问的（get 会把命中的挪到队尾）。
            self._entries.pop(next(iter(self._entries)))
            self._evictions += 1


#: 进程级单例。缓存本来就该是"全进程一份" ——
#: 每个处理器实例各存一份等于没有缓存。
_cache: InProcessToolCache | None = None


def get_tool_cache() -> InProcessToolCache:
    """取进程级工具缓存。"""

    global _cache

    if _cache is None:
        _cache = InProcessToolCache()

    return _cache
