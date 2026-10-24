"""
工具结果缓存：Redis 主实现 + 进程内兜底。

两种实现，同一套异步接口
------------------------
    RedisToolCache      生产：跨实例共享，TTL 交给 Redis
    InProcessToolCache  兜底：没有 Redis 时（脚本 / 冒烟 / 本地没起 Redis），
                        或测试里显式注入

应用启动时由 `app/bootstrap/tool_cache.py` 决定用哪一种：
拿得到 redis 数据源就用 Redis，拿不到就退回进程内并打 warning。
测试与冒烟脚本不启动 lifespan，因此 `get_tool_cache()` 默认给进程内实现。

⚠️ 缓存是**旁路**，失败必须 fail-open
----------------------------------
Redis 抖一下、反序列化失败、值不可 JSON 化 —— 都不能让工具调用失败。
`RedisToolCache` 的每个操作都捕获异常、记 warning、按"未命中 / 未写入"
处理。缓存丢一次只是少省一次调用，把工具打挂才是事故。

⚠️ 只存 JSON 能表达的值
----------------------
Redis 里存的是 `json.dumps` 之后的文本。写入方（cache 拦截器）会先用
`_execution.execution_result` 做规范化；仍然不可序列化的值会在这里被
`default=str` 兜底成字符串，或者直接放弃写入。进程内实现原样存对象，
因此两种实现在"值的形态"上可能不同 —— 这是 Redis 的代价，不是 bug。

⚠️ TTL 用 Redis 的 EX
--------------------
`ttl_seconds <= 0` 表示不缓存；正值直接交给 `SET ... EX`。进程内实现
用 `time.monotonic()`（不是 `time.time()`），理由见原实现：系统校时
不该让缓存突然全过期或永不过期。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "CacheStats",
    "InProcessToolCache",
    "RedisToolCache",
    "ToolCache",
    "configure_tool_cache",
    "get_tool_cache",
]

logger = logging.getLogger(__name__)

#: Redis 键的固定前缀。工具键已经在 sha256 里，不会串工具。
REDIS_KEY_PREFIX = "tool_cache:"


class ToolCache(Protocol):
    """
    缓存实现的最小接口。

    全部 async：Redis 实现天然异步；进程内实现虽然不 await 任何东西，
    也保持同一形状，免得调用方按实现分支。
    """

    async def get(self, key: str) -> tuple[bool, Any]:
        """返回 (是否命中, 值)。命中值可能是 None。"""
        ...

    async def put(
            self,
            key: str,
            value: Any,
            *,
            ttl_seconds: int,
    ) -> None:
        """写入。ttl_seconds <= 0 表示不缓存。"""
        ...

    async def invalidate(self, key: str) -> bool:
        """删一条；返回它原本在不在。"""
        ...

    async def clear(self) -> int:
        """清空；返回清掉的条数（Redis 实现可能只返回尽力而为的数量）。"""
        ...


@dataclass
class CacheStats:
    """进程内缓存的运行计数，给观测用（字段可增减）。"""

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


# ============================================================
# 进程内兜底
# ============================================================


@dataclass
class _Entry:
    value: Any
    expires_at: float


class InProcessToolCache:
    """
    带 TTL 与容量上限的进程内缓存（LRU）。

    不是"生产共享缓存" —— 多实例各存各的、重启清零。它只用于：

        - 没有 Redis 的本地 / 冒烟 / 测试
        - 应用启动阶段拿不到 redis 数据源时的降级

    容量到了先清过期条目，仍然超了才按 LRU 淘汰。
    """

    def __init__(self, *, max_entries: int = 1024) -> None:

        if max_entries < 1:
            raise ValueError("max_entries 至少为 1。")

        self._max_entries = max_entries

        self._entries: OrderedDict[str, _Entry] = OrderedDict()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        # 保线程交错下的字典完整性。**不**解决跨进程共享。
        self._lock = threading.Lock()

    async def get(self, key: str) -> tuple[bool, Any]:

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

            self._entries.move_to_end(key)
            self._hits += 1

            return True, entry.value

    async def put(
            self,
            key: str,
            value: Any,
            *,
            ttl_seconds: int,
    ) -> None:

        if ttl_seconds <= 0:
            return

        with self._lock:

            self._entries[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + ttl_seconds,
            )
            self._entries.move_to_end(key)

            self._evict_locked()

    async def invalidate(self, key: str) -> bool:

        with self._lock:
            return self._entries.pop(key, None) is not None

    async def clear(self) -> int:

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

    def _evict_locked(self) -> None:

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
            self._entries.pop(next(iter(self._entries)))
            self._evictions += 1


# ============================================================
# Redis
# ============================================================


class RedisToolCache:
    """
    Redis 实现。

    client 由 `RedisDataSource` 持有，这里只借用 —— **不要**在这里
    调 close / aclose：那会连带关掉别的使用者的连接池。

    fail-open：任何 Redis / 序列化异常都只记 warning，按 miss / 未写入
    处理。工具调用绝不能因为缓存坏了而失败。
    """

    def __init__(
            self,
            client: Any,
            *,
            key_prefix: str = REDIS_KEY_PREFIX,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix

    async def get(self, key: str) -> tuple[bool, Any]:

        try:
            raw = await self._client.get(self._key_prefix + key)

            if raw is None:
                return False, None

            return True, json.loads(raw)

        except Exception:  # noqa: BLE001 - 见类 docstring

            logger.warning(
                "工具缓存读取失败，按未命中处理（key=%s）。",
                key,
                exc_info=True,
            )

            return False, None

    async def put(
            self,
            key: str,
            value: Any,
            *,
            ttl_seconds: int,
    ) -> None:

        if ttl_seconds <= 0:
            return

        try:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

            await self._client.set(
                self._key_prefix + key,
                payload,
                ex=ttl_seconds,
            )

        except Exception:  # noqa: BLE001 - 见类 docstring

            logger.warning(
                "工具缓存写入失败，本次不缓存（key=%s）。",
                key,
                exc_info=True,
            )

    async def invalidate(self, key: str) -> bool:

        try:
            deleted = await self._client.delete(
                self._key_prefix + key
            )

            return bool(deleted)

        except Exception:  # noqa: BLE001

            logger.warning(
                "工具缓存删除失败（key=%s）。",
                key,
                exc_info=True,
            )

            return False

    async def clear(self) -> int:
        """
        尽力清空本前缀下的键。

        Redis 没有"按前缀清空"的原子命令，这里用 SCAN + UNLINK。
        返回实际删掉的条数；扫描失败就当 0，不抛。
        """

        deleted = 0

        try:
            batch: list[Any] = []

            async for key in self._client.scan_iter(
                    match=f"{self._key_prefix}*",
                    count=500,
            ):
                batch.append(key)

                if len(batch) >= 500:
                    deleted += int(
                        await self._client.unlink(*batch)
                    )
                    batch.clear()

            if batch:
                deleted += int(
                    await self._client.unlink(*batch)
                )

        except Exception:  # noqa: BLE001

            logger.warning(
                "工具缓存清空失败（prefix=%s）。",
                self._key_prefix,
                exc_info=True,
            )

        return deleted


# ============================================================
# 进程级入口
# ============================================================

#: 当前缓存实现。None = 还没配置，`get_tool_cache()` 会装进程内兜底。
_cache: ToolCache | None = None


def configure_tool_cache(cache: ToolCache | None) -> None:
    """
    设置进程级缓存实现。

    由 `app/bootstrap/tool_cache.py` 在启动时调用：有 Redis 用
    `RedisToolCache`，没有就装 `InProcessToolCache`。测试可以注入
    自己的实现（或显式装进程内实现来隔离状态）。
    """

    global _cache

    _cache = cache


def get_tool_cache() -> ToolCache:
    """
    取进程级缓存。

    没配置过时自动装进程内实现 —— 不启动 lifespan 的脚本 / 冒烟 / 测试
    因此不需要额外装配。生产由 bootstrap 在启动时改配 Redis。
    """

    global _cache

    if _cache is None:
        _cache = InProcessToolCache()

    return _cache
