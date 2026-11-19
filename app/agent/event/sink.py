"""
一条事件流的**写入运行时**：有界队列 + 分片写线程 + 丢弃策略。

为什么每个 sink 一个实例
------------------------
旧实现是“一个进程一个队列池，按 stream key 分片”，前提是只有一条逻辑
流（chat）。v2 需要 chat 与 trace **各自独立**：trace 的 span 洪峰不能
吃掉 chat 的逐 token 缓冲。因此把“队列 + 线程 + 丢弃策略”抽成
`SinkRuntime`，每个 sink 一个实例。

它**不**知道：
    - 事件长什么样（收 `(event, key)`，serializer 自己决定字段）
    - 键怎么算（router 的 `SinkPolicy` 算好传进来）
    - 哪些事件该来（router 的放行集合决定）

因此测试可以只换一个 `EventSink`（记录器）和一个 serializer，把线程、
丢弃、排空这些行为跑完，不必碰真 Redis。

⚠️ 队列有界，且一旦丢就丢到排空为止
----------------------------------
理由与旧发布器逐字相同（见 `publisher.py` 的模块 docstring）：
    - 无界队列在 Redis 停摆时是内存泄漏
    - “丢一条、后面接着收”会在正文中间撕出一个洞
所以一旦某个分片满了，就把它置为 dropping，直到该分片排空才恢复。
写出去的那段永远是一个连续前缀。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass

from app.common.event.spec import Event

from .config import SinkConfig
from .publisher import EventSink

__all__ = ["SinkRuntime"]


logger = logging.getLogger(__name__)


#: 事件 -> Redis Stream 字段。chat 在过渡期可以传 v1 形状的 serializer；
#: trace 始终传 `Event.to_fields`。
Serializer = Callable[[Event], dict[str, str]]


@dataclass(frozen=True, slots=True)
class _Write:
    key: str
    fields: dict[str, str]
    maxlen: int | None
    ttl_seconds: int | None


@dataclass(slots=True)
class _Shard:
    queue: queue.Queue
    thread: threading.Thread | None = None

    #: 这一分片的丢弃期。丢一条后一直丢到该分片排空为止。
    dropping: bool = False


class SinkRuntime:
    """
    一个 sink 的写入端。

    生命周期：
        start()     起写线程
        enqueue()   事件循环线程调用，只做 put_nowait
        drain()     测试用，等队列排空
        aclose()    先排空再停线程
    """

    __slots__ = (
        "_name",
        "_sink",
        "_config",
        "_serializer",
        "_shards",
        "_started",
        "_stopping",
        "_lock",
        "_failures",
        "_rejected",
    )

    def __init__(
            self,
            *,
            name: str,
            sink: EventSink,
            config: SinkConfig,
            serializer: Serializer,
            queue_size: int | None = None,
            workers: int | None = None,
    ) -> None:

        self._name = name
        self._sink = sink
        self._config = config
        self._serializer = serializer

        worker_count = max(1, int(workers or config.write_workers))

        self._shards: list[_Shard] = [
            _Shard(
                queue=queue.Queue(
                    maxsize=queue_size or config.write_queue_size,
                )
            )
            for _ in range(worker_count)
        ]

        self._started = False
        self._stopping = threading.Event()

        self._lock = threading.Lock()
        self._failures = 0
        self._rejected = 0

    # ========================================================
    # 状态
    # ========================================================

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def rejected(self) -> int:
        with self._lock:
            return self._rejected

    @property
    def pending(self) -> int:
        return sum(shard.queue.qsize() for shard in self._shards)

    def ping(self) -> bool:
        return self._sink.ping()

    # ========================================================
    # 生命周期
    # ========================================================

    def start(self) -> None:
        if self._started:
            return

        self._stopping.clear()

        for index, shard in enumerate(self._shards):
            shard.thread = threading.Thread(
                target=self._consume,
                args=(shard,),
                name=f"event-sink-{self._name}-{index}",
                daemon=True,
            )
            shard.thread.start()

        self._started = True

    async def aclose(self, timeout: float = 10.0) -> None:
        if not self._started:
            return

        threads = [
            shard.thread
            for shard in self._shards
            if shard.thread is not None
        ]

        for shard in self._shards:
            shard.thread = None

        self._started = False
        self._stopping.set()

        def _join() -> None:
            deadline = time.monotonic() + timeout

            for thread in threads:
                thread.join(max(0.0, deadline - time.monotonic()))

        try:
            await asyncio.wait_for(
                asyncio.to_thread(_join),
                timeout + 1,
            )

        except Exception:  # noqa: BLE001 —— 排不空不是异常路径
            logger.warning(
                "事件 sink %s 的写线程未能在 %.0f 秒内退出，剩余 %d 条未写出。",
                self._name,
                timeout,
                self.pending,
            )

    async def drain(self, timeout: float = 5.0) -> None:
        def _join_all() -> None:
            for shard in self._shards:
                shard.queue.join()

        await asyncio.wait_for(
            asyncio.to_thread(_join_all),
            timeout,
        )

    # ========================================================
    # 入队
    # ========================================================

    def enqueue(self, event: Event, key: str) -> bool:
        """
        收下一条事件，返回**是否已入队**。**从不抛异常。**

        与旧发布器一致：返回 True 只表示“已接收”，写入结果不回报。
        """

        if not self._config.enabled:
            return False

        if not self._started or self._stopping.is_set():
            return False

        try:
            fields = self._serializer(event)

        except Exception as exc:  # noqa: BLE001
            self._record_failure(exc, key)
            return False

        shard = self._shards[
            zlib.crc32(key.encode("utf-8")) % len(self._shards)
        ]

        if shard.dropping:
            if not shard.queue.empty():
                self._record_rejection(key, shard.queue.maxsize)
                return False

            shard.dropping = False

        item = _Write(
            key=key,
            fields=fields,
            maxlen=self._config.maxlen,
            ttl_seconds=self._config.ttl_seconds,
        )

        try:
            shard.queue.put_nowait(item)

        except queue.Full:
            shard.dropping = True
            self._record_rejection(key, shard.queue.maxsize)
            return False

        return True

    # ========================================================
    # 写线程
    # ========================================================

    def _consume(self, shard: _Shard) -> None:
        while True:

            try:
                item = shard.queue.get(timeout=0.2)

            except queue.Empty:
                if self._stopping.is_set():
                    return

                continue

            try:
                self._sink.write(
                    item.key,
                    item.fields,
                    maxlen=item.maxlen,
                    ttl_seconds=item.ttl_seconds,
                )

            except Exception as exc:  # noqa: BLE001 —— 单条失败不能停线程
                self._record_failure(exc, item.key)

            finally:
                shard.queue.task_done()

    # ========================================================
    # 留痕
    # ========================================================

    def _record_failure(self, exc: BaseException, key: str) -> None:
        with self._lock:
            self._failures += 1
            count = self._failures

        if count & (count - 1):
            return

        logger.warning(
            "事件 sink %s 投递失败 %d 次（最近一次 key=%s）：%s: %s。"
            "事件是旁路，不影响运行本身。",
            self._name,
            count,
            key,
            type(exc).__name__,
            exc,
        )

    def _record_rejection(self, key: str, queue_size: int) -> None:
        with self._lock:
            self._rejected += 1
            count = self._rejected

        if count & (count - 1):
            return

        logger.warning(
            "事件 sink %s 队列积压，已丢弃 %d 条（最近一次 key=%s，"
            "队列上限 %d）。会一直丢到该分片排空为止。",
            self._name,
            count,
            key,
            queue_size,
        )
