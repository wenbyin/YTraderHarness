"""
进程内调度器实现。

一个 asyncio 队列 + 一个后台 worker 协程，全应用一套。

定位与代价
----------
这是**本期**的实现，不需要 Redis / Celery / 独立 worker 进程，
改动最小。代价必须说清楚：

    进程重启会丢掉队列里还没跑完的任务。

丢掉的任务不会有任何记录（run 行都还没建），因此表现为
"点了处理但没反应"。缓解方式是 reprocess —— 它是设计好的
重跑入口，不是权宜之计。

跨重启的可靠性需要把队列换成 outbox 表 + worker 轮询，
那时本类被替换，端口与调用方都不用动。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Final

from .dispatcher import (
    DispatchEnvelope,
    DispatchHandle,
    GraphDispatcher,
)

logger = logging.getLogger(__name__)

__all__ = ["InProcessGraphDispatcher"]

#: 队列容量。满了就拒绝并明确告知，而不是无限堆积把内存吃光。
DEFAULT_QUEUE_SIZE: Final[int] = 64

_Handler = Callable[[DispatchEnvelope], Awaitable[None]]

_STOP: Final[object] = object()


class InProcessGraphDispatcher(GraphDispatcher):

    def __init__(
            self,
            *,
            queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=queue_size
        )
        self._handlers: dict[str, _Handler] = {}
        self._worker: asyncio.Task | None = None
        self._closing = False

    # ========================================================
    # Registration
    # ========================================================

    def register(
            self,
            graph_name: str,
            handler: _Handler,
    ) -> None:
        """
        为某个图注册执行体。

        业务模块在组合根里调用一次。新增 pipeline 只需多注册一行，
        不需要新的 dispatcher。
        """

        if graph_name in self._handlers:
            raise ValueError(
                "Handler already registered for graph: "
                f"{graph_name}"
            )

        self._handlers[graph_name] = handler

    # ========================================================
    # Lifecycle
    # ========================================================

    async def start(self) -> None:
        """启动后台 worker。应用 lifespan 启动阶段调用。"""

        if self._worker is not None:
            return

        self._closing = False
        self._worker = asyncio.create_task(
            self._consume(),
            name="graph-dispatcher",
        )

        logger.info(
            "Graph dispatcher started; registered graphs: %s",
            sorted(self._handlers),
        )

    async def aclose(self) -> None:
        """
        停止接收并等待在跑的任务结束。

        先置 _closing 再入队哨兵：这样排在哨兵后面的任务不会再被
        接收，而哨兵之前已经在跑的任务会被 await 到。
        """

        worker = self._worker

        if worker is None:
            return

        self._closing = True
        self._worker = None

        await self._queue.put(_STOP)

        try:
            await worker

        except asyncio.CancelledError:
            pass

        except Exception:
            logger.exception(
                "Graph dispatcher worker terminated abnormally."
            )

    # ========================================================
    # Dispatch
    # ========================================================

    async def dispatch(
            self,
            envelope: DispatchEnvelope,
    ) -> DispatchHandle:

        if self._closing:
            return DispatchHandle(
                pipeline_run_id=envelope.pipeline_run_id,
                thread_id=envelope.thread_id,
                accepted=False,
                message="Dispatcher is shutting down.",
            )

        if envelope.graph_name not in self._handlers:
            return DispatchHandle(
                pipeline_run_id=envelope.pipeline_run_id,
                thread_id=envelope.thread_id,
                accepted=False,
                message=(
                    f"No handler registered for graph "
                    f"{envelope.graph_name!r}."
                ),
            )

        try:
            self._queue.put_nowait(envelope)

        except asyncio.QueueFull:
            # 明确拒绝而不是阻塞调用方 —— 调用方在 HTTP 请求里，
            # 阻塞会把它一起拖住。
            return DispatchHandle(
                pipeline_run_id=envelope.pipeline_run_id,
                thread_id=envelope.thread_id,
                accepted=False,
                message=(
                    "Dispatcher queue is full; retry later."
                ),
            )

        return DispatchHandle(
            pipeline_run_id=envelope.pipeline_run_id,
            thread_id=envelope.thread_id,
            accepted=True,
        )

    # ========================================================
    # Worker
    # ========================================================

    async def _consume(self) -> None:

        while True:
            envelope = await self._queue.get()

            if envelope is _STOP:
                self._queue.task_done()
                return

            try:
                handler = self._handlers[
                    envelope.graph_name
                ]

                await handler(envelope)

            except asyncio.CancelledError:
                raise

            except Exception:
                # 单次运行失败不能杀死 worker —— 否则后续所有任务
                # 都静默地不再执行，而队列看起来还是好的。
                # 运行失败本身已经记在 run 行上了。
                logger.exception(
                    "Graph run failed: graph=%s run_id=%s",
                    envelope.graph_name,
                    envelope.pipeline_run_id,
                )

            finally:
                self._queue.task_done()

    # ========================================================

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def registered_graphs(self) -> list[str]:
        return sorted(self._handlers)
