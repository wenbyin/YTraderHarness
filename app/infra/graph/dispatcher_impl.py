"""
进程内调度器实现。

一个 asyncio 队列 + 一个后台 worker 协程，全应用一套。

两个闸门，各管各的
------------------
    queue_size        **在等**的上限，满了拒绝（DEFAULT_QUEUE_SIZE）
    max_concurrency   **在跑**的上限，超出的留在队列里等
                      （DEFAULT_MAX_CONCURRENCY）

分开是必须的：只有一个"容量"时，要么不能限制并发（来者不拒全开跑），
要么把正在跑的也算进容量（于是拒绝的理由说不清是"排队太深"还是
"跑得太多"）。两个数字合起来的在途上限是 queue_size + max_concurrency。

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
import os
from collections.abc import Awaitable, Callable
from typing import Final

from uuid import UUID

from .cancellation import get_run_cancellation_registry
from .dispatcher import (
    DispatchEnvelope,
    DispatchHandle,
    GraphDispatcher,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_QUEUE_SIZE",
    "InProcessGraphDispatcher",
    "build_graph_dispatcher",
    "get_graph_dispatcher",
    "reset_graph_dispatcher",
]

#: 队列容量（**在等**的信封数）。满了就拒绝并明确告知，而不是无限
#: 堆积把内存吃光。在跑的那些不算在里面 —— 见 DEFAULT_MAX_CONCURRENCY。
DEFAULT_QUEUE_SIZE: Final[int] = int(
    os.environ.get("GRAPH_DISPATCHER_QUEUE_SIZE", "4096")
)

#: 同时**在跑**的图数量上限。超出的留在队列里等，不拒绝。
#:
#: 为什么是 32 这个量级，而不是"越大越好"
# ------------------------------------
#: 每个 run 都要吃 CPU：文档流水线的 embedding 阶段是本地前向，实测约
#: 26 ms/条，**单个 run 就能把 torch 的线程池吃满**（见
#: embedder/bge_small_zh_local.py 与 129 条 chunk 的实测）。所以这个
#: 数字不是吞吐旋钮 —— 并行度提上去，多个 run 只是分同一批核，各自
#: 变慢，总吞吐基本不变。
#:
#: 它的作用是让若干个 run **交替推进**：一个在算向量时，另一个正好在
#: 等 Milvus 写入 / 等模型调用，把空隙填上。调到接近核数只会让每个
#: run 的延迟都变差。
#:
#: 默认 768。Windows 上 `select()` 有 512 socket 上限，真实压测时用
#: GRAPH_DISPATCHER_MAX_CONCURRENCY 压到 128 留余量；Linux 上可以
#: 提到 700+（见 scripts/run_server_stress_linux.sh）。
#: 文档流水线的 embedding 是本地 CPU 前向，和 Agent 对话共用这个
#: dispatcher；跑重文档管线时可以临时调回较小值。
DEFAULT_MAX_CONCURRENCY: Final[int] = int(
    os.environ.get("GRAPH_DISPATCHER_MAX_CONCURRENCY", "768")
)

_Handler = Callable[[DispatchEnvelope], Awaitable[None]]

_STOP: Final[object] = object()


class InProcessGraphDispatcher(GraphDispatcher):

    def __init__(
            self,
            *,
            queue_size: int = DEFAULT_QUEUE_SIZE,
            max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if max_concurrency < 1:
            # 0 是个特别隐蔽的取值：队列照收（dispatch 返回 accepted），
            # 但 worker 永远停在等信号量上，一个都不跑，也没有任何报错。
            raise ValueError(
                f"max_concurrency 必须 >= 1，收到 {max_concurrency} —— "
                "0 会让所有派发被受理却永不执行。"
            )

        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=queue_size
        )

        self._max_concurrency = max_concurrency

        #: 在跑并发度。**先占位、再取信封**（见 _consume）—— 顺序反过来
        #: 会让信封离开队列却仍在等位，queue_size 报的积压量就比真实的小，
        #: 关停时还会凭空丢一个已经取出来、还没排上队的信封。
        self._slots = asyncio.Semaphore(max_concurrency)

        self._handlers: dict[str, _Handler] = {}
        self._worker: asyncio.Task | None = None
        self._closing = False

        #: `run_id -> 正在跑它的那个**子任务**`。同时最多 max_concurrency 条。
        #:
        #: ⚠️ 取消必须打在子任务上，不能打在 worker 上
        # ------------------------------------------
        # worker 只负责取信封、起子任务。在它身上调 `cancel()` 会把循环
        # 本身也干掉，此后所有任务都静默地不再执行（而队列看起来还是好的）。
        # 所以每次运行起一个独立 task，取消只打在那一个上。
        self._running: dict[UUID, asyncio.Task] = {}

        #: 已派发但还没跑完的 run_id。用来回答"这个 run 我认得吗" ——
        #: 排队中的那些也在里面（它们还没有 task 可取消，只能先登记，
        #: 等 handler 开工时自己看）。
        self._known: set[UUID] = set()

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
            "Graph dispatcher started; registered graphs: %s, "
            "queue_size=%d, max_concurrency=%d",
            sorted(self._handlers),
            self._queue.maxsize,
            self._max_concurrency,
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

        self._known.add(envelope.pipeline_run_id)

        return DispatchHandle(
            pipeline_run_id=envelope.pipeline_run_id,
            thread_id=envelope.thread_id,
            accepted=True,
        )

    # ========================================================
    # Cancel
    # ========================================================

    async def cancel(self, pipeline_run_id: UUID) -> bool:
        """
        请求取消一次运行。见 `GraphDispatcher.cancel` 的契约。

        两条路，都是一个动作：**先登记、再动手**（或先登记、等 handler 自己看）。

            正在跑    登记 + `task.cancel()`
            还在队列   只登记 —— 它还没有 task 可取消。handler 拿到
                       envelope 后第一件事就是来看这个登记，看到就立刻
                       收尾（图一次都不跑）

        顺序不能反：先 cancel 后登记的话，handler 可能在登记之前就已经
        读过了，于是它按"没人取消过"收尾 —— 记成 failed 而不是 cancelled，
        而且 rollback 也不会做。
        """

        registry = get_run_cancellation_registry()

        task = self._running.get(pipeline_run_id)

        if task is not None:
            registry.request(pipeline_run_id)
            task.cancel()
            return True

        if pipeline_run_id in self._known:
            registry.request(pipeline_run_id)
            return True

        # 不认得：跑完了、或者从来不是这个调度器派的。
        # 不登记 —— 登记一个永远不会被读的 id 就是一条内存泄漏。
        return False

    # ========================================================
    # Liveness
    # ========================================================

    async def is_active(self, pipeline_run_id: UUID) -> bool:
        """
        见 `GraphDispatcher.is_active` 的契约。

        ⚠️ `_running` 与 `_known` **两个都要看**，不能只看前者
        --------------------------------------------------
        `_running` 只在 handler **开工那一刻**才登记（见 _consume 里那句
        `self._running[run_id] = task`），而 `_known` 在派发被**受理**时就
        登记了（见 dispatch 末尾）。于是"排队中"的 run 只在前者里没有、
        在后者里有 —— 那是"建完运行立刻打开 SSE"最常见的时序，只看
        `_running` 会把它当场判死。

        ⚠️ 并发放开之后这个窗口**更宽了**：在跑的占满 10 格时，后面
        那些会一直待在队列里，`_known` 里有、`_running` 里没有的时间
        可能长达几分钟。只看 `_running` 会把这些流全部关掉。

        两者都在 `_run` 的 finally 里摘掉，所以跑完的 run 在这里
        返回 False。

        这里没有 await，因此读两个集合是原子的一个回合（单线程事件循环，
        同 thread_lock.py 的论证）。
        """

        return (
            pipeline_run_id in self._running
            or pipeline_run_id in self._known
        )

    # ========================================================
    # Worker
    # ========================================================

    async def _consume(self) -> None:
        """
        取信封、起子任务。**同时最多 `max_concurrency` 条在跑**。

        为什么用信号量封顶、而不是"取一个起一个"
        --------------------------------------
        上限的作用是把积压留在**队列**里 —— queue_size 才是那个会拒绝
        人的闸门。来者不拒地全起起来，就是拿队列容量当摆设：拒绝的理由
        变成了"同时在跑太多"，而调用方对此无能为力（它只想排队）。

        信号量在 `queue.get()` **之前**获取：先占位、再取信封，队列深度
        才等于真实积压量。反过来（先取再等位）信封已经离开队列却还没开工，
        `queue_size` 看着是空的，实际有任务在排；关停时那个信封还会
        凭空消失。
        """

        while True:

            await self._slots.acquire()

            try:
                envelope = await self._queue.get()

            except BaseException:
                # 被取消（应用关闭）时把位置还回去。进程都要退了，这一格
                # 其实无所谓 —— 但保持"占位 <=> 有任务在跑"这条不变量，
                # 读的人不必去脑补例外。
                self._slots.release()
                raise

            if envelope is _STOP:
                self._slots.release()
                self._queue.task_done()
                await self._drain()
                return

            run_id = envelope.pipeline_run_id

            # ⚠️ 起一个**子任务**，但**不 await** —— 并发由上面那个信号量
            # 封顶。取消仍然必须打在这一次运行上（理由见 `_running`），
            # 不能打在 worker 身上。
            #
            # create_task 与下面那行赋值之间没有 await，所以不存在
            # "任务已经跑完、_running 里却还留着"的窗口。
            task = asyncio.create_task(
                self._run(envelope),
                name=f"graph-run:{run_id}",
            )

            self._running[run_id] = task

    async def _run(self, envelope: DispatchEnvelope) -> None:
        """
        跑一个信封，并在收尾时把占用的并发位还回去。

        每个信封恰好走这里一次：`_consume` 起任务、本方法收尾。
        释放信号量放在 finally —— 正常结束、抛异常、被取消三条路都必须
        还，漏掉一条就会让并发度**永久**少一格，而且没有任何报错。

        为什么不让 worker 直接 await handler
        ------------------------------------
        那样并发度就恒为 1。这里把 handler 包进一个任务里，`_running`
        登记的就是这个任务 —— `cancel()` 打的是它，worker 不受影响。
        """

        run_id = envelope.pipeline_run_id

        try:
            handler = self._handlers[
                envelope.graph_name
            ]

            try:
                await handler(envelope)

            except asyncio.CancelledError:
                # ⚠️ 这里**不能无条件 re-raise**
                # ----------------------------------
                # 被取消的是**这一次运行**，而 worker 与其它在跑的 run
                # 都还活着。无条件抛出去会让这次运行以"被取消"收尾 ——
                # 那不是问题所在；问题在于用登记表区分两种取消：
                # 这一次运行被取消（有人登记过），与整个进程在关闭
                # （没人登记过）。
                if not get_run_cancellation_registry().is_requested(
                        run_id
                ):
                    raise

                logger.info(
                    "Graph run cancelled: graph=%s run_id=%s",
                    envelope.graph_name,
                    run_id,
                )

        except Exception:
            # 单次运行失败不能影响别人 —— 否则后续所有任务都静默地
            # 不再执行，而队列看起来还是好的。运行失败本身已经记在
            # run 行上了。
            logger.exception(
                "Graph run failed: graph=%s run_id=%s",
                envelope.graph_name,
                run_id,
            )

        finally:
            self._slots.release()

            # `_known` 在这里才摘掉：那之前 `cancel()` 都要能认出这个
            # run（包括"已经被取消、正在收尾"的那一小段时间）。
            self._running.pop(run_id, None)
            self._known.discard(run_id)
            self._queue.task_done()

    async def _drain(self) -> None:
        """
        等在跑的那些收尾。

        `aclose` 的契约是"停止接收，但不打断在跑的"（见 `aclose` 的
        docstring）—— 并发之后这件事要显式做一遍，worker 退出时手里
        可能还有最多 max_concurrency 条在跑。

        循环是因为 `_running` 在 gather 期间会被各个任务的 finally 改写；
        取一次快照、等它们，再看还有没有剩的。
        """

        while self._running:
            await asyncio.gather(
                *list(self._running.values()),
                return_exceptions=True,
            )

    # ========================================================

    @property
    def queue_size(self) -> int:
        """队列里**还在等**的信封数。不含在跑的（见 running_count）。"""

        return self._queue.qsize()

    @property
    def running_count(self) -> int:
        """正在跑的 run 数。上限是 max_concurrency。"""

        return len(self._running)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def registered_graphs(self) -> list[str]:
        return sorted(self._handlers)


# ============================================================
# 进程级单例
# ============================================================
#
# 谁需要它：事件流订阅端的**探活探针**（见 app/agent/event/run_stream.py）。
# 它要问"本进程手里还有这个 run 吗"，而那个判断只有调度器答得出来。
#
# ⚠️ 为什么不做成构造参数一路传下去
# ------------------------------
# 探针是在**装配事件流读取端**的时候造出来的，而那时调度器还没造 ——
# 组合根里的先后顺序是"先 Redis 与读取端，后注册图与 dispatcher"
# （app/main.py）。把顺序倒过来只是把耦合换个方向，而且读取端本来也
# 不该依赖调度器。
#
# 所以这里留一个进程级访问器：探针闭包在**被调用时**才去取，那时
# lifespan 早就装配完了。与读取端自己的 build_/get_/reset_ 三件套同形。
#
# 未装配时返回 None —— 探针据此退回"只信数据库状态"，而不是把
# "问不到"当成"没有这个 run"（那会把所有流都关掉）。
_dispatcher: InProcessGraphDispatcher | None = None


def build_graph_dispatcher(
        dispatcher: InProcessGraphDispatcher | None,
) -> None:
    """装配进程级调度器引用。**由组合根调用一次**（app/main.py 的 lifespan）。"""

    global _dispatcher

    _dispatcher = dispatcher


def get_graph_dispatcher() -> InProcessGraphDispatcher | None:
    """取进程级调度器。未装配时返回 None。"""

    return _dispatcher


def reset_graph_dispatcher() -> None:
    """清掉单例（测试用）。"""

    global _dispatcher

    _dispatcher = None
