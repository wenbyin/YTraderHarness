"""
第一版的 TaskDispatcher 实现：**DB 轮询**（spec §6 允许）。

端口在 app/common/task/protocols.py::TaskDispatcher。业务代码只认那个
端口，因此将来换成 MQ Dispatcher，调用方一行都不用改。

投递物只有 task_id
-----------------
dispatch() 只传一个 UUID，不传 Task、不传参数、更不传图状态。原因不是
"消息队列怕大包"，而是**不能让同一份数据有两处副本**：一份在 messages
里、一份在 tasks 里，两份迟早不一致，而不一致时没有任何机制知道该信谁。
DB 是事实来源，投递只是"叫一下"（spec §6）。

两个循环，各管一段，缺一不可
--------------------------
    _consume()        消费 dispatch() 推进来的 task_id，交给 Worker
    _sweep_forever()  定时唤醒 + 兜底扫描

其中 _sweep_forever() 有两副面孔，取决于此刻有没有定时任务：

    没有定时任务  固定节拍（sweep_interval）地扫，纯兜底
    有定时任务    绑定最近的那一条，**精确睡到它那一刻**再醒

后者是"延迟任务该在什么时候跑"这件事的精度来源 —— 没有它，一条
30 秒后的定时任务要等到下一个节拍才可能被捞到，延迟误差等于节拍长度。
醒来之后还要核对绑定的 generation（防抖：被重排过的定时作废），
理由写在那个方法的 docstring 里。

其中兜底那一面为什么**不能省**：

    1. 延迟任务。execute_at 在未来的任务在提交时就被投递了一次，那次
       投递时它还没到点 —— 如果没人再叫它，它会一直躺在库里。
       本进程内的唤醒队列不带定时能力（那要另造一个 scheduler，
       而 spec §7 明确要求延迟执行由 Task Infrastructure 统一处理）。

    2. 丢失的投递。dispatch() 只是往内存队列里放一个 UUID。进程重启、
       队列满被丢弃、或者 post-commit 钩子还没跑就崩了 —— 这些投递都
       没了。它们在 task_outbox 里可能是已投递状态（比如 relay 标记完
       published 就崩了），因此 relay **不会**再补，只有这里能捞回来。

所以：relay 负责"第一次投递"，sweep 负责"该跑却没被跑的"，consume 负责
"投过来了就跑"。三者重叠是**有意的**，重叠的部分靠 Worker 的 claim CAS
去重 —— 重复投递不会导致重复执行。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from app.common.repository.factory import RepositoryFactory
from app.common.task.protocols import TaskDispatcher
from app.services.task.repository import TaskStore

from .worker import Worker

__all__ = ["DbPollingTaskDispatcher", "DEFAULT_QUEUE_SIZE"]

logger = logging.getLogger(__name__)

#: 进程内唤醒队列的容量。
#:
#: 有界是刻意的：无界队列在上游疯狂提交时会一直吃内存，直到 OOM ——
#: 而 OOM 会把**正在跑**的任务一起带走（它们已经 claim 了，要等租约过期
#: 才能回来）。满了就丢，丢了有 outbox 与 sweep 兜底，代价只是一点延迟。
DEFAULT_QUEUE_SIZE = 1024

#: 睡眠下限（秒）。防止"剩余时间已经算成 0 或负数"时忙等 —— 那种情况下
#: 顶部那一轮已经把它投出去了，不需要原地转圈。
_MIN_SLEEP = 0.05


@dataclass(frozen=True, slots=True)
class _TimerBinding:
    """
    本进程当前绑定的那份定时（一个闹钟）。

    它记录的是**睡之前看到的**样子。醒来后要拿它与库里的现状比对：
    generation 变了就说明这份定时已经被重排取代，本次唤醒作废。

    只活在一次循环往复之间，因此不落库、不进查询接口。
    """

    task_id: UUID

    #: 睡之前那条任务的代际。醒来后与库里的比对。
    generation: int

    #: 睡之前那条任务的执行时刻。只用于日志（"从哪个时间改到了哪个"）。
    execute_at: datetime

    #: 还差多少秒到点（算出来时的快照）。
    remaining: float


class DbPollingTaskDispatcher:
    """
    进程内的任务投递器：DB 轮询 + 内存唤醒队列。

    它**不是** Task 的事实来源 —— 那只在库里。本类丢一次投递、甚至整个
    进程崩掉，任务都不会丢（靠 task_outbox + sweep）。这是本实现与既有
    InProcessGraphDispatcher 最大的区别：那个的队列就是它的全部，
    进程一重启，排队中的任务全没了。
    """

    def __init__(
            self,
            *,
            worker: Worker,
            repositories: RepositoryFactory,
            concurrency: int = 4,
            sweep_interval: float = 15.0,
            batch_size: int = 100,
            queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        """
        Args:
            sweep_interval: **最长睡多久**（兜底节拍），不是固定轮询间隔。
                实际睡眠会被"下一条定时任务的时刻"缩短（精确唤醒），
                这个值只做上限 —— 它同时是"别的实例把时间往前改"这件事
                被本进程发现的最大延迟。调小更灵敏、更费查询。
        """

        self._worker = worker
        self._repositories = repositories
        self._concurrency = max(1, concurrency)
        self._sweep_interval = sweep_interval
        self._batch_size = batch_size

        self._queue: asyncio.Queue[UUID] = asyncio.Queue(
            maxsize=queue_size
        )

        #: 已经在队列里（或正在被消费）的 task_id。
        #:
        #: 只为了**减少无用功**：同一条任务被 post-commit 钩子、relay、
        #: sweep 各推一次是常态，重复入队会让 Worker 反复去 claim 同一条
        #: 并全部落空。正确性不依赖它（claim 是 CAS），它只是省几次查询。
        self._queued: set[UUID] = set()

        self._consumers: list[asyncio.Task] = []
        self._sweeper: asyncio.Task | None = None
        self._closing = False

    # ========================================================
    # 生命周期
    # ========================================================

    async def start(self) -> None:

        if self._consumers:
            return

        self._closing = False

        self._consumers = [
            asyncio.create_task(
                self._consume(index),
                name=f"task-worker-{index}",
            )
            for index in range(self._concurrency)
        ]

        self._sweeper = asyncio.create_task(
            self._sweep_forever(),
            name="task-sweeper",
        )

        logger.info(
            "[Task] 投递器已启动：%d 个消费者，扫描间隔 %.1fs",
            self._concurrency,
            self._sweep_interval,
        )

    async def aclose(self) -> None:
        """
        停掉所有循环。

        先置 _closing 再取消：消费循环在 CancelledError 里要能分辨
        "正在关闭"与"单条任务被取消"，否则关闭时每条在跑的任务都会被
        当成用户取消 —— 那会把它们的收尾逻辑走错。
        """

        self._closing = True

        for task in self._consumers:
            task.cancel()

        if self._sweeper is not None:
            self._sweeper.cancel()
            self._sweeper = None

        # return_exceptions：取消会让协程抛 CancelledError，
        # 不吞掉的话这里会中断，剩下的消费者就没人等了。
        await asyncio.gather(
            *self._consumers,
            return_exceptions=True,
        )

        self._consumers = []
        self._queued.clear()

        logger.info("[Task] 投递器已停止")

    # ========================================================
    # 投递
    # ========================================================

    async def dispatch(
            self,
            task_id: UUID,
    ) -> None:
        """
        让某个 Worker **尽快**去看看这条任务。

        ⚠️ 这是一个"叫一下"，不是"送达保证"：

            - 队列满         -> 丢掉，记一条 warning
            - 进程要关了     -> 丢掉
            - 进程随后崩了   -> 队列内容一起没了

        三种情况都不影响任务最终被执行 —— task_outbox 里那行还在，
        sweep 也会捞。所以这里**绝不抛异常**：它的调用点之一是
        ServiceFactory 的 post-commit 钩子，在那里抛会把一次已经成功的
        请求变成失败（任务其实已经落库了，用户却看到 500）。

        ⚠️ 也不要 await 它的返回值来推断"任务会跑"：投递成功与执行成功
        之间隔着 claim、解析、业务逻辑。
        """

        if self._closing:
            return

        if task_id in self._queued:
            return

        try:
            self._queue.put_nowait(task_id)
            self._queued.add(task_id)

        except asyncio.QueueFull:
            # 不抛、不重试：满了说明积压严重，而这个投递只是
            # "加速"而非保证。让它安静地降落，交给 outbox / sweep。
            logger.warning(
                "[Task] 唤醒队列已满（%d），放弃这次投递：%s。"
                "任务不会丢 —— task_outbox 与兜底扫描会补上。",
                self._queue.maxsize,
                task_id,
            )

    @property
    def queue_size(self) -> int:
        """当前排队条数（冒烟与运维观察用）。"""

        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return bool(self._consumers) and not self._closing

    # ========================================================
    # 消费
    # ========================================================

    async def _consume(
            self,
            index: int,
    ) -> None:

        while True:

            task_id = await self._queue.get()

            # 先摘掉标记再处理：处理期间同一条又被人投递的话，应当允许它
            # 重新入队（那次投递可能来自 relay，代表一个**新**的投递意图）。
            self._queued.discard(task_id)

            try:
                await self._worker.run(task_id)

            except asyncio.CancelledError:
                # 关闭流程或运行期取消。此时**不写任何状态**：
                # 取消的收尾（写 cancelled）由 TaskService.cancel 负责，
                # 这里再写一次只会与其竞争。
                raise

            except Exception:  # noqa: BLE001

                # 单个任务把消费者打死 = 并发度悄悄降一格，而没人会发现。
                # Worker 自己已经处理了执行失败，能走到这里的都是它没预料
                # 到的（比如写回时数据库断了）—— 记下来，继续下一条。
                logger.exception(
                    "[Task] 消费者 %d 处理 %s 时出现未预期的异常",
                    index,
                    task_id,
                )

    # ========================================================
    # 兜底扫描
    # ========================================================

    async def _sweep_forever(self) -> None:
        """
        兜底扫描 + **精确定时唤醒**（同一个循环）。

        没有定时任务时它是固定节拍的扫描；有定时任务时它会绑定下一条
        最近要跑的，并精确睡到那一刻 —— 而不是让人等到下一个节拍。

        ⚠️ 为什么要绑定 generation（防抖）
        --------------------------------
        延迟任务的执行时间可以被**反复重排**（典型是审批超时：每次有新
        动静就把截止时间往后推）。重排会让**上一个已经生成的定时失效**。

        所以每次睡之前先绑定 (task_id, generation, execute_at)：

            醒来 -> 读回库里的 generation
                    一致   -> 这次唤醒仍然有效，走正常的投递
                    不一致 -> 说明这条定时在睡眠期间被重排过，
                              本次唤醒**作废**，下一轮按新时刻重新绑定

        只比对 execute_at 不行：把时间重排到**同一个时刻**是合法的一次
        重排，那时前后相等，只有计数器能区分新旧定时。

        ⚠️ 硬保证不在这一层
        ----------------
        "不让旧定时提前执行"真正靠的是 TaskStore.claim 里的到期判断
        （execute_at <= now()）—— 即使这里判断失误、或者投递被重复，
        一条还没到点的任务也 claim 不下来。这一层负责的是**精度**：
        到点就醒、被重排就立刻改绑，而不是空转到下一个节拍。
        """

        while True:

            binding: _TimerBinding | None = None
            delay = self._sweep_interval

            try:
                # 先扫一轮再睡，而不是反过来。
                #
                # 这一轮就是"服务重启恢复"里那个**恢复**：上一次进程崩时
                # 遗留在库里的到期任务（包括被 post-commit 钩子漏掉、
                # 以及租约过期的）在这一刻被捞起来重新投递。若先 sleep，
                # 重启后要干等一个扫描周期才有任务开始跑。
                await self.sweep_once()

                binding = await self._bind_next_timer()

                if binding is not None and binding.remaining is not None:
                    # 睡到那一刻，但不超过兜底节拍 —— 上限是必要的：
                    # 别的实例**把时间往前改**时，本进程只有在下一轮重新
                    # 绑定时才会发现，节拍就是这个发现延迟的上界。
                    delay = min(
                        self._sweep_interval,
                        max(binding.remaining, _MIN_SLEEP),
                    )

            except asyncio.CancelledError:
                raise

            except Exception:  # noqa: BLE001

                # 扫描失败绝不能终止循环：数据库抖一下就让兜底机制永久
                # 停摆，是最难发现的一类故障（一切看起来都正常，只是
                # 延迟任务再也不跑了）。
                logger.exception("[Task] 兜底扫描出错，将在下轮重试")

            try:
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                raise

            # 醒来：核对绑定的那份定时还有效吗。
            #
            # 作废时**不做任何事**就回到循环顶部 —— 顶部会重新绑定，
            # 而重新绑定读的是库里的 execute_at，也就是那个新时刻。
            # "不一致就继续 sleep 到新的执行时间"就落在这一步。
            if binding is not None:
                await self._check_binding(binding)

    async def _bind_next_timer(self) -> "_TimerBinding | None":
        """
        绑定下一个闹钟：取最近一条还没到点的可跑任务。

        读的是**库**，不是本进程的内存 —— 因此别的实例刚提交或刚重排的
        定时也看得见（前提是它在我们的下一次重新绑定之前落库，节拍就是
        那个上界）。
        """

        async with self._repositories.scope(begin=False) as repos:

            upcoming = await repos.create(TaskStore).next_scheduled()

        if upcoming is None or upcoming.execute_at is None:
            return None

        remaining = (
            upcoming.execute_at - datetime.now(timezone.utc)
        ).total_seconds()

        return _TimerBinding(
            task_id=upcoming.task_id,
            generation=upcoming.generation,
            execute_at=upcoming.execute_at,
            remaining=remaining,
        )

    async def _check_binding(
            self,
            binding: "_TimerBinding",
    ) -> bool:
        """
        醒来后核对：我绑定的那份定时还是当前有效的那个吗？

        Returns:
            True  仍然一致（顶部那一轮已经/即将按它投递）。
            False 已被取代 —— 什么都不做，顶部会按新时刻重新绑定。
        """

        async with self._repositories.scope(begin=False) as repos:

            current = await repos.create(TaskStore).get(binding.task_id)

        if current is None:
            # 任务没了（本系统不删任务，真出现只能是人为清理）。
            # 当"作废"处理：让顶部重新绑定，别在这里猜。
            logger.info(
                "[Task] 定时 %s 已不存在，本次唤醒作废", binding.task_id
            )
            return False

        if current.generation != binding.generation:

            logger.info(
                "[Task] 定时 %s 已被重排（generation %d -> %d，"
                "执行时间 %s -> %s），本次唤醒作废，按新时刻重新绑定",
                binding.task_id,
                binding.generation,
                current.generation,
                binding.execute_at.isoformat(),
                (
                    current.execute_at.isoformat()
                    if current.execute_at is not None
                    else "立即"
                ),
            )

            return False

        # 一致。注意**不要**在这里投递：顶部那一轮的 sweep_once 会按
        # "到期与否"统一处理，比在这里特判一条更不容易漏（比如它刚好
        # 被别的实例抢走了）。
        return True

    async def sweep_once(self) -> tuple[int, int]:
        """
        做一轮兜底扫描。返回 (投递的到期任务数, 收回的过期租约数)。

        独立成公开方法而不是藏在循环里：冒烟要能**只跑一轮**并断言结果，
        不必等 15 秒的定时器，也不必去猜轮询节奏。
        """

        delivered = 0
        reclaimed = 0

        async with self._repositories.scope() as repos:

            store = repos.create(TaskStore)

            # ⚠️ 先收再取，顺序不能反。
            #
            # 收回会把任务从 running 改成 retryable（或 failed），也就是
            # 把它放回"可 claim"集合。若先取到期列表，那些刚收回的任务
            # 这一轮就漏掉了，要白等一个扫描周期。
            reclaimed_ids = await store.reclaim_expired(
                limit=self._batch_size
            )

            reclaimed = len(reclaimed_ids)

            if reclaimed:
                logger.warning(
                    "[Task] 收回 %d 条租约过期的任务"
                    "（执行它们的 Worker 可能已经崩溃）",
                    reclaimed,
                )

            due = await store.list_due(limit=self._batch_size)

        for task_id in due:
            await self.dispatch(task_id)
            delivered += 1

        return delivered, reclaimed
