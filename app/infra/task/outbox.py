"""
OutboxRelay —— 把 task_outbox 里"该投递但还没投出去"的记录交出去。

它兑现的是 spec §5 的那句保证：

    DB commit 成功
    但 MQ publish 失败
    -> 任务不会永久丢失

做法是让 submit 把"要投递"与"任务本身"写进**同一个事务**（见
app/services/task/application/task_service_impl.py），投递放到提交之后。
于是"投了没有"这件事有了一行持久记录，投递失败就只是那行还停在
pending，本 relay 下一轮再试。

⚠️ 它管的是**第一次投递**，不是"保证被执行"
------------------------------------------
这两件事很容易混。relay 只负责"让某个 Worker 去看看这条任务"：

    投递成功 -> 把 outbox 行标成 published，**从此不再管它**

如果那次投递之后 Worker 没跑成（进程重启丢了内存队列、队列满被丢弃、
或标记 published 之后立刻崩了），outbox 里那行已经 published，本 relay
**不会**再补投。补那是 DbPollingTaskDispatcher 的兜底扫描负责的
（它按 tasks 表的 status + execute_at 捞，与 outbox 无关）。

分工：

    OutboxRelay   第一次投递（"这条任务还没被叫过"）
    Sweep         该跑却没跑的（"叫过了但它没跑成"）

两者都会导致重复投递，而重复投递是安全的 —— Worker 的 claim 是 CAS。
"""

from __future__ import annotations

import asyncio
import logging

from app.common.repository.factory import RepositoryFactory
from app.common.task.protocols import TaskDispatcher
from app.services.task.repository import TaskOutboxStore

__all__ = ["OutboxRelay", "DEFAULT_MAX_PUBLISH_RETRIES"]

logger = logging.getLogger(__name__)

#: 单条投递记录的重试上限。到顶转 failed 并告警 ——
#: 一直留在 pending 的话，下一轮照样会捞它，看起来一切正常，
#: 而它其实已经失败了几十次。
DEFAULT_MAX_PUBLISH_RETRIES = 10


class OutboxRelay:
    """
    定期把未投递的 outbox 记录交给 TaskDispatcher。

    与 Dispatcher 分成两个组件（而不是让 Dispatcher 自己去读 outbox），
    是因为它们回答的是两个问题：

        OutboxRelay  "还有哪条任务从没被叫过？"
        Dispatcher   "哪条任务该跑？（以及怎么跑）"

    将来换 MQ 时，变的只有 relay 交出去的目标（改成 publish 到 broker），
    而"哪条该跑"的判断完全不动。反过来把两者揉在一起的话，换 MQ 就得
    连任务调度逻辑一起改。
    """

    def __init__(
            self,
            *,
            dispatcher: TaskDispatcher,
            repositories: RepositoryFactory,
            interval: float = 10.0,
            batch_size: int = 100,
            max_retry_count: int = DEFAULT_MAX_PUBLISH_RETRIES,
    ) -> None:

        self._dispatcher = dispatcher
        self._repositories = repositories
        self._interval = interval
        self._batch_size = batch_size
        self._max_retry_count = max_retry_count

        self._loop: asyncio.Task | None = None
        self._closing = False

    # ========================================================
    # 生命周期
    # ========================================================

    async def start(self) -> None:

        if self._loop is not None:
            return

        self._closing = False

        self._loop = asyncio.create_task(
            self._run(),
            name="task-outbox-relay",
        )

        logger.info(
            "[Task] outbox relay 已启动：间隔 %.1fs，投递重试上限 %d",
            self._interval,
            self._max_retry_count,
        )

    async def aclose(self) -> None:

        self._closing = True

        if self._loop is not None:
            self._loop.cancel()
            await asyncio.gather(self._loop, return_exceptions=True)
            self._loop = None

        logger.info("[Task] outbox relay 已停止")

    @property
    def running(self) -> bool:
        return self._loop is not None and not self._closing

    # ========================================================
    # 循环
    # ========================================================

    async def _run(self) -> None:

        while True:

            try:
                # 先跑一轮再睡：进程刚起来时库里最可能积着未投递的记录
                # （上一次崩在投递之前），没有理由让它们再等一个周期。
                await self.relay_once()

                await asyncio.sleep(self._interval)

            except asyncio.CancelledError:
                raise

            except Exception:  # noqa: BLE001

                # 一轮出错不能让 relay 停摆：数据库抖一下就让"投递保证"
                # 永久失效，是最糟的失败方式 —— 没有任何症状，只是任务
                # 再也不跑了。
                logger.exception("[Task] outbox relay 一轮出错，将在下轮重试")

    async def relay_once(self) -> int:
        """
        做一轮投递，返回成功投出的条数。

        独立成公开方法，理由与 Dispatcher.sweep_once 一样：
        冒烟要能只跑一轮并断言结果，不必依赖定时器。
        """

        # 只读事务读一批出来。用 begin=False：读不需要写事务，
        # 而且紧接着就要做投递（外部动作），不该把事务开着跨过去。
        async with self._repositories.scope(begin=False) as repos:

            pending = await repos.create(
                TaskOutboxStore
            ).list_unpublished(
                limit=self._batch_size,
                max_retry_count=self._max_retry_count,
            )

        delivered = 0

        for record in pending:

            # ⚠️ 每条各自开事务，而不是整批一个。
            #
            # 整批一个事务的话，任何一条标记失败都会让整个事务进入
            # aborted 状态，后面所有条都写不进去 —— 于是"一条坏记录"
            # 会把整批投递卡死，而那些本来能投出去的记录要等下一轮。
            #
            # 另外：投递（dispatch）必须在**事务外**调用。它是外部动作，
            # 放进事务里就会出现"投出去了但事务回滚了"—— 投递记录回到
            # pending，下一轮再投一次（重复投递，靠 claim 去重，
            # 但白白多跑一次）。
            try:
                await self._dispatcher.dispatch(record.task_id)

            except Exception as exc:  # noqa: BLE001

                await self._record_failure(record.outbox_id, exc)

                continue

            async with self._repositories.scope() as repos:

                await repos.create(TaskOutboxStore).mark_published(
                    record.outbox_id
                )

            delivered += 1

        if delivered:
            logger.debug("[Task] outbox relay 本轮投出 %d 条", delivered)

        return delivered

    # ========================================================
    # internals
    # ========================================================

    async def _record_failure(
            self,
            outbox_id,
            exc: BaseException,
    ) -> None:
        """记下投递失败。

        失败原因必须落库（last_error）：它是"本该被执行、却一直没被交出去"
        这件事**唯一**的痕迹。只写日志的话，日志滚动之后就再也查不出
        "这条任务为什么从来没跑过"。
        """

        message = f"{type(exc).__name__}: {exc}"

        logger.warning(
            "[Task] 投递失败（outbox=%s）：%s",
            outbox_id,
            message,
        )

        async with self._repositories.scope() as repos:

            await repos.create(TaskOutboxStore).mark_publish_failed(
                outbox_id,
                error=message,
                max_retry_count=self._max_retry_count,
            )
