"""
TaskService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback —— 与其余模块
一致。submit 能保证 tasks + task_outbox 同事务，靠的正是**不提交**：
两次写入落在 ServiceFactory.scope() 已经开好的同一个事务里，
谁都不提前提交，于是不可能出现"任务写了、投递意图没写"。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext
from app.common.task.exceptions import TaskStateError
from app.common.task.models import Task
from app.common.task.protocols import (
    TaskDispatcher,
    TaskHandlerRegistry,
)
from app.common.task.types import TaskStatus

from ..repository.task import TaskStore
from ..repository.task_outbox import TaskOutboxStore
from .task_service import TaskService

__all__ = ["TaskServiceImpl"]


class TaskServiceImpl(TaskService):

    def __init__(
            self,
            *,
            task_store: TaskStore,
            task_outbox_store: TaskOutboxStore,
            handler_registry: TaskHandlerRegistry,
            task_dispatcher: TaskDispatcher | None = None,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._tasks = task_store
        self._outbox = task_outbox_store
        self._registry = handler_registry

        #: 可选。没有它也能工作 —— 投递由 outbox relay 兜底
        #: （infra/task/outbox.py），只是延迟任务要等下一轮轮询。
        #: 有它时，提交会在**事务提交之后**立刻推一次，省掉这段等待。
        #:
        #: ⚠️ 参数名是 task_dispatcher，不是 dispatcher
        #: ------------------------------------------
        #: ServiceFactory 按**参数名**优先匹配池子，而池子里的 "dispatcher"
        #: 已经被 InProcessGraphDispatcher 占着（Agent 运行那条链路）。
        #: 若这里叫 dispatcher，注入进来的会是那个图调度器 —— 它虽然也有
        #: 个 dispatch()，但收的是 DispatchEnvelope 而不是 task_id，
        #: 于是失败点跑到"第一次提交任务"时，报的是一句令人费解的
        #: 签名不匹配。换个名字是唯一稳妥的做法：按名字匹配永远优先于
        #: 按类型匹配（见 ServiceFactory._build_kwargs）。
        self._dispatcher = task_dispatcher

    # ========================================================
    # 提交与生命周期
    # ========================================================

    async def submit(
            self,
            task: Task,
    ) -> UUID:

        # -----------------------------------------------------
        # 1. handler 与 params 在**提交这一刻**校验
        # -----------------------------------------------------
        #
        # 不校验的话，错误要等 Worker 从库里把它读出来才发现 ——
        # 那已经过了一次投递、一个事务、可能还有一段轮询等待。
        # 提交方看到的将是"任务提交成功了，但五分钟后失败了"，
        # 而真正的原因（handler 拼错 / 参数类传错）在两跳之外。
        definition = self._registry.resolve(task.handler)

        definition.parse_params(task)

        # -----------------------------------------------------
        # 2. 补齐 lineage
        # -----------------------------------------------------
        #
        # 只有这里知道"这条任务是不是链条起点"：调用方给的 Task 可能
        # 来自 Handler 的 child()（已带 lineage），也可能来自一个 HTTP
        # 请求（什么都没有）。
        task.seal_root()

        # -----------------------------------------------------
        # 3. 落库（两个仓储共享同一个 session，因此同事务）
        # -----------------------------------------------------
        saved = await self._tasks.add(task)

        # ⚠️ 只有**真的新建了**才登记投递意图。
        #
        # 幂等命中时 saved 是既有的那一条，它第一次提交时已经登记过；
        # 再插一行就是一次多余的投递，而且每次重复提交都会多一行。
        if saved.task_id == task.task_id:
            await self._outbox.add(saved.task_id)

        # -----------------------------------------------------
        # 4. 提交后投递
        # -----------------------------------------------------
        #
        # 幂等命中也登记钩子：调用方这次的心意是"我要它跑"，
        # 而既有那条可能正卡在"投出去了但没人跑"（比如 relay 标记
        # published 之后进程被杀）。再推一次是廉价的，且 Worker 的
        # claim 是 CAS，重复投递不会导致重复执行。
        #
        # ⚠️ 这是**尽力而为**的加速，不是保证：钩子在事务提交后才跑，
        # 进程死在这里的话投递就丢了 —— 而 task_outbox 里那行还在，
        # relay 会补投。保证在 outbox，不在这里。
        self._schedule_dispatch(saved.task_id)

        return saved.task_id

    async def cancel(
            self,
            task_id: UUID,
    ) -> TaskStatus:

        current = await self._require(task_id)

        # 已经结束了就什么都不做，如实回当前状态。
        # 报 409 会逼调用方先查一次再取消，而那中间仍有竞态 ——
        # 结果是一个必然偶发、无法复现的冲突错误。
        if current.status.is_terminal:
            return current.status

        await self._tasks.mark_cancelled(task_id)

        # 回读一次而不是直接返回 CANCELLED：状态可能刚好被执行者写成了
        # 终态（并发取消），那时真实的答案是"它跑完了"，不是"取消了"。
        refreshed = await self._tasks.get(task_id)

        return (
            current.status
            if refreshed is None
            else refreshed.status
        )

    async def get(
            self,
            task_id: UUID,
    ) -> Task:

        return await self._require(task_id)

    async def retry(
            self,
            task_id: UUID,
    ) -> UUID:

        current = await self._require(task_id)

        revived = await self._tasks.schedule_retry(task_id)

        if revived is None:
            # 走到这里说明状态在这几毫秒里变了，或者本来就不允许。
            # 用**刚读到的** current.status 报错，便于定位。
            raise TaskStateError(
                f"任务当前是 {current.status.value}，不能重试。"
                "只有 failed / cancelled 的任务可以人工重跑："
                "正在跑或排队的重跑会与执行中的那次冲突，"
                "已成功的重跑会产生第二次外部副作用（需要重跑请提交新任务）。"
            )

        # 重跑同样要投递，否则它回到 pending 之后只能等下轮轮询 ——
        # 而人工重跑恰恰是"我现在就想让它再跑一次"。
        self._schedule_dispatch(revived.task_id)

        return revived.task_id

    async def invoke(
            self,
            task_id: UUID,
    ) -> Task:

        current = await self._require(task_id)

        # 复用 reschedule 而不是新写一条 SQL：它已经保证了
        # "只在可 claim 时生效 + generation 自增"，而这两件事对
        # "立即触发"一条定时任务同样是必须的 —— 少了 generation +1，
        # 已经睡在原来那个时刻的定时器醒来时会认不出自己已被取代。
        updated = await self._tasks.reschedule(
            task_id,
            execute_at=None,
        )

        if updated is None:
            # 与 retry / reschedule 同一套路：用**刚读到的**状态报错。
            raise TaskStateError(
                f"任务当前是 {current.status.value}，不能立即触发。"
                "只有还没开始跑的（pending / retryable）任务可以提前触发："
                "正在跑的任务马上就会写回结果；已终结的再触发会产生"
                "第二次外部副作用（失败/取消的重跑请用 retry）。"
            )

        # execute_at 已经是 null，这条路必然是"现在就该跑"，
        # 无条件推一次，不必像 reschedule 那样再判一次时间。
        self._schedule_dispatch(updated.task_id)

        return updated

    async def reschedule(
            self,
            task_id: UUID,
            *,
            execute_at: datetime | None,
    ) -> Task:

        current = await self._require(task_id)

        updated = await self._tasks.reschedule(
            task_id,
            execute_at=execute_at,
        )

        if updated is None:
            # 与 retry 同一个套路：用**刚读到的**状态报错，便于定位。
            # 走到这里通常是因为它已经在跑（或者在读与写之间跑掉了）。
            raise TaskStateError(
                f"任务当前是 {current.status.value}，不能重排执行时间。"
                "只有还在排队（pending / retryable）的任务可以改时间："
                "正在跑的任务马上就会写回结果，改时间没有意义；"
                "已终结的改了会让「已完成」变成「还会再跑」。"
            )

        # ⚠️ 只有"新时间已经到点"才立刻投递一次。
        #
        # 未来的时间由定时器负责（它会绑定这条任务并精确睡到那一刻），
        # 现在推一次是白推 —— Worker 会去 claim，而 claim 的到期判断会
        # 拒掉它（还没到点），结果是白跑一次数据库。
        #
        # 但"重排到一个已经过去的时刻"（例如把截止时间直接提前到此刻、
        # 或提前到一个过去的时间点）必须立刻动起来，否则要等定时器下一
        # 轮重新绑定才发现 —— 那是最多一个兜底节拍的延迟，而调用方的
        # 意图恰恰是"现在就按新时间办"。
        if execute_at is None or execute_at <= datetime.now(timezone.utc):
            self._schedule_dispatch(updated.task_id)

        return updated

    # ========================================================
    # 查询
    # ========================================================

    async def list_tasks(
            self,
            *,
            status: TaskStatus | None = None,
            handler: str | None = None,
            root_task_id: UUID | None = None,
            parent_task_id: UUID | None = None,
            limit: int = 50,
            offset: int = 0,
    ) -> tuple[list[Task], int]:

        items = await self._tasks.list(
            status=status,
            handler=handler,
            root_task_id=root_task_id,
            parent_task_id=parent_task_id,
            limit=limit,
            offset=offset,
        )

        total = await self._tasks.count(
            status=status,
            handler=handler,
            root_task_id=root_task_id,
            parent_task_id=parent_task_id,
        )

        return items, total

    async def get_lineage(
            self,
            root_task_id: UUID,
    ) -> list[Task]:

        return await self._tasks.list_lineage(root_task_id)

    # ========================================================
    # internals
    # ========================================================

    async def _require(
            self,
            task_id: UUID,
    ) -> Task:

        task = await self._tasks.get(task_id)

        if task is None:
            raise NotFoundError(f"任务不存在：{task_id}")

        return task

    def _schedule_dispatch(
            self,
            task_id: UUID,
    ) -> None:
        """
        登记一个"事务提交之后再投递"的钩子。

        为什么必须等到提交之后
        --------------------
        在事务里投递的话，Worker 可能在事务还没提交时就去 claim ——
        它看不到那行（未提交），claim 返回 None，于是这次投递白费，
        而任务要等下一轮轮询才会被真正捞起。更糟的是它看起来"投递成功"
        （没有异常），只是没跑。

        钩子的执行点是 ServiceFactory.scope() 提交之后那一段
        （见 app/common/service/factory.py），与既有 agent_run 的
        _schedule 走的是同一条路。
        """

        dispatcher = self._dispatcher

        if dispatcher is None:
            return

        async def _hook() -> None:

            await dispatcher.dispatch(task_id)

        self.post_commit.append(_hook)
