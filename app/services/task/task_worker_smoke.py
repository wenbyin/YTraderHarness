"""
端到端冒烟：submit -> tasks + task_outbox -> relay/扫描 -> Worker ->
Registry -> ServiceFactory -> Service.method(params, ctx)。

跑法：
    python -m app.services.task.task_worker_smoke

它验的是 spec §18 那条完整链路，以及几个**只有把整条链接起来才验得到**
的性质：

    立即任务          submit 之后真的被执行（不是只落库）
    延迟任务          execute_at 没到就不跑，到了才跑
    Task -> Task      Handler 通过 ctx 派生子任务，lineage 自动接上
    Retry             失败后按策略重排、再跑、最终成功
    永久失败          到上限就终态，不再试
    TaskPermanentError Handler 说"别试了"就真的不试
    幂等              同 key 重复提交只跑一次
    取消              协作式取消：跑完之后不得把状态改回成功
    重复投递          第二次投递拿不到 claim，不会重复执行
    重启恢复          没人投递过的任务靠兜底扫描被捞起来
    投递保证          未投递的 outbox 记录被 relay 补投
    参数校验          参数类传错在 submit 那一刻就报

⚠️ 为什么必须用真 Postgres + 真 Worker
------------------------------------
这些性质几乎全部依赖**跨事务、跨协程**的真实行为：claim 的 CAS、
outbox 与 tasks 同事务、execute_at 由数据库时钟判断、租约、以及
"重新读一次库里到底变成了什么"。用桩仓储验出来的只是桩的 if 分支。

⚠️ 本文件用**探针 Handler**，不接任何业务
--------------------------------------
这是刻意的（本版的交付范围不含业务接入）：探针服务定义在本文件里、
只在本进程的注册表里注册，因此它同时证明了"Handler 由组合根显式登记"
这条契约确实成立 —— Task 核心不需要认识任何业务模块。

⚠️ 时序
------
Dispatcher 与 relay 是常驻协程，所以断言靠**轮询等状态**（_wait_for），
不靠 sleep 一个经验值。退避策略显式关掉抖动（jitter=0.0）并压到 50ms，
让重试不必把冒烟拖成几十秒。
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.common.factories import build_factories, reset_factories
from app.common.repository.factory import RepositoryFactory
from app.common.service.base import BaseService
from app.common.task import Task, TaskPermanentError, TaskStatus
from app.common.task.models import RawTaskParams
from app.common.task.types import OutboxStatus
from app.infra.task import (
    DbPollingTaskDispatcher,
    OutboxRelay,
    RetryPolicy,
    TaskHandlerRegistryImpl,
    Worker,
)
from app.services.task.application import TaskService
from app.services.task.models.sqlalchemy.task import Task as TaskModel
from app.services.task.models.sqlalchemy.task_outbox import (
    TaskOutbox as TaskOutboxModel,
)
from app.services.task.repository import TaskOutboxStore, TaskStore

DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

PROBE = "task_smoke"

#: 单条用例等待状态变化的上限。超了就当失败 —— 免得冒烟挂死。
WAIT_SECONDS = 12.0

FAILS: list[str] = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        FAILS.append(label)


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ============================================================
# 探针：参数与 Handler
# ============================================================


class ProbeParams(BaseModel):
    """探针 Handler 的参数。真实业务里每个 Handler 有自己的参数类。"""

    label: str = ""


class OtherParams(BaseModel):
    """故意"传错"的另一个参数类，用来验 submit 期的参数校验。"""

    something_else: str = ""


#: 每次 Handler 被真正调用时记一笔。模块级而不是实例级 ——
#: ServiceFactory 每次 create() 都造新实例，实例级记不下来。
CALLS: list[tuple[str, str]] = []

#: 需要"第一次失败"的 Handler 用的计数器。
ATTEMPTS: dict[str, int] = {}

#: 让某个 Handler 卡住不返回，用来测"执行中被取消"。
BLOCK = asyncio.Event()


class ProbeService(BaseService):
    """探针业务服务。

    刻意做成"契约 + 实现"两个类，与仓库里其余服务一致 ——
    ServiceFactory 靠 resolve_impl 反射找实现，一个类自成一体的写法
    会让它找不到。
    """

    @abstractmethod
    async def record(self, params: ProbeParams, ctx: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def spawn_child(self, params: ProbeParams, ctx: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def flaky(self, params: ProbeParams, ctx: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def always_fails(self, params: ProbeParams, ctx: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def gives_up(self, params: ProbeParams, ctx: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def slow(self, params: ProbeParams, ctx: Any) -> None:
        raise NotImplementedError


class ProbeServiceImpl(ProbeService):

    async def record(self, params: ProbeParams, ctx: Any) -> None:
        CALLS.append(("record", params.label))

    async def spawn_child(self, params: ProbeParams, ctx: Any) -> None:
        """派生一条子任务 —— spec §8 的 Task -> Task。"""

        CALLS.append(("spawn_child", params.label))

        # 用 ctx.submit 而不是自己拼 parent/root：lineage 的接续只有一处
        # 实现，Handler 想漏填也漏不了。
        await ctx.submit(
            handler=f"{PROBE}.record",
            params=ProbeParams(label=f"child-of-{params.label}"),
        )

    async def flaky(self, params: ProbeParams, ctx: Any) -> None:
        """第一次失败、第二次成功。"""

        ATTEMPTS[params.label] = ATTEMPTS.get(params.label, 0) + 1
        CALLS.append(("flaky", params.label))

        if ATTEMPTS[params.label] == 1:
            raise RuntimeError("上游 502（模拟暂时性失败）")

    async def always_fails(self, params: ProbeParams, ctx: Any) -> None:
        CALLS.append(("always_fails", params.label))
        raise ValueError("这段逻辑就是坏的")

    async def gives_up(self, params: ProbeParams, ctx: Any) -> None:
        CALLS.append(("gives_up", params.label))
        # 声明"这次失败是永久性的"。Worker 应当直接判 FAILED、
        # 不消耗 max_attempts 里的另外两次。
        raise TaskPermanentError("单据已作废，再试也没用")

    async def slow(self, params: ProbeParams, ctx: Any) -> None:
        CALLS.append(("slow", params.label))
        await BLOCK.wait()


def build_registry() -> TaskHandlerRegistryImpl:

    registry = TaskHandlerRegistryImpl()

    for method in (
        "record",
        "spawn_child",
        "flaky",
        "always_fails",
        "gives_up",
        "slow",
    ):
        registry.register(
            handler=f"{PROBE}.{method}",
            service_contract=ProbeService,
            method=method,
            params_type=ProbeParams,
        )

    return registry


# ============================================================
# harness
# ============================================================


class Harness:

    def __init__(self, engine, session_factory, repositories) -> None:
        self.engine = engine
        self.session_factory = session_factory
        self.repositories = repositories
        self.registry = build_registry()
        self.services = build_factories(
            session_factory,
            pool={
                "handler_registry": self.registry,
            },
        )
        self.worker = Worker(
            repositories=repositories,
            services=self.services,
            registry=self.registry,
            policy=RetryPolicy(
                # 抖动静默掉：要精确断言"重试发生在什么时候"就必须确定。
                base_delay=0.05,
                multiplier=1.0,
                max_delay=0.05,
                jitter=0.0,
            ),
            lease_seconds=60.0,
            identity="smoke-worker",
        )
        self.dispatcher = DbPollingTaskDispatcher(
            worker=self.worker,
            repositories=repositories,
            concurrency=2,
            sweep_interval=0.15,
            batch_size=50,
        )
        self.relay = OutboxRelay(
            dispatcher=self.dispatcher,
            repositories=repositories,
            interval=0.15,
            batch_size=50,
        )

        # submit 的 post-commit 钩子要靠池子里的 dispatcher 才生效 ——
        # 池子在 build_factories 时就抓走了，所以现在补进去。
        self.services._pool["dispatcher"] = self.dispatcher

        self.created: list[UUID] = []

    async def get(self, task_id: UUID) -> Task | None:

        async with self.repositories.scope(begin=False) as repos:

            return await repos.create(TaskStore).get(task_id)

    async def submit(self, task: Task) -> UUID:
        """走**正常的业务入口**提交，而不是直接调仓储。

        这样才顺带验到 submit 自己的两件事：补齐 lineage、以及登记
        提交后投递的钩子（所以下面这些用例里，任务提交完就会自己跑起来，
        不依赖 relay 或扫描）。
        """

        async with self.services.scope(TaskService) as service:

            task_id = await service.submit(task)

        self.created.append(task_id)

        return task_id

    async def wait_for(
            self,
            task_id: UUID,
            *statuses: TaskStatus,
            timeout: float = WAIT_SECONDS,
    ) -> Task | None:
        """轮询等任务走到某个状态。超时返回最后一次读到的样子。"""

        deadline = (
            asyncio.get_running_loop().time() + timeout
        )

        last = None

        while asyncio.get_running_loop().time() < deadline:

            last = await self.get(task_id)

            if last is not None and last.status in statuses:
                return last

            await asyncio.sleep(0.025)

        return last


async def main() -> int:

    logging.basicConfig(level=logging.ERROR)

    engine = create_async_engine(DSN)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    from app.infra.datasource.initializer import (
        import_all_models,
        initialize_database,
    )

    import_all_models()
    await initialize_database(engine)

    repositories = RepositoryFactory(session_factory=session_factory)

    harness = Harness(engine, session_factory, repositories)

    await harness.dispatcher.start()

    # ⚠️ 刻意**不**启动 relay 的常驻循环。
    #
    # 前面每一节靠的都是另外两条路径：submit 的 post-commit 钩子（立即
    # 任务）、以及兜底扫描（重试 / 延迟 / 派生 / 恢复）。relay 的循环若
    # 一直开着，它会抢在断言之前把 outbox 记录标成 published ——
    # "relay 补投"那一节就变成了一个时序赌注（谁先动手取决于调度）。
    #
    # 所以那一节手动调 relay_once()，最后再单独起一次循环验循环本身。

    try:

        # =====================================================
        section("立即任务：submit 之后真的被执行了")

        plain = await harness.submit(
            Task(handler=f"{PROBE}.record", params=ProbeParams(label="plain"))
        )

        done = await harness.wait_for(plain, TaskStatus.SUCCESS)

        check("走到了 success",
              done is not None and done.status is TaskStatus.SUCCESS,
              f"实际 {done.status if done else None}")
        check("Handler 真的被调用且参数原样送达",
              ("record", "plain") in CALLS)
        check("attempt 记成 1", done is not None and done.attempt == 1)
        check("finished_at 有值", done is not None and done.finished_at is not None)
        # 终态必须清掉租约：lease_expires_at 是"当前这次执行的租约到期时刻"，
        # 任务已经结束却还留着一个未来的时刻，会让查询接口看起来像
        # "有人正在跑"。（leased_by 同时清空同理 —— 它的定义是**租约持有者**，
        # 没有租约就没有持有者。它随执行期存在，见下面"取消"那节的断言。）
        check("终态后租约已清（不会让人以为还在跑）",
              done is not None
              and done.lease_expires_at is None
              and done.leased_by is None)

        # =====================================================
        section("重复投递：同一条任务不会被跑第二次")

        again = await harness.worker.run(plain)

        check("第二次投递抢不到 claim",
              again.ran is False, f"ran={again.ran}")
        check("且它不谎报失败（没抢到 ≠ 失败）",
              again.status is None, f"status={again.status}")

        # =====================================================
        section("幂等：同 key 重复提交只跑一次")

        key = "smoke-idempotency-1"

        # 同一个 key 提交两次，断言它们落到同一条任务上。
        first = await harness.submit(
            Task(handler=f"{PROBE}.record",
                 params=ProbeParams(label="idem"),
                 idempotency_key=key)
        )
        second = await harness.submit(
            Task(handler=f"{PROBE}.record",
                 params=ProbeParams(label="idem"),
                 idempotency_key=key)
        )

        check("同 key 两次提交返回同一个 task_id", first == second,
              f"{first} vs {second}")

        await harness.wait_for(first, TaskStatus.SUCCESS)
        await asyncio.sleep(0.3)  # 给"万一它多跑一次"留出暴露的时间

        check("Handler 只被调用了 1 次",
              CALLS.count(("record", "idem")) == 1,
              f"实际 {CALLS.count(('record', 'idem'))} 次")

        # =====================================================
        section("Task -> Task：Handler 派生子任务并接上 lineage")

        parent = await harness.submit(
            Task(handler=f"{PROBE}.spawn_child",
                 params=ProbeParams(label="p1"))
        )

        parent_done = await harness.wait_for(parent, TaskStatus.SUCCESS)

        check("父任务成功",
              parent_done is not None
              and parent_done.status is TaskStatus.SUCCESS)

        # 子任务要等一会儿才被提交 + 执行
        deadline = asyncio.get_running_loop().time() + WAIT_SECONDS
        child_id = None

        while asyncio.get_running_loop().time() < deadline:

            async with repositories.scope(begin=False) as repos:
                chain = await repos.create(TaskStore).list_lineage(parent)

            if len(chain) >= 2 and all(
                t.status is TaskStatus.SUCCESS for t in chain
            ):
                child_id = chain[1].task_id
                break

            await asyncio.sleep(0.05)

        check("子任务被创建并执行成功", child_id is not None)

        if child_id is not None:

            async with repositories.scope(begin=False) as repos:
                chain = await repos.create(TaskStore).list_lineage(parent)

            harness.created.append(child_id)

            child = chain[1]

            check("子任务的 parent 是父任务",
                  child.parent_task_id == parent)
            check("子任务的 root 也是父任务（链的起点）",
                  child.root_task_id == parent)
            check("整条链按顺序两条",
                  [t.handler for t in chain] == [
                      f"{PROBE}.spawn_child",
                      f"{PROBE}.record",
                  ])
            check("子任务看到了派生者给的参数",
                  ("record", "child-of-p1") in CALLS)

        # 顺带验一次查询接口背后的那条路（list/count 的口径一致）
        async with harness.services.scope(TaskService) as service:

            items, total = await service.list_tasks(
                root_task_id=parent, limit=10
            )

        check("按 root 查得到整条链", len(items) == 2, f"实际 {len(items)}")
        check("总数与列表一致", total == 2, f"实际 {total}")

        # =====================================================
        section("重试：失败后重排、再跑、最终成功")

        flaky = await harness.submit(
            Task(handler=f"{PROBE}.flaky",
                 params=ProbeParams(label="flaky-1"),
                 max_attempts=3)
        )

        flaky_done = await harness.wait_for(flaky, TaskStatus.SUCCESS)

        check("最终成功",
              flaky_done is not None
              and flaky_done.status is TaskStatus.SUCCESS,
              f"实际 {flaky_done.status if flaky_done else None}")
        check("attempt 记成 2（第一次失败 + 第二次成功）",
              flaky_done is not None and flaky_done.attempt == 2,
              f"实际 {flaky_done.attempt if flaky_done else None}")
        check("Handler 被调用了 2 次",
              CALLS.count(("flaky", "flaky-1")) == 2,
              f"实际 {CALLS.count(('flaky', 'flaky-1'))} 次")
        check("上一次失败的原因被保留（回答「为什么重试过」）",
              flaky_done is not None
              and "上游 502" in (flaky_done.error or ""),
              f"error={flaky_done.error if flaky_done else None}")

        # =====================================================
        section("永久失败：到上限就终态")

        dead = await harness.submit(
            Task(handler=f"{PROBE}.always_fails",
                 params=ProbeParams(label="dead-1"),
                 max_attempts=2)
        )

        dead_done = await harness.wait_for(dead, TaskStatus.FAILED)

        check("最终是 failed",
              dead_done is not None
              and dead_done.status is TaskStatus.FAILED,
              f"实际 {dead_done.status if dead_done else None}")
        check("跑满了 max_attempts",
              dead_done is not None and dead_done.attempt == 2,
              f"实际 {dead_done.attempt if dead_done else None}")
        check("error 里带着异常类型与消息",
              dead_done is not None
              and "ValueError" in (dead_done.error or ""),
              f"error={dead_done.error if dead_done else None}")
        check("finished_at 有值",
              dead_done is not None and dead_done.finished_at is not None)

        # =====================================================
        section("TaskPermanentError：Handler 说别试了就不试")

        given_up = await harness.submit(
            Task(handler=f"{PROBE}.gives_up",
                 params=ProbeParams(label="giveup-1"),
                 max_attempts=3)
        )

        given_up_done = await harness.wait_for(given_up, TaskStatus.FAILED)
        await asyncio.sleep(0.3)

        check("直接 failed",
              given_up_done is not None
              and given_up_done.status is TaskStatus.FAILED)
        check("**没有**浪费另外两次尝试（attempt 停在 1）",
              given_up_done is not None and given_up_done.attempt == 1,
              f"实际 {given_up_done.attempt if given_up_done else None}")
        check("只调用了一次",
              CALLS.count(("gives_up", "giveup-1")) == 1,
              f"实际 {CALLS.count(('gives_up', 'giveup-1'))} 次")

        # =====================================================
        section("延迟任务：没到点不跑")

        due_at = datetime.now(timezone.utc) + timedelta(seconds=1.5)

        delayed = await harness.submit(
            Task(handler=f"{PROBE}.record",
                 params=ProbeParams(label="delayed"),
                 execute_at=due_at)
        )

        await asyncio.sleep(0.6)

        mid = await harness.get(delayed)

        check("到点之前仍是 pending（只落库，不执行）",
              mid is not None and mid.status is TaskStatus.PENDING,
              f"实际 {mid.status if mid else None}")
        check("到点之前 Handler 没被调用",
              ("record", "delayed") not in CALLS)

        delayed_done = await harness.wait_for(delayed, TaskStatus.SUCCESS)

        check("到点之后被跑掉",
              delayed_done is not None
              and delayed_done.status is TaskStatus.SUCCESS,
              f"实际 {delayed_done.status if delayed_done else None}")
        check("Handler 这时才被调用", ("record", "delayed") in CALLS)

        # =====================================================
        section("取消：协作式，且跑完之后不得改回成功")

        BLOCK.clear()

        running = await harness.submit(
            Task(handler=f"{PROBE}.slow",
                 params=ProbeParams(label="slow-1"),
                 max_attempts=1)
        )

        running_task = await harness.wait_for(
            running, TaskStatus.RUNNING
        )

        check("先真的跑起来了（才能验「执行中被取消」）",
              running_task is not None
              and running_task.status is TaskStatus.RUNNING,
              f"实际 {running_task.status if running_task else None}")
        # running 期间才有租约，这是 leased_by 的语义窗口 ——
        # 也正因为有它，"这条卡住的任务当时是谁在跑"才答得出来。
        check("执行期间写着租约持有者（排查卡住的任务要靠它）",
              running_task is not None
              and running_task.leased_by == "smoke-worker"
              and running_task.lease_expires_at is not None,
              f"leased_by={running_task.leased_by if running_task else None}")

        async with harness.services.scope(TaskService) as service:

            status_after = await service.cancel(running)

        check("cancel 返回真实的当前状态",
              status_after is TaskStatus.CANCELLED,
              f"实际 {status_after}")

        # 放行那个卡住的 Handler：它会正常返回，然后 Worker 试图写成功。
        BLOCK.set()

        await asyncio.sleep(0.5)

        final = await harness.get(running)

        check("执行者写回被拒绝，最终停在 cancelled（否则取消形同虚设）",
              final is not None
              and final.status is TaskStatus.CANCELLED,
              f"实际 {final.status if final else None}")

        # 取消一条已经结束的任务：不报错，如实回它的状态
        async with harness.services.scope(TaskService) as service:

            late = await service.cancel(plain)

        check("取消一条已成功的任务是幂等的（不报错）",
              late is TaskStatus.SUCCESS, f"实际 {late}")

        # =====================================================
        section("重启恢复：没人投递过的任务被兜底扫描捞起来")

        # 直接写库，绕开 submit —— 模拟"上一次进程崩在投递之前"：
        # 任务落库了、outbox 里有记录，但没有任何人投递过。
        stranded_id = (
            Task(
                handler=f"{PROBE}.record",
                params=ProbeParams(label="stranded"),
            )
            .seal_root()
            .task_id
        )

        async with repositories.scope() as repos:

            store = repos.create(TaskStore)
            await store.add(
                Task(
                    handler=f"{PROBE}.record",
                    params=ProbeParams(label="stranded"),
                    task_id=stranded_id,
                ).seal_root()
            )
            await repos.create(TaskOutboxStore).add(stranded_id)

        harness.created.append(stranded_id)

        async with repositories.scope(begin=False) as repos:
            row = await repos.create(TaskStore).get(stranded_id)

        check("它确实躺在库里没跑",
              row is not None and row.status is TaskStatus.PENDING)
        check("且 Handler 还没被调用",
              ("record", "stranded") not in CALLS)

        # 兜底扫描应当把它捞起来（这是"服务重启恢复"的实质）
        recovered = await harness.wait_for(stranded_id, TaskStatus.SUCCESS)

        check("被扫描捞起来并执行成功",
              recovered is not None
              and recovered.status is TaskStatus.SUCCESS,
              f"实际 {recovered.status if recovered else None}")
        check("Handler 被调用了", ("record", "stranded") in CALLS)

        # =====================================================
        section("投递保证：relay 补投未投递的 outbox 记录")

        # 现造一条"落了库、但没人投递过"的记录 —— 正是"进程崩在投递
        # 之前"留下的那个形状。
        #
        # 不复用上面那条 stranded：它已经被兜底扫描处理过了，
        # 它的 outbox 状态取决于谁先动手，拿它断言就是赌时序。
        undelivered_id = UUID(int=7)

        async with repositories.scope() as repos:

            store = repos.create(TaskStore)

            await store.add(
                Task(
                    handler=f"{PROBE}.record",
                    params=ProbeParams(label="undelivered"),
                    task_id=undelivered_id,
                ).seal_root()
            )
            await repos.create(TaskOutboxStore).add(undelivered_id)

        harness.created.append(undelivered_id)

        delivered = await harness.relay.relay_once()

        check("relay 把未投递的记录投出去了",
              delivered >= 1, f"本轮投出 {delivered}")

        async with repositories.scope(begin=False) as repos:
            record = await repos.create(TaskOutboxStore).get_by_task(
                undelivered_id
            )

        check("投递记录被标成 published",
              record is not None
              and record.status is OutboxStatus.PUBLISHED,
              f"实际 {record.status if record else None}")
        check("published_at 有值",
              record is not None and record.published_at is not None)

        # 补投之后确实被执行了 —— 投出去不是终点，"任务跑了"才是。
        undelivered_done = await harness.wait_for(
            undelivered_id, TaskStatus.SUCCESS
        )
        check("补投之后任务被执行成功",
              undelivered_done is not None
              and undelivered_done.status is TaskStatus.SUCCESS,
              f"实际 {undelivered_done.status if undelivered_done else None}")

        # 再验一次**常驻循环**本身（上面手动调的是它的单步）。
        looped_id = UUID(int=8)

        async with repositories.scope() as repos:

            store = repos.create(TaskStore)

            await store.add(
                Task(
                    handler=f"{PROBE}.record",
                    params=ProbeParams(label="looped"),
                    task_id=looped_id,
                ).seal_root()
            )
            await repos.create(TaskOutboxStore).add(looped_id)

        harness.created.append(looped_id)

        await harness.relay.start()

        await asyncio.sleep(0.5)

        async with repositories.scope(begin=False) as repos:
            looped = await repos.create(TaskOutboxStore).get_by_task(
                looped_id
            )

        check("常驻循环也会自己把记录投出去",
              looped is not None
              and looped.status is OutboxStatus.PUBLISHED,
              f"实际 {looped.status if looped else None}")

        await harness.relay.aclose()

        # =====================================================
        section("参数校验：传错参数类在 submit 就报")

        reported = False
        detail = ""

        try:
            await harness.submit(
                Task(
                    handler=f"{PROBE}.record",
                    params=OtherParams(something_else="x"),
                )
            )
        except ValueError as exc:
            reported = True
            detail = type(exc).__name__

        check("submit 就抛错，而不是等执行时才炸", reported,
              f"异常={detail}")
        check("报的是参数类对不上（不是 UnknownHandler）",
              detail == "TaskParamsError", f"实际 {detail}")

        # =====================================================
        section("未注册的 handler：submit 就报")

        unregistered = False

        try:
            await harness.submit(
                Task(handler="nope.does_not_exist",
                     params=ProbeParams(label="x"))
            )
        except ValueError as exc:
            unregistered = "UnknownHandler" in type(exc).__name__

        check("submit 期就拒绝未注册的 handler", unregistered)

        # =====================================================
        section("查询接口背后的服务：lineage 与 404 语义")

        async with harness.services.scope(TaskService) as service:

            chain = await service.get_lineage(parent)
            missing_empty = await service.get_lineage(
                UUID(int=0)
            )

        check("lineage 查得到父与子", len(chain) == 2, f"实际 {len(chain)}")
        check("不存在的 root 返回空列表而不是报错",
              missing_empty == [])

    finally:

        await harness.relay.aclose()
        await harness.dispatcher.aclose()

        # -----------------------------------------------------
        # 清理：按 task_id 精确删，不做前缀匹配。
        # -----------------------------------------------------
        if harness.created:
            async with session_factory() as session:
                await session.execute(
                    delete(TaskOutboxModel).where(
                        TaskOutboxModel.task_id.in_(harness.created)
                    )
                )
                # 先断开自引用再删（父行在子行还引用它时删不掉，
                # 且 FK 刻意没有 CASCADE）。
                await session.execute(
                    update(TaskModel)
                    .where(TaskModel.task_id.in_(harness.created))
                    .values(parent_task_id=None)
                )
                await session.execute(
                    delete(TaskModel).where(
                        TaskModel.task_id.in_(harness.created)
                    )
                )
                await session.commit()

            print(f"\n  已清理 {len(harness.created)} 条探针任务")

        reset_factories()
        await engine.dispose()

    if FAILS:
        print(f"\n失败 {len(FAILS)} 项：")
        for f in FAILS:
            print(f"  - {f}")
        return 1

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
