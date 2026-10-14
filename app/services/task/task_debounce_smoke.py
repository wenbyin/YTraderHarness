"""
防抖的延迟任务冒烟：重排执行时间 + generation 失效令牌 + 精确定时唤醒。

跑法：
    python -m app.services.task.task_debounce_smoke

要验的是这一件事：

    同一个逻辑定时被反复重排时，**只有最后一次的时间生效**，
    并且在那个时刻之前一次都不许执行。

场景上它就是"审批超时"：每来一次新动静就把截止时间往后推，推了几次
之后只有一个截止时间是真的。

它分三层验，因为这三层各自会独立出错：

    仓储层    reschedule 只在可 claim 时生效；generation 每次 +1；
              next_scheduled 取的是最早的那条未来任务
    定时器层  醒来后能识别"我绑定的那份定时已被取代"（generation 不一致）
    端到端    重排到更晚之后，**旧时刻到点时不执行**；最终在最后一次
              的时间上只执行一次

⚠️ sweep_interval 被刻意调得很大（30 秒）
---------------------------------------
这是本文件最要紧的一处配置。兜底扫描 30 秒才转一圈，因此"延迟 1 秒的
任务在 1 秒左右跑掉"这件事**只可能**由精确定时唤醒做到 —— 如果它坏了，
任务会等到 30 秒才被兜底捞走，断言立刻失败。

调小这个值（比如默认的 15 秒也一样，但越大越明显）会让这个文件失去
判别力：兜底扫描自己就能在节拍内把任务捞起来，看不出定时器有没有工作。

⚠️ 本文件用探针 Handler，不接任何业务
------------------------------------
与 task_worker_smoke 共用同一套探针（导入它的 __main__ 守卫是安全的），
因此它同样证明了"Handler 由组合根显式登记"这条契约。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.common.factories import build_factories, reset_factories
from app.common.repository.factory import RepositoryFactory
from app.common.task import Task, TaskStatus
from app.infra.task import (
    DbPollingTaskDispatcher,
    OutboxRelay,
    RetryPolicy,
    TaskHandlerRegistryImpl,
    Worker,
)
from app.infra.task.dispatcher import _TimerBinding
from app.services.task.application import TaskService
from app.services.task.models.sqlalchemy.task import Task as TaskModel
from app.services.task.models.sqlalchemy.task_outbox import (
    TaskOutbox as TaskOutboxModel,
)
from app.services.task.repository import TaskOutboxStore, TaskStore
from app.services.task.task_worker_smoke import (
    CALLS,
    PROBE,
    ProbeParams,
    build_registry,
)

DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

#: 兜底节拍。刻意大 —— 见模块 docstring。
SWEEP_INTERVAL = 30.0

WAIT_SECONDS = 12.0

FAILS: list[str] = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        FAILS.append(label)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _in(seconds: float) -> datetime:
    return _now() + timedelta(seconds=seconds)


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

    # 复用 worker 冒烟的探针注册表（同一套 handler 名与参数类）。
    probe = build_registry()
    registry = TaskHandlerRegistryImpl()
    for handler in probe.known():
        definition = probe.resolve(handler)
        registry.register(
            handler=handler,
            service_contract=definition.service_contract,
            method=definition.method,
            params_type=definition.params_type,
        )

    services = build_factories(
        session_factory,
        pool={"handler_registry": registry},
    )

    worker = Worker(
        repositories=repositories,
        services=services,
        registry=registry,
        policy=RetryPolicy(jitter=0.0),
        lease_seconds=60.0,
        identity="debounce-smoke",
    )

    dispatcher = DbPollingTaskDispatcher(
        worker=worker,
        repositories=repositories,
        concurrency=2,
        sweep_interval=SWEEP_INTERVAL,
    )

    relay = OutboxRelay(
        dispatcher=dispatcher,
        repositories=repositories,
        interval=SWEEP_INTERVAL,
    )

    services._pool["task_dispatcher"] = dispatcher

    created: list[UUID] = []

    async def submit(task: Task) -> UUID:

        async with services.scope(TaskService) as service:
            task_id = await service.submit(task)

        created.append(task_id)

        return task_id

    async def reschedule(task_id: UUID, execute_at: datetime | None) -> Task:

        async with services.scope(TaskService) as service:
            return await service.reschedule(task_id, execute_at=execute_at)

    async def get(task_id: UUID) -> Task | None:

        async with repositories.scope(begin=False) as repos:
            return await repos.create(TaskStore).get(task_id)

    async def wait_for(task_id: UUID, *statuses, timeout=WAIT_SECONDS):

        deadline = asyncio.get_running_loop().time() + timeout

        while asyncio.get_running_loop().time() < deadline:

            row = await get(task_id)

            if row is not None and row.status in statuses:
                return row

            await asyncio.sleep(0.02)

        return None

    try:

        # =====================================================
        section("仓储层：reschedule 的条件与 generation 递增")

        target = await submit(
            Task(
                handler=f"{PROBE}.record",
                params=ProbeParams(label="store-level"),
                execute_at=_in(600),
            )
        )

        row = await get(target)

        check("新提交的延迟任务 generation = 1",
              row is not None and row.generation == 1,
              f"实际 {row.generation if row else None}")

        moved = await reschedule(target, _in(900))

        check("重排后 generation = 2", moved.generation == 2,
              f"实际 {moved.generation}")
        check("重排后执行时间变了",
              moved.execute_at is not None
              and moved.execute_at > _in(800))
        check("重排不动 attempt（它是改时间，不是重试）",
              moved.attempt == 0)
        check("重排不动 status", moved.status is TaskStatus.PENDING)

        again = await reschedule(target, _in(1200))
        check("再重排一次 generation = 3", again.generation == 3,
              f"实际 {again.generation}")

        # ---- 只在可 claim 时生效 ----
        #
        # ⚠️ 先把它改到"已到点"，才能 claim 成 running。
        # claim 带到期判断（execute_at <= now()），拿一条 execute_at 在
        # 未来 20 分钟的任务去 claim 只会拿到 None —— 那是 claim 的正确
        # 行为，不是这里要验的东西。第一版冒烟就踩了这个坑：
        # 任务一直停在 pending，于是"running 时不许重排"根本没被验到。
        await reschedule(target, None)

        async with repositories.scope() as repos:
            claimed = await repos.create(TaskStore).claim(
                target, leased_by="probe", lease_seconds=60
            )

        check("前置：已到点的任务能被 claim 成 running",
              claimed is not None
              and claimed.status is TaskStatus.RUNNING,
              f"实际 {claimed.status if claimed else None}")

        async with repositories.scope(begin=False) as repos:
            running = await repos.create(TaskStore).reschedule(
                target, execute_at=_in(9999)
            )

        check("正在跑的任务重排不生效（仓储层返回 None）", running is None)

        async with repositories.scope() as repos:
            finished = await repos.create(TaskStore).mark_success(target)

        check("前置：running 的任务能写回成功",
              finished is not None
              and finished.status is TaskStatus.SUCCESS,
              f"实际 {finished.status if finished else None}")

        async with repositories.scope(begin=False) as repos:
            terminal = await repos.create(TaskStore).reschedule(
                target, execute_at=_in(9999)
            )

        check("已终结的任务重排不生效", terminal is None)

        # ---- 服务层把它翻成明确的错误 ----
        state_error = False

        try:
            await reschedule(target, _in(9999))
        except ValueError as exc:
            state_error = type(exc).__name__ == "TaskStateError"

        check("服务层对已终结的重排报 TaskStateError",
              state_error)

        # =====================================================
        section("仓储层：next_scheduled 取最早的那条未来任务")

        far_id = await submit(
            Task(handler=f"{PROBE}.record",
                 params=ProbeParams(label="far"),
                 execute_at=_in(3000))
        )
        near_id = await submit(
            Task(handler=f"{PROBE}.record",
                 params=ProbeParams(label="near"),
                 execute_at=_in(2000))
        )

        async with repositories.scope(begin=False) as repos:
            nxt = await repos.create(TaskStore).next_scheduled()

        check("取到的是两条里更早的那条",
              nxt is not None and nxt.task_id == near_id,
              f"实际 {nxt.task_id if nxt else None}（near={near_id}）")

        # 把 near 推到 far 后面，下一个就应当是 far
        await reschedule(near_id, _in(4000))

        async with repositories.scope(begin=False) as repos:
            nxt2 = await repos.create(TaskStore).next_scheduled()

        check("重排之后下一个闹钟换成剩下最早的那条",
              nxt2 is not None and nxt2.task_id == far_id,
              f"实际 {nxt2.task_id if nxt2 else None}（far={far_id}）")

        # 两个方法的条件是互补的：next_scheduled 只给"还没到点的"，
        # list_due 只给"已经到点的"。一条任务不该同时出现在两边 ——
        # 否则定时器会为了它睡 0 秒空转。
        async with repositories.scope(begin=False) as repos:
            due = await repos.create(TaskStore).list_due(limit=200)

        check("还没到点的任务不在 list_due 里（两个方法条件互补）",
              near_id not in due and far_id not in due)

        # =====================================================
        section("定时器层：醒来能识别「我绑定的那份已被取代」")

        async with repositories.scope(begin=False) as repos:
            fresh = await repos.create(TaskStore).get(near_id)

        # 白盒调用：直接构造一个"绑定了旧 generation"的定时，
        # 看它能否识别出自己已失效。这是防抖机制的核心判断。
        stale = _TimerBinding(
            task_id=near_id,
            generation=fresh.generation - 1,      # 故意落后一代
            execute_at=_in(1),
            remaining=1.0,
        )

        valid = await dispatcher._check_binding(stale)

        check("generation 落后 -> 判定为已作废（返回 False）",
              valid is False)

        current = _TimerBinding(
            task_id=near_id,
            generation=fresh.generation,          # 与库里一致
            execute_at=_in(1),
            remaining=1.0,
        )

        check("generation 一致 -> 判定为仍然有效（返回 True）",
              await dispatcher._check_binding(current) is True)

        # =====================================================
        section("端到端：反复重排，旧时刻到点不执行")

        await dispatcher.start()

        before = CALLS.count(("record", "debounced"))

        # T0 = 1 秒后。定时器会绑定它并睡到那时。
        debounced = await submit(
            Task(
                handler=f"{PROBE}.record",
                params=ProbeParams(label="debounced"),
                execute_at=_in(1.0),
            )
        )

        # 先让它绑定上 (generation=1, T0)，然后**推到更晚**。
        await asyncio.sleep(0.4)
        await reschedule(debounced, _in(1.6))      # generation=2, T1

        # 再推一次，让最终时刻是 T2。
        await asyncio.sleep(0.4)
        await reschedule(debounced, _in(1.6))      # generation=3, T2

        final = await get(debounced)

        check("多次重排后 generation = 3（提交 1 次 + 重排 2 次）",
              final is not None and final.generation == 3,
              f"实际 {final.generation if final else None}")

        # ---- 关键断言：等过最初的 T0，但它必须还没跑 ----
        await asyncio.sleep(0.6)

        midway = await get(debounced)

        check("旧时刻（T0）过去了，任务**没有**被执行",
              midway is not None
              and midway.status is TaskStatus.PENDING
              and CALLS.count(("record", "debounced")) == before,
              f"status={midway.status if midway else None}, "
              f"调用次数={CALLS.count(('record', 'debounced')) - before}")

        # ---- 最终时刻到了：应该在很短时间内精确醒来 ----
        started = asyncio.get_running_loop().time()

        done = await wait_for(debounced, TaskStatus.SUCCESS)

        elapsed = asyncio.get_running_loop().time() - started

        check("最终时刻到了之后被执行",
              done is not None and done.status is TaskStatus.SUCCESS,
              f"实际 {done.status if done else None}")

        check("只执行了一次（重排没有让它跑多遍）",
              CALLS.count(("record", "debounced")) - before == 1,
              f"实际 {CALLS.count(('record', 'debounced')) - before} 次")

        # 兜底节拍是 30 秒，所以"这么点时间就跑掉"只可能是定时器按时醒的
        check("是**精确**唤醒，而不是等兜底节拍",
              elapsed < 3.0,
              f"实际等了 {elapsed:.2f}s（兜底节拍 {SWEEP_INTERVAL:.0f}s）")

        # =====================================================
        section("端到端：重排到「立即」会立刻跑")

        immediate = await submit(
            Task(handler=f"{PROBE}.record",
                 params=ProbeParams(label="immediate"),
                 execute_at=_in(600))
        )

        mark = CALLS.count(("record", "immediate"))

        await reschedule(immediate, None)     # 改成立即

        done_now = await wait_for(immediate, TaskStatus.SUCCESS)

        check("改成立即之后很快执行",
              done_now is not None
              and CALLS.count(("record", "immediate")) - mark == 1,
              f"实际 {done_now.status if done_now else None}")

    finally:

        await relay.aclose()
        await dispatcher.aclose()

        if created:
            async with session_factory() as session:
                await session.execute(
                    delete(TaskOutboxModel).where(
                        TaskOutboxModel.task_id.in_(created)
                    )
                )
                await session.execute(
                    update(TaskModel)
                    .where(TaskModel.task_id.in_(created))
                    .values(parent_task_id=None)
                )
                await session.execute(
                    delete(TaskModel).where(
                        TaskModel.task_id.in_(created)
                    )
                )
                await session.commit()

            print(f"\n  已清理 {len(created)} 条探针任务")

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
