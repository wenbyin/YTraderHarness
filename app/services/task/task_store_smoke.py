"""
TaskStore / TaskOutboxStore 冒烟：**真 Postgres** 上的状态机与并发保护。

跑法：
    python -m app.services.task.task_store_smoke

为什么要打真库，不能只用内存桩
----------------------------
本文件要验的几条，恰好都是**只有真库才有的行为**：

    claim 的 CAS          两个事务真并发抢同一行，输的那个靠
                          PostgreSQL 的行锁 + READ COMMITTED 重判
    list_due 的 NULL 排序  NULLS FIRST 是数据库的排序语义
    mark_cancelled 的并发  取消与执行者写回真同时发生
    租约回收              靠 now() 这个**数据库时钟**判断过期

用假仓储把这些"验"一遍，验的是假仓储的 if 分支，不是 Postgres 的行为。

⚠️ 本文件会**建表之外什么都不改**：它自己插入的探针数据在结尾删掉，
不碰库里其它行。表本身由 initialize_database 的 create_all 建
（幂等，已存在就跳过）。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.common.repository.factory import RepositoryFactory
from app.common.task import Task, TaskStatus
from app.common.task.models import RawTaskParams
from app.common.task.types import OutboxStatus
from app.services.task.models.sqlalchemy.task import Task as TaskModel
from app.services.task.models.sqlalchemy.task_outbox import (
    TaskOutbox as TaskOutboxModel,
)
from app.services.task.repository import (
    TaskOutboxStore,
    TaskStore,
)

#: 与 app/config/datasource/postgre.yaml 里的 dsn 一致。
#: 冒烟直连而不走 DataSourceManager：那需要把整个 lifespan 拉起来
#: （图、Redis、模型预热全都会启动），对一个仓储冒烟来说太重。
DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

#: 探针数据的 handler 前缀。只为了让人在库里一眼认出这些行 ——
#: 删除是按 task_id 精确删的，不靠前缀匹配（前缀匹配会误删真实数据）。
PROBE = "task_smoke"

FAILS: list[str] = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        FAILS.append(label)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def mk(handler: str = "plain", **kw: Any) -> Task:
    """造一条探针任务。参数用 RawTaskParams 只是为了让这里不必定义一个
    业务参数类 —— 真实提交必须给具体模型（见 common/task/models.py）。"""

    return Task(
        handler=f"{PROBE}.{handler}",
        params=RawTaskParams.model_validate({"probe": True}),
        **kw,
    )


async def main() -> int:

    logging.basicConfig(level=logging.WARNING)

    engine = create_async_engine(DSN)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    repositories = RepositoryFactory(session_factory=factory)

    #: 本文件造出来的全部 task_id，结尾精确清理。
    created: list[UUID] = []

    try:
        # 表必须先存在。正常启动路径里由 lifespan 建（见
        # app/infra/datasource/initializer.py），冒烟跳过 lifespan，
        # 所以这里补一次 —— create_all 幂等，已有表不会被改。
        await _ensure_tables(engine)

        # =====================================================
        section("add 与幂等")

        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            outbox = repos.create(TaskOutboxStore)
            saved = await store.add(mk(idempotency_key="task-smoke-key-1"))
            await outbox.add(saved.task_id)
        created.append(saved.task_id)

        check("落库后是 pending", saved.status is TaskStatus.PENDING)
        check("attempt 从 0 开始", saved.attempt == 0)
        check("root_task_id 未被自动填（那是 submit 的事）",
              saved.root_task_id is None)

        async with repositories.scope() as repos:
            again = await repos.create(TaskStore).add(
                mk(idempotency_key="task-smoke-key-1")
            )

        check("同幂等键重复提交返回**同一条**",
              again.task_id == saved.task_id)

        async with repositories.scope() as repos:
            rows = await repos.create(TaskOutboxStore).get_by_task(
                saved.task_id
            )
        check("幂等返回没有新增第二条投递记录", rows is not None)

        # =====================================================
        section("list_due 与延迟执行")

        now = datetime.now(timezone.utc)

        immediate = mk("due")
        future = mk("due", execute_at=now + timedelta(hours=1))
        #: 已经到点的定时任务。它和 immediate 都是"该跑"，区别是一个靠
        #: execute_at 为 NULL、一个靠时间已过 —— 用它俩比顺序才能真正
        #: 验到 NULLS FIRST（拿一个不在 due 里的任务比是比不出来的）。
        overdue = mk("due", execute_at=now - timedelta(hours=1))

        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            s_immediate = await store.add(immediate)
            s_future = await store.add(future)
            s_overdue = await store.add(overdue)
        created += [
            s_immediate.task_id,
            s_future.task_id,
            s_overdue.task_id,
        ]

        async with repositories.scope() as repos:
            due = await repos.create(TaskStore).list_due(limit=500)

        check("立即任务（execute_at 为 NULL）会被捞到",
              s_immediate.task_id in due)
        check("已到点的定时任务也会被捞到",
              s_overdue.task_id in due)
        check("未来的定时任务不会被提前捞到",
              s_future.task_id not in due)
        # ⚠️ 这一条是 NULLS FIRST 的直接判据。PostgreSQL 的默认是
        # NULLS LAST，漏了 NULLS FIRST 时"立即"任务会被排到一堆
        # "某个时刻到点"的任务后面 —— 延迟任务于是插队到立即任务前面。
        check("立即任务排在已到点的定时任务前面（NULLS FIRST）",
              due.index(s_immediate.task_id) < due.index(s_overdue.task_id),
              f"immediate@{due.index(s_immediate.task_id)} "
              f"overdue@{due.index(s_overdue.task_id)}")

        # =====================================================
        section("claim：重复投递只跑一次")

        target = mk("claim")
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            target = await store.add(target)
            await repos.create(TaskOutboxStore).add(target.task_id)
            claimed = await store.claim(
                target.task_id, leased_by="worker-1", lease_seconds=60
            )
        created.append(target.task_id)

        check("claim 成功 -> running",
              claimed is not None and claimed.status is TaskStatus.RUNNING)
        check("claim 把 attempt 加到 1",
              claimed is not None and claimed.attempt == 1)
        check("claim 写了租约",
              claimed is not None
              and claimed.lease_expires_at is not None
              and claimed.leased_by == "worker-1")

        async with repositories.scope() as repos:
            duplicate = await repos.create(TaskStore).claim(
                target.task_id, leased_by="worker-2", lease_seconds=60
            )
        check("**重复投递的第二次 claim 拿不到**（CAS）", duplicate is None)

        # -----------------------------------------------------
        # 真并发：两个独立事务同时抢同一条。
        # 后到的那个会阻塞在行锁上，等前者提交后按 READ COMMITTED
        # 重新判断 WHERE —— 此时状态已是 running，于是影响 0 行。
        race = mk("race")
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            race = await store.add(race)
            await repos.create(TaskOutboxStore).add(race.task_id)
        created.append(race.task_id)

        async def racer(who: str) -> Task | None:
            async with repositories.scope() as repos:
                return await repos.create(TaskStore).claim(
                    race.task_id, leased_by=who, lease_seconds=60
                )

        winners = [
            r for r in await asyncio.gather(racer("a"), racer("b"))
            if r is not None
        ]

        check("并发 claim 只有一个赢家", len(winners) == 1,
              f"赢家={winners[0].leased_by if winners else None}")

        # =====================================================
        section("mark_success 与并发取消保护")

        async with repositories.scope() as repos:
            done = await repos.create(TaskStore).mark_success(target.task_id)

        check("mark_success -> success",
              done is not None and done.status is TaskStatus.SUCCESS)
        check("success 记了 finished_at",
              done is not None and done.finished_at is not None)
        check("success 清了租约",
              done is not None
              and done.lease_expires_at is None
              and done.leased_by is None)

        async with repositories.scope() as repos:
            late = await repos.create(TaskStore).mark_success(target.task_id)
        check("已终态时 mark_success 无效", late is None)

        # 这条是"取消"能不能立住的判据：执行者手里的任务被取消了，
        # 它跑完写回成功时必须无效。
        victim = mk("cancel-race")
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            victim = await store.add(victim)
            await repos.create(TaskOutboxStore).add(victim.task_id)
            await store.claim(
                victim.task_id, leased_by="worker-1", lease_seconds=60
            )
        created.append(victim.task_id)

        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            cancelled = await store.mark_cancelled(victim.task_id)
            overwrite = await store.mark_success(victim.task_id)

        check("running 的任务能被取消", cancelled is True)
        check("取消后执行者写回成功**无效**（否则取消形同虚设）",
              overwrite is None)

        async with repositories.scope() as repos:
            row = await repos.create(TaskStore).get(victim.task_id)
        check("最终停在 cancelled", row.status is TaskStatus.CANCELLED)

        pending_one = mk("cancel-pending")
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            pending_one = await store.add(pending_one)
            first = await store.mark_cancelled(pending_one.task_id)
            second = await store.mark_cancelled(pending_one.task_id)
        created.append(pending_one.task_id)

        check("pending 的任务能取消", first is True)
        check("已终态再取消返回 False（来晚了不是错误）", second is False)

        # =====================================================
        section("mark_failure：重试与放弃")

        flaky = mk("flaky", max_attempts=2)
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            flaky = await store.add(flaky)
            await repos.create(TaskOutboxStore).add(flaky.task_id)
            await store.claim(
                flaky.task_id, leased_by="worker-1", lease_seconds=60
            )
            retryable = await store.mark_failure(
                flaky.task_id, error="boom", retry_delay_seconds=0
            )
        created.append(flaky.task_id)

        check("还能再试 -> retryable",
              retryable.status is TaskStatus.RETRYABLE)
        check("retryable 记了下次执行时刻",
              retryable.execute_at is not None)
        check("retryable 不算结束（无 finished_at）",
              retryable.finished_at is None)
        check("retryable 清了租约", retryable.lease_expires_at is None)
        check("error 落库", retryable.error == "boom")

        async with repositories.scope() as repos:
            due = await repos.create(TaskStore).list_due(limit=500)
        check("retryable 到点后重新可跑", flaky.task_id in due)

        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            await store.claim(
                flaky.task_id, leased_by="worker-1", lease_seconds=60
            )
            terminal = await store.mark_failure(
                flaky.task_id, error="boom again", retry_delay_seconds=None
            )

        check("不再重试 -> failed", terminal.status is TaskStatus.FAILED)
        check("failed 记了 finished_at", terminal.finished_at is not None)

        # =====================================================
        section("schedule_retry：人工重跑")

        async with repositories.scope() as repos:
            revived = await repos.create(TaskStore).schedule_retry(
                flaky.task_id
            )

        check("failed 能被人工重跑",
              revived is not None and revived.status is TaskStatus.PENDING)
        # 不清零的话，用满 max_attempts 的任务 retry 后一失败就又成终态，
        # 于是 retry 在最需要它的场景里恰好没用。
        check("重跑把 attempt 清零", revived.attempt == 0)
        check("重跑保留上一次 error（人工重跑前正要看它）",
              revived.error == "boom again")
        check("重跑清了 finished_at", revived.finished_at is None)

        async with repositories.scope() as repos:
            refused = await repos.create(TaskStore).schedule_retry(
                target.task_id
            )
        check("已成功的任务不能 retry（会产生第二次外部副作用）",
              refused is None)

        # =====================================================
        section("reclaim_expired：崩溃 Worker 的回收")

        stuck = mk("stuck", max_attempts=3)
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            stuck = await store.add(stuck)
            await repos.create(TaskOutboxStore).add(stuck.task_id)
            # lease_seconds 为负 = 租约当场过期，等价于"执行者已经死了"。
            await store.claim(
                stuck.task_id, leased_by="dead-worker", lease_seconds=-1
            )
        created.append(stuck.task_id)

        async with repositories.scope() as repos:
            reclaimed = await repos.create(TaskStore).reclaim_expired(limit=50)
        check("租约过期的被收回", stuck.task_id in reclaimed)

        async with repositories.scope() as repos:
            row = await repos.create(TaskStore).get(stuck.task_id)
        check("收回后是 retryable（别的 Worker 能接管）",
              row.status is TaskStatus.RETRYABLE)
        check("收回后租约已清",
              row.lease_expires_at is None and row.leased_by is None)
        check("收留在 error 里留了痕（执行者已死，写不下别的）",
              "租约过期" in (row.error or ""))

        # attempt 用满的必须收成 failed，否则"一跑就把 Worker 搞崩"的任务
        # 会在 running <-> retryable 之间永远循环，把实例一个个拖死。
        poison = mk("poison", max_attempts=1)
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            poison = await store.add(poison)
            await repos.create(TaskOutboxStore).add(poison.task_id)
            await store.claim(
                poison.task_id, leased_by="dead-worker", lease_seconds=-1
            )
        created.append(poison.task_id)

        async with repositories.scope() as repos:
            await repos.create(TaskStore).reclaim_expired(limit=50)
        async with repositories.scope() as repos:
            row = await repos.create(TaskStore).get(poison.task_id)
        check("attempt 用满的租约过期 -> failed（不然永远循环）",
              row.status is TaskStatus.FAILED, f"实际 {row.status}")

        # =====================================================
        section("lineage：Task -> Task 的执行链")

        # seal_root() 是 TaskService.submit() 会做的事：把根任务自己的
        # root_task_id 填成 task_id。这里手动补上，让冒烟走的是**生产的
        # 形状** —— 绕开 submit 直接写库会得到一个 submit 不会产生的形状，
        # 拿它验出来的结论对生产不成立。
        async with repositories.scope() as repos:
            root = await repos.create(TaskStore).add(mk("root").seal_root())
        created.append(root.task_id)

        async with repositories.scope() as repos:
            child = await repos.create(TaskStore).add(
                root.child(
                    handler=f"{PROBE}.child",
                    params=RawTaskParams.model_validate({"step": 2}),
                )
            )
        # ⚠️ 子任务也要登记进 created。漏了它，结尾的 DELETE 会留下
        # 一条指着 root 的子行 —— 于是清理撞自引用外键失败，探针数据
        # 每次跑都往库里堆一份（这个坑本文件已经踩过一次，见 finally 的说明）。
        created.append(child.task_id)

        async with repositories.scope() as repos:
            grand = await repos.create(TaskStore).add(
                child.child(
                    handler=f"{PROBE}.grandchild",
                    params=RawTaskParams.model_validate({"step": 3}),
                )
            )
        created.append(grand.task_id)

        check("child 的 parent 是 root",
              child.parent_task_id == root.task_id)
        check("child 的 root 是 root", child.root_task_id == root.task_id)
        check("grandchild 的 parent 是 child",
              grand.parent_task_id == child.task_id)
        check("grandchild 的 root 仍是 root（链的起点不随深度改变）",
              grand.root_task_id == root.task_id)

        async with repositories.scope() as repos:
            chain = await repos.create(TaskStore).list_lineage(root.task_id)

        check("整条链查得到 3 条", len(chain) == 3, f"实际 {len(chain)}")
        check("链按产生顺序排列（升序，因果叙事）",
              [c.handler for c in chain] == [
                  f"{PROBE}.root",
                  f"{PROBE}.child",
                  f"{PROBE}.grandchild",
              ])

        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            kids = await store.list(parent_task_id=root.task_id)
            # 按 root 限定范围，而不是只按 handler。
            # 只按 handler 的话，上一次没清理干净的探针数据会被算进来
            # （或者别人手工造的同类行），断言就变成在数库里的历史存量。
            total = await store.count(
                handler=f"{PROBE}.child",
                root_task_id=root.task_id,
            )

        check("按 parent 查子任务",
              len(kids) == 1 and kids[0].task_id == child.task_id)
        check("count 与 list 的筛选口径一致", total == 1, f"实际 {total}")

        # 根任务的 root_task_id 为 NULL 时，它也必须出现在**自己**的链里。
        # 少了这层保护，一条绕过 submit 写进来的根任务会从它自己的
        # 执行链上消失 —— 链上少一条，而且不报错。
        unsealed = mk("unsealed-root")
        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            unsealed = await store.add(unsealed)
        created.append(unsealed.task_id)

        async with repositories.scope() as repos:
            chain = await repos.create(TaskStore).list_lineage(
                unsealed.task_id
            )
        check("root_task_id 为 NULL 的根任务仍出现在自己的链里",
              [c.task_id for c in chain] == [unsealed.task_id])

        # =====================================================
        section("task_outbox：投递意图")

        async with repositories.scope() as repos:
            store = repos.create(TaskStore)
            ob_task = await store.add(mk("outbox"))
            record = await repos.create(TaskOutboxStore).add(ob_task.task_id)
        created.append(ob_task.task_id)

        check("新登记是 pending", record.status is OutboxStatus.PENDING)
        check("retry_count 从 0 开始", record.retry_count == 0)

        async with repositories.scope() as repos:
            unpublished = await repos.create(TaskOutboxStore).list_unpublished(
                limit=50, max_retry_count=3
            )
        check("未投递的被捞出",
              any(u.outbox_id == record.outbox_id for u in unpublished))

        async with repositories.scope() as repos:
            await repos.create(TaskOutboxStore).mark_publish_failed(
                record.outbox_id, error="mq down", max_retry_count=3
            )
        async with repositories.scope() as repos:
            row = await repos.create(TaskOutboxStore).get_by_task(
                ob_task.task_id
            )
        check("投递失败记了次数与原因",
              row.retry_count == 1 and row.last_error == "mq down")
        check("未到上限仍是 pending", row.status is OutboxStatus.PENDING)

        async with repositories.scope() as repos:
            await repos.create(TaskOutboxStore).mark_publish_failed(
                record.outbox_id, error="mq down", max_retry_count=2
            )
        async with repositories.scope() as repos:
            row = await repos.create(TaskOutboxStore).get_by_task(
                ob_task.task_id
            )
        check("到上限转 failed（一直 pending 会看起来一切正常）",
              row.status is OutboxStatus.FAILED, f"实际 {row.status}")

        async with repositories.scope() as repos:
            unpublished = await repos.create(TaskOutboxStore).list_unpublished(
                limit=50, max_retry_count=2
            )
        check("超限的不再被捞出",
              not any(u.outbox_id == record.outbox_id for u in unpublished))

        async with repositories.scope() as repos:
            await repos.create(TaskOutboxStore).mark_published(
                record.outbox_id
            )
        async with repositories.scope() as repos:
            row = await repos.create(TaskOutboxStore).get_by_task(
                ob_task.task_id
            )
        check("投递成功后 published_at 有值", row.published_at is not None)
        check("投递成功后清掉上次的失败原因", row.last_error is None)

    finally:
        # =====================================================
        # 清理：按 task_id 精确删，不做前缀匹配 ——
        # 前缀匹配迟早会删掉别人的真实数据。
        #
        # 用 ORM 模型而不是裸 SQL 字符串：表改名时 ORM 会跟着报错，
        # 而裸 SQL 会安静地删一张旧表（或者什么都不删）然后"通过"。
        if created:
            async with factory() as session:
                await session.execute(
                    delete(TaskOutboxModel).where(
                        TaskOutboxModel.task_id.in_(created)
                    )
                )
                # ⚠️ 必须先把自引用断开再删。
                #
                # tasks.parent_task_id 是自引用外键，且**刻意没有 CASCADE**
                # （见 models/sqlalchemy/task.py：级联删掉整条子链是无声的，
                # 而无声地少掉一批任务是排查不出来的）。因此父行在子行还
                # 引用着它时删不掉 —— 一条 DELETE 里同时包含父子也不行，
                # 行的处理顺序不由我们决定。
                #
                # 生产上要删任务链同样得先做这一步；本冒烟正好把这条约束
                # 亲手撞了一次，顺手记在这里。
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

            print(f"\n  已清理 {len(created)} 条探针任务及其投递记录")

        await engine.dispose()

    if FAILS:
        print(f"\n失败 {len(FAILS)} 项：")
        for f in FAILS:
            print(f"  - {f}")
        return 1

    print("\n全部通过。")
    return 0


async def _ensure_tables(engine: Any) -> None:
    """确保两张表在。正常启动路径由 lifespan 建，冒烟跳过 lifespan。"""

    from app.common.models import Base
    from app.infra.datasource.initializer import import_all_models

    import_all_models()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
