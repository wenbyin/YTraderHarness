"""
PostgreSQL implementation of TaskStore.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。

⚠️ 本文件的写法与仓库里其余仓储**看起来不一样**，这是刻意的
--------------------------------------------------------
capability / agent_run 那些实现走 ORM 对象（session.get / session.add /
model.attr = x），本文件直接用 Core（select / insert / update 作用于
`Task.__table__`）。两个理由，都不是风格问题：

1. **状态推进必须是条件 UPDATE。**
   claim 要原子地"只有还是 pending/retryable 才改成 running"，
   mark_* 要原子地"只有还是 running 才写回"。用 ORM 的
   "读出来 -> 改属性 -> flush" 表达不了这个条件，那中间有窗口。

2. **不碰 identity map 就消灭了一整类 bug。**
   ORM 对象一旦加载进 session，同一 session 后续的 session.get 会直接
   返回那个**可能已过期**的实例；而这里大量使用 Core UPDATE，它绕过
   identity map 且不会更新它。两种写法混用就会出现"读到的还是旧状态"，
   且不报错。所以本文件读也走 `select(table)`，从不 session.get ——
   于是 identity map 全程为空，这类问题不可能发生。

   顺带还避开了 self-referencing relationship 在 async session 上的
   惰性加载（MissingGreenlet）—— 本模型压根没有 relationship。

返回值一律是 app/common/task/models.py::Task 实体；下面是两者的转换。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Interval,
    case,
    func,
    insert,
    literal,
    null,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError

from app.common.task.models import RawTaskParams, Task
from app.common.task.types import TaskStatus

from ..models.sqlalchemy.task import Task as TaskModel
from .task import TaskStore

#: 表的 Core 句柄。所有语句都作用于它，见模块 docstring 的第 2 条理由。
_T = TaskModel.__table__


#: 可被 claim 的状态。从枚举派生而不是手写字符串 ——
#: 以后给 TaskStatus 加一个可 claim 的状态，这里自动跟上。
_CLAIMABLE_VALUES: tuple[str, ...] = tuple(
    s.value for s in TaskStatus if s.is_claimable
)

#: 非终态（cancel 能拦下来的范围）。
_OPEN_VALUES: tuple[str, ...] = tuple(
    s.value for s in TaskStatus if not s.is_terminal
)

#: 租约过期收回时写进 error 的一句话。
#:
#: 这是这种失败**唯一**的痕迹：执行它的进程已经死了，没机会写下任何
#: 东西。所以它必须落库，而不是只进日志。
_LEASE_LOST_NOTE = "租约过期未完成，已收回"


# ============================================================
# 幂等键的生命周期（2026-10-10 定）
# ============================================================
#
# 键的默认语义是**一次性**的：任务进终态之后那个键仍然占着它，"重复提交
# 拿回同一条"永远成立。库里 45 条任务全是 `generation = 1`、从来没发生过
# 一次重排，就是这个语义的直接结果 —— 稳定键只能用一次。
#
# 需要"同一个键反复用"的调用方（当前只有 L1：`memory.l1.extract:{session_id}`，
# 一个会话长期占着它、每来一句就把执行时间往后推）**显式**声明
# `reusable_key=True`：那样它一进终态就把键写回 NULL，键的含义变成
# "**此刻**哪个任务代表这件事"，下一个窗口就能重新提交。
#
# ⚠️ 为什么是开关，而不是统一释放
# --------------------------
# 这是**共享基础设施**。统一释放会顺带改掉 L0 的行为：它的键是 per-run 的，
# "任务跑完之后再收尾一次"在旧语义下什么都不做，改完会多排一条。
# 默认 False 让老调用方一行不动。
#
# ⚠️ 为什么 retryable **不**释放
# -------------------------
# 它还要再跑一次。这两次之间同一个键的重复提交必须仍然折叠成同一条，
# 否则会排出两条都跑的任务、两条都写同一份数据。
#
# ⚠️ 谁来释放
# --------
# `_release_key()` 是唯一落点，四处终态写入共用它：
# mark_success / mark_failure(试满) / mark_cancelled / reclaim(试满)。
# 它返回 SQL 表达式，所以批量 reclaim 也能按每行自己的开关取值。


def _release_key() -> Any:
    """
    进终态时把幂等键写成什么 —— **由任务自己声明的 `reusable_key` 决定**。

    ⚠️ 为什么不无条件释放
    -----------------
    这是共享基础设施。无条件释放会顺带改掉 L0 的行为：它的键是 per-run 的
    一次性键（`memory.l0.extract:{run_id}`），"任务已经跑完之后再收尾一次"
    这件事在旧语义下是**什么都不做**，无条件释放之后会多排一条。

    所以默认 `reusable_key = False`（老语义，一行不动），只有显式声明
    可以复用的那些才释放 —— 当前只有 L1。

    ⚠️ 返回 SQL 表达式而不是 Python 值
    ------------------------------
    reclaim 那条 UPDATE 是**批量**的，一次可能动多行。要按每行自己的
    `reusable_key` 决定，就得逐行读出来再算 —— 而那就是"读-改-写"，
    正是本模块开头明确拒绝的写法（并发下会覆盖别人的结果）。
    """

    return case(
        (_T.c.reusable_key.is_(True), null()),
        else_=_T.c.idempotency_key,
    )


def _lease_deadline(
        seconds: float,
) -> Any:
    """
    "现在 + N 秒"的 SQL 表达式。

    用 func.now()（**数据库**的时钟）而不是 Python 的 datetime.now()：
    租约是多个实例共同判断的（A 写、B 判过期），用各自的本地时钟就会
    因机器间的时间偏移而误判 —— 快的那些会不停地把别人正在跑的任务
    抢走。让唯一的一个时钟（DB）来做这件事，偏移问题就不存在了。

    同理，execute_at 的比较也一律用 func.now()（见 list_due）。
    只有重试退避的秒数在 Python 里算，但它算出来的是**时长**，不是时刻。
    """

    return func.now() + literal(
        timedelta(seconds=seconds),
        Interval,
    )


def _to_task(
        row: Mapping[str, Any],
) -> Task:
    """一行 -> Task 实体。

    params 用 RawTaskParams 装载：库里只有序列化结果，具体是哪个参数类
    要等 Registry 解析出 handler 才知道（见 common/task/models.py）。
    """

    return Task(
        task_id=row["task_id"],
        handler=row["handler"],
        params=RawTaskParams.model_validate(row["params_json"]),
        status=TaskStatus(row["status"]),
        execute_at=row["execute_at"],
        attempt=row["attempt"],
        max_attempts=row["max_attempts"],
        generation=row["generation"],
        parent_task_id=row["parent_task_id"],
        root_task_id=row["root_task_id"],
        causation_id=row["causation_id"],
        idempotency_key=row["idempotency_key"],
        reusable_key=row["reusable_key"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        lease_expires_at=row["lease_expires_at"],
        leased_by=row["leased_by"],
        finished_at=row["finished_at"],
        error=row["error"],
    )


def _insert_values(
        task: Task,
) -> dict[str, Any]:
    """Task 实体 -> 一条 INSERT 的列值。

    ⚠️ params 走 params_dict 而不是 task.model_dump()["params"]。
    两者现在都能用（Task.params 标了 SerializeAsAny），但 params_dict 是
    从**实例本身** dump 的，语义更直接 —— 别再挂上一层"整条 Task 序列化"
    的中间结果。

    租约两列刻意不在这里：新提交的任务还没有人持租约，让它们保持 NULL。
    """

    return {
        "task_id": task.task_id,
        "handler": task.handler,
        "params_json": task.params_dict,
        "status": task.status.value,
        "execute_at": task.execute_at,
        "attempt": task.attempt,
        "max_attempts": task.max_attempts,
        "generation": task.generation,
        "parent_task_id": task.parent_task_id,
        "root_task_id": task.root_task_id,
        "causation_id": task.causation_id,
        "idempotency_key": task.idempotency_key,
        "reusable_key": task.reusable_key,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "error": task.error,
    }


class TaskStoreImpl(TaskStore):

    # ========================================================
    # 写入
    # ========================================================

    async def add(
            self,
            task: Task,
    ) -> Task:

        # 先查一次：绝大多数重复提交都发生在"已经提交过了"这种平静的
        # 情况下，一次 SELECT 就能挡掉，不必每次都付 SAVEPOINT 的代价。
        # 它挡不住并发（两次查询都可能为空），所以下面还有一层。
        if task.idempotency_key is not None:

            existing = await self.get_by_idempotency_key(
                task.idempotency_key
            )

            if existing is not None:
                return existing

        statement = (
            insert(_T)
            .values(**_insert_values(task))
            .returning(*_T.c)
        )

        try:
            # ⚠️ SAVEPOINT 不是可选的。
            # 唯一索引冲突在 PostgreSQL 里会让**整个事务**进入 aborted
            # 状态，后续任何语句都报 "current transaction is aborted"
            # —— 包括下面那次重新查询。套一层 begin_nested 之后，
            # 冲突只回滚到这个 SAVEPOINT，外层事务照常可用。
            async with self._session.begin_nested():

                row = (await self._session.execute(
                    statement
                )).one()

        except IntegrityError:

            # 走到这里只可能是并发插入撞了幂等键（主键冲突意味着
            # task_id 重复，那是调用方自造了一个已存在的 UUID，属于
            # 编程错误，不该被这里善意地吞掉）。
            existing = None

            if task.idempotency_key is not None:
                existing = await self.get_by_idempotency_key(
                    task.idempotency_key
                )

            if existing is None:
                raise

            return existing

        return _to_task(row._mapping)

    async def claim(
            self,
            task_id: UUID,
            *,
            leased_by: str,
            lease_seconds: float,
    ) -> Task | None:

        statement = (
            update(_T)
            .where(
                _T.c.task_id == task_id,
                # 这一行就是并发保护本身：状态不是"可跑"时，
                # UPDATE 影响 0 行，RETURNING 拿不到东西 -> 返回 None。
                # 两个实例同时收到同一条消息时，只有一个能命中。
                _T.c.status.in_(_CLAIMABLE_VALUES),
                # ⚠️ 到期判断**必须**在这里，不能只放在 list_due。
                #
                # dispatch() 是"叫一下"，而**每一条**任务在提交时都会被
                # 叫一下 —— 包括 execute_at 在未来 1 小时的延迟任务。
                # 若 claim 不看 execute_at，那条任务会当场被抢走并执行，
                # 延迟执行彻底失效（先落库、再立刻跑，比不延迟还糟：
                # 它看起来还是"定时"的）。
                #
                # 这个条件与 list_due 的那条是同一个判断，重复是必要的：
                # list_due 决定"扫描时捞谁"，这里决定"被叫到时能不能跑"。
                # 投递的路径不止扫描一条（post-commit 钩子、relay 都会推），
                # 所以到期这件事必须由**执行前的最后一道闸**来保证。
                or_(
                    _T.c.execute_at.is_(None),
                    _T.c.execute_at <= func.now(),
                ),
            )
            .values(
                status=TaskStatus.RUNNING.value,
                # 自增写在 SQL 里，不是在 Python 里 +1：
                # 后者要先把值读出来，那中间又有窗口。
                attempt=_T.c.attempt + 1,
                started_at=func.now(),
                lease_expires_at=_lease_deadline(lease_seconds),
                leased_by=leased_by,
                updated_at=func.now(),
            )
            .returning(*_T.c)
        )

        row = (await self._session.execute(
            statement
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    async def mark_success(
            self,
            task_id: UUID,
    ) -> Task | None:

        statement = (
            update(_T)
            .where(
                _T.c.task_id == task_id,
                # 只在 running 时生效：任务可能已经被取消，
                # 那时"执行者改回成功"会让取消形同虚设。
                _T.c.status == TaskStatus.RUNNING.value,
            )
            .values(
                status=TaskStatus.SUCCESS.value,
                finished_at=func.now(),
                lease_expires_at=None,
                leased_by=None,
                # 终态**释放幂等键** —— 见模块顶部「幂等键的生命周期」。
                idempotency_key=_release_key(),
                updated_at=func.now(),
            )
            .returning(*_T.c)
        )

        row = (await self._session.execute(
            statement
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    async def mark_failure(
            self,
            task_id: UUID,
            *,
            error: str,
            retry_delay_seconds: float | None,
    ) -> Task | None:

        retryable = retry_delay_seconds is not None

        values: dict[str, Any] = {
            "status": (
                TaskStatus.RETRYABLE.value
                if retryable
                else TaskStatus.FAILED.value
            ),
            # error 不因后续成功而清掉；失败成功与否都先记下这次的原因。
            "error": error,
            "lease_expires_at": None,
            "leased_by": None,
            "updated_at": func.now(),
            # 只有终态才记 finished_at：retryable 的任务还没结束。
            "finished_at": None if retryable else func.now(),
        }

        if retryable:
            values["execute_at"] = _lease_deadline(
                float(retry_delay_seconds)
            )

        else:
            # 终态（试满了）才释放幂等键；retryable 那条**必须留着** ——
            # 否则它等重试的这段时间里，同一个键的重复提交会再建一条任务，
            # 而两条都会跑。见模块顶部「幂等键的生命周期」。
            values["idempotency_key"] = _release_key()

        statement = (
            update(_T)
            .where(
                _T.c.task_id == task_id,
                _T.c.status == TaskStatus.RUNNING.value,
            )
            .values(**values)
            .returning(*_T.c)
        )

        row = (await self._session.execute(
            statement
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    async def mark_cancelled(
            self,
            task_id: UUID,
    ) -> bool:

        statement = (
            update(_T)
            .where(
                _T.c.task_id == task_id,
                # 只拦非终态。已经跑完/失败的任务"取消"来晚了，
                # 不是错误，返回 False 让调用方自己决定怎么答。
                _T.c.status.in_(_OPEN_VALUES),
            )
            .values(
                status=TaskStatus.CANCELLED.value,
                finished_at=func.now(),
                lease_expires_at=None,
                leased_by=None,
                # 终态**释放幂等键** —— 取消也是终态，见模块顶部的说明。
                idempotency_key=_release_key(),
                updated_at=func.now(),
            )
            .returning(_T.c.task_id)
        )

        row = (await self._session.execute(
            statement
        )).one_or_none()

        return row is not None

    async def schedule_retry(
            self,
            task_id: UUID,
    ) -> Task | None:

        statement = (
            update(_T)
            .where(
                _T.c.task_id == task_id,
                # 只从"已终止且不是成功"进入。success 的重跑会产生
                # 第二次外部副作用 —— 那该提交一条新任务，不是 retry。
                _T.c.status.in_((
                    TaskStatus.FAILED.value,
                    TaskStatus.CANCELLED.value,
                )),
            )
            .values(
                status=TaskStatus.PENDING.value,
                # attempt 清零，见契约里的说明：不清零的话，
                # 跑满次数的任务 retry 后一失败就又成终态，接口等于没用。
                attempt=0,
                execute_at=func.now(),
                finished_at=None,
                lease_expires_at=None,
                leased_by=None,
                updated_at=func.now(),
                # ⚠️ error 刻意**不**清：它回答的是"上一次为什么失败"，
                # 而人工重跑之前正需要看到这个。
            )
            .returning(*_T.c)
        )

        row = (await self._session.execute(
            statement
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    # ========================================================
    # 读取
    # ========================================================

    async def reschedule(
            self,
            task_id: UUID,
            *,
            execute_at: datetime | None,
    ) -> Task | None:

        statement = (
            update(_T)
            .where(
                _T.c.task_id == task_id,
                # 只在可 claim 时生效。跑到一半的任务改时间没有意义
                # （它马上就写回结果了），终态的更不能改 —— 那会让
                # "已完成"变成"还会再跑"。
                _T.c.status.in_(_CLAIMABLE_VALUES),
            )
            .values(
                execute_at=execute_at,
                # 自增写在 SQL 里，不是在 Python 里读出来 +1：
                # 两个调用方同时重排时，后者必须拿到比前者更大的值，
                # 否则两个定时会认为自己都有效。
                generation=_T.c.generation + 1,
                updated_at=func.now(),
            )
            .returning(*_T.c)
        )

        row = (await self._session.execute(
            statement
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    async def next_scheduled(
            self,
    ) -> Task | None:

        statement = (
            select(*_T.c)
            .where(
                _T.c.status.in_(_CLAIMABLE_VALUES),
                # 还没到点的。execute_at 为 NULL 的任务属于"立即"，
                # 它会出现在 list_due 里，不该被当成"下一个闹钟" ——
                # 否则定时器会为了它睡 0 秒空转。
                _T.c.execute_at.is_not(None),
                _T.c.execute_at > func.now(),
            )
            .order_by(
                _T.c.execute_at.asc(),
                # 同一时刻的多条要有稳定顺序，否则每轮绑定的可能是不同的
                # 那一条，日志上看起来像在反复抖动。
                _T.c.created_at.asc(),
                _T.c.task_id.asc(),
            )
            .limit(1)
        )

        row = (await self._session.execute(
            statement
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    async def get(
            self,
            task_id: UUID,
    ) -> Task | None:

        row = (await self._session.execute(
            select(*_T.c).where(_T.c.task_id == task_id)
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    async def get_by_idempotency_key(
            self,
            idempotency_key: str,
    ) -> Task | None:

        row = (await self._session.execute(
            select(*_T.c).where(
                _T.c.idempotency_key == idempotency_key
            )
        )).one_or_none()

        return None if row is None else _to_task(row._mapping)

    async def list_due(
            self,
            *,
            limit: int,
    ) -> list[UUID]:

        statement = (
            select(_T.c.task_id)
            .where(
                _T.c.status.in_(_CLAIMABLE_VALUES),
                or_(
                    # execute_at 为 NULL = 立即执行。注意不能写成
                    # `execute_at <= now()` —— SQL 里 NULL <= x 是 NULL，
                    # 会把所有立即任务静默漏掉（这是这类查询最常见的 bug）。
                    _T.c.execute_at.is_(None),
                    _T.c.execute_at <= func.now(),
                ),
            )
            .order_by(
                # NULLS FIRST：立即任务排在定时任务前面。少了它，
                # PostgreSQL 默认 NULLS LAST，一堆"立即"任务会被
                # 排到一堆"未来某个时刻"的任务后面。
                _T.c.execute_at.asc().nulls_first(),
                # 同刻的按创建顺序，保证结果稳定（否则同一批里
                # 谁先谁后由执行计划决定，测试会随机失败）。
                _T.c.created_at.asc(),
            )
            .limit(limit)
        )

        result = await self._session.execute(statement)

        return list(result.scalars())

    async def reclaim_expired(
            self,
            *,
            limit: int,
    ) -> list[UUID]:

        # UPDATE 在 PostgreSQL 里不支持 LIMIT，所以先按条件选出要收的
        # 一批 task_id，再拿它当 UPDATE 的筛选条件。子查询在同一张表上
        # 是允许的：它只求值一次，与 UPDATE 的目标行集不互相影响。
        expired = (
            select(_T.c.task_id)
            .where(
                _T.c.status == TaskStatus.RUNNING.value,
                # lease_expires_at 为 NULL 的 running 行不该存在
                # （claim 一定写租约），但真出现了也不该被这里当成
                # "过期"无限次收回 —— 显式排除掉更安全。
                _T.c.lease_expires_at.is_not(None),
                _T.c.lease_expires_at <= func.now(),
            )
            .order_by(_T.c.lease_expires_at.asc())
            .limit(limit)
        )

        # attempt 已用满的收回为 failed，而不是 retryable ——
        # 否则一个"一跑就把 Worker 搞崩"的任务会在
        # running -> retryable -> running 之间永远循环，把实例一个个拖死。
        exhausted = _T.c.attempt >= _T.c.max_attempts

        statement = (
            update(_T)
            .where(_T.c.task_id.in_(expired))
            .values(
                status=case(
                    (exhausted, TaskStatus.FAILED.value),
                    else_=TaskStatus.RETRYABLE.value,
                ),
                finished_at=case(
                    (exhausted, func.now()),
                    else_=None,
                ),
                # 追加而不是覆盖：上一次失败的原因还有价值，
                # 而"租约过期"是这次的新事实。原来没有 error 时
                # 直接写这句（避免出现开头孤零零一个分号）。
                error=case(
                    (
                        _T.c.error.is_(None),
                        literal(_LEASE_LOST_NOTE),
                    ),
                    else_=func.concat(
                        _T.c.error,
                        literal("; " + _LEASE_LOST_NOTE),
                    ),
                ),
                lease_expires_at=None,
                leased_by=None,
                # 与上面的状态分支**同步**：试满了（终态）释放幂等键，
                # 还能重试的留着 —— 理由见模块顶部「幂等键的生命周期」。
                idempotency_key=case(
                    (exhausted, _release_key()),
                    else_=_T.c.idempotency_key,
                ),
                updated_at=func.now(),
            )
            .returning(_T.c.task_id)
        )

        result = await self._session.execute(statement)

        return list(result.scalars())

    # ========================================================
    # 查询
    # ========================================================

    def _conditions(
            self,
            *,
            status: TaskStatus | None,
            handler: str | None,
            root_task_id: UUID | None,
            parent_task_id: UUID | None,
    ) -> list[Any]:
        """list 与 count 共用的筛选条件。

        抽出来是为了让两者**永远一致** —— 分开写的话，以后加一个筛选
        条件很容易只加到一处，结果是"分页总数和实际列表对不上"。
        """

        conditions: list[Any] = []

        if status is not None:
            conditions.append(_T.c.status == status.value)

        if handler is not None:
            conditions.append(_T.c.handler == handler)

        if root_task_id is not None:
            conditions.append(_T.c.root_task_id == root_task_id)

        if parent_task_id is not None:
            conditions.append(
                _T.c.parent_task_id == parent_task_id
            )

        return conditions

    async def list(
            self,
            *,
            status: TaskStatus | None = None,
            handler: str | None = None,
            root_task_id: UUID | None = None,
            parent_task_id: UUID | None = None,
            limit: int = 50,
            offset: int = 0,
    ) -> list[Task]:

        statement = select(*_T.c)

        conditions = self._conditions(
            status=status,
            handler=handler,
            root_task_id=root_task_id,
            parent_task_id=parent_task_id,
        )

        if conditions:
            statement = statement.where(*conditions)

        # created_at 单列不足以稳定排序（同一批同时插入的会并列），
        # 补一个 task_id 让分页不会漏行或重复行。
        statement = statement.order_by(
            _T.c.created_at.desc(),
            _T.c.task_id.desc(),
        ).limit(limit).offset(offset)

        result = await self._session.execute(statement)

        return [_to_task(r._mapping) for r in result]

    async def count(
            self,
            *,
            status: TaskStatus | None = None,
            handler: str | None = None,
            root_task_id: UUID | None = None,
            parent_task_id: UUID | None = None,
    ) -> int:

        statement = select(func.count()).select_from(_T)

        conditions = self._conditions(
            status=status,
            handler=handler,
            root_task_id=root_task_id,
            parent_task_id=parent_task_id,
        )

        if conditions:
            statement = statement.where(*conditions)

        return (await self._session.execute(
            statement
        )).scalar_one()

    async def list_lineage(
            self,
            root_task_id: UUID,
    ) -> list[Task]:

        statement = (
            select(*_T.c)
            .where(
                or_(
                    _T.c.root_task_id == root_task_id,
                    # ⚠️ 这个 OR 是刻意的保护，不是冗余。
                    #
                    # root_task_id 由 TaskService.submit() 补齐（seal_root），
                    # 正常情况下根任务自己的 root_task_id == task_id。
                    # 但若某条根任务的这个字段是 NULL（历史数据、或将来
                    # 有人绕过 submit 直接写库），单靠上面那个条件就会让
                    # **根任务自己从它自己的执行链里消失** —— 链上少一条，
                    # 而且不报错。
                    #
                    # 调用方问的是"以 X 为根的那条链，含 X"，这里就照这个
                    # 意思回答。or_ 不会造成重复行（是行级筛选，不是连接）。
                    _T.c.task_id == root_task_id,
                )
            )
            # 升序：链条是因果叙事，倒着读没有意义。
            # 同样补 task_id 保证同刻插入的多条有稳定顺序。
            .order_by(
                _T.c.created_at.asc(),
                _T.c.task_id.asc(),
            )
        )

        result = await self._session.execute(statement)

        return [_to_task(r._mapping) for r in result]
