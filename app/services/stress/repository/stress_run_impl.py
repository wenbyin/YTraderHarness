"""
PostgreSQL implementation of StressRunStore.

写法与 task_impl.py 一致：直接走 Core（select / insert / update 作用于
模型表），不碰 ORM 的 identity map。

为什么不用 ORM 对象
------------------
状态推进必须是**条件 UPDATE**（例如「只有还在 queued 才改成 running」），
而 ORM 的「读出来 -> 改属性 -> flush」表达不了这个条件：读与写之间
存在窗口，取消可能刚好落进去。

读也走 ``select(table)`` 而不是 ``session.get``，于是 identity map
全程为空 —— 这里大量 Core UPDATE 绕过 identity map 且不会更新它，
两种写法混用会出现「读到的还是旧状态」且不报错。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, insert, or_, select, update

from ..domain.entities import (
    StressRunCursor,
    StressRunMetricsRecord,
    StressRunRecord,
)
from ..domain.schemas.stress import AggregationStatus, StressRunStatus
from ..models.sqlalchemy.stress_run import (
    StressRun as StressRunModel,
)
from ..models.sqlalchemy.stress_run import (
    StressRunMetrics as StressRunMetricsModel,
)
from .stress_run import StressRunStore

__all__ = ["StressRunStoreImpl"]

#: 运行表的 Core 句柄。
_R = StressRunModel.__table__

#: 聚合快照表的 Core 句柄。
_M = StressRunMetricsModel.__table__

#: 还能被取消 / 判失败的状态。已终结的状态不在里面。
_OPEN_VALUES: tuple[str, ...] = (
    StressRunStatus.QUEUED.value,
    StressRunStatus.RUNNING.value,
)


# ============================================================
# 行 -> 实体
# ============================================================


def _to_run(
        row: Mapping[str, Any],
) -> StressRunRecord:
    """stress_runs 的一行 -> StressRunRecord。"""

    return StressRunRecord(
        run_id=row["run_id"],
        name=row["name"],
        status=StressRunStatus(row["status"]),
        config=row["config"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_by=row["created_by"],
    )


def _to_metrics(
        row: Mapping[str, Any],
) -> StressRunMetricsRecord:
    """
    stress_run_metrics 的一行 -> StressRunMetricsRecord。

    ⚠️ NUMERIC 列从 asyncpg 回来是 ``Decimal``。在这里转成 float，而不是
    让 DTO 去接 —— Pydantic 的 float 字段虽然也接受 Decimal，但那样
    「实体里是什么类型」就取决于驱动，冒烟与测试看不到同一形状。
    """

    def _num(value: Any) -> float | None:
        return None if value is None else float(value)

    return StressRunMetricsRecord(
        run_id=row["run_id"],
        aggregation_status=AggregationStatus(row["aggregation_status"]),
        aggregation_algorithm=row["aggregation_algorithm"],
        finalized_at=row["finalized_at"],
        sessions_started=row["sessions_started"],
        sessions_completed=row["sessions_completed"],
        sessions_failed=row["sessions_failed"],
        sessions_abandoned=row["sessions_abandoned"],
        achieved_sessions_per_second=_num(
            row["achieved_sessions_per_second"]
        ),
        session_latency=row["session_latency"],
        sampled_turns_per_session=row["sampled_turns_per_session"],
        turns_per_session=row["turns_per_session"],
        turns_started=row["turns_started"],
        turns_completed=row["turns_completed"],
        turns_succeeded=row["turns_succeeded"],
        turns_failed=row["turns_failed"],
        turns_timed_out=row["turns_timed_out"],
        turns_cancelled=row["turns_cancelled"],
        achieved_turns_per_second=_num(
            row["achieved_turns_per_second"]
        ),
        turn_error_rate=_num(row["turn_error_rate"]),
        turn_latency=row["turn_latency"],
        time_to_first_output=row["time_to_first_output"],
        turn_latency_by_index=row["turn_latency_by_index"],
        first_turn_latency=row["first_turn_latency"],
        later_turn_latency=row["later_turn_latency"],
        trace_counts=row["trace_counts"],
        model_metrics=row["model_metrics"],
        node_metrics=row["node_metrics"],
        tool_metrics=row["tool_metrics"],
        fanout_metrics=row["fanout_metrics"],
        model_input_tokens_sum=row["model_input_tokens_sum"],
        model_output_tokens_sum=row["model_output_tokens_sum"],
        events_seen=row["events_seen"],
        events_deduplicated=row["events_deduplicated"],
        events_unmatched=row["events_unmatched"],
        events_late=row["events_late"],
    )


def _insert_run_values(
        run: StressRunRecord,
) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "name": run.name,
        "status": run.status.value,
        "config": run.config,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error_code": run.error_code,
        "error_message": run.error_message,
        "created_by": run.created_by,
    }


def _metrics_update_values(
        metrics: StressRunMetricsRecord,
) -> dict[str, Any]:
    """
    运行中/中断快照的列值。

    与 ``_metrics_values`` 的唯一区别是 ``finalized_at``：collecting
    快照必须保持 NULL（运行还没结束），incomplete 才写终止时刻。不能用
    ``metrics.finalized_at or func.now()`` —— 那会把每次周期刷新都标成
    “已定稿”。
    """

    values = _metrics_values(metrics)

    values["finalized_at"] = (
        None
        if metrics.aggregation_status == AggregationStatus.COLLECTING
        else metrics.finalized_at
    )

    return values


def _metrics_values(
        metrics: StressRunMetricsRecord,
) -> dict[str, Any]:
    """实体 -> 快照更新的列值。"""

    return {
        "aggregation_status": metrics.aggregation_status.value,
        "aggregation_algorithm": metrics.aggregation_algorithm,
        # None 时用数据库时钟，而不是 Python 的 now()：
        # 快照定稿时刻应当与 finished_at 用同一个时钟。
        "finalized_at": metrics.finalized_at or func.now(),
        "sessions_started": metrics.sessions_started,
        "sessions_completed": metrics.sessions_completed,
        "sessions_failed": metrics.sessions_failed,
        "sessions_abandoned": metrics.sessions_abandoned,
        "achieved_sessions_per_second": (
            metrics.achieved_sessions_per_second
        ),
        "session_latency": metrics.session_latency,
        "sampled_turns_per_session": (
            metrics.sampled_turns_per_session
        ),
        "turns_per_session": metrics.turns_per_session,
        "turns_started": metrics.turns_started,
        "turns_completed": metrics.turns_completed,
        "turns_succeeded": metrics.turns_succeeded,
        "turns_failed": metrics.turns_failed,
        "turns_timed_out": metrics.turns_timed_out,
        "turns_cancelled": metrics.turns_cancelled,
        "achieved_turns_per_second": (
            metrics.achieved_turns_per_second
        ),
        "turn_error_rate": metrics.turn_error_rate,
        "turn_latency": metrics.turn_latency,
        "time_to_first_output": metrics.time_to_first_output,
        "turn_latency_by_index": metrics.turn_latency_by_index,
        "first_turn_latency": metrics.first_turn_latency,
        "later_turn_latency": metrics.later_turn_latency,
        "trace_counts": metrics.trace_counts,
        "model_metrics": metrics.model_metrics,
        "node_metrics": metrics.node_metrics,
        "tool_metrics": metrics.tool_metrics,
        "fanout_metrics": metrics.fanout_metrics,
        "model_input_tokens_sum": metrics.model_input_tokens_sum,
        "model_output_tokens_sum": metrics.model_output_tokens_sum,
        "events_seen": metrics.events_seen,
        "events_deduplicated": metrics.events_deduplicated,
        "events_unmatched": metrics.events_unmatched,
        "events_late": metrics.events_late,
        "updated_at": func.now(),
    }


# ============================================================
# 仓储实现
# ============================================================


class StressRunStoreImpl(StressRunStore):

    # --------------------------------------------------------
    # 写入
    # --------------------------------------------------------

    async def add_run(
            self,
            run: StressRunRecord,
    ) -> StressRunRecord:

        statement = (
            insert(_R)
            .values(**_insert_run_values(run))
            .returning(*_R.c)
        )

        row = (await self._session.execute(statement)).one()

        return _to_run(row._mapping)

    async def add_metrics(
            self,
            run_id: UUID,
    ) -> StressRunMetricsRecord:

        # 只给 run_id，其余列全部交给 server_default 填 —— 这样「新快照
        # 应该长什么样」只有 schema 一处定义，不在这里再抄一份默认值。
        statement = (
            insert(_M)
            .values(run_id=run_id)
            .returning(*_M.c)
        )

        row = (await self._session.execute(statement)).one()

        return _to_metrics(row._mapping)

    async def mark_running(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:

        statement = (
            update(_R)
            .where(
                _R.c.run_id == run_id,
                # 条件本身就是并发保护：已取消时影响 0 行。
                _R.c.status == StressRunStatus.QUEUED.value,
            )
            .values(
                status=StressRunStatus.RUNNING.value,
                started_at=func.now(),
            )
            .returning(*_R.c)
        )

        row = (await self._session.execute(statement)).one_or_none()

        return None if row is None else _to_run(row._mapping)

    async def mark_cancelled(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:

        statement = (
            update(_R)
            .where(
                _R.c.run_id == run_id,
                _R.c.status.in_(_OPEN_VALUES),
            )
            .values(
                status=StressRunStatus.CANCELLED.value,
                finished_at=func.now(),
            )
            .returning(*_R.c)
        )

        row = (await self._session.execute(statement)).one_or_none()

        if row is None:
            return None

        # 快照永远等不到定稿了，如实标成 incomplete。
        # 只动还在 collecting 的行：已经 final 的不该被取消改掉。
        await self._session.execute(
            update(_M)
            .where(
                _M.c.run_id == run_id,
                _M.c.aggregation_status
                == AggregationStatus.COLLECTING.value,
            )
            .values(
                aggregation_status=AggregationStatus.INCOMPLETE.value,
                finalized_at=func.now(),
                updated_at=func.now(),
            )
        )

        return _to_run(row._mapping)

    async def mark_failed(
            self,
            run_id: UUID,
            *,
            error_code: str | None,
            error_message: str | None,
    ) -> StressRunRecord | None:

        statement = (
            update(_R)
            .where(
                _R.c.run_id == run_id,
                _R.c.status.in_(_OPEN_VALUES),
            )
            .values(
                status=StressRunStatus.FAILED.value,
                finished_at=func.now(),
                error_code=error_code,
                error_message=error_message,
            )
            .returning(*_R.c)
        )

        row = (await self._session.execute(statement)).one_or_none()

        if row is None:
            return None

        await self._session.execute(
            update(_M)
            .where(
                _M.c.run_id == run_id,
                _M.c.aggregation_status
                == AggregationStatus.COLLECTING.value,
            )
            .values(
                aggregation_status=AggregationStatus.INCOMPLETE.value,
                finalized_at=func.now(),
                updated_at=func.now(),
            )
        )

        return _to_run(row._mapping)

    async def update_metrics(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunMetricsRecord | None:

        statement = (
            update(_M)
            .where(
                _M.c.run_id == run_id,
                # final 不可覆盖；collecting / incomplete 都可以继续写。
                _M.c.aggregation_status
                != AggregationStatus.FINAL.value,
            )
            .values(**_metrics_update_values(metrics))
            .returning(*_M.c)
        )

        row = (await self._session.execute(statement)).one_or_none()

        return None if row is None else _to_metrics(row._mapping)

    async def complete_run(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunRecord | None:

        statement = (
            update(_R)
            .where(
                _R.c.run_id == run_id,
                # 只从 running 进入。取消 / 失败先到的话，这里影响 0 行，
                # 下面的快照也不会被写成 final —— 取消赢在竞态里。
                _R.c.status == StressRunStatus.RUNNING.value,
            )
            .values(
                status=StressRunStatus.COMPLETED.value,
                finished_at=func.now(),
            )
            .returning(*_R.c)
        )

        row = (await self._session.execute(statement)).one_or_none()

        if row is None:
            return None

        await self._session.execute(
            update(_M)
            .where(_M.c.run_id == run_id)
            .values(**_metrics_values(metrics))
        )

        return _to_run(row._mapping)

    # --------------------------------------------------------
    # 读取
    # --------------------------------------------------------

    async def get_run(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:

        row = (await self._session.execute(
            select(*_R.c).where(_R.c.run_id == run_id)
        )).one_or_none()

        return None if row is None else _to_run(row._mapping)

    async def get_metrics(
            self,
            run_id: UUID,
    ) -> StressRunMetricsRecord | None:

        row = (await self._session.execute(
            select(*_M.c).where(_M.c.run_id == run_id)
        )).one_or_none()

        return None if row is None else _to_metrics(row._mapping)

    async def list_runs(
            self,
            *,
            status: StressRunStatus | None = None,
            limit: int = 20,
            cursor: StressRunCursor | None = None,
    ) -> list[StressRunRecord]:

        conditions: list[Any] = []

        if status is not None:
            conditions.append(_R.c.status == status.value)

        if cursor is not None:
            # keyset 分页：严格排在游标之后的行。
            #
            # 两列都要比较。只比 created_at 时，同一微秒创建的多条要么
            # 全被跳过、要么全被重复取到 —— 生产上很难复现，用户只会
            # 觉得「翻页偶尔少一条」。
            conditions.append(
                or_(
                    _R.c.created_at < cursor.created_at,
                    and_(
                        _R.c.created_at == cursor.created_at,
                        _R.c.run_id < cursor.run_id,
                    ),
                )
            )

        statement = select(*_R.c)

        if conditions:
            statement = statement.where(*conditions)

        statement = (
            statement
            .order_by(
                _R.c.created_at.desc(),
                _R.c.run_id.desc(),
            )
            .limit(limit)
        )

        result = await self._session.execute(statement)

        return [_to_run(row._mapping) for row in result]
