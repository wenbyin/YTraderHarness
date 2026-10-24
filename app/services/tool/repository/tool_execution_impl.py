"""
PostgreSQL implementation of ToolExecutionRepository.

事务规则
--------
本仓储**从不**调用 session.commit() / session.rollback() ——
与其余仓储同一条规则，事务边界由调用方决定。

⚠️ advisory lock 与读 / 写必须在同一个事务里
------------------------------------------
`pg_advisory_xact_lock` 的锁随事务结束释放。调用方（服务层）在
一个 `factory.scope(begin=True)` 里依次调用 lock -> 读 -> 写，
提交时一起释放。拆到两个事务里等于没加锁。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import ToolExecutionStatus
from ..models.sqlalchemy.tool_execution import (
    ToolExecution as ToolExecutionModel,
)
from .tool_execution import ToolExecutionRepository

__all__ = ["ToolExecutionRepositoryImpl"]


def _to_entity(
        model: ToolExecutionModel,
) -> ToolExecution:

    return ToolExecution.model_validate(model)


class ToolExecutionRepositoryImpl(ToolExecutionRepository):

    # ========================================================
    # 并发
    # ========================================================

    async def lock(self, match_key: str) -> None:
        """
        事务级 advisory lock。

        `hashtextextended(text, 0)` 返回 bigint，正好是
        `pg_advisory_xact_lock(bigint)` 要的形状。用扩展版而不是
        `hashtext`：后者是 int4，碰撞概率更高；而且这里只需要
        "同一个 key 串行化"，碰撞只会多排队，不会出错。
        """

        await self._session.execute(
            select(
                func.pg_advisory_xact_lock(
                    func.hashtextextended(match_key, 0)
                )
            )
        )

    # ========================================================
    # 读
    # ========================================================

    async def find_by_call_id(
            self,
            *,
            run_id: UUID,
            tool_call_id: str,
    ) -> ToolExecution | None:

        result = await self._session.execute(
            select(ToolExecutionModel)
            .where(
                ToolExecutionModel.run_id == run_id,
                ToolExecutionModel.tool_call_id == tool_call_id,
            )
            .order_by(
                ToolExecutionModel.created_at.desc(),
                ToolExecutionModel.execution_id.desc(),
            )
            .limit(1)
        )

        row = result.scalar_one_or_none()

        return None if row is None else _to_entity(row)

    async def status_counts(
            self,
            *,
            match_key: str,
            window_start: datetime | None,
    ) -> dict[ToolExecutionStatus, int]:

        conditions = [
            ToolExecutionModel.match_key == match_key,
            ToolExecutionModel.counted.is_(True),
        ]

        if window_start is not None:
            conditions.append(
                ToolExecutionModel.created_at >= window_start
            )

        result = await self._session.execute(
            select(
                ToolExecutionModel.status,
                func.count(),
            )
            .where(*conditions)
            .group_by(ToolExecutionModel.status)
        )

        return {
            ToolExecutionStatus(status): int(count)
            for status, count in result.all()
        }

    # ========================================================
    # 写
    # ========================================================

    async def insert(
            self,
            execution: ToolExecution,
    ) -> ToolExecution:

        now = datetime.now(timezone.utc)

        statement = (
            insert(ToolExecutionModel)
            .values(
                execution_id=execution.execution_id,
                tool_key=execution.tool_key,
                tool_id=execution.tool_id,
                session_id=execution.session_id,
                run_id=execution.run_id,
                user_id=execution.user_id,
                scope_type=execution.scope_type.value,
                scope_key=execution.scope_key,
                execution_type=execution.execution_type.value,
                param_hash=execution.param_hash,
                match_key=execution.match_key,
                status=execution.status.value,
                counted=execution.counted,
                tool_call_id=execution.tool_call_id,
                result=execution.result,
                error=execution.error,
                created_at=now,
                updated_at=now,
            )
            .returning(ToolExecutionModel)
        )

        result = await self._session.execute(statement)

        row = result.scalar_one_or_none()

        if row is None:
            raise LookupError(
                "执行记录插入失败：数据库没有返回行。"
            )

        return _to_entity(row)

    async def reopen(
            self,
            execution_id: UUID,
    ) -> ToolExecution:

        statement = (
            update(ToolExecutionModel)
            .where(
                ToolExecutionModel.execution_id == execution_id
            )
            .values(
                status=ToolExecutionStatus.PENDING.value,
                counted=True,
                result=None,
                error=None,
                updated_at=func.now(),
            )
            .returning(ToolExecutionModel)
        )

        result = await self._session.execute(statement)

        row = result.scalar_one_or_none()

        if row is None:
            raise LookupError(
                f"执行记录不存在：{execution_id}。"
                "它是本次调用查出来的，取不到说明 execution_id 对不上，"
                "或那一行被删了。"
            )

        return _to_entity(row)

    async def settle(
            self,
            execution_id: UUID,
            *,
            status: ToolExecutionStatus,
            counted: bool,
            result: Any = None,
            error: dict[str, Any] | None = None,
    ) -> ToolExecution:

        statement = (
            update(ToolExecutionModel)
            .where(
                ToolExecutionModel.execution_id == execution_id
            )
            .values(
                status=status.value,
                counted=counted,
                result=result,
                error=error,
                updated_at=func.now(),
            )
            .returning(ToolExecutionModel)
        )

        result_row = await self._session.execute(statement)

        row = result_row.scalar_one_or_none()

        if row is None:
            raise LookupError(
                f"tool_execution 记录不存在：{execution_id}。"
                "它是本次调用插入的，取不到说明 execution_id "
                "对不上，或那一行被删了。"
            )

        return _to_entity(row)
