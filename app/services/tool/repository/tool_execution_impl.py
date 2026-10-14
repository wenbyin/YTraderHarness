"""
PostgreSQL implementation of ToolExecutionRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback() ——
与其余仓储同一条规则，事务边界由调用方决定。
"""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import (
    RETRYABLE_EXECUTION_STATUSES,
    ToolExecutionStatus,
)
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
    # 抢占
    # ========================================================

    async def claim(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            tool_key: str,
            param_hash: str,
    ) -> ToolExecution | None:

        # 一条语句同时做"查"与"占"：
        #
        #     没有冲突          -> INSERT，抢到
        #     有冲突、且状态可重跑 -> DO UPDATE 重置成 PENDING，抢到
        #     有冲突、且状态不可重跑 -> WHERE 不成立，DO UPDATE 被跳过，
        #                             RETURNING 不给行 -> 没抢到
        #
        # ⚠️ 第三条是这个设计的全部要点：**没抢到不是异常，是返回
        # None**。用捕获 IntegrityError 的写法做不到这件事 ——
        # 那只区分"插入成功/失败"，说不出"失败是因为别人在跑"还是
        # "因为已经跑成功了"，而这两者对调用方的含义完全不同。
        #
        # 用 `constraint=` 而不是 `index_elements=`：唯一键是真约束
        # （见 ORM 的 __table_args__），写死名字比列推断稳。
        retryable = [s.value for s in RETRYABLE_EXECUTION_STATUSES]

        statement = (
            insert(ToolExecutionModel)
            .values(
                execution_id=uuid.uuid4(),
                session_id=session_id,
                run_id=run_id,
                tool_key=tool_key,
                param_hash=param_hash,
                status=ToolExecutionStatus.PENDING.value,
                result=None,
                error=None,
            )
            .on_conflict_do_update(
                constraint="uq_tool_execution_identity",
                set_={
                    "status": ToolExecutionStatus.PENDING.value,
                    # 上一轮尝试的痕迹一起清掉 —— 一行只表达当前这次。
                    "result": None,
                    "error": None,
                    "updated_at": func.now(),
                },
                where=(
                    ToolExecutionModel.status.in_(retryable)
                ),
            )
            .returning(ToolExecutionModel)
        )

        result = await self._session.execute(statement)

        row = result.scalar_one_or_none()

        return None if row is None else _to_entity(row)

    # ========================================================
    # 读
    # ========================================================

    async def find(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            tool_key: str,
            param_hash: str,
    ) -> ToolExecution | None:

        result = await self._session.execute(
            select(ToolExecutionModel).where(
                ToolExecutionModel.session_id == session_id,
                ToolExecutionModel.run_id == run_id,
                ToolExecutionModel.tool_key == tool_key,
                ToolExecutionModel.param_hash == param_hash,
            )
        )

        row = result.scalar_one_or_none()

        return None if row is None else _to_entity(row)

    # ========================================================
    # 落终态
    # ========================================================

    async def settle(
            self,
            execution_id: UUID,
            *,
            status: ToolExecutionStatus,
            result: Any = None,
            error: dict[str, Any] | None = None,
    ) -> ToolExecution:

        # ⚠️ 用 `update(...).returning(...)` 而不是"取回来改再 flush"
        # ------------------------------------------------------
        # 后者有两个坑：
        #
        #   多一次往返     get() 再 flush() 是两次，而这里一次就够
        #   updated_at    列上的 onupdate=func.now() 是**数据库侧**算的。
        #                 取回来改的写法拿到的是 flush 前的旧值，想拿新值
        #                 得手写 `model.updated_at = func.now()` —— 而那样
        #                 属性上坐的是一个 SQL 表达式对象，pydantic 立刻
        #                 拒（它要 datetime），或者触发一次隐式 refresh
        #                 在 async session 上抛 MissingGreenlet。
        #
        # RETURNING 一次往返同时解决两件事：DB 算好的 updated_at、
        # 以及"到底改到没有"。
        #
        # ⚠️ 改不到就抛：那一行是我们 claim 出来的，它不在只可能是
        # execution_id 传错（装配错误）或有人手工删了行。静默返回一个
        # 假实体比抛错危险得多 —— 调用方会以为状态落库了。
        statement = (
            update(ToolExecutionModel)
            .where(
                ToolExecutionModel.execution_id == execution_id
            )
            .values(
                status=status.value,
                result=result,
                error=error,
            )
            .returning(ToolExecutionModel)
        )

        result_row = await self._session.execute(statement)

        row = result_row.scalar_one_or_none()

        if row is None:
            raise LookupError(
                f"tool_execution 记录不存在：{execution_id}。"
                "它是本次调用 claim 出来的，取不到说明 execution_id "
                "对不上，或那一行被删了。"
            )

        return _to_entity(row)
