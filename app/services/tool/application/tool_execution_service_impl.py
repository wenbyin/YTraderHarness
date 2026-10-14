"""
ToolExecutionService 的默认实现。

几乎只是转发 —— 值得留下的判断都在契约与枚举那边文档化了。
本层真正做的事只有两件：

    把 arguments 算成判重键（_execution.param_hash_of）
    把异常规范成 error 列的形状（_execution.execution_error）

两件都必须在**一个**地方做，否则它们错了不会报错，只会让判重
静默失效。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from app.common.service.base import ServiceContext

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import ToolExecutionStatus
from ..repository.tool_execution import ToolExecutionRepository
from ._execution import execution_error, param_hash_of
from .tool_execution_service import ToolExecutionService

__all__ = ["ToolExecutionServiceImpl"]


class ToolExecutionServiceImpl(ToolExecutionService):

    def __init__(
            self,
            *,
            tool_execution_repository: ToolExecutionRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._executions = tool_execution_repository

    # ========================================================
    # 抢占
    # ========================================================

    async def claim(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            tool_key: str,
            arguments: Mapping[str, Any],
    ) -> ToolExecution | None:

        return await self._executions.claim(
            session_id=session_id,
            run_id=run_id,
            tool_key=tool_key,
            param_hash=param_hash_of(arguments),
        )

    # ========================================================
    # 读
    # ========================================================

    async def find(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            tool_key: str,
            arguments: Mapping[str, Any],
    ) -> ToolExecution | None:

        return await self._executions.find(
            session_id=session_id,
            run_id=run_id,
            tool_key=tool_key,
            param_hash=param_hash_of(arguments),
        )

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
            error_exc: BaseException | None = None,
    ) -> ToolExecution:

        # ⚠️ PENDING 是 claim 的产物，不是能 settle 过去的状态 ——
        # 放它进来等于开一条"把正在跑的记录退回起点"的路，而别的
        # 分支随后就会抢走执行权、把同一个副作用跑第二遍。
        #
        # 在服务层拦而不是在仓储：这是**语义**约束，不是表结构约束
        # （DB 的 CHECK 只认取值合法，说不出"谁能写它"）。
        if status is ToolExecutionStatus.PENDING:
            raise ValueError(
                "settle 不接受 PENDING：它是 claim 的产物。"
                "要把一行置于'刚抢到、还没跑'，用 claim；"
                "要表达'跑起来了但结果不明'，用 RUNNING。"
            )

        if error is None and error_exc is not None:
            error = execution_error(error_exc)

        return await self._executions.settle(
            execution_id,
            status=status,
            result=result,
            error=error,
        )
