"""
ToolExecutionService 的默认实现。

真正做判断的地方有两块：

    _execution.decide_existing  同一次 tool_call_id 已有记录
    _execution.decide_new       新调用的状态 / 次数策略

两个都是纯函数，因此策略矩阵可以脱离数据库验证。本层只负责：
加锁、把记录读出来、按判定写回。

⚠️ 顺序：lock -> 读 existing -> 读 latest/count -> 写
---------------------------------------------------
锁必须在**所有**读之前拿。先读再锁的写法里，两个并发分支可能
同时读到「还没满」，然后都在锁内插入 —— 锁一点用都没有。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from app.common.service.base import ServiceContext

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import (
    ExecutionDecision,
    ExecutionMatch,
    GuardAction,
    ToolExecutionStatus,
)
from ..domain.schemas.execution import (
    ExecutionGuardRequest,
    ExecutionReservation,
)
from ..repository.tool_execution import ToolExecutionRepository
from ._execution import (
    decide_existing,
    decide_new,
    execution_error,
    match_key_of,
    param_hash_of,
    scope_key_of,
)
from .tool_execution_service import ToolExecutionService

__all__ = ["ToolExecutionServiceImpl"]

logger = logging.getLogger(__name__)


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
    # 预占
    # ========================================================

    async def reserve(
            self,
            request: ExecutionGuardRequest,
    ) -> ExecutionReservation:

        param_hash = (
            param_hash_of(request.arguments)
            if request.execution is ExecutionMatch.INVOKE
            else ""
        )

        # scope_key 缺失会抛 ValueError —— 拦截器把它转成 fail-closed
        # 的拒绝。不要在这里兜底造一个键：那会把不同的会话 / 用户
        # 混进同一个计数桶，而且不会报错。
        scope_key = scope_key_of(
            request.scope,
            session_id=request.session_id,
            run_id=request.run_id,
            user_id=request.user_id,
        )

        match_key = match_key_of(
            tool_key=request.tool_key,
            scope=request.scope,
            scope_key=scope_key,
            execution=request.execution,
            param_hash=param_hash,
        )

        window_start = (
            None
            if request.window_seconds is None
            else datetime.now(timezone.utc)
            - timedelta(seconds=request.window_seconds)
        )

        # ⚠️ 锁必须在任何读之前。事务由调用方的 scope 提供。
        await self._executions.lock(match_key)

        existing = None

        if request.run_id is not None and request.tool_call_id:

            existing = await self._executions.find_by_call_id(
                run_id=request.run_id,
                tool_call_id=request.tool_call_id,
            )

        if existing is not None:

            decision, reason = decide_existing(existing)

            if decision is ExecutionDecision.REPLAY:
                return ExecutionReservation(
                    decision=ExecutionDecision.REPLAY,
                    attempt=existing,
                )

            if decision is ExecutionDecision.BLOCK:
                return ExecutionReservation(
                    decision=ExecutionDecision.BLOCK,
                    attempt=existing,
                    reason=reason,
                )

            # EXECUTE：resume / 可重试的旧调用。继续走配额判定 ——
            # 额度可能已经被别的调用用掉了。

        counts = await self._executions.status_counts(
            match_key=match_key,
            window_start=window_start,
        )

        decision, reason = decide_new(
            counts=counts,
            success=request.success,
            error=request.error,
            unknown=request.unknown,
        )

        if decision is ExecutionDecision.BLOCK:

            blocked = existing

            if blocked is None:
                blocked = await self._record_block(
                    request,
                    scope_key=scope_key,
                    match_key=match_key,
                    param_hash=param_hash,
                    reason=reason,
                )

            logger.warning(
                "执行守卫拦截 %s（scope=%s, execution=%s, counts=%s）：%s",
                request.tool_key,
                request.scope.value,
                request.execution.value,
                {
                    status.value: count
                    for status, count in counts.items()
                },
                reason,
            )

            return ExecutionReservation(
                decision=ExecutionDecision.BLOCK,
                attempt=blocked,
                reason=reason,
            )

        self._log_allowed_overflow(request, counts)

        if existing is not None:
            attempt = await self._executions.reopen(
                existing.execution_id
            )
        else:
            now = datetime.now(timezone.utc)

            attempt = await self._executions.insert(
                ToolExecution(
                    execution_id=uuid4(),
                    tool_key=request.tool_key,
                    tool_id=request.tool_id,
                    session_id=request.session_id,
                    run_id=request.run_id,
                    user_id=request.user_id,
                    scope_type=request.scope,
                    scope_key=scope_key,
                    execution_type=request.execution,
                    param_hash=param_hash,
                    match_key=match_key,
                    status=ToolExecutionStatus.PENDING,
                    counted=True,
                    tool_call_id=request.tool_call_id,
                    result=None,
                    error=None,
                    created_at=now,
                    updated_at=now,
                )
            )

        return ExecutionReservation(
            decision=ExecutionDecision.EXECUTE,
            attempt=attempt,
        )

    # ========================================================
    # 落终态
    # ========================================================

    async def settle(
            self,
            execution_id: UUID,
            *,
            status: ToolExecutionStatus,
            counted: bool,
            result: Any = None,
            error: dict[str, Any] | None = None,
            error_exc: BaseException | None = None,
    ) -> ToolExecution:

        # ⚠️ PENDING 是 reserve 的产物，不是能 settle 过去的状态 ——
        # 放它进来等于开一条"把正在跑的记录退回起点"的路。
        if status is ToolExecutionStatus.PENDING:
            raise ValueError(
                "settle 不接受 PENDING：它是 reserve 的产物。"
                "要表达'跑起来了但结果不明'，用 RUNNING / UNKNOWN。"
            )

        if error is None and error_exc is not None:
            error = execution_error(error_exc)

        return await self._executions.settle(
            execution_id,
            status=status,
            counted=counted,
            result=result,
            error=error,
        )

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _log_allowed_overflow(
            request: ExecutionGuardRequest,
            counts: dict[ToolExecutionStatus, int],
    ) -> None:
        """
        action=ALLOW 的超限只告警不拦。

        仍然打 warning，否则「灰度模式」就没有观察价值 —— 上线前
        要靠它判断正式切 BLOCK 会不会误伤。
        """

        success_count = (
            counts.get(ToolExecutionStatus.SUCCESS, 0)
            + counts.get(ToolExecutionStatus.PENDING, 0)
            + counts.get(ToolExecutionStatus.RUNNING, 0)
        )

        for label, rule, count in (
            ("success", request.success, success_count),
            ("unknown", request.unknown,
             counts.get(ToolExecutionStatus.UNKNOWN, 0)),
            ("error", request.error,
             counts.get(ToolExecutionStatus.FAILED, 0)),
        ):

            if (
                    rule.limit is not None
                    and count >= rule.limit
                    and rule.action is GuardAction.ALLOW
            ):
                logger.warning(
                    "执行守卫已超限但策略为放行：%s（scope=%s, "
                    "execution=%s, status=%s, count=%s, limit=%s）",
                    request.tool_key,
                    request.scope.value,
                    request.execution.value,
                    label,
                    count,
                    rule.limit,
                )

    async def _record_block(
            self,
            request: ExecutionGuardRequest,
            *,
            scope_key: str,
            match_key: str,
            param_hash: str,
            reason: str | None,
    ) -> ToolExecution:
        """
        为新工具调用写一条 BLOCKED 审计行。

        counted=False：被拦下的调用没有执行，不占额度。它存在的意义
        是让运维看得到「模型在什么时候、被什么策略挡住」。

        ⚠️ 只在 `existing is None` 时调用。同一次 tool_call_id 已经有
        记录时再插一行会撞上部分唯一索引 —— 那种情况下旧行本身就是
        审计记录。
        """

        now = datetime.now(timezone.utc)

        return await self._executions.insert(
            ToolExecution(
                execution_id=uuid4(),
                tool_key=request.tool_key,
                tool_id=request.tool_id,
                session_id=request.session_id,
                run_id=request.run_id,
                user_id=request.user_id,
                scope_type=request.scope,
                scope_key=scope_key,
                execution_type=request.execution,
                param_hash=param_hash,
                match_key=match_key,
                status=ToolExecutionStatus.BLOCKED,
                counted=False,
                tool_call_id=request.tool_call_id,
                result=None,
                error={
                    "type": "ExecutionGuardBlocked",
                    "message": reason or "执行守卫拦截",
                },
                created_at=now,
                updated_at=now,
            )
        )
