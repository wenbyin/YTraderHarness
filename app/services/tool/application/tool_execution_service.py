"""
Service contract for ToolExecution.

执行守卫的判定全部在这里：给定一次调用、维度配置与已有记录，
回答「执行 / 拦截 / 回放」。调用方（AROUND 拦截器）只负责把
身份与参数递进来、按结果驱动工具、最后 settle。

为什么调用方递的是 `arguments` 而不是 `param_hash`
-----------------------------------------------
判重键 / INVOKE 维度是"这次调用是不是同一件事"的定义，**属于本
服务**。让调用方自己算摘要，等于把定义复制到每一个调用点 ——
而它错了不会报错，只会静静地每次都判成"新调用"，也就是守卫完全
失效。

事务边界
--------
由调用方决定，本服务内部不 commit / rollback（与其余服务同一条）。

⚠️ `reserve` 必须在**一个事务**里完成 lock -> 读 -> 写
-----------------------------------------------------
`factory.scope(ToolExecutionService, begin=True)` 就是这个事务。
把读和写拆到两个 scope 里，并发窗口会重新打开 —— 症状是
「两个相同调用都执行了」，没有任何报错。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import ToolExecutionStatus
from ..domain.schemas.execution import (
    ExecutionGuardRequest,
    ExecutionReservation,
)

__all__ = ["ToolExecutionService"]


class ToolExecutionService(BaseService):

    # ========================================================
    # 预占
    # ========================================================

    @abstractmethod
    async def reserve(
            self,
            request: ExecutionGuardRequest,
    ) -> ExecutionReservation:
        """
        原子地判定并预占一次调用。

        返回：

            EXECUTE  attempt 非空。调用方**必须**在工具跑完（或中断 /
                     取消 / 拒绝）后调 `settle`。
            BLOCK    工具不执行，reason 说明原因。
            REPLAY   同一次工具调用已经成功过，attempt.result 是当时
                     的结果；调用方直接回放，不再执行。

        ⚠️ 并发安全
        ----------
        本方法内部先拿按 `match_key` 的 advisory lock，再读计数并写
        记录。调用方**不要**在它外面再做"先查后执行" —— 那条路的
        并发窗口正是本方法要关掉的东西。
        """

        raise NotImplementedError

    # ========================================================
    # 落终态
    # ========================================================

    @abstractmethod
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
        """
        把一条预占记录推进到终态。

        ⚠️ 只允许推进到终态，**不接受 PENDING**
        -------------------------------------
        PENDING 是 reserve 的产物。允许 settle 到 PENDING 等于开一条
        把"正在跑的记录"退回起点的路，而那条路一旦被走到，守卫的
        重放/恢复判定就会以为「还没开始」。

        `counted` 在这一步修正：拒绝 / 中断 / 取消 / 缓存命中传 False，
        真正执行过的传 True。

        error 与 error_exc 二选一
        ------------------------
        `error_exc` 是给"手上正好有个异常对象"的调用方用的便捷入口，
        本服务会把它经 _execution.execution_error 规范成存库的形状。
        两个都传以 `error` 为准（它是已经规范好的那份）。
        """

        raise NotImplementedError
