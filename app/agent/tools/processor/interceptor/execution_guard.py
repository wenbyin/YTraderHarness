"""
执行守卫：按维度计数，达到上限 / 命中状态策略时拦下工具调用。

它替代了旧版在 `graph/react/tools.py` 里手写的 `tool_execution` 账本。
旧账本回答的是「同一件事是不是已经做过」（一个判重键一行），
守卫回答的是「这个维度上已经发生过几次、这次还让不让做」
（一次执行一行，按维度聚合）。

两个维度
--------
    scope      SESSION / RUN / USER / GLOBAL
    execution  METHOD（只看工具）/ INVOKE（工具 + 相同参数）

组合出的 8 种命中方式由 `ExecutionGuardRequest` 落成 `match_key`。
计数、加锁、索引都按它走。

处理策略：三种状态独立配额（不是三选一）
----------------------------------------
    success_limit / success_action   SUCCESS 的次数上限与动作
    error_limit   / error_action     FAILED  的次数上限与动作
    unknown_limit / unknown_action   UNKNOWN 的次数上限与动作

    limit=1 + BLOCK  -> 允许 1 次，第 2 次开始拦
    limit=3 + BLOCK  -> 允许 3 次，第 4 次开始拦
    limit=None       -> 不按这个状态限制
    action=ALLOW     -> 仍然计数，但不拦（灰度 / 告警）

字段保持**扁平**：前端直接按 config_schema 的 properties 生成控件，
嵌套模型会输出 $ref，当前前端解析不了。

判定顺序：success -> unknown -> error。某个状态达到上限且动作是
BLOCK 就停；ALLOW 则继续看下一个。默认值：

    success: limit=3, BLOCK
    error:   limit=3, BLOCK
    unknown: limit=1, BLOCK（fail-closed）

⚠️ success 的计数包含 PENDING / RUNNING
--------------------------------------
在途尝试先占一个成功名额，否则两个并发分支会同时看到「还没有
成功」而都执行。落终态后这一行会变成 SUCCESS / FAILED / UNKNOWN，
名额随之转移。

⚠️ submit 工具（退款 / 下单）
----------------------------
「成功一次后永久禁止再次提交；失败最多重试 3 次」：

    scope=USER, execution=INVOKE, window_seconds=None,
    success_limit=1, success_action=BLOCK,
    error_limit=3,   error_action=BLOCK,
    unknown_limit=1, unknown_action=BLOCK

同一用户 + 同一组参数成功过一次就永久拦住；失败可以再试，
第 4 次失败开始拦；unknown 一次就 fail-closed。

⚠️ 为什么它必须是 AROUND
-----------------------
它要在工具跑**之前**预占一次额度，在跑完（或中断/取消/拒绝）之后
落终态。BEFORE 拿不到结果，AFTER 已经来不及阻止执行。

⚠️ 中断 / 取消 / 拒绝的记账
--------------------------
    GraphInterrupt（工具未开始）  INTERRUPT, counted=False，原样抛出
    GraphInterrupt（工具已开始）  UNKNOWN,   counted=True，原样抛出
    CancelledError（未开始）      CANCELLED, counted=False，原样抛出
    CancelledError（已开始）      UNKNOWN,   counted=True，原样抛出
    ToolCallDenied               REJECTED,  counted=False
    普通异常                      FAILED,    counted=True
    成功                          SUCCESS,   counted=True
    缓存命中（outcome.cached）     SUCCESS,   counted=False

`tool.started` 由 `ToolExecutor._base_call` 在真正调用工具方法前写入
（见 processor/context.py）。

⚠️ 它只在挂了这条处理器的工具上生效
--------------------------------
处理器链按 `tool_id` 从库里读，没有链（工具还没同步进库、链读取
失败）时 AROUND 根本不会执行。这是「把账本搬进链里」的代价：
改造后，没挂守卫的工具**没有**执行记录。上线时需要给需要保护的工具
挂上本处理器（AROUND 两条记录、SYNC）。

⚠️ 与其它 AROUND 的顺序
---------------------
守卫在外层：一次用户调用占一格，内层重试消耗的是同一次额度。
守卫在内层：每次重试各占一格。默认建议放最外层（sequence 最小），
这样「次数」的语义是「模型要求做这件事几次」，而不是「底层重试几次」。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import UUID

from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, ConfigDict, Field

from app.common.factories import get_service_factory
from app.services.tool.application import ToolExecutionService
from app.services.tool.application._execution import execution_result
from app.services.tool.domain.schemas.enum import (
    ExecutionDecision,
    ExecutionMatch,
    ExecutionScope,
    GuardAction,
    ProcessorHook,
    ExecutionMode,
    ToolExecutionStatus,
)
from app.services.tool.domain.schemas.execution import (
    ExecutionGuardRequest,
    ExecutionReservation,
    ExecutionStatusLimit,
)

from ..base import ToolInterceptor
from ..trace import emit_tool_idempotency
from ..context import (
    CALLER_CONTEXT_KEY,
    RUN_CONTEXT_KEY,
    SESSION_CONTEXT_KEY,
    TOOL_STARTED_CONTEXT_KEY,
    AsyncToolCall,
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
    ToolTimeout,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "ExecutionGuardConfig",
    "ExecutionGuardProcessor",
]

logger = logging.getLogger(__name__)


class ExecutionGuardConfig(BaseModel):
    """
    守卫参数。title / description 是对外契约（前端表单）。

    三种状态各自独立计数、独立配额、独立动作 —— 不是三选一。
    判定顺序：success -> unknown -> error；某个状态超限且动作是
    BLOCK 就停，ALLOW 则继续看下一个。

    ⚠️ 字段必须保持**扁平**
    ---------------------
    管理面的前端直接读 `get_config_schema()` 生成表单。嵌套的
    pydantic 模型会输出 `$defs` + `$ref`，当前前端解析不了，表现
    为「success 是一个无法识别的对象、控件置空」。因此这里用
    `success_limit` / `success_action` 这样的平铺字段，不要改成
    `success: {limit, action}`。

    extra="forbid"：旧版嵌套配置（success/error/unknown 对象）会
    被明确拒绝，而不是被静默忽略、悄悄退回默认值。
    """

    model_config = ConfigDict(extra="forbid")

    scope: Literal["SESSION", "RUN", "USER", "GLOBAL"] = Field(
        default="SESSION",
        title="计数维度",
        description=(
            "SESSION 同一会话 / RUN 同一次运行 / USER 同一个用户 / "
            "GLOBAL 所有用户。⚠️ GLOBAL 是跨租户的：一个用户刷满额度"
            "会把所有人挡住，除非业务明确要全局熔断，不要用它。"
        ),
    )

    execution: Literal["METHOD", "INVOKE"] = Field(
        default="METHOD",
        title="命中方式",
        description=(
            "METHOD 只看工具（不同参数合并计数）/ INVOKE 工具 + "
            "相同参数（只有完全相同的调用才合并）。"
        ),
    )

    window_seconds: int | None = Field(
        default=None,
        ge=1,
        title="计数窗口（秒）",
        description=(
            "只数最近多少秒内的记录。留空 = 整个 scope 生命周期。"
            "⚠️ 对 USER / GLOBAL 来说，留空就是**永久**："
            "退款、下单这类 submit 工具常用「永久只成功一次」，"
            "但普通的读工具不要这么配，否则用户用几次就再也调不动。"
        ),
    )

    success_limit: int | None = Field(
        default=3,
        ge=0,
        title="成功次数上限",
        description=(
            "允许 SUCCESS 出现的次数；count >= limit 时按 "
            "success_action 处理。1 = 成功一次后禁止再次提交；"
            "0 = 一次都不允许成功；留空 = 不限制成功次数。"
            "⚠️ PENDING / RUNNING 也先占一个成功名额，防止并发"
            "分支同时通过。"
        ),
    )

    success_action: Literal["BLOCK", "ALLOW"] = Field(
        default="BLOCK",
        title="成功超限动作",
        description=(
            "成功次数达到上限后的动作。ALLOW 是灰度 / 告警模式："
            "仍然计数，但不拦。"
        ),
    )

    error_limit: int | None = Field(
        default=3,
        ge=0,
        title="失败次数上限",
        description=(
            "允许 FAILED 出现的次数；count >= limit 时按 "
            "error_action 处理。3 = 允许 3 次失败，第 4 次开始拦；"
            "0 = 一失败就拦；留空 = 不限制失败重试。"
        ),
    )

    error_action: Literal["BLOCK", "ALLOW"] = Field(
        default="BLOCK",
        title="失败超限动作",
        description=(
            "失败次数达到上限后的动作。ALLOW 是灰度 / 告警模式："
            "仍然计数，但不拦。"
        ),
    )

    unknown_limit: int | None = Field(
        default=1,
        ge=0,
        title="unknown 次数上限",
        description=(
            "允许 UNKNOWN 出现的次数；count >= limit 时按 "
            "unknown_action 处理。1 = 出现一次 unknown 后下次就拦"
            "（fail-closed）；留空 = 不限制。"
        ),
    )

    unknown_action: Literal["BLOCK", "ALLOW"] = Field(
        default="BLOCK",
        title="unknown 超限动作",
        description=(
            "unknown 次数达到上限后的动作。ALLOW 是灰度 / 告警模式："
            "仍然计数，但不拦。"
        ),
    )


class ExecutionGuardProcessor(ToolInterceptor):
    """
    AROUND 拦截器：预占一次额度 -> 执行 -> 落终态。

    实现只做三件事：取身份、驱动服务、把结果翻译成 ToolOutcome。
    策略判定在 `ToolExecutionService.reserve`，纯函数在
    `application/_execution.py`，两者都不依赖 langgraph。
    """

    name = "execution_guard_interceptor"

    display_name = "执行守卫"

    description = (
        "按会话 / 运行 / 用户 / 全局维度统计同名工具（或同名同参调用）"
        "的执行次数，达到上限或命中 unknown / error 策略时拦截。"
    )

    config_model: ClassVar[type[BaseModel]] = ExecutionGuardConfig

    #: 只有 AROUND：它要在执行前预占、执行后落终态。
    supported_hooks: ClassVar[frozenset[ProcessorHook]] = frozenset({
        ProcessorHook.AROUND,
    })

    #: AROUND 必然 SYNC（基类会校验）。
    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.SYNC})

    # ========================================================
    # AROUND
    # ========================================================

    async def around(
            self,
            invocation: ToolInvocation,
            next_call: AsyncToolCall,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:

        parsed = self.config_model.model_validate(config)

        identity = self._identity(invocation, runtime)

        missing = self._missing_identity(parsed, identity)

        if missing is not None:
            return self._deny(invocation, missing)

        request = ExecutionGuardRequest(
            tool_key=invocation.tool_key,
            tool_id=invocation.tool_id,
            session_id=identity["session_id"],
            run_id=identity["run_id"],
            user_id=identity["user_id"],
            tool_call_id=self._tool_call_id(invocation),
            scope=ExecutionScope(parsed.scope),
            execution=ExecutionMatch(parsed.execution),
            arguments=dict(invocation.arguments),
            window_seconds=parsed.window_seconds,
            success=ExecutionStatusLimit(
                limit=parsed.success_limit,
                action=GuardAction(parsed.success_action),
            ),
            error=ExecutionStatusLimit(
                limit=parsed.error_limit,
                action=GuardAction(parsed.error_action),
            ),
            unknown=ExecutionStatusLimit(
                limit=parsed.unknown_limit,
                action=GuardAction(parsed.unknown_action),
            ),
        )

        try:
            reservation = await _reserve(request)

        except Exception as exc:  # noqa: BLE001 - 一律 fail-closed

            # 账本用不了 -> **不执行**。一次没有记录的副作用不可追溯，
            # 而不可追溯比失败更糟（与旧账本同一条立场）。
            logger.exception(
                "执行守卫账本不可用，%s 未执行。", invocation.tool_key
            )

            return self._deny(
                invocation,
                f"执行守卫账本不可用（{type(exc).__name__}），"
                "因此未执行工具。",
            )

        return await self._after_reservation(
            invocation,
            next_call,
            reservation,
            runtime=runtime,
            scope=parsed.scope,
        )

    # ========================================================
    # 预占之后
    # ========================================================

    async def _after_reservation(
            self,
            invocation: ToolInvocation,
            next_call: AsyncToolCall,
            reservation: ExecutionReservation,
            *,
            runtime: Any = None,
            scope: Any = None,
    ) -> ToolOutcome:

        invocation.context["execution_guard.decision"] = (
            reservation.decision.value
        )

        # ---- trace: 幂等 / 配额决策 ----
        #
        # 现有实现只有三种决策（EXECUTE / BLOCK / REPLAY），提示词里的
        # 7 个值中只有 5 个能真实发生。BLOCK 再按 attempt 状态细分：
        # PENDING / RUNNING -> already_running；其余 -> quota_rejected。
        # `blocked_unknown` / `deduplicated` 没有独立分支，**不发**。
        attempt = reservation.attempt
        status = getattr(attempt, "status", None)
        status_value = (
            status.value if hasattr(status, "value") else (
                None if status is None else str(status)
            )
        )

        if reservation.decision is ExecutionDecision.REPLAY:
            decision = "already_succeeded"
        elif reservation.decision is ExecutionDecision.BLOCK:
            decision = (
                "already_running"
                if status_value in ("PENDING", "RUNNING")
                else "quota_rejected"
            )
        else:
            decision = "execute"

        emit_tool_idempotency(
            runtime,
            tool_name=invocation.tool_key,
            decision=decision,
            scope=str(scope) if scope is not None else None,
            existing_execution_status=status_value,
        )

        if reservation.decision is ExecutionDecision.BLOCK:
            return self._deny(
                invocation,
                reservation.reason or "执行守卫拦截了这次调用。",
            )

        if reservation.decision is ExecutionDecision.REPLAY:

            # 同一次工具调用已经成功过：回放当时的结果，不再执行。
            # cached=True 让观测侧把它与"一次真实执行"区分开。
            return ToolOutcome(
                value=(
                    reservation.attempt.result
                    if reservation.attempt is not None
                    else None
                ),
                cached=True,
            )

        attempt = reservation.attempt

        if attempt is None:
            # 服务层返回 EXECUTE 却没给记录 —— 装配错误。
            # 没有记录就不执行，与账本不可用同一条立场。
            return self._deny(
                invocation,
                "执行守卫没有返回预占记录，因此未执行工具。",
            )

        invocation.context["execution_guard.attempt_id"] = str(
            attempt.execution_id
        )

        try:
            outcome = await next_call()

        except GraphInterrupt:

            started = self._started(invocation)

            # 审批是 BEFORE：正常情况下工具没开始，记 INTERRUPT 且
            # 不占额度，resume 时才能继续。若将来出现 AFTER 中断，
            # started=True 会把它记成 UNKNOWN —— 可能已产生副作用。
            await _settle(
                attempt.execution_id,
                status=(
                    ToolExecutionStatus.UNKNOWN
                    if started
                    else ToolExecutionStatus.INTERRUPT
                ),
                counted=started,
            )

            raise

        except asyncio.CancelledError as exc:

            started = self._started(invocation)

            await _settle(
                attempt.execution_id,
                status=(
                    ToolExecutionStatus.UNKNOWN
                    if started
                    else ToolExecutionStatus.CANCELLED
                ),
                counted=started,
                error_exc=exc,
            )

            raise

        except BaseException as exc:  # noqa: BLE001 - 先记账再原样抛

            # 不是 Exception 的失败（KeyboardInterrupt / SystemExit /
            # 其它 BaseException）：执行可能已经开始，保守记 UNKNOWN。
            await _settle(
                attempt.execution_id,
                status=ToolExecutionStatus.UNKNOWN,
                counted=self._started(invocation),
                error_exc=exc,
            )

            raise

        status, counted = self._classify(
            outcome,
            started=self._started(invocation),
        )

        invocation.context["execution_guard.status"] = status.value

        # 成功的结果要落库（供 replay / 审计）；失败只记 error。
        await _settle(
            attempt.execution_id,
            status=status,
            counted=counted,
            result=(
                execution_result(outcome.value)
                if status is ToolExecutionStatus.SUCCESS
                else None
            ),
            error_exc=(
                None
                if status is ToolExecutionStatus.SUCCESS
                else outcome.error
            ),
        )

        return outcome

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _classify(
            outcome: ToolOutcome,
            *,
            started: bool,
    ) -> tuple[ToolExecutionStatus, bool]:
        """
        一次正常返回的 ToolOutcome -> (状态, 是否占额度)。

        `started` 是工具方法是否真的开始执行（`tool.started`），
        只有超时需要它来区分两种完全不同的情形。
        """

        if outcome.cached:
            # 缓存命中：没有真实执行，不占额度。
            return ToolExecutionStatus.SUCCESS, False

        if outcome.ok:
            return ToolExecutionStatus.SUCCESS, True

        if isinstance(outcome.error, ToolCallDenied):
            # 工具**没有跑**（审批 / 鉴权 / 参数拒绝）。
            return ToolExecutionStatus.REJECTED, False

        if isinstance(outcome.error, ToolTimeout):

            if started:
                # 已经开始执行、被 timeout 取消：外部副作用可能已经
                # 发生了，按 UNKNOWN 处理（fail-closed）。
                return ToolExecutionStatus.UNKNOWN, True

            # 还没进工具方法就超时：确认没有副作用，记 FAILED 但不占
            # 额度，避免一次"没跑"的等待消耗 error 配额。
            return ToolExecutionStatus.FAILED, False

        # 工具跑了并抛了异常。下一次是否放行由 error 配额决定；
        # 这一次仍然占 error 额度。
        return ToolExecutionStatus.FAILED, True

    @staticmethod
    def _started(invocation: ToolInvocation) -> bool:
        """工具方法是否真的开始执行过。"""

        return bool(
            invocation.context.get(TOOL_STARTED_CONTEXT_KEY, False)
        )

    @staticmethod
    def _tool_call_id(
            invocation: ToolInvocation,
    ) -> str | None:

        raw = invocation.context.get("tool_call_id")

        return str(raw) if raw else None

    @staticmethod
    def _identity(
            invocation: ToolInvocation,
            runtime: ToolRuntime | None,
    ) -> dict[str, Any]:
        """
        取 (session_id, run_id, user_id)。

        两个来源，优先级固定：

            invocation.context   执行器抄下来的（图外直调也有）
            runtime.context      兜底（老路径 / 处理器单测）

        ⚠️ 拿不到就返回 None，由 `_missing_identity` fail-closed。
        不要回落到"用会话反查用户"或造一个 UUID —— 前者多一条可能
        忘的路径，后者会把不同的会话混进同一个计数桶。
        """

        context = invocation.context

        session_id = _as_uuid(context.get(SESSION_CONTEXT_KEY))
        run_id = _as_uuid(context.get(RUN_CONTEXT_KEY))
        user_id = _user_id(context.get(CALLER_CONTEXT_KEY))

        graph_context = getattr(runtime, "context", None)

        if session_id is None:
            session_id = _as_uuid(
                getattr(graph_context, "session_id", None)
            )

        if run_id is None:
            run_id = _as_uuid(
                getattr(graph_context, "pipeline_run_id", None)
            )

        if user_id is None:
            user_id = _user_id(
                getattr(graph_context, "caller_identity", None)
            )

        return {
            "session_id": session_id,
            "run_id": run_id,
            "user_id": user_id,
        }

    @staticmethod
    def _missing_identity(
            config: ExecutionGuardConfig,
            identity: Mapping[str, Any],
    ) -> str | None:
        """scope 需要的身份缺失时，给出一条能直接看懂的原因。"""

        if (
                config.scope == ExecutionScope.SESSION.value
                and identity.get("session_id") is None
        ):
            return (
                "执行守卫需要 session_id 才能按会话计数，但这次调用"
                "没有会话身份，因此未执行工具。"
            )

        if (
                config.scope == ExecutionScope.RUN.value
                and identity.get("run_id") is None
        ):
            return (
                "执行守卫需要 run_id 才能按运行计数，但这次调用"
                "没有运行身份，因此未执行工具。"
            )

        if (
                config.scope == ExecutionScope.USER.value
                and not identity.get("user_id")
        ):
            return (
                "执行守卫需要 user_id 才能按用户计数，但这次调用"
                "没有调用者身份，因此未执行工具。"
            )

        return None

    @staticmethod
    def _deny(
            invocation: ToolInvocation,
            reason: str,
    ) -> ToolOutcome:
        """
        守卫拒绝：工具没有执行。

        用 ToolCallDenied 而不是普通 Exception：tools 节点据此把它
        写成"调用被拒绝（工具未执行）"，观测侧也会把 DENIED 与
        FAILED 分开。文案会经 ToolMessage 回到模型，所以不要带堆栈、
        路径或别人的数据。
        """

        invocation.context["execution_guard.reason"] = reason

        return ToolOutcome(error=ToolCallDenied(reason))


# ============================================================
# 服务访问
# ============================================================
#
# 单独两个模块级函数，和旧 tools.py 的 _claim_execution /
# _settle_execution 同一个理由：用例可以把它们换成内存桩，从而在没有
# Postgres 的环境里验证 AROUND 的中断 / 取消 / 结算路径。


class _GuardUnavailable(Exception):
    """守卫账本当前用不了（应用没装配 / 数据库断了）。"""


async def _reserve(
        request: ExecutionGuardRequest,
) -> ExecutionReservation:
    """开一段独立事务做预占（lock + insert 必须一起提交）。"""

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        raise _GuardUnavailable(
            f"应用未装配，执行守卫账本不可用（{exc}）。"
        ) from exc

    async with factory.scope(ToolExecutionService, begin=True) as service:
        return await service.reserve(request)


async def _settle(
        execution_id: UUID,
        *,
        status: ToolExecutionStatus,
        counted: bool,
        result: Any = None,
        error_exc: BaseException | None = None,
) -> None:
    """
    落终态。**永不抛**。

    走到这里工具已经跑过了（或已经被中断/取消）。把一次"已经发生的
    事"因为记账失败而变成一次失败，是拿观测去推翻事实。失败记
    ERROR 级日志 —— 它会让那一行停在 PENDING，下一次同名调用会
    被当成"执行中"而拒绝，所以必须看得见。
    """

    try:
        factory = get_service_factory()

        async with factory.scope(
                ToolExecutionService, begin=True
        ) as service:

            await service.settle(
                execution_id,
                status=status,
                counted=counted,
                result=result,
                error_exc=error_exc,
            )

    except Exception:  # noqa: BLE001 - 见 docstring

        logger.exception(
            "执行守卫落终态失败（execution_id=%s, status=%s）。"
            "工具本身已经跑过了；这一行会停在 PENDING，"
            "下一次同名调用会被当成执行中而拒绝。",
            execution_id,
            status.value,
        )


# ============================================================
# 小题
# ============================================================


def _as_uuid(value: Any) -> UUID | None:
    """把 context 里的值归一成 UUID；认不出来当缺失。"""

    if value is None or isinstance(value, UUID):
        return value

    try:
        return UUID(str(value))

    except (TypeError, ValueError):
        return None


def _user_id(value: Any) -> str | None:
    """
    从 caller 快照里取 user_id。

    `build_caller_context` 把缺失的 user_id 写成空串 —— 这里归一成
    None，让 `_missing_identity` 的 `not user_id` 判断成立。
    """

    if not isinstance(value, Mapping):
        return None

    raw = value.get("user_id")

    return str(raw) if raw else None
