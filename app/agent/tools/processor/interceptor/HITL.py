"""
人工审批（Human In The Loop）拦截器。

挂 BEFORE：审批的意义就是"决定这次调用要不要发生"，那正是 BEFORE
唯一能做而其它两段做不到的事（AFTER 时工具已经跑完了）。

⚠️ 本次是**占位实现**，接不上真实的审批
----------------------------------------
真正的人机审批要三样东西，现在一样都没有：

    挂起   调用要能停在"等审批"这个状态上 —— 需要工具调用路径
           （仓库里目前没有 invoke 入口，app/infra/agent 是空的）
    通道   审批请求要送达某个人，结果要送回来 —— 需要审批存储
           与通知（邮件 / IM / 前端待办）
    恢复   审批通过后从挂起处继续 —— 需要可持久化的执行上下文

因此本实现**只做"读配置 → 记日志 → 按超时策略返回"**：它不等待、
不挂起、也不假装审批过了。真接上之前，它的实际行为等于"审批通道
一直超时"，也就是 `on_timeout` 那一条分支。

默认 DENY 是**刻意的 fail-closed**：一个挂在链上、却没有任何人
真的审批过的关卡，唯一安全的默认是不放行。把默认设成 ALLOW 会让
"我挂了审批"变成一句谎话 —— 而日志里一切正常，没人会发现。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field

from ..base import ToolInterceptor
from ..context import (
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "HITLConfig",
    "HITLToolProcessor",
]

logger = logging.getLogger(__name__)


class HITLConfig(BaseModel):
    """审批参数。"""

    prompt: str = Field(
        default="是否允许执行该工具调用？",
        min_length=1,
        max_length=500,
        title="审批提示",
        description="展示给审批人的问题，应当说明这次调用要做什么。",
    )

    approver_role: str | None = Field(
        default=None,
        max_length=255,
        title="审批角色",
        description=(
            "谁有权审批。留空表示不限定角色。"
            "注意这只是**路由依据**，不是权限本身 —— "
            "权限由审批系统判定，这里写什么都不构成授权。"
        ),
    )

    wait_seconds: int = Field(
        default=300,
        ge=1,
        le=86_400,
        title="等待时长",
        description="最多等审批人多久（秒），超时按下面的策略处理。",
    )

    on_timeout: Literal["DENY", "ALLOW"] = Field(
        default="DENY",
        title="超时策略",
        description=(
            "无人审批时的处理。**默认 DENY（fail-closed）**："
            "没有任何人真的审批过就放行，等于这道关卡不存在。"
        ),
    )


class HITLToolProcessor(ToolInterceptor):
    """
    调用前要人点头。

    返回值遵循 BEFORE 的契约：放行返回 invocation，拒绝返回
    `ToolOutcome(error=ToolCallDenied(...))`（短路，工具不会被调用）。
    重试拦截器据此判断"不该重试" —— 拒绝是对这次调用的判定，
    重试只会再被拒一次。
    """

    name = "hitl_processor"

    display_name = "人工审批"

    description = (
        "调用前送人工审批，未通过则拒绝该次调用。"
    )

    config_model: ClassVar[type[BaseModel]] = HITLConfig

    supported_hooks: ClassVar[frozenset[ProcessorHook]] = (
        frozenset({ProcessorHook.BEFORE})
    )

    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.SYNC})

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation | ToolOutcome:
        """
        当前实现**不会真的等人**，见模块 docstring。

        走的是超时分支，因此行为完全由 `on_timeout` 决定。
        """

        parsed = self.config_model.model_validate(config)

        if parsed.on_timeout == "ALLOW":

            logger.warning(
                "%s：审批通道尚未接通，%s 的调用按 on_timeout=ALLOW "
                "放行。（要真正审批需要工具调用路径 + 审批存储 + "
                "可恢复的执行上下文，见本模块 docstring。）",
                self.get_key(),
                invocation.tool_key,
            )

            return invocation

        logger.warning(
            "%s：审批通道尚未接通，%s 的调用按 on_timeout=DENY 拒绝。"
            "（要真正审批需要工具调用路径 + 审批存储 + 可恢复的"
            "执行上下文，见本模块 docstring。）",
            self.get_key(),
            invocation.tool_key,
        )

        return ToolOutcome(
            error=ToolCallDenied(
                f"{self.get_key()} 未获批准（审批通道尚未接通，"
                f"按 on_timeout={parsed.on_timeout} 处理）。"
            )
        )
