"""
调用者鉴权拦截器：这个工具**只对某些角色/群组开放**。

它拦在工具链的最前面（BEFORE），不满足就短路成 `ToolCallDenied` ——
工具一次都不会被调用。

与 auth 层的分工
---------------
    app/common/auth          **登录门槛**：token 有效吗、在不在
                             `required_groups` 里。不过就 403，
                             整个应用都进不来。
    本处理器                  **工具门槛**：这个人调**这一个工具**
                             够不够格。要求是挂在工具上的配置，
                             不是全局策略。

所以两者检查的是同一批数据（token 的 claim），但粒度与后果不同：
全局那道拦不住"能登录、但不该动某个工具"的情形，而那正是这里要处理的。

⚠️ 身份是**快照**，不是此刻的 Casdoor 状态
----------------------------------------
agent 运行是异步派发的（请求提交后才跑），跑的时候手里没有 token，
而 Casdoor 是用户的唯一真相、本地没有用户表 —— 所以角色/群组是在
**建运行的那一刻**从 claim 快照下来的（见 agent_run.caller_identity），
由执行器放进 `invocation.context`（见 `CALLER_CONTEXT_KEY`）。

因此它反映的是"发起这次运行的人当时的身份"。token 里 `groups` 本来
就只在登录/刷新时更新，两者是同一种新鲜度。

⚠️ 拿不到身份时默认**拒绝**
--------------------------
`on_missing` 默认 `deny`，与审批拦截器在拿不到审批通道时的默认是同一条
理由：一个挂在链上、却无从判断的关卡，唯一安全的默认是不放行。
"没有身份"与"身份里没有这个角色"是两件不同的事，但对**放行**这个
决定来说结论一样 —— 都不能证明够格。真要在没有身份时放行（比如
离线脚本），显式把 `on_missing` 配成 `allow`。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

from ..base import ToolInterceptor
from ..context import (
    CALLER_CONTEXT_KEY,
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = ["AuthConfig", "AuthProcessor"]

logger = logging.getLogger(__name__)


class AuthConfig(BaseModel):
    """鉴权参数。title / description 是对外契约（前端表单）。"""

    require_roles: list[str] = Field(
        default_factory=list,
        title="要求的角色",
        description=(
            "命中其中**任意一个**即通过。留空 = 不按角色限制。"
            "与 require_group 同时配置时两者都要满足。"
        ),
    )

    require_group: str | None = Field(
        default=None,
        title="要求的群组",
        description=(
            "必须是调用者所属的群组之一。留空 = 不按群组限制。\n"
            "写 token 里的原样形式（`/auth/me` 返回的 groups 就是它）——"
            "群组名可能带 owner 前缀，形如 `y-trader/Y-Trader-User`。"
        ),
    )

    on_missing: Literal["deny", "allow"] = Field(
        default="deny",
        title="拿不到身份时",
        description=(
            "deny（默认）= 拒绝，fail-closed；allow = 放行。\n"
            "只有在调用方本来就带不了身份的场景（离线脚本、内部批处理）"
            "才该改成 allow —— 那等于把这个关卡关掉。"
        ),
    )

    @model_validator(mode="after")
    def _require_at_least_one(self) -> "AuthConfig":
        """
        两个条件都空 = 这个处理器什么都不做。

        ⚠️ 在**挂载期**报错，而不是运行期放行：挂一个不做事的鉴权关卡
        比没挂更危险 —— 链上看起来有鉴权，实际每次调用都放过去，而且
        没有任何迹象。挂载校验会把 config 过一遍这个模型（见
        tool_processor_service_impl._validate_mount），因此这里抛的
        ValidationError 会变成挂载接口的 400，写配置的人当场就知道。
        """

        # ⚠️ 顺序要紧：**先归一化，再判空**。
        #
        # 反过来（先判空后归一）会让 `require_group="   "` 从缝里过去：
        # 判空时那是个非空字符串，于是"至少配一个"通过；紧接着它被归一
        # 成 None —— 结果是一个两个条件都空的配置**合法地**落了库，
        # 也就是一个不拦任何东西的鉴权关卡。这正是上面那段要防的东西。
        #
        # 空字符串与"没配"是同一件事：`require_group=""` 不会命中任何
        # 群组，读起来却像是配了。
        if self.require_group is not None and not self.require_group.strip():
            self.require_group = None

        if not self.require_roles and not self.require_group:
            raise ValueError(
                "checkauth 至少要配 require_roles 或 require_group 之一。"
                "两个都空等于不鉴权 —— 要放行所有调用就不要挂这个"
                "处理器，而不是挂一个空的上来。"
            )

        return self


def _matches_required_group(
        groups: list[str],
        required: str,
) -> bool:
    """
    `required` 是否在调用者的群组里。

    **精确匹配**，两种情况都算（与 config 的说明一致）：

        required == 调用者某个群组名              原样写
        required 是 `owner/name` 形式且原样相等    限定名原样写

    刻意**不做**"取 / 之后那段来比"的宽松匹配：那会让**任何** owner
    下的同名群组都通过（`evil-org/Y-Trader-User` 也算），而群组名是
    管理员造的 —— auth 层拒绝过这种写法，同一条理由在这里同样成立。

    代价是配置里必须写成 token 里的那个形式。auth 层之所以能容忍两种
    形式，是因为它手里有 `org_name` 可以拼出限定名；工具级这里只有
    快照，没有那个上下文。`/auth/me` 返回的 groups 就是该写的原文。
    """

    target = required.strip()

    return bool(target) and target in groups


class AuthProcessor(ToolInterceptor):
    """
    按调用者的角色/群组决定放不放行。

    ⚠️ 放行 = 返回原样的 `invocation`（不碰参数）。本处理器**不改
    数据**，它只是决定这次调用能不能发生 —— 改数据是 transformer 的
    事（见 base.py 对三个 type 的分工）。
    """

    name = "auth_interceptor"

    display_name = "调用者鉴权"

    description = (
        "按调用者的角色 / 群组放行或拒绝这次工具调用。"
        "拿不到身份时默认拒绝。"
    )

    config_model: ClassVar[type[BaseModel]] = AuthConfig

    #: 只做 BEFORE：它是"要不要让这次调用发生"的判断，跑在工具之前。
    #: 没有 AFTER / AROUND 可做 —— 调用已经过去了再判合不合规没有意义。
    supported_hooks: ClassVar[frozenset[ProcessorHook]] = frozenset(
        {ProcessorHook.BEFORE}
    )

    #: 只支持 SYNC：ASYNC 的返回值会被丢弃（见 base.py），而本处理器
    #: 靠**返回值**短路 —— 不被 await 的拒绝等于没拒绝，那会让
    #: "配了鉴权却没生效"变成一个静默事实。
    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.SYNC})

    # ========================================================
    # BEFORE：判
    # ========================================================

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation | ToolOutcome:

        parsed = self.config_model.model_validate(config)

        caller = invocation.context.get(CALLER_CONTEXT_KEY)

        if not isinstance(caller, dict):
            return self._missing(
                parsed, invocation, reason="invocation.context 里没有调用者身份"
            )

        roles = [
            str(item) for item in (caller.get("roles") or ())
        ]
        groups = [
            str(item) for item in (caller.get("groups") or ())
        ]

        if parsed.require_roles and not set(
                parsed.require_roles
        ) & set(roles):
            return self._deny(
                parsed,
                invocation,
                reason="角色不足",
                detail=(
                    f"要求 {parsed.require_roles}，"
                    f"调用者持有 {roles}"
                ),
            )

        if (
                parsed.require_group is not None
                and not _matches_required_group(
            groups, parsed.require_group
        )
        ):
            return self._deny(
                parsed,
                invocation,
                reason="群组不符",
                detail=(
                    f"要求 {parsed.require_group!r}，"
                    f"调用者属于 {groups}"
                ),
            )

        return invocation

    # ========================================================
    # internals
    # ========================================================

    def _deny(
            self,
            parsed: AuthConfig,
            invocation: ToolInvocation,
            *,
            reason: str,
            detail: str,
    ) -> ToolOutcome:
        """
        拒绝。

        ⚠️ 给模型的文案**刻意含糊**（与 auth 层 403 同一条口径：只说
        不符合，不说要求什么），细节全部留在服务端日志里。理由：这段
        文案会经 ToolMessage 回到模型、进而可能被转述给用户，而"这个
        工具需要哪些角色"属于部署的权限结构，不该由一次失败的调用
        泄出去。排查的人看日志，那里有完整的比对结果。
        """

        logger.warning(
            "checkauth：拒绝 %s（%s）—— %s。",
            invocation.tool_key,
            reason,
            detail,
        )

        return ToolOutcome(
            error=ToolCallDenied(
                f"{self.get_key()}：当前调用者不满足 {invocation.tool_key} "
                "所要求的角色 / 群组，未执行。"
            )
        )

    def _missing(
            self,
            parsed: AuthConfig,
            invocation: ToolInvocation,
            *,
            reason: str,
    ) -> ToolInvocation | ToolOutcome:
        """
        拿不到身份。按 `on_missing` 决定 —— 默认拒绝（见模块 docstring）。
        """

        if parsed.on_missing == "allow":
            logger.warning(
                "checkauth：%s 放行了一个**无法鉴权**的调用（%s）——"
                "on_missing=allow。",
                invocation.tool_key,
                reason,
            )

            return invocation

        logger.warning(
            "checkauth：拒绝 %s —— %s（on_missing=deny）。",
            invocation.tool_key,
            reason,
        )

        return ToolOutcome(
            error=ToolCallDenied(
                f"{self.get_key()}：无法确认调用者身份，"
                f"{invocation.tool_key} 未执行。"
            )
        )
