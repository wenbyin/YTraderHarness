"""
工具结果缓存：AROUND 一条，执行前查、成功后回填。

挂法
----
只支持 `AROUND + SYNC`。执行前查缓存：

    命中 -> 直接返回 ToolOutcome(value=..., cached=True)，工具不执行
    未命中 -> await next_call()，成功才回填

为什么不再支持 BEFORE + AFTER
---------------------------
拆成两半时缓存落在调用链的固定位置，看起来与重试无关；但它也把
"查"与"回填"拆成两条配置，两边的 `ttl_seconds` 必须手写一致，
漏一处就会出现"存进去的键与查的键不是一个"——表现是命中率恒为 0，
且没有任何报错。AROUND 一条记录没有这个问题。

⚠️ 不干扰流程
------------
    没有缓存       -> 正常执行
    执行失败       -> **不缓存**，原样返回失败结果
    缓存读写失败   -> 运行时实现（RedisToolCache）fail-open：
                      按未命中 / 未写入处理，只打 warning

缓存是优化，不是正确性的一环。任何缓存层的问题都不该让工具调用失败。

⚠️ 只缓存成功
------------
把失败也缓存起来会让一次偶发故障在 TTL 内变成持续故障，而调用方
看到的是"这个工具一直坏着"。真要抑制故障时的重复调用，那是熔断，
不是缓存 —— 两者的失效条件不一样。

⚠️ 凭什么认定"同一个调用"
------------------------
键 = **方法名 + 全部参数的摘要**，由缓存自己算，**不由 config 传**：

    cache_key = f"{tool_key}:{param_hash_of(arguments)}"

因此：

    参数不同        不命中（对）
    参数相同、链不同  命中（对：缓存的是**工具的结果**）
    参数含不可序列化对象  用 str 兜底（可能不稳定，见 param_hash_of）

不再提供 `key_fields`：判重维度只有一个——"同方法 + 完全相同的
参数"，少一个配置项就少一处"库里存的是 A、查的是 B"的漂移。

⚠️ 不含租户 / 用户 / 会话
------------------------
调用者身份在 `invocation.context[CALLER_CONTEXT_KEY]`，不在
`invocation.arguments` 里（工具入参 schema 不能暴露 user_id，
否则模型能填别人的 id）。因此：

    结果依赖 `current_tool_caller()` 的工具**不能挂缓存** ——
    参数相同就会命中同一条缓存，A 用户可能拿到 B 用户的结果。

这类工具要么别挂缓存，要么让参数里带一个能区分租户 / 用户的字段
（由工具自己的业务参数决定，缓存不替它判断）。

⚠️ 与执行守卫的顺序
------------------
守卫在外层、缓存在内层：命中时守卫已经预占了一条记录，命中结果会
让守卫把它落成 `SUCCESS, counted=False`，因此**不占**执行配额。

缓存在外层、守卫在内层：命中时守卫根本不会运行 —— 历史执行记录里
不会出现这一次调用。两种都成立，但审计口径不同，挂载时要想清楚。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.services.tool.application._execution import (
    execution_result,
    param_hash_of,
)

from ..base import ToolInterceptor
from ..context import (
    AsyncToolCall,
    ToolInvocation,
    ToolOutcome,
)
from ..runtime.cache import get_tool_cache
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "CacheConfig",
    "CacheToolProcessor",
]

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """
    缓存参数。title / description 是对外契约（前端表单）。

    ⚠️ 判重维度不开放配置
    --------------------
    缓存键固定是「方法名 + 全部参数的 sha256」，由处理器自己算。
    不再提供 `key_fields` —— 少一个配置项就少一处键漂移。

    `extra="forbid"`：旧配置里如果还带 `key_fields`，会被明确拒绝；
    `around` 捕获配置错误后按"跳过缓存"处理，不影响工具调用。
    """

    model_config = ConfigDict(extra="forbid")

    ttl_seconds: int = Field(
        default=300,
        ge=0,
        le=86_400,
        title="有效期",
        description=(
            "Redis TTL（秒）。0 表示**不缓存**（等于把这个处理器关掉，"
            "但保留在链上）。超过一天的值不建议：工具结果多半与"
            "实时数据有关，缓存一整天的东西没人敢用。"
        ),
    )


class CacheToolProcessor(ToolInterceptor):
    """
    执行前查缓存；未命中则执行，**成功**后回填。

    AROUND-only。缓存实现由运行时提供（生产 Redis，见
    runtime/cache.py），本处理器只认 get / put 两个动作。
    """

    name = "cache_interceptor"

    display_name = "结果缓存"

    description = (
        "缓存工具调用结果，命中时直接返回、不执行工具。"
    )

    config_model: ClassVar[type[BaseModel]] = CacheConfig

    #: 只有 AROUND：查与回填必须在同一个处理器里，键才不会漂。
    supported_hooks: ClassVar[frozenset[ProcessorHook]] = frozenset({
        ProcessorHook.AROUND,
    })

    #: AROUND 必须 SYNC（基类会校验）。
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

        try:
            parsed = self.config_model.model_validate(config)

        except Exception:  # noqa: BLE001 - 缓存配置坏了也不能拦工具

            # 旧配置里可能还带 key_fields（新 schema 已 forbid）。
            # 缓存是旁路：配置不合法就这一次不缓存，工具照常执行。
            logger.warning(
                "%s 的缓存配置无效，本次跳过缓存：%s",
                invocation.tool_key,
                config,
                exc_info=True,
            )

            return await next_call()

        key = self._cache_key(invocation)

        hit, value = await get_tool_cache().get(key)

        if hit:
            logger.debug(
                "%s：%s 命中缓存。", self.get_key(), invocation.tool_key
            )

            # cached=True 是必须的：观测侧靠它区分"真的执行了一次"与
            # "一次命中"，执行守卫也靠它把这次预占标成不占额度。
            from ..trace import emit_tool_idempotency

            emit_tool_idempotency(
                runtime,
                tool_name=invocation.tool_key,
                decision="cache_hit",
                scope="cache",
            )

            return ToolOutcome(value=value, cached=True)

        outcome = await next_call()

        # 只回填成功结果；失败 / 已缓存的结果都不写。
        if outcome.ok and not outcome.cached:
            await get_tool_cache().put(
                key,
                execution_result(outcome.value),
                ttl_seconds=parsed.ttl_seconds,
            )

        return outcome

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _cache_key(
            invocation: ToolInvocation,
    ) -> str:
        """
        方法名 + 全部参数的摘要 —— 由缓存自己算，不由 config 传。

            cache_key = f"{tool_key}:{param_hash_of(arguments)}"

        `param_hash_of` 与执行账本的判重键是同一个算法
        （canonical_json + sha256）：同一份参数在缓存和账本里得到
        同一个摘要，不会出现"账本判成同一次、缓存判成两次"。

        ⚠️ **不包含** session / run / user —— 调用者身份不在
        arguments 里，见模块 docstring 的隔离警告。参数含不可
        JSON 化对象时 `param_hash_of` 用 str 兜底；默认 repr 含
        内存地址，可能导致命中率恒为 0，所以工具参数请用可序列化
        的形态。
        """

        return (
            f"{invocation.tool_key}:"
            f"{param_hash_of(invocation.arguments)}"
        )
