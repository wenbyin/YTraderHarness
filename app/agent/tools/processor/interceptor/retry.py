"""
工具重试：包住 `next_call()`，按配置决定要不要再试、试几次、退避多久。

挂法
----
`AROUND + SYNC`。与 timeout 拆开：

    retry (outer) -> timeout (inner)   每次尝试各有自己的超时（推荐）
    timeout (outer) -> retry (inner)   整个重试组共享一个总预算

与执行守卫的顺序
---------------
    retry (outer) -> guard (inner)     每次尝试都走守卫，失败/成功分别计数
    guard (outer) -> retry (inner)     整个重试组只占一次额度

推荐 retry 在外层：否则中间失败的尝试对守卫不可见，error 配额也不会
按尝试累计。

⚠️ 只重试显式配置的错误
----------------------
默认 `retry_on_error=false`，即只重试超时（`ToolTimeout` /
`TimeoutError`）。其它异常（HTTP 5xx、业务错误、参数错误）默认
**不重试**；确认工具幂等后，把 `retry_on_error` 打开即可重试所有
`outcome.error`。

永远不重试：
    ToolCallDenied      工具根本没跑；再试一次也只会再被拒
    GraphInterrupt      langgraph 的控制流，重试会把审批变成循环
    CancelledError      取消不是失败

⚠️ 副作用工具默认不要挂
----------------------
每次 `next_call()` 都会重新走一遍内层链。工具自己的 `factory.scope`
事务会回滚，但**外部副作用不会**（退款、下单已经发出去了）。只读 /
幂等工具才适合重试。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, ConfigDict, Field

from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

from ..base import ToolInterceptor
from ..context import (
    AsyncToolCall,
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "RetryConfig",
    "RetryProcessor",
]

logger = logging.getLogger(__name__)

#: 无论如何都不重试的异常。
#:
#: 这三个不是"工具失败了"，而是控制流 / 未执行 —— 重试它们只会
#: 制造循环或掩盖取消。
_NEVER_RETRY: tuple[type[BaseException], ...] = (
    ToolCallDenied,
    GraphInterrupt,
    asyncio.CancelledError,
)


class RetryConfig(BaseModel):
    """重试参数。字段保持扁平，前端直接按 properties 生成控件。"""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        title="最大尝试次数",
        description=(
            "包含第一次在内的总尝试次数。1 = 不重试；3 = 最多执行 "
            "3 次（2 次重试）。"
        ),
    )

    retry_on_error: bool = Field(
        default=False,
        title="重试所有失败",
        description=(
            "关闭（默认）：只重试超时（ToolTimeout / TimeoutError）。"
            "打开：任何 outcome.error 都重试。"
            "ToolCallDenied / GraphInterrupt / CancelledError 永远不重试。"
        ),
    )

    backoff_seconds: float = Field(
        default=0.5,
        ge=0,
        le=60,
        title="首次退避（秒）",
        description=(
            "第一次重试前等待的秒数。按 backoff_multiplier 逐次放大。"
        ),
    )

    backoff_multiplier: float = Field(
        default=2.0,
        ge=1,
        le=10,
        title="退避倍数",
        description="每次重试的等待时间乘数。",
    )

    max_backoff_seconds: float = Field(
        default=30.0,
        ge=0,
        le=300,
        title="退避上限（秒）",
        description="等待时间不会超过这个值。",
    )



class RetryProcessor(ToolInterceptor):
    """
    AROUND 重试处理器。

    只负责"再试几次、什么时候再试"；单次能等多久由 timeout 处理器负责。
    """

    name = "retry_interceptor"

    display_name = "工具重试"

    description = (
        "按配置重试失败的工具调用：默认只重试超时，也可重试所有错误；支持指数退避。"
    )

    config_model: ClassVar[type[BaseModel]] = RetryConfig

    supported_hooks: ClassVar[frozenset[ProcessorHook]] = frozenset({
        ProcessorHook.AROUND,
    })

    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.SYNC})

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

        except Exception:  # noqa: BLE001 - 配置坏了也不该打挂工具

            # 典型是旧配置（还带 retry_on 白名单数组）。这里记 ERROR
            # 后按「不重试」执行一次，至少让工具本身跑起来，而不是
            # 让整轮对话 failed。
            logger.error(
                "retry_interceptor 配置无效，本次不重试：%s",
                config,
                exc_info=True,
            )

            return await next_call()

        last: ToolOutcome | None = None

        for attempt in range(1, parsed.max_attempts + 1):

            outcome = await next_call()

            if outcome.ok:
                return outcome

            last = outcome

            if not self._retryable(
                    outcome.error, parsed.retry_on_error
            ):
                return outcome

            if attempt >= parsed.max_attempts:
                break

            delay = min(
                parsed.backoff_seconds
                * (parsed.backoff_multiplier ** (attempt - 1)),
                parsed.max_backoff_seconds,
            )

            logger.warning(
                "%s：%s 第 %d/%d 次失败（%s），%.2f 秒后重试。",
                self.get_key(),
                invocation.tool_key,
                attempt,
                parsed.max_attempts,
                type(outcome.error).__name__,
                delay,
            )

            from ..trace import emit_tool_retry

            emit_tool_retry(
                runtime,
                tool_name=invocation.tool_key,
                next_attempt=attempt + 1,
                delay_ms=int(delay * 1000),
                reason=(
                    type(outcome.error).__name__
                    if outcome.error is not None
                    else None
                ),
            )

            if delay > 0:
                await asyncio.sleep(delay)

        return last if last is not None else ToolOutcome()

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _retryable(
            error: BaseException | None,
            retry_on_error: bool,
    ) -> bool:
        """
        这个 outcome.error 要不要重试。

            retry_on_error=False   只重试超时（ToolTimeout / TimeoutError）
            retry_on_error=True    任何 outcome.error 都重试

        两种模式都先排除 ToolCallDenied / GraphInterrupt /
        CancelledError —— 它们不是"失败得不够久"。
        """

        if error is None:
            return False

        if isinstance(error, _NEVER_RETRY):
            return False

        if retry_on_error:
            return True

        # ToolTimeout 是 TimeoutError 的子类，一个 isinstance 覆盖两者。
        return isinstance(error, TimeoutError)
