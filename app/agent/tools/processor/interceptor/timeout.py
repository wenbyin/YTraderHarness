"""
工具执行超时：包住 `next_call()`，超过预算就取消并返回 `ToolTimeout`。

挂法
----
`AROUND + SYNC`。与 retry 拆开：

    retry (outer) -> timeout (inner)   每次尝试各有自己的超时
    timeout (outer) -> retry (inner)   整个重试组共享一个总预算

默认建议 timeout 在 retry 内层。

⚠️ 它能取消什么、不能取消什么
---------------------------
`asyncio.wait_for` 取消的是**协程**。工具里的 `await`（HTTP、数据库、
`asyncio.sleep`）会被取消；阻塞式 I/O 与 CPU 密集代码**不会**被中断，
超时只能等到它自己返回。给这类工具配 timeout 时要知道它形同虚设。

⚠️ 超时 ≠ 没有副作用
------------------
工具可能已经把请求发出去了，只是没等到响应。因此超时返回的是
`ToolOutcome(error=ToolTimeout)`，执行守卫会按 `tool.started` 把它
归成 UNKNOWN（可能已产生副作用），而不是普通 FAILED。

⚠️ 不要用 shield
---------------
`asyncio.shield` 会让超时后工具继续在后台跑 —— 既没有被取消，调用方
又已经收到失败，是最坏的组合。这里刻意不用。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

from ..base import ToolInterceptor
from ..context import (
    AsyncToolCall,
    ToolInvocation,
    ToolOutcome,
    ToolTimeout,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "TimeoutConfig",
    "TimeoutProcessor",
]

logger = logging.getLogger(__name__)


class TimeoutConfig(BaseModel):
    """超时参数。字段保持扁平，前端直接按 properties 生成控件。"""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=3600,
        title="超时秒数",
        description=(
            "单次工具调用最多等待多少秒。超时后取消调用并返回 "
            "ToolTimeout；工具可能已经产生副作用，守卫会按 UNKNOWN "
            "处理。"
        ),
    )


class TimeoutProcessor(ToolInterceptor):
    """
    AROUND 超时处理器。

    只负责"等多久"；重试次数与退避由 retry 处理器负责。
    """

    name = "timeout_interceptor"

    display_name = "工具超时"

    description = (
        "限制单次工具调用的执行时长，超时返回 ToolTimeout。"
    )

    config_model: ClassVar[type[BaseModel]] = TimeoutConfig

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

        parsed = self.config_model.model_validate(config)

        try:
            return await asyncio.wait_for(
                next_call(),
                timeout=parsed.timeout_seconds,
            )

        except asyncio.TimeoutError:

            logger.warning(
                "%s：%s 超过 %.2f 秒，已取消。",
                self.get_key(),
                invocation.tool_key,
                parsed.timeout_seconds,
            )

            return ToolOutcome(
                error=ToolTimeout(parsed.timeout_seconds)
            )
