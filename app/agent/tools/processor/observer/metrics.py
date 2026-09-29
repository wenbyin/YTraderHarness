"""
调用次数 / 耗时 / 成功率 / 错误率。

挂两半
------
    BEFORE  记下开始时刻
    AFTER   记一次调用（成败、耗时、错误类型）

两半靠 **invocation.context** 里一个带命名空间的键传递开始时刻 ——
它们共享同一个 invocation 对象，因此这是可靠的。键名带
`metrics.` 前缀，避免与调用方放进去的东西撞上。

耗时取哪个
----------
优先用 `outcome.duration_ms`（执行侧填的，只算工具本身），
拿不到才回落到"从 BEFORE 到现在"的墙钟。两者的区别很实在：
墙钟包含链上其它处理器的开销，用它当"工具耗时"会随着链变长而变大。
回落值因此在指标里**没有单独标记** —— 它就是个近似，
真要看准得让执行侧填 duration_ms。

为什么没有 config
-----------------
这个观察者没有什么值得调的：口径由 runtime/metrics.py 定死，
记哪些工具由"它挂在谁的链上"决定。硬造几个开关（采样率、
开关某几项）只会让数字变得不可解释 —— "调用次数少了"究竟
是没调用还是被采样掉了？

无 config 也让能力目录返回空 object（前端隐藏表单）这条路径
有了真实用例。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from ..base import ToolObserver
from ..context import (
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)
from ..runtime.metrics import get_tool_metrics
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = ["ToolMetricsObserver"]

logger = logging.getLogger(__name__)

#: 前后两半传递开始时刻用的键。带前缀以免撞上调用方放的东西。
STARTED_AT_KEY = "metrics.started_at"


class ToolMetricsObserver(ToolObserver):
    """
    把每次调用记进进程级指标表。

    被拒绝的调用（`ToolCallDenied`）**单独计数**，不算失败 ——
    "调用失败率 20%"在"后端在报错"与"审批拦下了一批"两种情况下
    要做的事完全不同（见 runtime/metrics.py 的说明）。
    """

    name = "metrics_observer"

    display_name = "指标"

    description = (
        "记录调用次数、耗时、成功率与错误分布。"
    )

    #: 没有可调参数 —— 见模块 docstring。
    config_model: ClassVar[type | None] = None

    supported_hooks: ClassVar[frozenset[ProcessorHook]] = (
        frozenset({
            ProcessorHook.BEFORE,
            ProcessorHook.AFTER,
        })
    )

    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.ASYNC})

    # ========================================================
    # BEFORE：记开始时刻
    # ========================================================

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation | ToolOutcome:

        # 就地写进 context：它是这次调用共享的临时区，AFTER 那一半
        # 会读到同一个对象。用 monotonic —— 墙钟被校时会算出负数。
        invocation.context[STARTED_AT_KEY] = time.monotonic()

        return invocation

    # ========================================================
    # AFTER：记一次调用
    # ========================================================

    async def after(
            self,
            invocation: ToolInvocation,
            outcome: ToolOutcome,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:

        get_tool_metrics().record(
            invocation.tool_key,
            duration_ms=self._duration_ms(invocation, outcome),
            error=outcome.error,
            denied=isinstance(outcome.error, ToolCallDenied),
        )

        # context 里的临时键用完就清掉：同一次调用可能挂多个指标
        # 处理器，留着不影响正确性，但一个"调用结束后还挂在那里的
        # 开始时刻"以后会被人当成别的东西用。
        invocation.context.pop(STARTED_AT_KEY, None)

        # 观察者不改结果：原样返回。
        return outcome

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _duration_ms(
            invocation: ToolInvocation,
            outcome: ToolOutcome,
    ) -> float | None:
        """工具耗时（毫秒）。拿不到就返回 None，不瞎估。"""

        if outcome.duration_ms is not None:
            return outcome.duration_ms

        started = invocation.context.get(STARTED_AT_KEY)

        if started is None:
            return None

        return (time.monotonic() - started) * 1000
