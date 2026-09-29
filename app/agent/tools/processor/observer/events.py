"""
工具生命周期事件观察者（给 SSE / Agent Server / 前端用）。

挂两半，各管一头
----------------
    BEFORE  发 STARTED
    AFTER   发 SUCCEEDED / FAILED / DENIED

两者都是 **ASYNC**：事件是旁路，收不到不该影响调用。这也意味着
"发了没发出去"没有任何反馈 —— 因此队列满时总线会记日志
（见 runtime/events.py），不会静默丢。

DENIED 从哪来
-------------
`ToolOutcome.error` 是 `ToolCallDenied` 时它表示**工具根本没跑**
（被别的拦截器拒了）。这与"跑了但失败"在界面上要给人看的解释
完全不同，因此分成两种事件类型，而不是笼统的 FAILED。

runtime 只用来取 tool_call_id
-----------------------------
langgraph 给每次工具调用一个 tool_call_id，它是"这条事件"与
"图里那次调用"对上号的**唯一**凭据 —— UI 按它在时间线上定位，
重放时按它配对。处理器本身不需要 state / store，因此这里对
runtime 的依赖极浅：取一个字段，拿不到就算了。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from ..base import ToolObserver
from ..context import (
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)
from ..runtime.events import (
    ToolEvent,
    ToolEventType,
    get_tool_event_bus,
)
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "EventObserverConfig",
    "ToolEventObserver",
]

logger = logging.getLogger(__name__)

#: 调用链往 invocation.context 里塞链上位置用的键。
#:
#: 处理器**看不到自己挂在第几条** —— 那是挂载的属性，不是实现的
#: 属性（同一个实现可以在一条链上挂多次）。所以位置只能由执行链
#: 的那一层填进 context，这里读出来放进事件，前端才能把事件归位。
SEQUENCE_CONTEXT_KEY = "tool_processor_sequence"


class EventObserverConfig(BaseModel):
    """事件参数。"""

    include_arguments: bool = Field(
        default=False,
        title="附带参数",
        description=(
            "事件里要不要带上这次调用的参数。**默认不带** —— "
            "事件会流向浏览器，而参数里可能有业务数据甚至个人信息。"
            "只在可信的订阅者场景下打开。"
        ),
    )

    event_types: list[ToolEventType] | None = Field(
        default=None,
        title="只发这些事件",
        description=(
            "留空 = 全发。想只看失败时可以填 [\"FAILED\"]，"
            "这样 SSE 上就只有异常流水，不会被正常调用刷掉。"
        ),
    )


class ToolEventObserver(ToolObserver):
    """
    把工具调用发布到进程内事件总线。

    ⚠️ 发布是**同步**的（总线非阻塞），但本处理器声明 ASYNC：
    它自己不该被等，而 publish 也不该被 await。

    ASYNC 处理器仍然拿得到 runtime：它由调用方 `create_task` 起，
    而任务会继承创建时的 contextvars —— 只要 runtime 是当参数传
    进来的（而不是存在 self 上），就没有丢失的风险。
    """

    name = "event_observer"

    display_name = "事件发布"

    description = (
        "把工具调用的开始与结束发布到事件总线（供 SSE 推送）。"
    )

    config_model: ClassVar[type[BaseModel]] = EventObserverConfig

    supported_hooks: ClassVar[frozenset[ProcessorHook]] = (
        frozenset({
            ProcessorHook.BEFORE,
            ProcessorHook.AFTER,
        })
    )

    #: 只有 ASYNC。旁路观测不该拖慢调用 —— 而 SYNC 会让一个卡住的
    #: 订阅者（慢的浏览器连接）间接拖住工具执行。
    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.ASYNC})

    # ========================================================
    # BEFORE -> STARTED
    # ========================================================

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation | ToolOutcome:

        parsed = self.config_model.model_validate(config)

        extra: dict[str, Any] = {}

        if parsed.include_arguments:
            extra["arguments"] = dict(invocation.arguments)

        self._correlate(extra, runtime)

        self._publish(
            ToolEventType.STARTED, invocation, parsed, extra=extra
        )

        # 观察者不改任何东西：原样放行。
        return invocation

    # ========================================================
    # AFTER -> SUCCEEDED / FAILED / DENIED
    # ========================================================

    async def after(
            self,
            invocation: ToolInvocation,
            outcome: ToolOutcome,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:

        parsed = self.config_model.model_validate(config)

        if outcome.error is None:
            event_type = ToolEventType.SUCCEEDED
        elif isinstance(outcome.error, ToolCallDenied):
            event_type = ToolEventType.DENIED
        else:
            event_type = ToolEventType.FAILED

        extra: dict[str, Any] = {}

        if outcome.cached:
            # 命中缓存也是一次"成功"，但界面该标出来 ——
            # 否则用户会以为工具真的跑了一次。
            extra["cached"] = True

        self._correlate(extra, runtime)

        self._publish(
            event_type, invocation, parsed, outcome, extra=extra
        )

        return outcome

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _correlate(
            extra: dict[str, Any],
            runtime: ToolRuntime | None,
    ) -> None:
        """
        把 langgraph 的 tool_call_id 放进事件。

        拿不到就**不放这个键**，而不是塞个空串或 null：
        空串看起来像一个有效的 id，前端按它配对会静默配错。
        键不在，前端只写一种判空。
        """

        if runtime is None:
            return

        tool_call_id = runtime.tool_call_id

        if tool_call_id:
            extra["tool_call_id"] = tool_call_id

    def _publish(
            self,
            event_type: ToolEventType,
            invocation: ToolInvocation,
            config: EventObserverConfig,
            outcome: ToolOutcome | None = None,
            *,
            extra: dict[str, Any],
    ) -> None:

        if (
                config.event_types is not None
                and event_type not in config.event_types
        ):
            return

        error = outcome.error if outcome is not None else None

        event = ToolEvent(
            type=event_type,
            tool_key=invocation.tool_key,
            tool_id=invocation.tool_id,
            occurred_at=datetime.now(timezone.utc),
            sequence=invocation.context.get(
                SEQUENCE_CONTEXT_KEY
            ),
            duration_ms=(
                outcome.duration_ms if outcome is not None else None
            ),
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            extra=extra,
        )

        delivered = get_tool_event_bus().publish(event)

        if not delivered:
            # 没有订阅者是**正常**的（没人开着界面）。这里不报警，
            # 否则没人看的时候日志全是噪音。真正的丢弃由总线报
            # （它有队列长度这个事实可依据）。
            logger.debug(
                "事件 %s（%s）没有订阅者。",
                event_type.value,
                invocation.tool_key,
            )
