"""
阶段 4：HISTORY_SELECTOR —— 选出这次要发的对话历史（规格 §十五）。

它是一个**纯窗口选择器**：摘要 + 最近 N 轮 + 当前用户消息，超预算就从最早的
轮次开始整轮丢。数据只有两个来源：Graph State 里的 `messages`，以及摘要里
那条"已覆盖到哪"的水位。

⚠️ 它**不再**自己去取"比 state 更多的历史"
-------------------------------------
这里曾经有两条这样的路，两条都**删掉了**：

    ConversationSource    INITIAL 时回库补齐更早的轮次
    语义历史检索           用当前问题去检索相关的历史轮次
                          （端口叫 `HistoryRetriever`）

删它们的理由有两条，各自都成立：

    没有消费者    两个端口从落地起就没有生产实现（默认全 None），
                  也就是说这两条路在生产里**从来没有跑过**
    有了替代      更早的内容现在归**记忆模块的 L0 层**：原始对话已经建了
                  索引，按需召回、带分数、分层降级。历史处理器再用另一种
                  口径捞一遍，只会让"模型看到的更早内容"有两个来源、
                  两套取舍 —— 而它们对"相关"的定义并不一致

于是这里只剩一条判据：**state 里有什么，就按窗口选什么**。

顺带把 §四B 那条约束变成了平凡真理：CONTINUATION 本来就只用 state，
而现在两种执行层级在这一点上完全一致 —— 没有"回库重放"这个分支可走错。
（那条不变量的守卫现在落在记忆模块上，见 processors/memory.py 的
`_replay`：一个运行时只召回一次。）

数字（轮次 / 预算）从哪读
---------------------
    `context.policy`    有效策略（resolver 阶段算好的）
    `context.budget`    已分配预算（resolver 阶段算好的）

它们是 resolver 阶段的正式产物、就在工作集上，因此这里直接读 ——
数据（messages / 摘要）仍然走声明式绑定。**不**去读
`context.base_state`：那才是"Processor 偷偷访问 Graph State"。
"""

from __future__ import annotations

from typing import Any

from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..history import HistorySelection
from ..messages import message_identity
from ..spec import ContextProcessorSpec, StateBinding
from ..values import SummarySnapshot

__all__ = ["HistoryProcessor"]


class HistoryProcessor(ContextProcessor):
    """按轮次窗口 + token 预算选出历史。"""

    spec = ContextProcessorSpec(
        name="history",
        order=40,
        stage=StageId.HISTORY_SELECTOR,
        when="context_policy.include_history",
        inputs={
            "messages": StateBinding("messages"),
            "current_message_id": StateBinding(
                "current_user_message_id", required=False
            ),
            "current_message": StateBinding(
                "current_user_message", required=False
            ),
            "summary": StateBinding("summary", required=False),
        },
        description=(
            "按最近 N 轮 + token 预算选择历史；整轮裁剪，保证工具调用"
            "与结果不被拆开。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        deps = dependencies_of(context)

        policy = context.policy
        budget = context.budget

        summary = SummarySnapshot.from_state(inputs.get("summary"))

        selection = deps.resolved_history_selector().select(
            list(inputs.get("messages") or ()),
            policy=policy,
            token_counter=deps.resolved_token_counter(),
            history_budget=budget.history_budget if budget else None,
            current_message_id=inputs.get("current_message_id"),
            summarized_until_message_id=(
                summary.summarized_until_message_id
                if summary is not None
                else None
            ),
        )

        context.add_history(selection.messages)

        # 当前用户消息**必须**在（见下面 `_ensure_current_message`）。
        _ensure_current_message(
            context, inputs.get("current_message"), selection
        )

        detail = dict(selection.as_metadata())

        context.set("history_selection", detail)

        context.metadata["history"] = detail

        return context


# ============================================================
# internals
# ============================================================


def _ensure_current_message(
        context: ContextBuildContext,
        current: Any,
        selection: HistorySelection,
) -> None:
    """
    当前用户消息**必须**出现在最终上下文里（规格 §十五）。

    它通常已经在选中的历史里（最后一轮的第一条）。不在的话（比如图
    state 的消息还没带上它）就作为"必须在的消息"补进去 ——
    少了它，这次调用就是"没有问题却在回答"。
    """

    if current is None:
        return

    key = message_identity(current)

    for message in selection.messages:

        if message_identity(message) == key:
            return

    context.add_required_message(current)
