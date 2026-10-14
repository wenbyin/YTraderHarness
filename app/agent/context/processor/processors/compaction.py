"""
阶段 10：CONTEXT_COMPACTOR —— 超预算时把"重叠"压成 0（规格 §十六）。

它只做一件事
-----------
history 处理器交上来的窗口里，最左边那几轮可能**已经被摘要覆盖**了
（`history.py` 的"你进我退"：窗口保底 `min_recent_turns` 轮，于是摘要
刚推进完的那几轮会既在摘要里、也在窗口里）。那几轮是**纯重复** ——
摘要里有，原文又发一遍。

本处理器只在**还超预算**时做一件事：

    把窗口最左推到 cursor —— 即把那段重叠整个切掉

切完 `摘要 ∪ 窗口` 仍然等于全部对话，只是从"重叠"变成"首尾相接"
（零重叠、零缺口）。**这就是它的上限**：终点是 cursor，而 cursor 是摘要
的边界，所以它连"能不能切过头"这个问题都不存在。

⚠️ 它**不生成摘要**
----------------
生成在 `processors/summary.py`（order 30），判据是轮次数，与 token 无关。
本处理器**不调模型** —— 全仓唯一在上下文构建里调模型的地方是那一个
阶段，实现住在 `summarization.py`（那个模块原来叫 compaction.py，
名字跟着职责一起改了）。

⚠️ 为什么用"窗口自己的 token"当判据，而不是全量对话
--------------------------------------------
旧实现拿**整段会话**的 token 去比 `compaction_threshold × history_budget`。
那是另一个时钟：它说的是"这段会话长不长"，而这里要回答的是"**这一次
交上去的窗口**装不装得下"。前者在长会话里永远为真（于是每轮都压），
在短而大的会话里永远为假（于是从没压过）。现在两者都不成立 —— 判据
就是这次真正要发出去的那几条消息的大小。

⚠️ 压完还是超，就到此为止
----------------------
`min_recent_turns` 是"无预算压力时的下限"，不是硬保证；但压到 cursor
之后**没有别的杠杆**了（再往前就是缺口）。剩下的只能交给校验器去判
"这一次的输入本身就超出了模型窗口"（规格 §二十八：Fatal）——本处理器
负责把"压不动了"这件事以一条 warning 的形式留在快照里，而不是让它
静默通过。

⚠️ 顺序：它在 prompt_assembler(80) / token_budget(90) 之后
------------------------------------------------------
所以切完要**重建 draft**（与 token_budget 同一套 `replace`），否则这一
次发出去的仍然是切之前的那份，而快照会说"切了 2 轮"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..messages import is_system_message, message_identifier
from ..spec import ContextProcessorSpec

__all__ = ["CompactionProcessor"]


class CompactionProcessor(ContextProcessor):
    """超预算时把窗口与摘要的重叠压成 0（纯切片，不调模型）。"""

    spec = ContextProcessorSpec(
        name="compaction",
        order=95,
        stage=StageId.CONTEXT_COMPACTOR,
        when="context_policy.allow_compaction",
        # 没有声明输入：它读的是**工作集**（`context.history` 与
        # `context.summary`），不是图 state 的通道。历史窗口是上一个阶段
        # 算出来的产物，绑到 state 上反而不对（state 里没有它）。
        inputs={},
        description=(
            "超预算时把历史窗口最左推到摘要的边界（重叠压成 0）；"
            "不生成摘要、不调模型。"
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

        if policy is None or budget is None:
            return context

        counter = deps.resolved_token_counter()

        history = list(context.history)

        detail: dict[str, Any] = {
            "history_tokens": counter.count_messages(history),
            "history_budget": budget.history_budget,
            "compacted": False,
        }

        before = detail["history_tokens"]

        if not history:

            detail["reason"] = "no_history"

            return self._record(context, detail)

        if before <= budget.history_budget:

            detail["reason"] = "within_budget"

            return self._record(context, detail)

        cursor = self._cursor(context)

        if cursor is None:

            detail["reason"] = "no_summary"

            context.warn(
                "历史超预算，但没有摘要可以对齐（cursor 为空），无法压重叠。"
                "这一轮会带着超出 history_budget 的历史发出去。"
            )

            return self._record(context, detail)

        index = _index_of_id(history, cursor)

        if index is None:

            # 摘要说覆盖到某条消息，但那条消息不在这次交上来的窗口里 ——
            # 说明窗口整体都在摘要之后（本来就没有重叠）。没有可压的东西。
            detail["reason"] = "no_overlap"

            context.warn(
                "历史超预算，但窗口里没有与摘要重叠的轮次（cursor 不在"
                "窗口内），无可压缩。"
            )

            return self._record(context, detail)

        # ---- 压：保留 system 消息 + cursor 之后的全部 --------------
        #
        # 切点落在 cursor 上，而 cursor 永远是一条**完整轮的末条消息**
        # （见 turns.py 的 fold_end），因此窗口的第一条一定是一轮的开头
        # —— 工具调用与结果不可能被拆开。

        kept = [
            message
            for position, message in enumerate(history)
            if position > index or is_system_message(message)
        ]

        context.add_history(kept)

        after = counter.count_messages(kept)

        detail.update(
            {
                "compacted": True,
                "reason": "overlap_cut",
                "cut_messages": len(history) - len(kept),
                "history_after": after,
            }
        )

        if after > budget.history_budget:

            context.warn(
                f"压掉重叠后历史仍然超预算（{after} > "
                f"{budget.history_budget}）：再往前切就会在摘要与窗口之间"
                "留下缺口。这一次的输入本身就超出了历史预算。"
            )

        _rebuild_draft(context)

        return self._record(context, detail)

    # --------------------------------------------------------

    def _cursor(self, context: ContextBuildContext) -> str | None:
        """当前摘要覆盖到哪一条消息（= 窗口切点的右边）。"""

        summary = context.summary

        if summary is None:
            return None

        return summary.summarized_until_message_id

    def _record(
            self,
            context: ContextBuildContext,
            detail: dict[str, Any],
    ) -> ContextBuildContext:

        context.set("compaction", detail)
        context.metadata["compaction"] = detail

        return context


# ============================================================
# internals
# ============================================================


def _index_of_id(
        messages: Sequence[Any],
        cursor_id: str,
) -> int | None:
    """
    cursor 在消息列表里的位置。

    用 `message_identifier` 而不是 `getattr(message, "id")` —— 没有 id 的
    消息写下的 cursor 是 `#<序号>`，与 history.py / turns.py 同一套标识。
    """

    for index, message in enumerate(messages):

        if message_identifier(message, index) == cursor_id:
            return index

    return None


def _rebuild_draft(context: ContextBuildContext) -> None:
    """
    用切完的消息重建 draft。

    与 `processors/budget.py` 同一手法（`replace` 而不是就地改：
    `ModelRequest` 是 frozen 的，而"改请求"必须留下痕迹）。draft 还不
    存在时就什么都不做 —— 校验阶段会自己报"看不到 draft"。
    """

    if context.draft is None:
        return

    context.draft = replace(
        context.draft,
        messages=tuple(context.assembled_messages()),
    )
