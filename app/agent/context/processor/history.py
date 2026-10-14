"""
历史选择（规格 §十五）。

窗口宽度从哪来
------------
**从摘要覆盖到哪来**（`turns.plan_window`）：

    u = 总轮数 − 摘要覆盖的轮数        N = clamp(u, min_recent_turns, limit(T))
    窗口 = 最后 N 轮                   limit 见 turns.window_limit

于是摘要在前、窗口在后地"你进我退"：摘要推进一轮，窗口就少留一轮，直到
撞上 `min_recent_turns` 这层下限。撞上之后继续推进，多出来的部分就是
**重叠**（那几轮既在摘要里、也在窗口里）——这是刻意的，"宁可重叠、
不可缺口"。

⚠️ 重叠不是浪费，缺口才是
----------------------
    overlap = max(0, N − u)     重叠：多发几轮原文，多花 token
    gap     = max(0, u − N)     **缺口：那几轮彻底没有地方放**

缺口是本设计里唯一不允许出现的状态。两个守卫各管一半：

    summary 阶段（order 30）   u 超过 limit 就生成新摘要 -> 不让缺口出现
    本选择器（order 40）        预算裁剪**不越过 cursor** -> 不制造缺口

⚠️ 预算裁剪的界（两个约束取更紧的那个）
---------------------------------
    不许越过 cursor        否则被丢的轮既不在摘要、也不在窗口 = 缺口
    至少留 min_recent_turns 轮   否则"最近的现场"没了

两者合起来：`start ≤ min(covered_turns, 总轮数 − min_recent_turns)`。

- 未覆盖轮数 ≥ 下限时（u 正常）：上界 = covered_turns —— 预算可以一路
  把重叠全裁掉，裁不动的地方正好是 cursor。
- 未覆盖轮数 < 下限时（刚折完）：上界 = 总轮数 − min_recent_turns ——
  **重叠被保下来了**。这正是留给后一阶段（compaction）的那件事：它超预算
  时才把这段重叠压成 0，见 processors/compaction.py。

⚠️ 本模块只选，不取
--------------
它**只按轮次与预算挑选**调用方给它的消息。它不去取"比 state 更多的
历史" —— 那条路（语义历史检索）已经删掉，更早的内容归记忆模块的 L0
层，详见 processors/history.py 顶部的说明。

`HistorySelector` 仍是端口（将来真要做另一种选择策略时替换它），但
"用检索挑历史"不再是一个选项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .messages import (
    is_system_message,
    message_identifier,
    repair_tool_pairs,
)
from .policy import ContextPolicy
from .turns import plan_window, split_turns

__all__ = [
    "HistorySelection",
    "HistorySelector",
    "TurnWindowHistorySelector",
    "split_turns",
]


@dataclass(frozen=True, slots=True)
class HistorySelection:
    """
    一次历史选择的结果。

    带 `dropped_*` 几个计数是刻意的：它们让"这次为什么少了三轮"在
    观测里**直接看得见**，而不必去比对两次调用的 messages。

    ⚠️ `overlap_turns` 与 `gap_turns` 的关系
    ------------------------------------
    两者互斥，且 `gap_turns` **必须恒为 0**（覆盖不变量）。它不为 0 只有
    一种解释：摘要没跟上（生成被关了 / 模型没能产出），而那是一件值得
    在快照里被看见的事——所以这里照实记，不在这里兜。
    """

    messages: tuple[Any, ...]

    message_ids: tuple[str, ...] = ()

    #: 保留下来的对话轮数（`Human -> ...` 算一轮）。
    turns: int = 0

    #: 因为窗口上限（`limit(T)`）没进去的轮数。
    dropped_turns: int = 0

    #: 因为 token 预算被丢掉的轮数。
    dropped_for_tokens: int = 0

    #: 被去掉的不成对工具往返（见 messages.repair_tool_pairs）。
    dropped_broken_tool_rounds: int = 0

    #: 已被摘要覆盖、但**没有**进窗口的轮数（既没发原文，摘要里也有）。
    skipped_summarized: int = 0

    token_count: int = 0

    # ---- 窗口与摘要的关系（观测用，见模块 docstring） ----------

    #: 总轮数（含正在飞的当前轮）。
    total_turns: int = 0

    #: 摘要覆盖的轮数。
    covered_turns: int = 0

    #: 未被摘要覆盖的轮数（= 总轮数 − 覆盖轮数）。
    uncovered_turns: int = 0

    #: 这一轮的窗口上限 `limit(T)`。
    turn_limit: int = 0

    #: 摘要与窗口重叠的轮数。
    overlap_turns: int = 0

    #: 窗口与摘要之间的缺口轮数（**必须为 0**）。
    gap_turns: int = 0

    def as_metadata(self) -> dict[str, Any]:

        return {
            "messages": len(self.messages),
            "turns": self.turns,
            "total_turns": self.total_turns,
            "covered_turns": self.covered_turns,
            "uncovered_turns": self.uncovered_turns,
            "turn_limit": self.turn_limit,
            "overlap_turns": self.overlap_turns,
            "gap_turns": self.gap_turns,
            "dropped_turns": self.dropped_turns,
            "dropped_for_tokens": self.dropped_for_tokens,
            "dropped_broken_tool_rounds": self.dropped_broken_tool_rounds,
            "skipped_summarized": self.skipped_summarized,
            "token_count": self.token_count,
        }


@runtime_checkable
class HistorySelector(Protocol):
    """
    "从一堆历史里选出这次要发的那几条"。

    实现必须是**同步纯函数**（不查库、不调模型）：它跑在每次 LLM 调用
    之前，任何 IO 都会变成调用延迟，而且会让"为什么少了三轮"无法从
    输入复现。
    """

    def select(
            self,
            messages: Sequence[Any],
            *,
            policy: ContextPolicy,
            token_counter: Any,
            history_budget: int | None = None,
            current_message_id: str | None = None,
            summarized_until_message_id: str | None = None,
    ) -> HistorySelection:
        ...


class TurnWindowHistorySelector:
    """
    默认选择器：轮次窗口 + token 预算。

    算法（顺序即优先级）：

        1. 算出窗口：`plan_window(messages, cursor, min, max)`
        2. 摘出 system 消息（**永不**参与裁剪）
        3. 取窗口覆盖的那几轮（最后 N 轮，可能包含已被摘要覆盖的轮）
        4. token 超预算时从**最早**的轮整轮丢，但**不越过 cursor**、
           也**不留少于 `min_recent_turns` 轮**（见模块 docstring）
        5. `repair_tool_pairs`：去掉不成对的工具往返（上游给的历史
           可能从半路开始）
    """

    def select(
            self,
            messages: Sequence[Any],
            *,
            policy: ContextPolicy,
            token_counter: Any,
            history_budget: int | None = None,
            current_message_id: str | None = None,
            summarized_until_message_id: str | None = None,
    ) -> HistorySelection:

        source = list(messages)

        if not source:
            return HistorySelection(messages=())

        plan = plan_window(
            source,
            summarized_until_message_id,
            min_turns=policy.min_recent_turns,
            max_turns=policy.max_recent_turns,
        )

        # ---- 1. system 消息：单独摘出，永不参与裁剪 ---------------

        preserved = [
            message for message in source if is_system_message(message)
        ]

        # ---- 2. 取窗口 -------------------------------------------

        turns = plan.turns
        total = len(turns)

        start = max(total - plan.window_turns, 0)
        initial_start = start

        # 预算裁剪的两个界（见模块 docstring）：取更紧的那个。
        keep_at_least = max(policy.min_recent_turns, 1)

        start_cap = max(total - keep_at_least, 0)

        if plan.covered_turns > 0:
            start_cap = min(start_cap, plan.covered_turns)

        # ---- 3. token 预算 ---------------------------------------

        dropped_for_tokens = 0

        def _tokens(selected: Sequence[Sequence[Any]]) -> int:

            return token_counter.count_messages(
                [message for turn in selected for message in turn]
            )

        window = list(turns[start:])

        if history_budget is not None:

            total_tokens = _tokens(window)

            while (
                    total_tokens > history_budget
                    and start < start_cap
            ):

                total_tokens -= _tokens(window[:1])
                window = window[1:]
                start += 1
                dropped_for_tokens += 1

        # ---- 4. 工具配对修复 -------------------------------------

        flat = [
            message
            for turn in window
            for message in turn
            if not is_system_message(message)
        ]

        repaired, dropped_broken = repair_tool_pairs(flat)

        selection = [*preserved, *repaired]

        # ⚠️ 窗口与摘要的关系按**裁剪之后**的 `start` 报，不是计划时的那份
        # ------------------------------------------------------
        # 预算把窗口最左往右推了，重叠就少了（推到 cursor 处归零），而
        # 缺口在 `start ≤ covered_turns` 这条约束下**不可能**为正 —— 除非
        # 压根没有摘要（covered=0），那时任何被裁掉的轮都无处可去。后者
        # 是事实，照实报，不在这里抹平。
        overlap = max(plan.covered_turns - start, 0)
        gap = max(start - plan.covered_turns, 0)

        return HistorySelection(
            messages=tuple(selection),
            message_ids=tuple(
                message_identifier(message, index)
                for index, message in enumerate(selection)
            ),
            turns=len(window),
            dropped_turns=max(initial_start - 0, 0),
            dropped_for_tokens=dropped_for_tokens,
            dropped_broken_tool_rounds=dropped_broken,
            # 被摘要覆盖、又没进窗口的那几轮：原文没发，摘要里记着。
            skipped_summarized=max(plan.covered_turns - overlap, 0),
            token_count=token_counter.count_messages(selection),
            total_turns=plan.total_turns,
            covered_turns=plan.covered_turns,
            uncovered_turns=plan.uncovered_turns,
            turn_limit=plan.limit,
            overlap_turns=overlap,
            gap_turns=gap,
        )
