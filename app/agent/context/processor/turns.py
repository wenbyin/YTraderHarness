"""
轮次、窗口与摘要节奏的**全部数学**（纯函数、零 IO）。

为什么必须单独成模块
------------------
有三个阶段要用同一套算术：

    summary 处理器 (order 30)      算 u、判"该不该生成新摘要"、切出要折的轮
    history 处理器 (order 40)      算 N = clamp(u, min, limit)，定窗口宽度
    compaction 处理器 (order 95)   要知道 cursor 落在哪里（切窗口的终点）

如果每处各自数一遍 Human 消息，"什么是轮"就有三份实现，而它们漂移的
表现是**窗口宽度与摘要覆盖面差一轮**——静默、且只在长对话里才显形。
（`app/services/memory/domain/schemas/query.py` 里也提到 split_turns 是
"本仓库唯一一处"的轮次定义，那份唯一性现在就落在本模块。）

一条不变量，四个量
----------------
    T = 总轮数（含正在飞的当前轮）
    c = 摘要覆盖的轮数（覆盖 轮 1..c）
    u = T − c                 未被摘要覆盖的轮数
    N = 窗口实际保留的轮数

    overlap = max(0, N − u)   窗口前那几轮已经被摘要覆盖了（重复）
    gap     = max(0, u − N)   窗口够不着 cursor，中间那几轮没地方放

两者互斥：要么重叠、要么缺口，不可能同时为正。**覆盖成立 ⟺ gap == 0
⟺ u ≤ N。**

节奏
----
    limit(T) = min(max_turns, min_turns + T // 10)

即"起始 min_turns，每满 10 轮 +1，封顶 max_turns"。默认 3 → 5：
早期对话短，折得勤一点能让摘要跟上现场；后期摘要本身已承载大部分
语义、raw 窗口也够用，拉长间隔降的是重写频率（每次重写 = revision+1
→ context_hash 变 + 一次模型调用）。

⚠️ limit 是**上限**，不是窗口宽度
-----------------------------
窗口宽度是 `clamp(u, min_turns, limit)`。两者在 `u` 落在区间内时相等，
在两端不等——别把 limit 当成"这轮保留了几轮"（快照里两个都记）。

⚠️ 折叠边界永远落在**完整轮**的末尾
------------------------------
`fold_end = T − 1`（不含当前这一轮）。这保证折进去的那一段永远不会把
一次工具调用与它的结果拆开——因为每个轮都是 Human 起头的。当前轮
（用户刚问、还没答的那一轮）永远不折：把它折进去等于"把正在问的问题
变成二手转述"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .messages import is_human_message, message_identifier

__all__ = [
    "TURNS_PER_STEP",
    "TurnWindow",
    "covered_turns",
    "plan_window",
    "split_turns",
    "turn_count",
    "window_limit",
]

#: 每满这么多轮，窗口上限 +1（见模块 docstring 的"节奏"）。
TURNS_PER_STEP = 10


# ============================================================
# 轮
# ============================================================


def split_turns(messages: Sequence[Any]) -> list[list[Any]]:
    """
    按 Human 消息切轮。

    第一条 Human 之前的内容（如果有）自成一组并放在最前 —— 它通常是
    上一次会话留下的尾巴或工具结果，丢掉会让当前这一轮缺上下文，而
    把它并进第一轮又会让"轮"的定义变糊。
    """

    turns: list[list[Any]] = []
    current: list[Any] = []

    for message in messages:

        if is_human_message(message) and current:
            turns.append(current)
            current = [message]
            continue

        current.append(message)

    if current:
        turns.append(current)

    return turns


def turn_count(messages: Sequence[Any]) -> int:
    """消息列表里有几轮（含正在飞的当前轮）。"""

    return len(split_turns(messages))


def covered_turns(
        messages: Sequence[Any],
        cursor_id: str | None,
) -> int:
    """
    摘要覆盖到第几轮：**cursor 那条消息所在轮的序号**。

    每条 Human 起一轮，所以"cursor 之前（含）有几条 Human"就是轮数。

    认两种 cursor：

        真 id（langchain 消息带 id）      "3f2c..."
        `#<序号>`（消息没有 id）            "#7"

    两者都用 `message_identifier` 判 —— 与快照里的 `history_message_ids`
    同一套标识，因此"摘要说覆盖到 X"与"快照里第 X 条是哪一条"永远同源。
    （旧实现只比 `getattr(message, "id")`，于是没有 id 的消息写的 cursor
    永远匹配不上，静默退化成"没有覆盖"。这里一并修掉。）

    ⚠️ cursor 为空 / 在消息列表里找不到 -> 返回 **0**
    ----------------------------------------------
    这是**安全**的那一侧：c=0 让 u=T，触发条件立刻成立，于是这一轮
    重新生成一份摘要从头覆盖。跨会话留下的旧摘要、或将来消息被墓碑
    裁掉，都会走到这里——自愈，而不是"摘要说覆盖了但其实没有"。
    """

    if not cursor_id:
        return 0

    seen = 0

    for index, message in enumerate(messages):

        if is_human_message(message):
            seen += 1

        if message_identifier(message, index) == cursor_id:
            return seen

    return 0


def window_limit(
        *,
        min_turns: int,
        max_turns: int,
        total_turns: int,
) -> int:
    """
    这一轮的窗口**上限**（见模块 docstring 的"节奏"）。

    `min_turns` 是下限也是起点：早期 limit == min_turns，窗口是钉死的，
    直到对话长到 `TURNS_PER_STEP` 才长第一格。

    调用方传进来的 min/max 不合法时**不修**，取一个自洽的值：上限低于
    下限时以上限为准（"最多保留 5 轮"比"最少保留 8 轮"更像是笔误的
    一侧）。真正的一致性由 `ContextPolicy.__post_init__` 保证。
    """

    step = total_turns // TURNS_PER_STEP

    return min(max_turns, min_turns + step)


# ============================================================
# 窗口
# ============================================================


@dataclass(frozen=True, slots=True)
class TurnWindow:
    """
    一次"窗口怎么划"的完整答案（纯数据，无正文之外的东西）。

    生产者：`plan_window()`。消费者：summary / history / compaction 三个
    阶段，以及快照。**三个阶段读同一份**，因此它们对"第几轮"的理解
    不可能不一致。
    """

    #: 切好的轮（每轮一组消息）。窗口取它的后 `window_turns` 组。
    turns: tuple[tuple[Any, ...], ...]

    #: 摘要覆盖的轮数（c）。cursor 找不到时为 0，见 `covered_turns`。
    covered_turns: int

    min_turns: int
    max_turns: int

    #: 这一轮的窗口上限 limit(T)。
    limit: int

    #: 窗口实际保留的轮数 N = clamp(u, min_turns, limit)。
    window_turns: int

    # --------------------------------------------------------

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def uncovered_turns(self) -> int:
        """u = T − c，未被摘要覆盖的轮数（含当前轮）。"""

        return max(self.total_turns - self.covered_turns, 0)

    @property
    def overlap_turns(self) -> int:
        """
        摘要与窗口重叠的轮数。

        它**不是**副作用：`window_turns` 有下限（min_turns），当未覆盖的
        轮数还不到下限时，窗口只能往回够到已经被摘要覆盖的轮 —— 这是
        "宁可重叠、不可缺口"这条安全侧的定价。
        """

        return max(self.window_turns - self.uncovered_turns, 0)

    @property
    def gap_turns(self) -> int:
        """
        既不在摘要、也不在窗口里的轮数。

        **必须恒为 0**（覆盖不变量）。它为正只有一种解释：窗口上限太小，
        够不到 cursor —— 而那正是 `should_summarize` 的触发条件，所以
        正常情况下它来不及为正就被折掉了。
        """

        return max(self.uncovered_turns - self.window_turns, 0)

    @property
    def should_summarize(self) -> bool:
        """
        这一轮要不要生成新摘要。

        判据只有一条：**未覆盖的轮数超过了窗口上限**。再不动手，下一次
        窗口往前一滑，cursor 就落在窗口外了 —— 那几轮会既不在摘要、也不
        在窗口里（gap）。

        附带条件 `fold_end > fold_start`：没有可折的轮（对话刚开始、
        或只剩当前这一轮）就不调模型。
        """

        if self.uncovered_turns <= self.limit:
            return False

        return self.fold_end > self.fold_start

    # --------------------------------------------------------

    @property
    def fold_start(self) -> int:
        """要折进摘要的第一轮（下标，含）。"""

        return min(self.covered_turns, self.total_turns)

    @property
    def fold_end(self) -> int:
        """
        要折进去的最后一轮的下一位（不含）。

        **永远排除当前这一轮**（`total_turns - 1`）：当前轮是"用户刚问、
        还没答"的那一轮，折进去等于把正在问的问题变成二手转述。
        """

        return max(self.total_turns - 1, self.fold_start)

    def fold_messages(self) -> list[Any]:
        """要折进摘要的那个**连续切片**（空切片 = 不用调模型）。"""

        folded = self.turns[self.fold_start:self.fold_end]

        return [message for turn in folded for message in turn]

    def window_messages(self) -> list[Any]:
        """窗口保留的那几轮（展平后的消息列表）。"""

        if self.window_turns <= 0:
            return []

        return [
            message
            for turn in self.turns[len(self.turns) - self.window_turns:]
            for message in turn
        ]

    def as_metadata(self) -> dict[str, Any]:
        """进快照的那一份（数字，没有正文）。"""

        return {
            "total_turns": self.total_turns,
            "covered_turns": self.covered_turns,
            "uncovered_turns": self.uncovered_turns,
            "min_recent_turns": self.min_turns,
            "max_recent_turns": self.max_turns,
            "turn_limit": self.limit,
            "window_turns": self.window_turns,
            "overlap_turns": self.overlap_turns,
            "gap_turns": self.gap_turns,
            "should_summarize": self.should_summarize,
        }


def plan_window(
        messages: Sequence[Any],
        cursor_id: str | None,
        *,
        min_turns: int,
        max_turns: int,
) -> TurnWindow:
    """
    划一次窗口（见 `TurnWindow`）。

    ⚠️ `cursor_id` 传的是**摘要当前覆盖到哪**，不是"这一轮之后"
    ------------------------------------------------
    调用方（summary / history / compaction）应当传 state 里那份摘要的
    `summarized_until_message_id`。summary 阶段生成新摘要之后要把它写回
    工作集（`context.set_summary`），history 阶段因此读到的是**新的**
    cursor —— 这正是"摘要一前进、窗口立刻后退"的实现方式（overlay 赢过
    base_state，见 context.py 的 `state_root`）。
    """

    turns = split_turns(messages)

    total = len(turns)

    covered = min(covered_turns(messages, cursor_id), total)

    limit = window_limit(
        min_turns=min_turns, max_turns=max_turns, total_turns=total
    )

    uncovered = max(total - covered, 0)

    # clamp(u, min_turns, limit)，再被"实际有几轮"夹一次：对话刚开始时
    # 保底 min_turns 轮物理上不存在，不能凭空要求。
    window = min(max(uncovered, min_turns), limit, total)

    return TurnWindow(
        turns=tuple(tuple(turn) for turn in turns),
        covered_turns=covered,
        min_turns=min_turns,
        max_turns=max_turns,
        limit=limit,
        window_turns=window,
    )
