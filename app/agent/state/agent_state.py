"""
Agent Builder 对用户输入做的初次理解 —— 公共的取值域与结果形状。

谁写谁读
--------
    Agent Builder 节点写（它做那次分类调用）
    路由函数读（按 execution_plan 决定去哪个节点）
    后续节点与观测读（行为 / 模块 / 能力需求）

⚠️ 值一律存**普通字符串**，不存枚举成员
----------------------------------------
`BehaviorCategory` 这些是 `str, Enum`，成员本身也是 str，看起来可以直接
存。**不要**：

    图 state 会过 checkpointer 序列化（JSON）。枚举成员序列化后是
    "chat" 这样一个普通字符串，反序列化回来**不会**再变成
    `BehaviorCategory.CHAT`。于是同一条运行里，第一轮的 state 里是
    枚举、恢复之后变成 str —— `is BehaviorCategory.CHAT` 这类判断
    会在"看起来一样的两轮"里给出不同答案。

所以：写入时取 `.value`，比较时也比 `.value`。枚举只在**代码里**用于
约束取值域与写提示词，不进 state。

⚠️ 每个维度都有 `other`
----------------------
"归不了类"是一种正常结果，不是错误：模型可能面对一个不属于任何既有
类目的输入。因此

    ① behavior          枚举外的值 -> "other"
    ② modules           不在候选集合里 -> 落 "other"（见 MODULE_OTHER）
    ③ execution_plan    枚举外的值 -> "other"，路由按"没有特别指示"处理
    ④ capability_needs  不在候选集合里 -> 落 "other"（同一个 MODULE_OTHER）

硬塞进最近的类目会让下游（路由 / 工具选择）按一个错误的前提工作，而
错误前提比"我不知道"难查得多。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

__all__ = [
    "MODULE_OTHER",
    "AgentClassification",
    "BehaviorCategory",
    "ExecutionPlan",
    "coerce_behavior",
    "coerce_execution_plan",
    "empty_classification",
]

#: 维度②（业务域）与维度④（能力域接口）的**共享**兜底值。
#:
#: 两个维度的候选集都来自数据库（CapabilityService 读出的 code），是
#: **动态**的，因此没法用一个 Enum 表达。任一维度的取值落不进候选集就写
#: 这个值。
#:
#: ⚠️ 名字里的 "MODULE" 是历史命名（最初只有维度② 用它），**不要**为
#: 维度④ 加一个 `CAPABILITY_OTHER` 别名：两者必须是**同一个** "other" ——
#: 两个维度的 code 合起来查工具（见 agent_builder_node 的
#: `_tool_keys_for_capabilities`），过滤时按值比一次就够；有别名就等于
#: 宣称它们可以分叉，而分叉的那天会有一半的 "other" 漏进查询。
MODULE_OTHER = "other"


class BehaviorCategory(str, Enum):
    """维度①：这次输入要**干什么**。"""

    CHAT = "chat"
    QA = "qa"
    ANALYSIS = "analysis"
    CREATION = "creation"
    MODIFICATION = "modification"
    RESEARCH = "research"
    EXECUTION = "execution"
    PLANNING = "planning"
    DECISION_SUPPORT = "decision_support"

    #: 归不了类。见模块 docstring。
    OTHER = "other"


class ExecutionPlan(str, Enum):
    """维度③：这次该怎么**执行**。它驱动路由。"""
    DIRECT = "direct"

    #: ReAct 主循环：直接回答、或需要工具时边走边定。
    REACT = "react"

    #: 需要先出计划再执行 -> 走 planner 节点。
    PLAN = "plan"

    #: 判不出来。路由按"没有特别指示"处理（当前 = 走 agent）。
    OTHER = "other"


class AgentClassification(TypedDict):
    """
    Agent Builder 的一次分类结果。

    五个维度（① ③ ④ 实做，② ⑤ 只落字段）：

        ① behavior            行为分类
        ② modules             **业务域**分类 —— 本次恒为空，见下
        ③ execution_plan      执行方案（驱动路由）
        ④ capability_needs    能力需求（DB 候选集里的能力 code）——
                              **驱动工具选择**
        ⑤ capability_config   能力配置选择（尚未驱动行为）

    ⚠️ 能力 code 只有 ④ 一个出口，② 是**留着的空位**
    ---------------------------------------------
    Agent Builder 把能力 code 翻成 tool_keys 写进 context_policy，capability
    阶段据此决定给模型看哪些工具。

    ② 之所以空着：`capability` 表只有 code / parent_id / is_active，
    **没有字段**能区分"业务域"与"能力域接口"。曾经让模型按名称与描述分桶，
    实测分不对（把 `CAPABILITY` 填进了 modules）。既然判断依据不存在，就
    不该让它留在模型那边。等分类体系加上区分字段、② 开始有值时，查工具
    那条路（按两个字段的并集）不用改。

    degraded / notes 是**诚实性**字段：分类是一次模型调用，可能失败、
    可能降级（比如数据库读不到候选能力）。失败时给出一个"全是 other"
    的结果并标 degraded，比抛异常中断整轮对话合适 —— 分类是增强，
    不是对话的必要条件。
    """

    #: ① 行为分类，`BehaviorCategory` 的 value。
    behavior: str

    #: ② 业务域分类，能力 code 列表。**本次恒为 `[]`** —— 见模块 docstring
    #: 与 agent_builder_node 里那段：库里没有能区分业务域与能力域接口的
    #: 字段，分桶的依据不存在，所以不交给模型填。
    #:
    #: 字段留着是因为它是对外契约与 state 的一部分，而工具查找仍按 ②④ 的
    #: 并集做 —— 将来加上区分字段后这里开始有值，那条路不用改。
    modules: list[str]

    #: ③ 执行方案，`ExecutionPlan` 的 value。路由按它分支。
    execution_plan: str

    #: ④ 能力需求：候选集里的 capability code 列表，**唯一驱动工具选择**的
    #: 字段。认不出的取值落 `MODULE_OTHER`，查工具前会把它过滤掉。
    #:
    #: 精确匹配、不含子孙能力 —— 见 ToolDefinitionRepository 的契约。
    capability_needs: list[str]

    #: ⑤ 能力配置选择。**尚未**驱动任何行为。
    capability_config: dict[str, Any]

    #: ⑥ 这一轮要不要**预先**召回 L0 原始对话。判据见分类提示词。
    #:
    #: False 是保守的一侧：多召回一次要付一次向量检索，漏召回的这一轮
    #: 主 agent 还挂着 L0/L1 的检索工具，能自己补。因此"判不出来"也落
    #: False（见 agent_builder_node 的 `_coerce_flag`）。
    #:
    #: ⚠️ 它**不是**最终生效的那份 —— 生效的是 `context_policy.memory_retrieve`
    #: （由 `_memory_retrieve` 拼出来，还要过 `agent_id` 非空这一道）。
    #: 留在这里是为了"这一轮发生过什么分类"能从 state 直接读出来。
    retrieve_l0: bool

    #: ⑦ 同上，针对 L1 原子记忆。
    retrieve_l1: bool

    #: 这次分类是否降级（模型调用失败 / 候选能力读不到 …）。
    degraded: bool

    #: 降级或纠偏的原因，给人看的。正常时为空。
    notes: list[str]


# ============================================================
# 纠偏与兜底
# ============================================================


def coerce_behavior(raw: Any) -> str:
    """
    把模型给的任意值收进 `BehaviorCategory` 的值域，收不进就 OTHER。

    接受 str 与 Enum（模型侧结构化解出来可能是两者之一），大小写与
    首尾空白不敏感 —— 模型写 "QA" 或 " qa " 显然是同一个意思，为这个
    判它"归不了类"没有道理。
    """

    return _coerce(raw, BehaviorCategory)


def coerce_execution_plan(raw: Any) -> str:
    """同 `coerce_behavior`，用于执行方案。"""

    return _coerce(raw, ExecutionPlan)


def _coerce(raw: Any, enum: type[Enum]) -> str:
    if isinstance(raw, Enum):
        raw = raw.value

    if isinstance(raw, str):
        candidate = raw.strip().lower()
        for member in enum:
            if member.value == candidate:
                return member.value

    return enum["OTHER"].value  # type: ignore[attr-defined]


def empty_classification(
        *,
        degraded: bool = True,
        note: str | None = None,
) -> AgentClassification:
    """
    一个"什么都没判出来"的结果。

    降级路径（模型调用失败、读不到候选模块、结构化输出解析不了）用它 ——
    五个维度全 OTHER，`degraded=True`。调用方据此知道"这是兜底，
    不是判断结果"，而不是把 OTHER 当成一次真实的分类。
    """

    return AgentClassification(
        behavior=BehaviorCategory.OTHER.value,
        modules=[],
        execution_plan=ExecutionPlan.OTHER.value,
        capability_needs=[],
        capability_config={},
        # 降级时**不召回**记忆：分类都失败了，"这一轮该不该查记忆"没有
        # 依据。与 `execution_plan` 落 other 同一条取舍（退回原样，
        # 而不是带病继续）。
        retrieve_l0=False,
        retrieve_l1=False,
        degraded=degraded,
        notes=[note] if note else [],
    )
