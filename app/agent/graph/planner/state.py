"""Planner 子图 state：独立 transcript + 结构化计划输出。

隔离边界
--------
- 输入用显式 `PlannerInput`：只带父图的对话、身份、摘要与已有 plan；
  父图的 `context_policy` / `current_task` / `tasks` / `team_decision`
  **不会**进入 planner，避免主 agent 的职责提示词与工具集泄漏进来。
- 输出用显式 `PlannerOutput`：只把 plan / tasks / current_task /
  team_decision 合并回父图。planner 自己的 messages / context_result
  留在子图 namespace，不串进主对话。

⚠️ 未声明的键会被 LangGraph 静默丢弃
-----------------------------------
与 ReactState / SubAgentState 同一条：节点返回的键必须在这里声明。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

__all__ = [
    "PlannerInput",
    "PlannerOutput",
    "PlannerState",
    "PlannerToolPayload",
]


class PlannerInput(TypedDict, total=False):
    """父图传给 planner 子图的初始 state。"""

    messages: list[AnyMessage]
    conversation_id: str
    user_id: str
    summary: dict[str, Any]
    plan: dict[str, Any]


class PlannerOutput(TypedDict, total=False):
    """planner 子图合并回父图的通道。

    刻意**不含** messages / context_result / context_policy：
    planner 的工具循环与中间推理只留在自己的 namespace。
    """

    plan: dict[str, Any]
    tasks: list[dict[str, Any]]
    current_task: dict[str, Any]
    team_decision: str


class PlannerState(TypedDict, total=False):
    """planner 子图的完整 state。"""

    #: planner 自己的 transcript：父对话作为初始历史，后续工具循环只写这里。
    messages: Annotated[list[AnyMessage], add_messages]

    conversation_id: str
    user_id: str
    summary: dict[str, Any]

    #: planner 自己的上下文策略（不从父图输入）。
    context_policy: dict[str, Any]

    exposed_tools: list[Any]
    context_result: dict[str, Any]

    #: 计划快照：输入用于 re-plan 的 plan_id / revision；输出覆盖父图。
    plan: dict[str, Any]
    tasks: list[dict[str, Any]]

    #: 计划发布时恒为空 / 空串；真实 claim 由父图 task_listener 做。
    current_task: dict[str, Any]
    team_decision: str

    #: 首次模型调用前解析一次默认能力，之后只复用 context_policy.tool_keys。
    planner_tools_resolved: bool

    #: 工具解析 / 探索阶段的降级说明，最终并入 plan.notes。
    planner_notes: list[str]


class PlannerToolPayload(TypedDict):
    """planner 子图 tool 扇出分支的 Send payload。

    Send payload **取代** state：分支看不到子图任何字段，因此这里列出
    分支真正需要的东西。
    """

    call: dict[str, Any]
    tool_keys: list[str]
    call_scope: str
    return_to: str
    is_controller: bool
