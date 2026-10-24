"""Planner 子图的装配。

拓扑
----
    START -> planner_llm
    planner_llm --(有 tool_calls)--> planner_tool × N（Send 扇出）
    planner_llm --(没有 tool_calls)--> task_generator
    planner_tool --(Command.goto)--> planner_llm
    task_generator -> END

作为父图节点编译
----------------
由 `app/agent/graph/react/graph.py` 直接 `add_node("planner",
build_planner_graph())` 编译进父图：

    interrupt   由父 Pregel 循环管理（冒泡 / resume）
    custom 事件 子图的 tool.* / plan.published 进入父运行的事件流
    checkpoint  父 thread 的嵌套 namespace
    输入隔离    `input_schema=PlannerInput`（不把父 context_policy 带进来）
    输出隔离    `output_schema=PlannerOutput`（只有 plan/tasks/current/team）
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.graph.react.context import ReactContext

from .catalog import (
    NODE_SPECS,
    PLANNER_LLM_NODE,
    TASK_GENERATOR_NODE,
    TOOL_NODE,
)
from .nodes import planner_llm_node, planner_tool_node, task_generator_node
from .routing import route_after_planner_llm
from .state import PlannerInput, PlannerOutput, PlannerState

__all__ = ["NODE_FUNCTIONS", "build_planner_graph"]


#: node_id -> 节点函数。单列成字典，便于冒烟脚本 monkey-patch
#: （与主图 / 子代理图同一条约定）。
NODE_FUNCTIONS: dict[str, Any] = {
    PLANNER_LLM_NODE: planner_llm_node,
    TOOL_NODE: planner_tool_node,
    TASK_GENERATOR_NODE: task_generator_node,
}


def build_planner_graph() -> StateGraph:
    """构造（未编译的）planner 子图。"""

    graph: StateGraph = StateGraph(
        PlannerState,
        input_schema=PlannerInput,
        output_schema=PlannerOutput,
        context_schema=ReactContext,
    )

    for spec in NODE_SPECS:
        graph.add_node(spec.node_id, NODE_FUNCTIONS[spec.node_id])

    graph.add_edge(START, PLANNER_LLM_NODE)

    # 有 tool_calls -> Send(planner_tool)；否则 -> task_generator。
    #
    # ⚠️ 第三参仍要写 TOOL_NODE：Send 绕过映射表，但裸字符串分支
    # （TASK_GENERATOR_NODE）必须能映射到；漏写会在运行期报 Unknown branch。
    graph.add_conditional_edges(
        PLANNER_LLM_NODE,
        route_after_planner_llm,
        [TOOL_NODE, TASK_GENERATOR_NODE],
    )

    graph.add_edge(TASK_GENERATOR_NODE, END)

    # planner_tool 没有静态出边：下一跳由 Command.goto 决定。

    return graph
