"""子代理子图的装配。

拓扑
----
    START -> task_listener
    task_listener --(领到任务)--> llm_node
    task_listener --(没有任务)--> END
    llm_node --(有 tool_calls)--> tool_node × N（Send 扇出）
    llm_node --(没有 tool_calls)--> task_listener   # 结算，再领下一个
    tool_node --(Command.goto)--> llm_node          # 工具结果回到模型

⚠️ tool_node 没有静态出边
-----------------------
下一跳由 `Command.goto` 决定（与主图 tools 节点同一条 handoff 约定）。
扇出的 N 个分支里只有第一个（控制器）返回 Command；其余只写消息。

作为**父图节点**编译
------------------
本图由 `app/agent/graph/react/graph.py` 直接 `add_node("subagent_worker",
build_subagent_graph())` 编译进父图：

    interrupt   由父 Pregel 循环管理（冒泡 / resume）
    custom 事件 子图的 tool.* / trace 进入父运行的事件流
    checkpoint  父 thread 的嵌套 namespace（不再有独立 child thread）
    输出隔离    `output_schema=SubAgentOutput`（空）—— 子图 state 不并回
                父 state，结果只走任务板

代价：原先按 task/attempt 的确定性 child thread 恢复不再成立；恢复跟
父 run 的 checkpoint 一起走。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.graph.react.context import ReactContext

from .catalog import LLM_NODE, NODE_SPECS, TASK_LISTENER_NODE, TOOL_NODE
from .nodes import llm_node, task_listener_node, tool_node
from .routing import route_after_listener, route_after_llm
from .state import SubAgentOutput, SubAgentState

__all__ = ["NODE_FUNCTIONS", "build_subagent_graph"]


#: node_id -> 节点函数。单列成字典，便于冒烟脚本 monkey-patch
#: （与主图 NODE_FUNCTIONS 同一条约定）。
NODE_FUNCTIONS: dict[str, Any] = {
    TASK_LISTENER_NODE: task_listener_node,
    LLM_NODE: llm_node,
    TOOL_NODE: tool_node,
}


def build_subagent_graph() -> StateGraph:
    """构造（未编译的）子代理图。"""

    graph: StateGraph = StateGraph(
        SubAgentState,
        context_schema=ReactContext,
        # ⚠️ output_schema **必须**显式给：子图作为父图节点运行时，它会按
        # 同名 channel 把输出合并回父 state（messages 会串进主对话）。
        # 空 schema = 什么也不回；结果走任务板。
        output_schema=SubAgentOutput,
    )

    for spec in NODE_SPECS:
        graph.add_node(spec.node_id, NODE_FUNCTIONS[spec.node_id])

    graph.add_edge(START, TASK_LISTENER_NODE)

    # 领到任务 -> llm_node；没有任务 -> END。
    graph.add_conditional_edges(
        TASK_LISTENER_NODE,
        route_after_listener,
        [LLM_NODE, END],
    )

    # 有 tool_calls -> Send(tool_node)；否则回 listener 结算。
    #
    # ⚠️ 第三参仍要写 TOOL_NODE：Send 绕过映射表，但裸字符串分支
    # （TASK_LISTENER_NODE）必须能映射到；漏写会在运行期报 Unknown branch。
    graph.add_conditional_edges(
        LLM_NODE,
        route_after_llm,
        [TOOL_NODE, TASK_LISTENER_NODE],
    )

    # tool_node 没有静态出边：下一跳由 Command.goto 决定。

    return graph
