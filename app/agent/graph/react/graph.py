"""
图的装配。

拓扑
----
    START -> builder
    builder --(execution_plan == plan)--> planner -> agent
    builder --(其余，含降级 / other)-----> agent
    agent   --(有 tool_calls)-----------> tools -> agent     循环
    agent   --(没有)--------------------> END

⚠️ builder 的分支列表与 route_after_builder 的返回值必须一致
----------------------------------------------------------
写漏一个，LangGraph 直到**运行到那条边**才报 `Unknown branch`（装配期
看不出来）。当前是 [AGENT_NODE, PLANNER_NODE] —— 注意 `direct` 与
`react` 都回 AGENT_NODE，所以列表里**没有**这两个名字：分支集合是"可能
走到的节点"，不是"可能返回的值"。理由见 routing.route_after_builder。

装配是**显式**写的，不遍历 `catalog.EDGES`
------------------------------------------
document 的 graph.py 能遍历 EDGES，是因为它那边除扇出外都是普通边。
这里不行：`agent -> tools` 是一条**条件边**，`add_edge` 表达不了它。

因此 `catalog.EDGES` 在本图里纯粹是**文档**（给拓扑指纹与前端画图
用），真装配在下面。两处必须一起改 —— 指纹算的是那边，跑的是这里。
这条代价是刻意的：反过来（从 EDGES 推导装配）就得让 EDGES 里带上
"这是条件边、条件是哪个函数"的信息，那等于把图的实现细节塞进一份
本该只描述形状的目录里。

返回**未编译**的图
------------------
由组合根决定用哪个 checkpointer 编译（生产 Postgres、冒烟
InMemory），图本身不该关心 —— 与 document 流水线同一约定。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .agent_builder_node import agent_builder_node
from .agent_node import agent_node
from .catalog import (
    AGENT_BUILDER_NODE,
    AGENT_NODE,
    NODE_SPECS,
    PLANNER_NODE,
)
from .context import ReactContext
from .planner_node import planner_node
from .routing import (
    TOOLS_NODE,
    route_after_agent,
    route_after_builder,
)
from .state import ReactState
from .tools import tools_node

__all__ = ["NODE_FUNCTIONS", "build_react_graph"]


#: 阶段 node_id -> 节点函数。
#:
#: 单列成字典而不是在装配时直接引用函数，是为了让冒烟脚本能像
#: document 那样 monkey-patch 掉某个节点（见 pipeline_smoke.py 的
#: stub-run）—— 不必为了跑一张"只有拓扑正确"的图去接真模型。
NODE_FUNCTIONS: dict[str, Any] = {
    AGENT_BUILDER_NODE: agent_builder_node,
    AGENT_NODE: agent_node,
    PLANNER_NODE: planner_node,
    TOOLS_NODE: tools_node,
}


def build_react_graph() -> StateGraph:
    """
    构造（未编译的）ReAct 图。

    ⚠️ 节点上**没有** retry_policy
    -------------------------------
    document 图给每个节点挂了重试（只重试 ConnectionError / TimeoutError
    / OSError）。本图刻意不挂，因为这里的重试已经**有主**了：

        模型失败   -> Model Runtime 内部重试（只重试可重试的错误，
                      且不会重跑节点、不重建上下文、不重放 state ——
                      见 app/agent/model/runtime.py）
        工具失败   -> 不重试。工具重试意味着副作用执行两遍（下单、
                      转账），那是业务必须自己决定的事。

    把重试挂到节点上会与 Runtime 的重试叠乘，而且它重跑的是**整个节点**
    （上下文重建 + 模型再调一次），而 Runtime 的重试只重发同一个请求。

    循环次数的兜底是 LangGraph 的 `recursion_limit`（默认 25 个
    super-step），不是这里 —— 要改成业务可理解的数字应当走
    `RunRequest` / `GraphRunner`（infra 的接线），在那里它是个
    显式参数，而不是节点上一个没人注意的默认值。
    """

    graph: StateGraph = StateGraph(
        ReactState,
        context_schema=ReactContext,
    )

    for spec in NODE_SPECS:
        graph.add_node(spec.node_id, NODE_FUNCTIONS[spec.node_id])

    # 入口是 builder，不是 agent：先理解输入，再决定怎么执行。
    graph.add_edge(START, AGENT_BUILDER_NODE)

    # 条件边一：Agent Builder 判出的执行方案决定去哪。
    # 分支列表是**可能走到的节点**（不是可能返回的值）：direct 与 react
    # 都回 AGENT_NODE，所以这里看不到它们。
    graph.add_conditional_edges(
        AGENT_BUILDER_NODE,
        route_after_builder,
        [AGENT_NODE, PLANNER_NODE],
    )

    # planner 当前是占位（什么都不做），放行到主循环。
    graph.add_edge(PLANNER_NODE, AGENT_NODE)

    # 条件边二：模型自己决定要不要调工具。
    #
    # ⚠️ 这条边**发的是 Send**：route_after_agent 对每个 tool_call 返回一条
    # `Send(TOOLS_NODE, payload)`，于是 tools 节点被实例化 N 次、每次只处理
    # 一个调用（为什么必须这样见 routing.route_after_agent 与 tools.py）。
    #
    # 第三参在这里的含义因此变了（实测 langgraph/graph/_branch.py：
    # `r if isinstance(r, Send) else self.ends[r]`）：
    #     Send        绕过这张映射表，目标名写在 Send 的第一个参数里
    #     裸字符串    仍然走它 —— 所以 `END` 必须留在这里，否则
    #                 "不用调工具"那条路会在运行期报 Unknown branch
    graph.add_conditional_edges(
        AGENT_NODE,
        route_after_agent,
        [TOOLS_NODE, END],
    )

    # 回环：带着工具结果回到模型。上界由 recursion_limit 兜底。
    #
    # ⚠️ 扇出后这条边同时是**扇入点**：N 个分支的返回值经 messages 上的
    # add_messages reducer 汇总进父状态，agent 在所有分支跑完之后只执行
    # 一次、看得到全部 ToolMessage（实测：顺序与 Send 顺序一致）。
    # 这也是本图不需要 document 那种"路由屏障节点"的原因 —— 全图只有
    # 一个扇出阶段。
    graph.add_edge(TOOLS_NODE, AGENT_NODE)

    return graph
