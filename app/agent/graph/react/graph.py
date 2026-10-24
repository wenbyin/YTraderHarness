"""
图的装配。

拓扑
----
    START -> builder
    builder --(合成 enter_plan_mode)--> tools
    builder --(plan 且没有任务 / 任务全完成)--> planner
    planner --(默认)-------------------------> task_listener -> agent
    planner --(STOP_AFTER_PLANNER=1)---------> END（只出计划，开发期）
    builder --(其余，含降级 / off)------------> agent
    agent   --(有 tool_calls)--> tools
    tools   --(Command.goto, 工具自己决定)--> planner | agent | subagent_dispatch
    agent   --(没有 tool_calls 且有任务)--> task_listener
    agent   --(没有 tool_calls 且无任务)--> END
    subagent_dispatch --(Send ×N, 预 claim)--> subagent_worker ×N
    subagent_worker   --(join)--------------> subagent_dispatch   # 下一波
    subagent_dispatch --(没有 ready task)----> task_listener       # 收尾

⚠️ builder 的分支列表与 route_after_builder 的返回值必须一致
----------------------------------------------------------
写漏一个，LangGraph 直到**运行到那条边**才报 `Unknown branch`（装配期
看不出来）。当前是 [AGENT_NODE, PLANNER_NODE, TOOLS_NODE] ——
注意 `react` 与 `other` 都回 AGENT_NODE，所以列表里**没有**这两个
名字：分支集合是"可能走到的节点"，不是"可能返回的值"。理由见
routing.route_after_builder。

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
    SUBAGENT_DISPATCH_NODE,
    SUBAGENT_WORKER_NODE,
    TASK_LISTENER_NODE,
)
from .context import ReactContext
from .routing import (
    TOOLS_NODE,
    route_after_agent,
    route_after_builder,
    route_after_planner,
)
from app.agent.graph.planner.graph import build_planner_graph
from app.agent.graph.subagent.graph import build_subagent_graph

from .state import ReactState
from .subagent_dispatch_node import subagent_dispatch_node
from .task_listener_node import task_listener_node
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
    TASK_LISTENER_NODE: task_listener_node,
    TOOLS_NODE: tools_node,
    SUBAGENT_DISPATCH_NODE: subagent_dispatch_node,
}

#: planner / subagent worker **不是**普通节点函数，而是编译进父图的子图。
#:
#: `NODE_FUNCTIONS["planner"]` 保留为**冒烟/测试替换点**：存在时按普通
#: 节点使用（见 agent_smoke.py 的 stub），不存在时挂编译子图。生产路径
#: 不往里写，于是始终走子图。
#:
#: 这样 interrupt 由父图的 Pregel 循环管理（冒泡 / resume），子图的
#: custom 事件也会进父运行的事件流。代价是子图输出会按同名 channel
#: 合并 —— 已用 `SubAgentOutput`（空 schema）隔掉。


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

        if spec.node_id in {SUBAGENT_WORKER_NODE, PLANNER_NODE}:
            continue

        graph.add_node(spec.node_id, NODE_FUNCTIONS[spec.node_id])

    # planner：编译好的子图作为节点。独立 PlannerState / transcript /
    # checkpoint namespace；只把 plan / tasks / current_task / team_decision
    # 合并回父图。
    #
    # ⚠️ 这里 compile() 不带 checkpointer：**子图会继承父图的**（实测，
    # 无 checkpointer 的子图作为节点时 interrupt / resume 仍按父 thread
    # 的嵌套 namespace 工作）。父图由组合根按生产/冒烟选择 saver。
    graph.add_node(
        PLANNER_NODE,
        NODE_FUNCTIONS.get(PLANNER_NODE) or build_planner_graph().compile(),
    )

    # worker：同上。
    graph.add_node(
        SUBAGENT_WORKER_NODE,
        build_subagent_graph().compile(),
    )

    # 入口是 builder，不是 agent：先理解输入，再决定怎么执行。
    graph.add_edge(START, AGENT_BUILDER_NODE)

    # 条件边一：控制调用 / plan_status + 计划任务状态决定去哪。
    #
    # 分支列表是**可能走到的节点**（不是可能返回的值）：react / other
    # 都回 AGENT_NODE，所以这里看不到它们。分类器的
    # execution_plan 只决定"要不要合成 enter_plan_mode 调用"，真正的
    # 模式切换在 tools 节点（见 routing.route_after_builder）。
    graph.add_conditional_edges(
        AGENT_BUILDER_NODE,
        route_after_builder,
        [AGENT_NODE, PLANNER_NODE, TOOLS_NODE],
    )

    # planner 之后：默认进 task_listener（领取一个 ready task，再进主
    # 循环）。开发期可以用开关停在 planner 之后，只检查计划：
    #
    #     YTRADER_PLAN_STOP_AFTER_PLANNER=1
    #
    # ⚠️ 默认**不能**是 END：planner 已经不发布任务板，停在 END 会得到
    # 一次「completed 但没有任何 assistant 回复」的运行。
    #
    # ⚠️ 条件边的分支列表必须包含两个目标；漏写会在运行期报
    # `Unknown branch`（装配期看不出来）。
    graph.add_conditional_edges(
        PLANNER_NODE,
        route_after_planner,
        [TASK_LISTENER_NODE, END],
    )
    graph.add_edge(TASK_LISTENER_NODE, AGENT_NODE)

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
        [TOOLS_NODE, TASK_LISTENER_NODE, END],
    )

    # tools **没有静态出边**。
    #
    # 下一跳由分支返回的 `Command.goto` 决定（LangGraph 的 handoff）：
    # **任何**工具都可以在 Tool 里写 `Command(goto=...)`；没返回 Command
    # 的普通工具由 tools 节点补 `Command(goto="agent")`。每批调用只有
    # 第一个调用（控制器分支）返回 Command，其余分支只写消息 / 状态 ——
    # 否则同一 super-step 会出现多个 goto，多个节点都被调度。
    #
    # ⚠️ 扇入仍然成立：N 个分支的返回值经 messages 上的 add_messages
    # reducer 汇总进父状态，控制器 Command 里指定的下一跳在所有分支
    # 跑完之后才执行（实测：顺序与 Send 顺序一致）。

    # 子代理扇出的 join：所有 worker 分支结束后回到 dispatch 重新调度
    # 下一波（实测：多个 Send 分支共用一个静态出边时，目标节点在**全部**
    # 分支结束后执行一次，不是每分支一次）。
    graph.add_edge(SUBAGENT_WORKER_NODE, SUBAGENT_DISPATCH_NODE)

    # subagent_dispatch 没有静态出边：
    #   有 ready task -> Command(goto=[Send("subagent_worker", ...)])
    #   没有          -> Command(goto="task_listener")
    # 与 tools 的 handoff 约定一致。

    return graph
