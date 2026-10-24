"""子代理子图的观测契约与模型路由键。

为什么是**另一张图**
------------------
worker 的节点身份、state schema、模型档位都与主图不同：

    node_id       llm_node / tool_node（不是 agent / tools）
    state         SubAgentState（不是 ReactState）
    agent_type    subagent（可以单独换模型）
    builder       SUB_AGENT（不是 AGENT_INVOKE）

放在同一张图里会让两件事变得说不清：

    1. RunEventProjector 的 `message_nodes={"agent"}` 会把 worker 的
       逐 token 正文当成用户可见回复推出去；
    2. worker 的 transcript 会落进父图 checkpoint，主图越来越重。

⚠️ 节点 id 与主图同名的那一个
---------------------------
`task_listener` 与主图的 `task_listener` **同名但是两个节点**：节点 id
是**每张图**的契约键，不是全局键。子图自己的目录与观测行按子图名区分。
"""

from __future__ import annotations

from uuid import UUID

from app.common.context import ContextBuilderType
from app.agent.model import ModelRole
from app.infra.graph import NodeSpec

__all__ = [
    "AGENT_TYPE",
    "CONTEXT_BUILDER_TYPE",
    "EDGES",
    "GRAPH_NAME",
    "GRAPH_VERSION",
    "LLM_NODE",
    "MODEL_ROLE",
    "NODE_SPECS",
    "SUBAGENT_TOOL_SCOPE_NAMESPACE",
    "TASK_LISTENER_NODE",
    "TOOL_NODE",
]


#: 图名。与主图的 `agent_react` 并列，前端 / 观测按它区分。
GRAPH_NAME = "agent_subagent"

#: 拓扑版本。0.x = 尚未接入父图派发。
GRAPH_VERSION = "0.1.0"


#: 任务领取 / 完成 / 退出节点。
TASK_LISTENER_NODE = "task_listener"

#: 模型调用节点（复用主图 agent_node 的核心实现）。
LLM_NODE = "llm_node"

#: 工具执行节点（Send 扇出，一次调用一个分支）。
TOOL_NODE = "tool_node"


#: 本图的 Agent 类型 —— Model Registry 的第一把路由键。
#:
#: 用 `subagent` 而不是 `react`：worker 常常是长工具循环，模型取舍与
#: 主对话不同，需要能单独换（见模型绑定 subagent + reasoning）。
AGENT_TYPE = "subagent"

#: 模型侧角色。worker 这次调用干的就是推理。
MODEL_ROLE = ModelRole.REASONING

#: 本图用的 Context Builder 类型（子任务 scope 的隔离上下文）。
CONTEXT_BUILDER_TYPE = ContextBuilderType.SUB_AGENT


#: 工具分支 scope 的 uuid5 命名空间。
#:
#: 与主图的 REACT_TOOL_SCOPE_NAMESPACE 分开：否则 worker 的 tool 分支
#: 与主图 tools 的分支可能算出同一个 branch_key（共享 call_scope 时），
#: 观测行会合并成一行。
SUBAGENT_TOOL_SCOPE_NAMESPACE = UUID(
    "b3e7d9a4-2c58-4f16-9a0d-6e1f8c5b7a42"
)


#: 节点目录。sequence 是登记序号，永不重编号。
NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        "task_listener",
        "任务领取 / 完成",
        "linear",
        0,
        "完成刚执行完的任务（写回任务板），再领取下一个 ready task；"
        "没有可领取的任务就退出。循环由 listener -> llm_node 的图回边"
        "表达，节点内不做阻塞 while。",
    ),
    NodeSpec(
        "llm_node",
        "模型调用",
        "linear",
        1,
        "用子任务自己的 messages / skill / capabilities 构建上下文并"
        "调用模型；复用主图 agent_node 的 run_agent_turn。",
    ),
    NodeSpec(
        "tool_node",
        "工具执行",
        "fanout",
        2,
        "执行一个工具调用（一个调用一个分支实例），结果回到 llm_node "
        "让模型看到 ToolMessage 后继续推理。",
    ),
)


#: 拓扑边（给指纹与前端画图用）。装配在 graph.py 里显式写。
EDGES: tuple[tuple[str, str], ...] = (
    ("task_listener", "llm_node"),   # 领到任务
    ("task_listener", "__end__"),    # 没有 ready task / 全部终态
    ("llm_node", "tool_node"),       # 模型要求调工具 —— Send 扇出
    ("llm_node", "task_listener"),   # 不再调工具：任务执行完毕，回来结算
    ("tool_node", "llm_node"),       # Command.goto：工具结果回到产生它的模型
)
