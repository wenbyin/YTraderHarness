"""Planner 子图的观测契约与模型路由键。

为什么是子图
------------
planner 需要自己的 transcript 与工具循环。把它作为父图节点直接跑，
工具调用与中间推理只能写进主对话 `messages` —— 会污染用户可见回复、
被 message_recorder 落库、被前端时间线渲染。子图给它独立的 state 与
checkpoint namespace，只把最终计划交回父图。

节点 id
-------
`planner_llm` 而不是 `planner`：父图节点已经叫 `planner`，同名会让
运行事件流里的 node.* 难以区分。`planner_tool` 也刻意与父图 `tools`、
子代理 `tool_node` 区分开。
"""

from __future__ import annotations

from uuid import UUID

from app.agent.model import ModelRole
from app.agent.tools.plan_tool import (
    ENTER_PLAN_TOOL_KEY,
    EXIT_PLAN_TOOL_KEY,
)
from app.agent.tools.task_tool import CREATE_TEAM_TOOL_KEY
from app.common.context import ContextBuilderType
from app.infra.graph import NodeSpec

__all__ = [
    "AGENT_TYPE",
    "CONTEXT_BUILDER_TYPE",
    "EDGES",
    "GRAPH_NAME",
    "GRAPH_VERSION",
    "MODEL_ROLE",
    "NODE_SPECS",
    "PLANNER_EXCLUDED_TOOL_KEYS",
    "PLANNER_LLM_NODE",
    "PLANNER_NODE_IDS",
    "PLANNER_TOOL_SCOPE_NAMESPACE",
    "TASK_GENERATOR_NODE",
    "TOOL_NODE",
]


#: 图名。与 `agent_react` / `agent_subagent` 并列。
GRAPH_NAME = "agent_planner"

#: 拓扑版本。子图刚拆出，从 0.1.0 开始。
GRAPH_VERSION = "0.1.0"


#: ReAct 规划回合（模型调用）。
PLANNER_LLM_NODE = "planner_llm"

#: 工具执行（一次调用一个 Send 分支）。
TOOL_NODE = "planner_tool"

#: 结构化计划生成（最终 JSON）。
TASK_GENERATOR_NODE = "task_generator"


#: 子图内合法的 Command.goto 目标。工具返回的其他目标（父图节点）必须
#: 被 planner_tool 拦下 —— 子图里 goto 不到父节点。
PLANNER_NODE_IDS: frozenset[str] = frozenset({
    PLANNER_LLM_NODE,
    TOOL_NODE,
    TASK_GENERATOR_NODE,
})


#: 虽然能力默认集里包含，但在子图里不能执行的工具：
#:
#:     create_team       -> Command(goto="subagent_dispatch")（父图节点）
#:     enter_plan_mode   -> Command(goto="planner")（父图节点）
#:     exit_plan_mode    -> Command(goto="agent")（父图节点）
#:
#: 这些工具的控制面属于父图；planner 调用它们会 goto 到一个不存在的
#: 节点。后续若新增返回父图 Command 的工具，planner_tool 的运行时守门
#: 会兜底，但这里先显式排除，避免模型看到必然失败的工具。
PLANNER_EXCLUDED_TOOL_KEYS: frozenset[str] = frozenset({
    CREATE_TEAM_TOOL_KEY,
    ENTER_PLAN_TOOL_KEY,
    EXIT_PLAN_TOOL_KEY,
})


#: 模型路由键：planner 子图用 `agent_type=planner + reasoning`。
AGENT_TYPE = "planner"

MODEL_ROLE = ModelRole.REASONING


#: 本图用的 Context Builder 类型（窄输入 + 可选工具）。
CONTEXT_BUILDER_TYPE = ContextBuilderType.PLANNER


#: 工具分支 scope 的 uuid5 命名空间。与主图 / 子代理分开，避免观测行
#: 按 branch_key 合并到同一条。
PLANNER_TOOL_SCOPE_NAMESPACE = UUID(
    "d4f8a1c2-7b36-4e59-9a0f-3c2d8e5b6a71"
)


#: 节点目录。sequence 是登记序号，永不重编号。
NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        PLANNER_LLM_NODE,
        "规划探索",
        "linear",
        0,
        "ReAct 规划回合：可用只读/运行时工具查目录或澄清；"
        "无 tool_calls 时进入 task_generator。",
    ),
    NodeSpec(
        TOOL_NODE,
        "规划工具",
        "fanout",
        1,
        "执行 planner 的一次工具调用；结果回到 planner_llm。"
        "返回父图 Command 的工具会被拦下。",
    ),
    NodeSpec(
        TASK_GENERATOR_NODE,
        "生成任务清单",
        "linear",
        2,
        "不接工具，严格 JSON 输出 TaskDraft；校验 DAG 后转成 A2A "
        "tasks 并发布 plan.published。",
    ),
)


#: 拓扑边（给指纹与前端画图用）。装配在 graph.py 里显式写。
EDGES: tuple[tuple[str, str], ...] = (
    ("__start__", PLANNER_LLM_NODE),
    (PLANNER_LLM_NODE, TOOL_NODE),          # 有 tool_calls：Send 扇出
    (PLANNER_LLM_NODE, TASK_GENERATOR_NODE),  # 没有 tool_calls：生成计划
    (TOOL_NODE, PLANNER_LLM_NODE),          # Command.goto 回到模型
    (TASK_GENERATOR_NODE, "__end__"),
)
