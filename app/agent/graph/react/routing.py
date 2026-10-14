"""
走向判断：builder 之后去哪、agent 之后去哪。

单独一个模块，理由与 document/pipeline/fanout.py 相同：它是**纯函数**
—— 只看 state，不碰 runtime、不碰数据库、不建图。因此能脱离图单独测
（喂一个 state 进去，断言返回哪个分支），而图里的条件边只能整张图跑。

    route_after_builder   Agent Builder 判出的执行方案 -> agent / planner
    route_after_agent     模型**有没有要求调工具** -> 一批 Send（一调用一分支）/ END

⚠️ ReactState 必须是**运行时** import，不能放进 TYPE_CHECKING
------------------------------------------------------------
LangGraph 在 `add_conditional_edges` 时会对这些路由函数调
`typing.get_type_hints()`（为了推它接受什么 state）。注解写在
`from __future__ import annotations` 下是字符串，get_type_hints 要去
模块全局里找那个名字 —— 只 import 在 TYPE_CHECKING 下的话找不到，
报的是 `NameError: name 'ReactState' is not defined`，
而堆栈指向 langgraph 的 _branch.py，离原因很远。

（节点函数没这个问题：`add_node` 不解析节点函数的注解。所以这条
陷阱只在**路由函数**上，而它看起来与节点函数一模一样。）

⚠️ 改成发 Send 之后，path_map 的语义变了
----------------------------------------
原文是"返回的字符串必须在 `add_conditional_edges` 第三参里"。现在
要分两种返回值（实测 `langgraph/graph/_branch.py`）：

    Send 对象       **绕过** path_map —— 目标节点名写在 `Send(node, …)`
                    的第一个参数里
    裸字符串        `END` 这种走 path_map，映射不到就 `Unknown branch`

所以 `graph.py` 的第三参仍然要写 `[TOOLS_NODE, END]`（`END` 得能
映射），但**不要再留一条"回 TOOLS_NODE 字符串"的返回路径** ——
那会同时存在两条进 tools 的路（一条给 Send、一条给普通边），
两者的 payload 不一样，早晚漂移。

⚠️ 绝不能返回空列表
------------------
空 destinations 时 LangGraph **不报错**，而是让这次运行**静默结束**
（document 的 `pipeline/fanout.py` 也是实测记下的）。表现是"图跑到
一半没了，没有终态、也没有报错"。本函数的形状天然保证非空
（没 tool_calls 就回 `END`，有就至少发一条 Send）—— 但**不要**
在这里加"过滤掉某些 call"的优化，那正是把静默结束引回来的写法。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Send

from app.agent.state import ExecutionPlan

from .state import ReactState

__all__ = ["route_after_agent", "route_after_builder"]


#: 要调工具时去的节点，必须是 catalog.NODE_SPECS 里的 node_id。
TOOLS_NODE = "tools"

#: 主循环节点。
AGENT_NODE = "agent"

#: 规划节点（占位，见 planner_node.py）。
PLANNER_NODE = "planner"


def route_after_builder(state: ReactState) -> str:
    """
    Agent Builder 判出的执行方案 -> 去哪。

        plan            -> planner（先规划再执行）
        其余（含 other）-> agent（正常主循环）

    ⚠️ 为什么 `direct` 与 `react` 走同一个节点
    ---------------------------------------
    两者的差别是"这次要不要给模型工具"，那是 `exposed_tools` 的产生
    路径（能力选择）该管的事，不是节点二选一。本图里 agent 节点既可能
    直接答复、也可能要求调工具（它自己按有没有 tool_calls 决定，见
    route_after_agent）—— 所以 `direct` 与 `react` 在**拓扑上**没有
    分岔点。真要把 direct 做成"不带工具"，改的是能力选择那一步。

    ⚠️ 没有分类结果 / 降级 / 判不出来，一律走 agent
    ---------------------------------------------
    那就是这次改动之前的行为。分类失败不该让对话走一条从没跑过的路。
    """

    classification = state.get("classification") or {}

    plan = classification.get("execution_plan")

    if plan == ExecutionPlan.PLAN.value:
        return PLANNER_NODE

    return AGENT_NODE


def _tool_keys(state: ReactState) -> list[str]:
    """
    本次暴露的工具 key（Agent Builder 写进 `context_policy` 的输入）。

    缺这个键就是"这次没有工具"（分类没判出模块、或降级了），与"没配"
    不是一回事 —— 两者都得到空集合，行为一致。

    本函数只**搬运原文**：允许集由 tools 分支用 `select_specs` 现算
    （见 `infra/agent/tool_executor.py`）。路由在这里算一次、分支再算
    一次的话，"给模型看什么"与"允许执行什么"就有了两个来源 ——
    而它们本该是同一个集合。
    """

    policy = state.get("context_policy") or {}

    raw = policy.get("tool_keys") if isinstance(policy, dict) else None

    if not isinstance(raw, (list, tuple)):
        return []

    return [str(item) for item in raw]


def _call_scope(
        message: AIMessage,
        index: int,
        call: Mapping[str, Any],
) -> str:
    """
    一个扇出分支的稳定身份（`branch_scope_id` 的 key）。

    ⚠️ 为什么不能只用 `tool_call_id`
    ---------------------------------
    很多 provider **每轮都重发** `"call_1"` 这种 id（实测过：同一张图
    跑两轮，两条 AIMessage 的 tool_call id 都是 `call_1`）。而观测行的
    唯一键是 `(pipeline_run_id, node_id, branch_key, attempt)` ——
    同一 run 的两轮用同一个 key，两个分支的行会**合并成一行**，
    表现是"这一轮的第二个工具在观测里不存在"。

    `AIMessage.id` 由 `add_messages` 归约时分配（没有的话补一个 uuid，
    实测确认），**每轮不同**，拼上它就把跨轮撞键这条路堵死了。

    `tool_call_id` 缺失的 provider（个别实现给空串）退化为下标：
    下标在**这一次** tool_calls 里稳定，而 message.id 已经保证了
    "这一次"与"上一次"不同。

    拼出来若形如 `None:call_1`，说明这条 AIMessage 没有 id —— 那只能
    是有人**绕过了 messages 的 reducer**（`add_messages` 会给没有 id
    的消息补一个 uuid，实测确认）。那种写法本身就是错的：绕过 reducer
    的消息不会进 state，下一轮也看不到。
    """

    call_id = str(call.get("id") or "")

    return f"{message.id}:{call_id or index}"


def route_after_agent(state: ReactState) -> str | list[Send]:
    """
    模型要求调工具就继续，否则收尾。

    ⚠️ 只看 `tool_calls`，**不看 `invalid_tool_calls`**
    --------------------------------------------------
    模型有时会产出一个参数不合 schema 的调用，langchain 把它放进
    `invalid_tool_calls` 而不是 `tool_calls`。那种调用**执行不了** ——
    路由到 tools 只会拿到一个空列表然后在别处炸，报错离原因很远。
    这里的取舍是当作"没要求调工具"直接收尾：调和的人能在消息里看到
    那个无效调用，而它本来也不是这一轮能成的。

    ⚠️ 只看**最后一条** AIMessage
    -----------------------------
    历史里的旧 AIMessage 早就被处理过了。遍历全部消息会让"上一轮调过
    工具"变成"这一轮也要调"，于是图会永远在循环里 —— 而且每次循环都
    带着正确的历史，看起来毫无异常。

    ⚠️ 这里**不设**循环次数上限
    ---------------------------
    模型反复要求调工具时，兜底的是 LangGraph 的 `recursion_limit`
    （默认 25 个 super-step，超出抛 GraphRecursionError）。要把它调成
    业务可理解的数字，应该改 `RunRequest` / `GraphRunner`（那是 infra
    的接线），而不是在这里藏一个计数器 —— 藏在这里的话，"怎么跑了
    25 轮"在日志里得先分清是哪个上限在起作用。

    ⚠️ 为什么是"一调用一分支"而不是"一次给一个节点一堆调用"
    ------------------------------------------------------
    后者（单节点里 for 循环）与 `interrupt()` 不兼容：interrupt 会
    丢弃该节点待写的状态、**从节点开头重跑**，于是中断在第 3 个工具
    时，前两个工具的真实副作用会再来一遍（实测：中断后 `['t1']`，
    resume 后 `['t1','t1','t2']`）。下单、转账这类工具不能这么跑。

    扇出后每个分支是一个独立 task，已完成的 task 在 checkpoint 里
    落定，resume 只重跑被中断的那一个（实测：`['t1','t2']`）。
    审批拦截器的 docstring 里"安全的前提是一个工具一个分支"说的就是这件事。
    """

    messages = state.get("messages") or ()

    if not messages:
        # 空对话走到 agent 之后是**接线错误**（入口该带着消息进来），
        # 但这里没有足够信息说清是哪一环，交给收尾让它尽早结束。
        return END

    last = messages[-1]

    if not (isinstance(last, AIMessage) and last.tool_calls):
        return END

    tool_keys = _tool_keys(state)

    return [
        Send(
            TOOLS_NODE,
            {
                "call": call,
                "tool_keys": tool_keys,
                "call_scope": _call_scope(last, index, call),
            },
        )
        for index, call in enumerate(last.tool_calls)
    ]
