"""
走向判断：builder 之后去哪、agent 之后去哪。

单独一个模块，理由与 document/pipeline/fanout.py 相同：它基本是**纯函数**
—— 只看 state、不碰数据库、不建图。因此能脱离图单独测（喂一个 state
进去，断言返回哪个分支），而图里的条件边只能整张图跑。

⚠️ 唯一的例外：`route_after_agent` 会发 trace 的 `fanout.*` / `graph.transition`
------------------------------------------------------------------------------
这些事件需要"这次 agent 执行的 task_id"和真实的 Send 分支清单 —— 两者
只有路由函数知道。它因此声明了 `runtime` 参数（langgraph 的 RunnableCallable
会注入），并用 `get_stream_writer()` 写 custom stream。trace 未开启时
直接返回，不产生任何副作用。

    route_after_builder   Agent Builder 判出的执行方案 -> agent / planner
    route_after_agent     模型**有没有要求调工具** -> 一批 Send（一调用一分支）
                          / task_listener（还有任务）/ END

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

import os
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Send

from app.common.event import EventType, SpanKind, SpanStatus, TraceEmitter
from app.common.event.trace_emitter import stable_span_id, tool_span_id
from app.infra.graph import execution_ids

from .context import ReactContext
from .plan_state import (
    PLAN_ACTIVE,
    PLAN_OFF,
    plan_needs_replan,
)
from .state import ReactState

__all__ = [
    "STOP_AFTER_PLANNER_ENV",
    "STOP_AFTER_PUBLISH_ENV",
    "TASK_LISTENER_NODE",
    "call_scope",
    "resolve_tool_keys",
    "route_after_agent",
    "route_after_builder",
    "route_after_planner",
]


#: 要调工具时去的节点，必须是 catalog.NODE_SPECS 里的 node_id。
TOOLS_NODE = "tools"

#: 主循环节点。
AGENT_NODE = "agent"

#: 规划节点（编译子图，见 app/agent/graph/planner/）。
PLANNER_NODE = "planner"

#: 任务领取 / 完成节点。
TASK_LISTENER_NODE = "task_listener"


def _route_for_plan(state: ReactState) -> str:
    """
    `plan_status` + 计划任务状态 -> 去 planner 还是 agent。

    这是本图**唯一**的"该不该重新规划"判据，builder 之后与 tools 之后
    共用同一份实现：

        非 plan 模式                     -> agent
        plan，且没有任务 / 任务全完成    -> planner（生成新计划）
        plan，且还有任务未完成           -> agent（继续执行）

    ⚠️ 不再看 `classification.execution_plan`
    ---------------------------------------
    分类只负责**提议**进入计划模式（builder 把它结算成 plan_status）。
    如果这里再读一次 classification，就出现两个判定源：分类说 plan、
    state 说 off 时，"路由把 agent 送去 planner 但状态没变"这种中间态
    又回来了。权威来源只有一个：`plan_status`。
    """

    if state.get("plan_status") != PLAN_ACTIVE:
        return AGENT_NODE

    if plan_needs_replan(state.get("plan")):
        return PLANNER_NODE

    return AGENT_NODE


#: 临时开关：planner 产出计划后**直接结束本轮**，便于只检查计划 / 不再
#: 往下执行。默认 **关**（1 = 停，0 = 继续）。
#:
#: 历史名字是 `YTRADER_PLAN_STOP_AFTER_PUBLISH`（当时 planner 还负责发布
#: 任务板）。现在发布已经移到 `create_team`，名字保留兼容；新名字更准确：
#:
#:     YTRADER_PLAN_STOP_AFTER_PLANNER=1
#:
#: ⚠️ 这是开发期开关，不是产品配置。接上 team_decision / worker 派发后
#: 应当删掉它，由决策节点决定下一跳。
STOP_AFTER_PLANNER_ENV = "YTRADER_PLAN_STOP_AFTER_PLANNER"

#: 兼容旧名字。
STOP_AFTER_PUBLISH_ENV = "YTRADER_PLAN_STOP_AFTER_PUBLISH"

_FALSY = frozenset({"0", "false", "no", "off"})


def route_after_planner(state: ReactState) -> str:
    """
    planner 之后去哪。

        默认            -> task_listener（继续执行）
        开关置 1        -> END（只出计划，便于检查）

    默认必须继续：planner 已经不发布任务板，停在 END 会得到一次
    「completed 但没有任何 assistant 回复」的运行 —— 前端看起来像卡死。

    继续时仍然走 `task_listener`，而不是直接 `agent`：listener 负责
    领取第一个 ready task 并把任务指令写进 messages，这一步不能省。
    """

    raw = os.environ.get(STOP_AFTER_PLANNER_ENV)

    if raw is None:
        raw = os.environ.get(STOP_AFTER_PUBLISH_ENV, "0")

    stop = str(raw).strip().lower() not in _FALSY

    return END if stop else TASK_LISTENER_NODE


def _tool_sends(
        state: ReactState,
        message: AIMessage,
        *,
        plan_origin: str,
        return_to: str = AGENT_NODE,
) -> list[Send]:
    """
    一条 AIMessage 里的全部 tool_call -> 一批 `Send(TOOLS_NODE, payload)`。

    builder 与 agent 共用：分类器合成的 enter 调用和模型自己的调用因此
    走**完全相同的**执行路径（tools 扇出 -> 工具执行 -> Command.goto）。

    空列表是**禁止**的：LangGraph 对空 destinations 不报错，而是让这次
    运行静默结束。调用方必须先确认 `message.tool_calls` 非空。
    """

    calls = list(message.tool_calls)

    if not calls:
        raise ValueError(
            "_tool_sends 收到一条没有 tool_calls 的 AIMessage —— "
            "调用方必须先判空（空 Send 会让整次运行静默结束）。"
        )

    tool_keys = resolve_tool_keys(state)
    plan_status = str(state.get("plan_status") or PLAN_OFF)
    plan = state.get("plan") or {}

    # 控制工具（create_team）要读完整任务列表：判有无待完成任务、查依赖环、
    # 发布任务板。普通工具用不到它，但 Send 的 payload 形状是统一的。
    #
    # ⚠️ 这是**临时方案**：payload 随 pending task 进 checkpoint，N 个工具
    # 分支就会有 N 份 tasks。等对齐官方 ToolRuntime.state（用 CONFIG_KEY_READ
    # 从 channels 水合）之后，这里可以去掉，见 tools.py 的 TODO。
    raw_tasks = state.get("tasks")
    tasks = (
        list(raw_tasks)
        if isinstance(raw_tasks, (list, tuple))
        else []
    )

    # 一个 super-step 只能有一个分支决定下一跳：**第一个调用**是
    # 控制器。这与具体工具无关 —— 任何工具返回 Command 都能通过它
    # 控制下一跳（不是只有计划工具）。
    #
    # ⚠️ 扇出的固有限制：同一批多个工具时，只有第一个工具的 Command
    # 会被采用。原版 ToolNode 在一个节点里执行全部调用，可以同时
    # 返回多个 Command；本图是一调用一分支，要完全对齐需要再加一个
    # 汇总屏障 —— 那是独立改动。
    return [
        Send(
            TOOLS_NODE,
            {
                "call": call,
                "tool_keys": tool_keys,
                "call_scope": call_scope(message, index, call),
                # 分支读不到父状态：控制工具要靠这几个值判断是不是
                # 一次真正的模式切换（enter 在 plan 模式下是幂等的）。
                "plan_status": plan_status,
                "plan_origin": plan_origin,
                "plan": plan,
                "tasks": tasks,
                # 普通工具（含 ask_user_question）执行完回到哪：
                # 主 agent -> "agent"；planner 自己的工具循环 -> "planner"。
                "return_to": return_to,
                "is_controller": index == 0,
            },
        )
        for index, call in enumerate(calls)
    ]


def route_after_builder(state: ReactState) -> str | list[Send]:
    """
    builder 之后去哪。

        最后一条是合成工具调用（分类判定要规划）-> Send 到 tools
        off                                      -> agent（正常主循环）
        plan，没有任务 / 任务全完成              -> planner（先规划再执行）
        plan，还有任务未完成                     -> agent（继续执行当前计划）

    `react` / `other` 在**拓扑上**没有分岔点：它们都回 agent，差别只在
    能力选择与 `exposed_tools`（见 agent_builder_node 的模块 docstring）。

    ⚠️ 分类器与模型走的是**同一条**进入路径
    -----------------------------------
    分类判定要规划时，builder 合成一条 `enter_plan_mode` 的 AIMessage
    （不是直接写 state）。这里把它 Send 到 tools —— 与模型自己调用
    控制工具完全同一条路。工具返回 `Command(goto="planner")`，tools
    节点写状态并把它交给 LangGraph。

    resume 不会重跑 builder；checkpoint 里的 plan_status 因此是模式
    在中断后仍然成立的依据。
    """

    messages = state.get("messages") or ()

    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:

        last = messages[-1]
        extra = getattr(last, "additional_kwargs", None) or {}
        origin = str(extra.get("plan_origin") or "tool")

        return _tool_sends(state, last, plan_origin=origin)

    return _route_for_plan(state)


def resolve_tool_keys(state: Mapping[str, Any]) -> list[str]:
    """
    本次暴露的工具 key。

    task 模式下优先用 SkillLoader 写回的 `task_tool_keys`：模型看到的
    是 skill 的工具，路由与 tools 节点必须用同一个集合，否则会把模型
    刚调用的工具判成“未暴露”。
    """

    if state.get("current_task"):
        task_keys = state.get("task_tool_keys")

        if isinstance(task_keys, (list, tuple)):
            return [str(item) for item in task_keys]

    return _normal_tool_keys(state)


#: 兼容旧名字：既有测试按 `_tool_keys` 引用它。实现只有一份。
_tool_keys = resolve_tool_keys


def _normal_tool_keys(state: Mapping[str, Any]) -> list[str]:
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


def call_scope(
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


def route_after_agent(
        state: ReactState,
        runtime: Runtime[ReactContext],
) -> str | list[Send]:
    """
    模型要求调工具就继续；否则有任务回 task_listener，没任务才收尾。

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
        # 还有任务（或还有未清掉的 current_task）就回 listener：
        # listener 负责把刚执行完的任务落终态、清 current_task，再领取
        # 下一个。两者都没有才 END。
        if state.get("tasks") or state.get("current_task"):
            return TASK_LISTENER_NODE

        return END

    calls = list(last.tool_calls)

    # Trace：fanout + 真实的 agent -> tools 控制流。
    #
    # 这里是唯一**可靠**知道"这批 Send 是谁、什么时候派出去的"的地方：
    # 分支函数作为 agent 节点的 writer 执行，`runtime` 由 langgraph 注入，
    # 因此拿得到本次 agent 执行的 task_id（= 投影层 node.* 的 span_id）。
    # 事件走 get_stream_writer()（提示词第 5 节的图内通道）。
    _emit_fanout_trace(runtime, last, calls)

    # enter / exit 也是普通工具：照常扇出执行，由 tools 节点结算状态。
    # 模型主动调用的来源永远是 "tool"；分类器合成的那条走
    # route_after_builder，plan_origin 由合成消息携带。
    return _tool_sends(state, last, plan_origin="tool")


def _emit_fanout_trace(
        runtime: Runtime[ReactContext],
        last: AIMessage,
        calls: list[Any],
) -> None:
    """
    从路由函数发出 fanout.* / graph.transition。

    ⚠️ 确定性 ID
    ------------
    路由函数在 resume / replay 时可能被重新求值。span_id 与 event_id
    都由 (trace_id, agent task_id, call_scope) 确定性生成，这样重复
    求值产生的重复事件能被 event_id 去重，而不是变成两条不同的记录。

    ⚠️ 不写业务 State
    ----------------
    分支关系全部在这里算出来并通过 custom stream 发出；Send payload
    里**没有** trace 字段（提示词第 4 节第 10 条）。
    """

    context = getattr(runtime, "context", None)
    trace_id = getattr(context, "trace_id", None)

    if not trace_id:
        return

    run_id = getattr(context, "pipeline_run_id", None)
    session_id = getattr(context, "session_id", None)

    ids = execution_ids(runtime)
    agent_task_id = ids.task_id or (
        str(run_id) if run_id is not None else "unknown"
    )

    emitter = TraceEmitter(
        trace_id=trace_id,
        run_id=str(run_id) if run_id is not None else "",
        thread_id=None if session_id is None else str(session_id),
        writer=get_stream_writer(),
    )

    fanout_id = stable_span_id("fanout", trace_id, agent_task_id)

    emitter.instant(
        EventType.FANOUT_STARTED,
        span_kind=SpanKind.FANOUT,
        span_name="tool_fanout",
        span_id=fanout_id,
        parent_span_id=agent_task_id,
        status=SpanStatus.RUNNING,
        payload={
            "fanout_type": "tool_calls",
            "dispatch_mode": "parallel",
            "expected_branch_count": len(calls),
        },
        event_id=stable_span_id("event", fanout_id, "started"),
    )

    for index, call in enumerate(calls):
        call_id = call.get("id") if isinstance(call, dict) else None
        branch_scope = call_scope(last, index, call)

        # 分支子 Span —— 与工具执行链里的 tool span 用同一个确定性 id，
        # 也与投影层从协议推 tool.* 时算出的 id 一致。
        branch_span_id = (
            tool_span_id(str(trace_id), str(call_id))
            if call_id is not None
            else stable_span_id("tool", trace_id, branch_scope)
        )

        emitter.instant(
            EventType.FANOUT_BRANCH_CREATED,
            span_kind=SpanKind.FANOUT,
            span_name="tool_fanout",
            span_id=fanout_id,
            parent_span_id=agent_task_id,
            payload={
                "branch_id": branch_scope,
                "branch_span_id": branch_span_id,
                "branch_index": index,
                "target_kind": "tool",
                "target_name": str(call.get("name") or ""),
            },
            event_id=stable_span_id(
                "event", fanout_id, branch_scope, "branch_created"
            ),
        )

        emitter.instant(
            EventType.GRAPH_TRANSITION,
            span_kind=SpanKind.NODE,
            span_name="graph.transition",
            span_id=agent_task_id,
            payload={
                "from_span_id": agent_task_id,
                "to_span_id": branch_span_id,
                "from_node": AGENT_NODE,
                "to_node": TOOLS_NODE,
                "transition_kind": "conditional",
                "route_key": "tool_calls",
            },
            event_id=stable_span_id(
                "event", fanout_id, branch_scope, "transition"
            ),
        )
