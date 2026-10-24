"""
计划模式工具：进入 / 退出。

它们是**真工具**，由 `tools` 节点正常执行
----------------------------------------
工具通过注入的 `ToolRuntime.state` 读取调用前的状态，返回 LangGraph
的 `Command`：

    enter_plan_mode -> Command(goto="planner" | "agent", update={...})
    exit_plan_mode  -> Command(goto="agent", update={...})

`goto` 与 `plan_status` / `plan_origin` / `plan` 都在工具里算好、
直接放进 `Command.update`。`tools` 节点只做两件事：

    1. 给这次调用补一条 ToolMessage
    2. 控制器分支把 Command 原样交给 LangGraph

于是没有 `route_after_tools`：下一跳由工具返回的 `goto` 决定。

⚠️ `runtime: ToolRuntime` 不会被模型看到
-------------------------------------
它是 LangChain / LangGraph 的**注入参数**：

    registry/schema.py      injected_args 识别它，strip_injected_args
                            把它从给模型的 JSON Schema 里剔除
    tool_executor.py        调用前把官方 ToolRuntime 注入 arguments

`ToolRuntime.state` 是本次 Send payload（ToolBranchPayload），里面有
`plan_status` / `plan` / `plan_origin` —— 工具因此能做幂等判断。

⚠️ 幂等与说明文案都在工具里
------------------------
工具自己读 state、自己决定：

    当前 off                        -> planner，写入 plan / 清空旧 plan
    当前 plan，任务未完成           -> agent，不重置现有计划
    当前 plan，任务全部完成         -> planner，重新规划

ToolMessage 的正文也由工具根据当前状态生成（"已进入" / "当前已处于
计划模式，未重置" …），节点不再追加说明。

⚠️ tool_key 由扫描器推导
------------------------
本模块在 `TOOL_SOURCE_PACKAGES` 的 `app.agent.tools` 下，`source_ref`
推导为 `agent`，tool_key 是 `agent.<函数名>`。下面的常量直接取工具对象
的 `name`，避免手写字符串与函数名漂移。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

__all__ = [
    "ENTER_PLAN_TOOL_KEY",
    "EXIT_PLAN_TOOL_KEY",
    "PLAN_CONTROL_TOOL_KEYS",
    "enter_plan_mode",
    "exit_plan_mode",
]


def _state_of(runtime: Any) -> Mapping[str, Any]:
    """
    `ToolRuntime.state` -> 本次 Send payload。

    ⚠️ 它不是完整的父 state
    ----------------------
    `tools` 是 Send 分支，分支拿到的是 payload（见 ToolBranchPayload）。
    里面带着调用前的 `plan_status` / `plan` / `plan_origin` —— 正是这两个
    工具需要的全部信息。
    """

    state = getattr(runtime, "state", None)

    return state if isinstance(state, Mapping) else {}


def _needs_replan(plan: Any) -> bool:
    """
    任务是否已经全部完成（该重新规划）。

    函数内 import：`plan_state` 在模块级 import 了本文件的 tool_key
    常量，顶层互相 import 会成环。
    """

    from app.agent.graph.react.plan_state import plan_needs_replan

    return plan_needs_replan(plan)


@tool
async def enter_plan_mode(
        reason: str = "",
        runtime: ToolRuntime = None,  # type: ignore[assignment]
) -> Command:
    """
    进入计划模式：先把任务拆解成计划，再执行。

    用于任务存在多种实现方案、需求不清、或改动影响面较大（需要先确认
    步骤）的场合。**重复调用是幂等的**：已经在计划模式时不会重置当前
    计划；只有计划里的任务全部完成后才会重新规划。

    参数 reason 是可选的进入理由，会写进工具结果供排查。
    """

    state = _state_of(runtime)
    current = str(state.get("plan_status") or "off")
    origin = str(state.get("plan_origin") or "tool")

    if current == "plan":

        if _needs_replan(state.get("plan")):
            goto = "planner"
            message = "当前计划的任务已全部完成，将重新生成计划。"
        else:
            goto = "agent"
            message = "当前已处于计划模式；未重置现有计划。"

        return Command(
            goto=goto,
            update={
                "plan_transition": "enter",
                "message": message,
                "reason": reason,
            },
        )

    return Command(
        goto="planner",
        update={
            # 状态更新直接在工具里给 —— tools 节点只负责把它并进
            # Command.update 并补 ToolMessage。
            "plan_status": "plan",
            "plan_origin": origin,
            "plan": {},
            # 与 plan 一起清空，避免上一份计划的 A2A 任务 /
            # current_task 被当成当前计划。
            "tasks": [],
            "current_task": {},
            "task_tool_keys": [],
            # 新计划重新做一次「要不要开团队」的决策。
            "team_decision": "",
            # 上一份计划的 subagent 模式也必须清掉，否则新计划的第一轮
            # 会继续按 Redis 任务板 claim / 派发旧任务。
            "subagent_mode": "off",
            "subagent_workers": 0,
            "plan_transition": "enter",
            "message": "已进入计划模式。",
            "reason": reason,
        },
    )


@tool
async def exit_plan_mode(
        reason: str = "",
        runtime: ToolRuntime = None,  # type: ignore[assignment]
) -> Command:
    """
    退出计划模式，恢复完整执行能力。

    计划模式内完成规划、或用户明确要求直接执行时调用。**直接退出**，
    不需要额外审批；重复调用同样是幂等的。

    参数 reason 是可选的退出理由，会写进工具结果供排查。
    """

    state = _state_of(runtime)
    current = str(state.get("plan_status") or "off")
    origin = str(state.get("plan_origin") or "tool")

    if current == "plan":
        return Command(
            goto="agent",
            update={
                "plan_status": "off",
                "plan_origin": origin,
                # 退出计划后，残留的待决策 / 派发标记没有意义。
                "team_decision": "",
                "subagent_mode": "off",
                "subagent_workers": 0,
                "plan_transition": "exit",
                "message": "已退出计划模式。",
                "reason": reason,
            },
        )

    return Command(
        goto="agent",
        update={
            "plan_transition": "exit",
            "message": "当前不在计划模式；退出是幂等的。",
            "reason": reason,
        },
    )


#: 两个控制工具的 tool_key。
#:
#: 由工具对象自己的 `name` 派生 —— scanner 用的也是这个值，手写字符串
#: 会在函数改名时悄悄失配（模型看得到、节点认不出）。
ENTER_PLAN_TOOL_KEY = f"agent.{enter_plan_mode.name}"
EXIT_PLAN_TOOL_KEY = f"agent.{exit_plan_mode.name}"

#: 控制工具集合。tools 节点据此跳过"查处理器链"那一步（它们没有链），
#: 路由据此判断该不该按计划状态改道。
PLAN_CONTROL_TOOL_KEYS = frozenset({
    ENTER_PLAN_TOOL_KEY,
    EXIT_PLAN_TOOL_KEY,
})
