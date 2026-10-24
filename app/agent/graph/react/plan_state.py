"""
计划模式的状态派生：**纯函数**，不碰数据库、不建图。

三个状态通道的权威定义在 `state.ReactState`：

    plan_status   "off" | "plan" —— 当前是否处于计划模式
    plan          最近一次 planner 产出的计划快照（含 tasks[].status）
    plan_origin   谁把模式推进来的（"classifier" / "tool"）

本模块只做两件事：

    1. `plan_needs_replan`  该不该（重新）生成计划（没任务 / 任务全完成）
    2. `plan_tool_keys`     哪些控制工具要暴露给模型

⚠️ 状态切换发生在 tools 节点
--------------------------
`enter_plan_mode` / `exit_plan_mode` 是**真工具**：它们返回
`Command(goto=..., update={...})`，`tools` 节点据此写 `plan_status` /
`plan_origin` / `plan`，并把本批第一个调用的 Command 交回 LangGraph
决定下一跳。控制器选择与工具类型无关 —— 任何工具返回 Command 都能
控制下一跳。
工具函数本身不写 state（它拿不到父状态），但跳转逻辑就在它的
`Command.goto` 里。

分类器触发的那次进入由 builder 合成一条同样的工具调用，走同一个
`tools` 节点 —— 没有第二条状态写入路径。

⚠️ 为什么判据在 state 上而不是在路由里现算
--------------------------------------
`route_after_builder` 与 `tools._plan_goto` 需要**同一个**判据。
各写一份的话，两条路在"任务全部完成"的边界上早晚给出不同答案，
而症状是"有时重新规划、有时不"—— 看起来像模型抽风。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.tools.hitl_tool import (
    ASK_USER_QUESTION_TOOL_KEY,
)
from app.agent.tools.plan_tool import (
    ENTER_PLAN_TOOL_KEY,
    EXIT_PLAN_TOOL_KEY,
)

__all__ = [
    "DONE_TASK_STATUSES",
    "PLAN_ACTIVE",
    "PLAN_OFF",
    "plan_needs_replan",
    "plan_tool_keys",
]


#: 计划模式的取值。**只存普通字符串**，不进枚举 —— 理由与
#: `agent_state` 模块开头那段相同：state 会被 JSON 序列化。
PLAN_OFF = "off"
PLAN_ACTIVE = "plan"


#: 算作"这个任务已完成"的状态值。
#:
#: 任务状态由任务工具 / 调度器写，形状尚未定死，因此这里收一个**集合**
#: 而不是单个值：`done` 是主值，`completed` 是常见的别名。认不出的一律
#: 算**未完成** —— 这是保守的一侧（宁可多跑一轮 planner，也不把没做完
#: 的任务当成做完）。
DONE_TASK_STATUSES = frozenset({"done", "completed"})


def _tasks(plan: Any) -> list[Mapping[str, Any]]:
    """从 plan 快照里取出任务列表；形状不对一律当空。"""

    if not isinstance(plan, Mapping):
        return []

    raw = plan.get("tasks")

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []

    return [item for item in raw if isinstance(item, Mapping)]


def plan_needs_replan(plan: Any) -> bool:
    """
    这次进入 planner 是不是"该（重新）生成计划"。

    两种情形都算：

        没有任务        第一次进入 / planner 还没产出 / 旧计划已被清空
        任务全部完成    上一轮计划做完了，EnterPlanTool 要求生成新任务

    任何任务未完成（pending / running / failed …）都返回 False ——
    计划模式里的正常流程是**继续执行**，不是每轮重新规划。
    """

    tasks = _tasks(plan)

    if not tasks:
        return True

    return all(
        str(task.get("status") or "").strip().lower()
        in DONE_TASK_STATUSES
        for task in tasks
    )


def plan_tool_keys(status: str | None = None) -> list[str]:
    """
    要暴露给模型的计划相关工具。

    **三个始终都暴露**，与当前模式无关：

        enter_plan_mode        请求进入计划模式
        exit_plan_mode         请求退出计划模式
        ask_user_question      规划/执行中缺信息时向用户提问（HITL）

    理由：builder 只在运行开头算一次工具集，而工具执行发生在运行中途。
    如果按模式动态收窄，模型在同一个 run 里进入计划模式后就看不到 exit
    / ask（要等下一轮用户输入才刷新）。三个都常驻、由工具自身保证幂等
    （off 时 exit 是无操作，plan 时 enter 不会重置计划），比维护一份
    运行中途会过期的工具清单简单。

    这些 key 会被并进 `context_policy.tool_keys` —— 与能力工具走
    **同一个来源**，这样"给模型看什么"与"允许执行什么"仍然恒等。
    """

    return [
        ENTER_PLAN_TOOL_KEY,
        EXIT_PLAN_TOOL_KEY,
        ASK_USER_QUESTION_TOOL_KEY,
    ]
