"""任务控制工具。

本模块提供工具名、参数与文档，让模型 / 调用方先能发现它们：

    create_task    创建一个任务（契约占位）
    claim_task     领取一个任务（契约占位）
    update_task    就地更新任务（契约占位）
    complete_task  标记任务完成并写结果（契约占位）
    create_team    把待完成任务发布到任务板（**已实现发布**）

⚠️ 只有 create_team 接了真实副作用
--------------------------------
它读 `ToolRuntime.state` 里的 `tasks`，判断有无待完成任务、检查依赖环，
然后把任务发布到 Redis 任务板。其余四个仍然返回显式的 `not_implemented`
—— 工具一旦静默返回成功，模型会以为任务真的创建/完成了；一个明确的
“还没接”比一个假成功安全。

`create_team` 审批通过后返回
`Command(goto="subagent_dispatch", update={"subagent_mode": "on", ...})`：
派发（预 claim + Send）在 `subagent_dispatch` 节点里，不在工具里 ——
审批拒绝时工具体根本不执行，也就不会切模式 / 起 worker。

后续接入其余工具时：
    - 数据源用 `app/agent/plan/listener.py` 的 `TaskSubscriber`
    - 写任务板走 `app/agent/plan/publish.py` 的 `TaskBoard`
    - 图内状态写回通过 `Command.update`，与计划工具同形

⚠️ 本模块在 `TOOL_SOURCE_PACKAGES` 的 `app.agent.tools` 下
--------------------------------------------------------
扫描器会把 tool_key 推导成 `agent.<函数名>`。下面的常量直接取工具
对象的 `name`，避免手写字符串与函数名漂移。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from app.agent.graph.react.catalog import SUBAGENT_DISPATCH_NODE
from app.agent.plan import build_task_board, publish_plan
from app.agent.plan.checks import (
    find_dependency_cycle,
    pending_tasks,
)

__all__ = [
    "CLAIM_TASK_TOOL_KEY",
    "COMPLETE_TASK_TOOL_KEY",
    "CREATE_TASK_TOOL_KEY",
    "CREATE_TEAM_TOOL_KEY",
    "TASK_TOOL_KEYS",
    "UPDATE_TASK_TOOL_KEY",
    "claim_task",
    "complete_task",
    "create_task",
    "create_team",
    "update_task",
]


def _not_implemented(
        tool_name: str,
        **echo: Any,
) -> dict[str, Any]:
    """统一的占位结果：明确告诉调用方“契约在、控制还没接”。"""

    return {
        "ok": False,
        "not_implemented": True,
        "tool": f"agent.{tool_name}",
        "message": f"{tool_name} 尚未实现；本工具当前只保留契约。",
        **echo,
    }


@tool
async def create_task(
        skill_key: str,
        description: str,
        params: dict[str, Any] | None = None,
        blocked_by: list[str] | None = None,
        block_policy: str = "all_succeeded",
        capabilities: list[str] | None = None,
) -> dict[str, Any]:
    """
    创建一个任务（契约占位；当前不会真的创建）。

    参数：
        skill_key      执行该任务的 skill 稳定标识
        description    自包含的任务说明
        params         机器可读参数（JSON object）
        blocked_by     依赖的任务 id 列表
        block_policy   入边策略；第一版只用 all_succeeded
        capabilities   planner 额外授予的能力 code 列表

    返回里会带 `not_implemented=true`，不会写入任何 state / Redis。
    """

    return _not_implemented(
        "create_task",
        skill_key=skill_key,
        description=description,
    )


@tool
async def claim_task(
        task_id: str,
        owner: str = "",
) -> dict[str, Any]:
    """
    领取一个任务（契约占位；当前不会真的 claim）。

    参数：
        task_id    要领取的任务 id
        owner      可选的领取者标识；空串 = 由运行时决定

    返回里会带 `not_implemented=true`，不会修改任务状态或 claim。
    """

    return _not_implemented(
        "claim_task",
        task_id=task_id,
        owner=owner,
    )


@tool
async def update_task(
        task_id: str,
        status: str | None = None,
        description: str | None = None,
        params: dict[str, Any] | None = None,
        blocked_by: list[str] | None = None,
        capabilities: list[str] | None = None,
) -> dict[str, Any]:
    """
    就地更新一个任务（契约占位；当前不会真的更新）。

    只传需要改的字段；未传的字段保持原值。`status` 用 A2A 状态字符串
    （如 TASK_STATE_WORKING / TASK_STATE_FAILED）。

    返回里会带 `not_implemented=true`，不会写入任何任务。
    """

    return _not_implemented(
        "update_task",
        task_id=task_id,
        status=status,
    )


@tool
async def complete_task(
        task_id: str,
        summary: str = "",
        result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    标记任务完成并写结果（契约占位；当前不会真的完成）。

    参数：
        task_id    要完成的任务 id
        summary    给父图和用户看的简短结果摘要
        result     结构化结果 / artifact 内容

    返回里会带 `not_implemented=true`，不会改状态、不会释放 claim。
    """

    return _not_implemented(
        "complete_task",
        task_id=task_id,
        summary=summary,
    )


@tool
async def create_team(
        worker_num: int,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
) -> dict[str, Any] | Command:
    """
    创建一个执行团队：把当前计划里**待完成**的任务发布到任务板，供
    worker 认领。

    参数：
        worker_num    worker 数量（也是最大并行数）。团队成员的技能由
                      各自认领的任务携带，不在这里指定。

    行为（顺序即判定顺序）：

        没有任务                    -> 跳过，`reason="no_tasks"`
        任务全部终态                -> 跳过，`reason="no_pending_tasks"`
        依赖存在环                  -> 拒绝，`reason="dependency_cycle"`
        任务板不可用 / 发布失败     -> `ok=false, degraded=true`
        正常                        -> 发布（幂等），返回 published / existing

    ⚠️ `runtime.state` 是 Send payload（`ToolBranchPayload`），不是完整
    ReactState。tasks / plan 由路由搬进 payload（见 `routing._tool_sends`）。
    这是临时方案：payload 会随 pending task 进 checkpoint。

    ⚠️ 本工具需要人工审批（由 tool processor chain 上的审批拦截器配置），
    不要在 `_CHAINLESS_TOOL_KEYS` 里给它开口子 —— 那会跳过整条链。

    TODO(subagent): 这里只发布任务板。切 `subagent_mode` / 派发 worker
    需要返回 `Command(goto="subagent_dispatch", update={...})`；等 dispatch
    节点落地后接上。
    """

    if (
            isinstance(worker_num, bool)
            or not isinstance(worker_num, int)
            or worker_num < 1
    ):
        return _tool_error(
            "create_team",
            reason="invalid_worker_num",
            message="worker_num 必须是 >= 1 的整数。",
            worker_num=worker_num,
        )

    state = _state_of(runtime)
    tasks = _task_list(state.get("tasks"))

    if not tasks:
        return _skipped(
            "no_tasks",
            worker_num=worker_num,
        )

    pending = pending_tasks(tasks)

    if not pending:
        return _skipped(
            "no_pending_tasks",
            worker_num=worker_num,
            task_count=len(tasks),
        )

    cycle = find_dependency_cycle(tasks)

    if cycle is not None:
        return _tool_error(
            "create_team",
            reason="dependency_cycle",
            message="任务依赖存在环，未发布到任务板。",
            worker_num=worker_num,
            cycle=cycle,
        )

    plan = state.get("plan")
    if not isinstance(plan, Mapping):
        plan = {}

    plan_id = str(plan.get("plan_id") or "").strip()

    if not plan_id:
        return _tool_error(
            "create_team",
            reason="missing_plan_id",
            message=(
                "当前 state 里没有 plan_id（不在计划模式？），"
                "未发布到任务板。"
            ),
            worker_num=worker_num,
        )

    context_id = _context_id(runtime, state, plan_id)
    result = await publish_plan(
        board=_board(runtime),
        context_id=context_id,
        plan_id=plan_id,
        revision=_revision(plan),
        tasks=tasks,
    )

    if result.degraded:
        return _tool_error(
            "create_team",
            reason="task_board_unavailable",
            message=result.reason,
            worker_num=worker_num,
            plan_id=plan_id,
            degraded=True,
        )

    # 真正会启动的 worker 数受待完成任务数限制；派发时用这个值，
    # 不要盲目按 worker_num 起空 worker。
    effective_workers = min(worker_num, len(pending))

    # 派发交给图节点，不在这里做：工具体只在**审批通过后**执行，
    # 而预 claim / Send 需要看到最新的任务板状态。返回 Command 之后，
    # tools 节点把 update 合进父状态并跳到 subagent_dispatch。
    #
    # ⚠️ update 里的 `message` 不是 state 通道：它只用于 tool result
    # 的正文（见 tools.py 的 _plan_message_text），LangGraph 会丢弃它。
    return Command(
        goto=SUBAGENT_DISPATCH_NODE,
        update={
            "subagent_mode": "on",
            "subagent_workers": effective_workers,
            "message": (
                f"已发布 {result.published} 个任务到任务板，"
                f"准备派发 {effective_workers} 个 worker"
                f"（复用已有 {result.skipped} 个）。"
            ),
        },
    )


# ============================================================
# create_team 的 internals
# ============================================================


def _state_of(runtime: Any) -> Mapping[str, Any]:
    """`ToolRuntime.state` -> 本次 Send payload；图外调用时是空 dict。"""

    state = getattr(runtime, "state", None)

    return state if isinstance(state, Mapping) else {}


def _task_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []

    return [
        dict(item)
        for item in raw
        if isinstance(item, Mapping)
    ]


def _revision(plan: Mapping[str, Any]) -> int:
    try:
        return int(plan.get("revision") or 0)
    except (TypeError, ValueError):
        return 0


def _context_id(
        runtime: Any,
        state: Mapping[str, Any],
        plan_id: str,
) -> str:
    """
    任务板索引键用的会话 id。

        state.conversation_id   图里显式给了就用它
        context.session_id      ReactContext 的业务会话
        plan_id                 最后兜底（至少不会把任务写进空键）
    """

    value = str(state.get("conversation_id") or "").strip()

    if value:
        return value

    context = getattr(runtime, "context", None)
    session_id = getattr(context, "session_id", None)

    if session_id is not None:
        return str(session_id)

    return plan_id


def _board(runtime: Any) -> Any:
    """优先用 ReactContext 注入的任务板；没有就按进程级 Redis 现造。"""

    context = getattr(runtime, "context", None)
    board = getattr(context, "task_board", None)

    return board if board is not None else build_task_board()


def _skipped(reason: str, **fields: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "skipped": True,
        "tool": "agent.create_team",
        "reason": reason,
        "message": "没有需要发布的任务，已跳过。",
        **fields,
    }


def _tool_error(
        tool_name: str,
        *,
        reason: str,
        message: str,
        **fields: Any,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": f"agent.{tool_name}",
        "reason": reason,
        "message": message,
        **fields,
    }


#: 本模块五个工具的 tool_key。scanner 推导的也是这个形状。
CREATE_TASK_TOOL_KEY = f"agent.{create_task.name}"
CLAIM_TASK_TOOL_KEY = f"agent.{claim_task.name}"
UPDATE_TASK_TOOL_KEY = f"agent.{update_task.name}"
COMPLETE_TASK_TOOL_KEY = f"agent.{complete_task.name}"
CREATE_TEAM_TOOL_KEY = f"agent.{create_team.name}"

#: 任务控制工具集合。后续把工具暴露给 planner / 主 agent 时按它取。
TASK_TOOL_KEYS = frozenset({
    CREATE_TASK_TOOL_KEY,
    CLAIM_TASK_TOOL_KEY,
    UPDATE_TASK_TOOL_KEY,
    COMPLETE_TASK_TOOL_KEY,
    CREATE_TEAM_TOOL_KEY,
})
