"""Task Listener 节点：完成上一个任务、领取下一个任务。

顺序（见 catalog / graph.py）：

    planner -> task_listener -> agent -> (有 tasks) -> task_listener -> ...

它负责**所有** task 列表判断：

    current_task WORKING 且路由回到这里 -> 上一个任务已完成，落终态
    没有可执行任务                    -> 清空 tasks / current_task
    有 >=2 个 ready 任务且还没问过     -> 不 claim，写 team_decision=pending，
                                         回 main agent 决定要不要开团队
    还有未完成任务                    -> claim 下一个，交给 agent

「问过」是一次性的：main agent 那一轮结束后如果 team_decision 仍是
pending（没有调用 create_team），这里会写 declined 并继续 claim ——
否则 listener -> agent -> listener 会无限空转。

真正的“循环监听”由**图上的回边**表达，节点里不要 `while True`：
LangGraph 节点是一个 super-step，阻塞循环会让父图的 fan-out/join 永远
等不到它，也挡着 checkpoint。

数据源
------
    state  source   主图默认；读 state["tasks"]
    Redis  source   子 agent / worker 模式；读任务板，Lua CAS claim

选择逻辑在 `app/agent/plan/listener.py`，两种 source 只是实现不同。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.runtime import Runtime

from app.agent.event.task_events import (
    publish_plan_cleared,
    publish_task_updated,
)
from app.agent.plan.listener import (
    TERMINAL,
    WORKING,
    TaskListener,
    build_task_subscriber,
    complete_task,
    has_unfinished,
    status_state,
    task_position,
)
from app.agent.plan.checks import ready_tasks
from app.agent.plan.task_prompt import (
    last_ai_text,
    task_instruction_message,
)
from app.agent.tools.task_tool import CREATE_TEAM_TOOL_KEY
from app.infra.graph import node_span

from .catalog import NODE_SPECS
from .context import ReactContext
from .state import ReactState

__all__ = ["task_listener_node"]

logger = logging.getLogger(__name__)

_SPEC = next(s for s in NODE_SPECS if s.node_id == "task_listener")

#: task history 里保留的 agent 摘要上限。任务完成后父图只需要摘要，
#: 不需要把一整段最终回答复制进任务 JSON。
_SUMMARY_LIMIT = 4000


async def task_listener_node(
        state: ReactState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """完成已执行的 current_task，再 claim 下一个 ready task。"""

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "task_count": len(state.get("tasks") or ()),
                "current_task": (
                    (state.get("current_task") or {}).get("id")
                ),
            },
    ) as span:

        context = runtime.context
        plan = state.get("plan") or {}
        context_id = _context_id(state, context)
        tasks = _tasks(state)
        current = state.get("current_task") or {}

        completed = False
        completed_task: dict[str, Any] | None = None
        full_result = last_ai_text(
            state.get("messages") or (), limit=None,
        )
        summary = full_result[:_SUMMARY_LIMIT]

        # 上一个任务由 agent 执行完（router_after_agent 只在没有
        # tool_calls 时回到这里）—— 落 COMPLETED，清 current_task。
        if isinstance(current, Mapping) and current:

            state_name = status_state(current)

            if state_name in WORKING:
                finished_id = str(current.get("id") or "")
                tasks = complete_task(
                    tasks,
                    task_id_value=finished_id,
                    summary=summary,
                    # 最终成果进 artifacts；summary 只做预览。
                    result_text=full_result,
                    context_id=context_id,
                )
                completed_task = next(
                    (
                        dict(task) for task in tasks
                        if str(task.get("id") or "") == finished_id
                    ),
                    None,
                )
                plan = _plan_mark_done(plan, finished_id)
                current = {}
                completed = True

            elif state_name in TERMINAL:
                current = {}

        plan_id, revision = _plan_ref(plan)
        plan_key = f"{plan_id}:{revision}" if plan_id else ""
        hist = list(state.get("hist_synthesis_keys") or ())

        # 全部终态：先判断是不是要进入“任务汇总”回合。
        #
        # 判定只有两条：
        #   1. task_list 全部终态；
        #   2. plan_key 不在 hist_synthesis_keys。
        # 不对比 pre_state —— listener 是并行 / 顺序两条路的汇合点，
        # 它只需要一个持久闩，不需要知道“这一拍是不是刚完成”。
        if not has_unfinished(tasks):

            if completed_task is not None:
                publish_task_updated(
                    plan_id=plan_id,
                    revision=revision,
                    task=completed_task,
                    current_task_id=None,
                )

            plan = _plan_mark_all_done(plan, tasks)

            if plan_key and tasks and plan_key not in hist:

                hist.append(plan_key)

                span.set_output_summary(
                    {
                        "completed": completed,
                        "claimed": None,
                        "remaining": 0,
                        "synthesis": "started",
                    }
                )
                return {
                    "current_task": {},
                    # 保留结果供 main agent 汇总；不要在这里清空。
                    "tasks": tasks,
                    "task_tool_keys": [],
                    "plan": plan,
                    "team_decision": "",
                    "subagent_mode": "off",
                    "subagent_workers": 0,
                    "hist_synthesis_keys": hist,
                    "synthesis_pending": True,
                    # 汇总回合不需要再开团队 / claim 任务。
                    "context_policy": _synthesis_policy(
                        state.get("context_policy")
                    ),
                }

            if state.get("synthesis_pending") and tasks:

                # 汇总回合中再次进入 listener：保持结果，继续交给 agent。
                # （正常情况下 agent 节点在汇总回答后会清空 tasks，不会
                # 再走到这里；这是 resume / 重入的兜底。）
                span.set_output_summary(
                    {
                        "completed": completed,
                        "claimed": None,
                        "remaining": 0,
                        "synthesis": "pending",
                    }
                )
                return {
                    "current_task": {},
                    "tasks": tasks,
                    "task_tool_keys": [],
                    "plan": plan,
                    "team_decision": "",
                    "subagent_mode": "off",
                    "subagent_workers": 0,
                    "hist_synthesis_keys": hist,
                    "synthesis_pending": True,
                    "context_policy": _synthesis_policy(
                        state.get("context_policy")
                    ),
                }

            publish_plan_cleared(
                plan_id=plan_id,
                revision=revision,
                reason=("completed" if tasks else "empty"),
            )

            span.set_output_summary(
                {
                    "completed": completed,
                    "claimed": None,
                    "remaining": 0,
                    "synthesis": "cleared",
                }
            )
            return {
                "current_task": {},
                "tasks": [],
                "task_tool_keys": [],
                "plan": plan,
                # 计划已结束，下一份计划重新决策；同时退出 subagent 模式，
                # 否则下一次普通对话会继续按 Redis 任务板 claim。
                "team_decision": "",
                "subagent_mode": "off",
                "subagent_workers": 0,
                "hist_synthesis_keys": hist,
                "synthesis_pending": False,
            }

        # subagent 模式：派发由 subagent_dispatch 负责，主 listener **不**
        # claim。否则它会抢走留给下一波 worker 的任务，或者把 WORKING
        # 的任务当成「主 agent 可以顺序执行」的活。
        #
        # 到达这里说明 dispatch 已经确认「当前没有 ready task」：
        #   - 还有 WORKING（外部 worker / 一次没走完的波次）-> 记 note 结束
        #   - 依赖死锁（无 ready、无 running）-> 记 note 结束，下一轮重规划
        # 两种情况都不 spin，也不清掉任务板（板才是权威）。
        if state.get("subagent_mode") == "on":

            remaining = sum(
                1 for task in tasks if status_state(task) not in TERMINAL
            )

            plan = _plan_note(
                plan,
                f"subagent 模式：本轮没有可派发的 ready task，"
                f"仍有 {remaining} 个未终态任务；结束本轮等待下一次调度。",
            )

            if completed_task is not None:
                publish_task_updated(
                    plan_id=plan_id,
                    revision=revision,
                    task=completed_task,
                    current_task_id=None,
                )

            publish_plan_cleared(
                plan_id=plan_id,
                revision=revision,
                reason="subagent_waiting",
            )

            span.set_output_summary({
                "completed": completed,
                "claimed": None,
                "remaining": remaining,
                "exit": "subagent_waiting",
            })

            return {
                "current_task": {},
                "tasks": [],
                "task_tool_keys": [],
                "plan": plan,
                "team_decision": "",
                "subagent_mode": "off",
                "subagent_workers": 0,
            }

        # 有并行 ready 任务时，先让 main agent 做一次「要不要开团队」的决策，
        # 本轮不 claim。
        #
        # 这是一次性询问：listener 写 pending 并回 agent；agent 那一轮结束后
        # 回到这里时，如果仍然是 pending（说明没有调用 create_team），就写
        # declined 并继续 claim —— 否则 listener -> agent -> listener 会
        # 无限空转。
        decision = str(state.get("team_decision") or "")
        ready = ready_tasks(tasks)
        declined = decision == "pending"

        if (
                decision not in {"pending", "declined"}
                and state.get("subagent_mode") != "on"
                and len(ready) >= 2
                # 没暴露 create_team 的 agent 不要被问一次注定会拒绝的
                # 决策：直接顺序执行。
                and _can_create_team(state)
        ):
            # 决策提示由 ContextBuilder 的 TASK_RESOLVER 注入（见
            # task_resolver.py）：listener 只负责把 current_task 清空、
            # 打上 pending 标记，然后回 main agent。

            # 如果这一趟刚完成上一个任务，完成态与事件必须先落下；
            # 决策回合只是「不 claim」，不是「不回写已完成的事」。
            if completed_task is not None:
                publish_task_updated(
                    plan_id=plan_id,
                    revision=revision,
                    task=completed_task,
                    current_task_id=None,
                )

            span.set_output_summary({
                "completed": completed,
                "claimed": None,
                "deferred": True,
                "ready": len(ready),
                "decision": "pending",
            })

            update: dict[str, Any] = {
                # planner 可能预写了 current_task；决策回合必须清掉，
                # 否则 ContextBuilder 会按任务模式构建上下文。
                "current_task": {},
                "task_tool_keys": [],
                "team_decision": "pending",
                "plan": plan,
            }

            if completed:
                update["tasks"] = tasks

            return update

        # 已经问过 main agent 但没调用 create_team：顺序执行。
        # 若它调用了 create_team，这里仍然是顺序 claim（worker 派发尚未
        # 实现），但至少不会重复询问同一份计划。
        decision_update = (
            {"team_decision": "declined"} if declined else {}
        )

        plan_id = str(plan.get("plan_id") or "").strip() or None

        subscriber = build_task_subscriber(
            {
                "tasks": tasks,
                "current_task": current,
                "subagent_mode": state.get("subagent_mode"),
            },
            context_id=context_id,
            plan_id=plan_id,
        )

        owner = f"{context.pipeline_run_id}:main"
        claimed = await TaskListener(subscriber).claim_next(owner=owner)

        if claimed is None:
            # 还有未完成任务，但没有 ready（依赖失败 / 全部被认领）。
            # 不能让 router 在 listener 和 agent 之间空转；结束本轮并
            # 把原因写进 plan.notes，保留可见性。
            plan = _plan_note(
                plan,
                "没有可执行的 ready task；本轮任务列表已清空。",
            )

            if completed_task is not None:
                publish_task_updated(
                    plan_id=plan_id,
                    revision=revision,
                    task=completed_task,
                    current_task_id=None,
                )

            publish_plan_cleared(
                plan_id=plan_id,
                revision=revision,
                reason="no_ready_task",
            )

            span.set_output_summary(
                {
                    "completed": completed,
                    "claimed": None,
                    "remaining": len(tasks),
                    "deadlock": True,
                }
            )
            return {
                "current_task": {},
                "tasks": [],
                "task_tool_keys": [],
                "plan": plan,
                # 任务列表已清空，下一份计划重新决策。
                "team_decision": "",
            }

        # 任务指令要和 ContextBuilder 的 required 消息同一份文本：
        # 负责人、任务列表进度都在里面。
        current_tasks = (
            claimed.tasks if claimed.tasks is not None else tasks
        )
        position, total = task_position(claimed.task, current_tasks)

        # 新任务先清掉上一轮的工具 key；本轮 ContextBuilder 的
        # SkillLoader 会重新算并写回。
        update: dict[str, Any] = {
            "current_task": claimed.task,
            "task_tool_keys": [],
            "plan": plan,
            # pending -> declined：main agent 已经有过决策回合但没开团队。
            **decision_update,
            # 任务指令写进 messages：任务面板 / 会话历史需要它，且
            # agent_message 落库时按 kind=task_instruction 识别。
            # id 稳定，同一 task 重跑不会堆多份。
            "messages": [
                task_instruction_message(
                    claimed.task,
                    owner=owner,
                    position=position,
                    total=total,
                )
            ],
        }

        if claimed.tasks is not None:
            update["tasks"] = claimed.tasks
        elif completed:
            update["tasks"] = tasks

        # 完成事件先于新 claim 事件，前端看到“旧任务完成 -> 新任务开始”。
        if completed_task is not None:
            publish_task_updated(
                plan_id=plan_id,
                revision=revision,
                task=completed_task,
                current_task_id=str(claimed.task.get("id") or "") or None,
            )

        publish_task_updated(
            plan_id=plan_id,
            revision=revision,
            task=claimed.task,
            current_task_id=str(claimed.task.get("id") or "") or None,
        )

        span.set_output_summary(
            {
                "completed": completed,
                "source": claimed.source,
                "claimed": claimed.task.get("id"),
                "remaining": len(update.get("tasks") or tasks),
                "attempt": (
                    (claimed.task.get("metadata") or {}).get("claim") or {}
                ).get("attempt"),
            }
        )

        return update


# ============================================================
# state helpers
# ============================================================


def _can_create_team(state: ReactState) -> bool:
    """
    本次运行暴露给 main agent 的工具里有没有 create_team。

    没有的话不做并行决策：那只会多一次注定被拒绝的 agent 回合。
    context_policy 是 builder 写的；缺它 / 形状不对一律按「没有」处理。
    """

    policy = state.get("context_policy")

    if not isinstance(policy, Mapping):
        return False

    keys = policy.get("tool_keys")

    if not isinstance(keys, (list, tuple)):
        return False

    return CREATE_TEAM_TOOL_KEY in {
        str(key) for key in keys
    }


def _tasks(state: ReactState) -> list[dict[str, Any]]:
    raw = state.get("tasks")

    if not isinstance(raw, (list, tuple)):
        return []

    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _plan_mark_done(
        plan: Mapping[str, Any],
        task_id: str,
) -> dict[str, Any]:
    """把 plan.tasks[].status 里对应的任务标成 done。"""

    if not plan:
        return {}

    tasks = plan.get("tasks")

    if not isinstance(tasks, (list, tuple)):
        return dict(plan)

    updated: list[Any] = []

    for task in tasks:

        item = task

        if isinstance(task, Mapping) and str(task.get("id") or "") == task_id:
            item = {**dict(task), "status": "done"}

        updated.append(item)

    return {**dict(plan), "tasks": updated}


def _plan_mark_all_done(
        plan: Mapping[str, Any],
        tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """把任务列表里所有终态任务在 plan 快照里标成 done。"""

    if not plan:
        return {}

    finished = {
        str(task.get("id") or "")
        for task in tasks
        if task.get("id") and status_state(task) in TERMINAL
    }

    if not finished:
        return dict(plan)

    result = dict(plan)

    for task_id in finished:
        result = _plan_mark_done(result, task_id)

    return result


def _synthesis_policy(raw: Any) -> dict[str, Any]:
    """汇总回合的 policy：保留提示词，但不再暴露工具。"""

    policy = dict(raw) if isinstance(raw, Mapping) else {}

    policy["tool_keys"] = []
    policy["include_capabilities"] = False

    return policy


def _plan_ref(plan: Mapping[str, Any]) -> tuple[str, int]:
    """事件信封用的 plan_id / revision。"""

    plan_id = str(plan.get("plan_id") or "").strip()

    try:
        revision = int(plan.get("revision") or 0)
    except (TypeError, ValueError):
        revision = 0

    return plan_id, revision


def _plan_note(
        plan: Mapping[str, Any],
        note: str,
) -> dict[str, Any]:
    result = dict(plan or {})
    notes = result.get("notes")

    if not isinstance(notes, list):
        notes = []

    result["notes"] = [*notes, note]

    return result


def _context_id(state: ReactState, context: ReactContext) -> str:
    value = str(state.get("conversation_id") or "").strip()

    if value:
        return value

    if context.session_id is not None:
        return str(context.session_id)

    return str(context.pipeline_run_id)
