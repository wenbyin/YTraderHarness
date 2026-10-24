"""子代理派发节点：预 claim + 动态 Send 扇出。

它做什么
--------
`create_team` 审批通过后返回：

    Command(goto="subagent_dispatch",
            update={"subagent_mode": "on", "subagent_workers": N})

本节点：

    1. 从任务板读本 plan 的 ready task
    2. 按并发上限 **预 claim**（CAS）最多 N 个，owner 按 worker 编号
    3. 用 `Command(goto=[Send("subagent_worker", payload), ...])` 扇出
    4. 没有 ready task -> 回 task_listener 收尾

worker 执行完、静态边回到本节点后会重新调度下一波。这条边是**每个
Send 分支结束时触发一次**（实测：多个分支共用一个静态出边，目标节点
在所有分支结束后执行一次），所以这里必须重新读板，不能缓存上一波。

⚠️ 为什么预 claim 而不是让 worker 自己抢
------------------------------------
两个 worker 同时抢板会互相踩（各自看到同一批 ready）；而且 Send 的
payload 取代 state，worker 看不到父图的 tasks。父图先 CAS 好、再把
已 claim 的任务放进 payload，谁跑哪个任务是确定的。

⚠️ 为什么「没有任务板」就退回顺序执行
----------------------------------
state 源没有跨分支的权威 claim：两个 worker 各自拿一份副本会重复执行。
因此 `build_task_subscriber` 退回 StateTaskSubscriber 时**不派发**，
把 `subagent_mode` 关掉并回 task_listener，走主 agent 顺序执行
（design_planner/task.md §7.6 / §7.11 的 fail-closed）。

⚠️ 任务板读取失败 -> 抛，不静默派发
----------------------------------
板读不到时既不能假装「没有 ready task」（会漏任务），也不能假装
「派发成功」。这里抛 RuntimeError，让运行 failed 并留下明确原因。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from app.agent.event.subagent_events import (
    publish_subagent_claimed,
    publish_subagent_completed,
    publish_subagent_created,
    publish_subagent_task_updated,
)
from app.agent.plan.listener import (
    SUBMITTED,
    TERMINAL,
    WORKING,
    ClaimedTask,
    StateTaskSubscriber,
    TaskListener,
    build_task_subscriber,
    status_state,
)
from app.infra.graph import node_span

from .catalog import (
    NODE_SPECS,
    SUBAGENT_WORKER_NODE,
    TASK_LISTENER_NODE,
)
from .context import ReactContext
from .state import ReactState

__all__ = ["subagent_dispatch_node"]

logger = logging.getLogger(__name__)

#: 本节点只读 ReactState 的一部分通道。
ReadingState = Mapping[str, Any]

_SPEC = next(
    spec for spec in NODE_SPECS if spec.node_id == "subagent_dispatch"
)

#: worker 的上下文基线。
#:
#: **不继承**父 agent 的 `system_prompt` / `tool_keys`：那会把主对话的
#: 职责提示词与主 agent 的工具带进子任务。TaskResolver / SkillLoader
#: 会在子图里按任务重算这两项；这里显式关掉 MVP 阶段不用的画像 / 记忆 /
#: 知识，只保留任务执行真正需要的。
_WORKER_POLICY: Mapping[str, Any] = {
    "include_history": True,
    "include_summary": True,
    "include_capabilities": True,
    "include_user_profile": False,
    "include_memory": False,
    "include_retrieved_context": False,
    "allow_compaction": True,
}

#: 可以从父策略继承的键（预算 / 历史窗口类，不含身份与工具）。
_INHERITED_POLICY_KEYS: tuple[str, ...] = (
    "max_recent_turns",
    "min_recent_turns",
    "max_history_tokens",
    "max_input_tokens",
)


async def subagent_dispatch_node(
        state: ReactState,
        runtime: Runtime[ReactContext],
) -> Command:
    """
    预 claim 一批 ready task，扇出给 worker；没有可派发的就回 listener。

    Returns
    -------
    Command
        有任务：`goto=[Send(...)]` + 同步后的 tasks；
        没任务：`goto="task_listener"`。
    """

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "tasks": len(_tasks(state)),
                "worker_limit": _worker_limit(state),
                "subagent_mode": state.get("subagent_mode"),
            },
    ) as span:

        context = runtime.context
        tasks = _tasks(state)
        plan = state.get("plan")
        plan_id = _plan_id(plan)
        context_id = _context_id(state, context)
        limit = _worker_limit(state)

        subscriber = build_task_subscriber(
            {
                "tasks": tasks,
                "current_task": {},
                "subagent_mode": "on",
            },
            context_id=context_id,
            plan_id=plan_id,
        )

        # 没有任务板 = 没有跨 worker 的 claim 权威：退回顺序执行。
        if isinstance(subscriber, StateTaskSubscriber):
            logger.warning(
                "subagent_dispatch：任务板不可用，退回主 agent 顺序执行"
                "（不会启动 worker）。"
            )
            span.set_output_summary({
                "dispatched": 0,
                "fallback": "state_source",
            })
            return Command(
                goto=TASK_LISTENER_NODE,
                update={
                    "current_task": {},
                    "tasks": tasks,
                    "subagent_mode": "off",
                    "subagent_workers": 0,
                },
            )

        listener = TaskListener(subscriber)

        try:
            snapshot = await listener.snapshot()

        except Exception as exc:  # noqa: BLE001 - 见模块 docstring
            raise RuntimeError(
                "subagent_dispatch：读取任务板失败，拒绝派发 worker。"
                "worker 模式下任务板是调度权威，读不到就不该猜。"
            ) from exc

        if snapshot is None:
            raise RuntimeError(
                "subagent_dispatch：任务板返回空快照，拒绝派发 worker。"
            )

        owner_root = str(context.pipeline_run_id)
        claimed: list[ClaimedTask] = []

        # 一次 claim 一个：Redis 的 Lua CAS 是逐 task 的。每轮会重新读板，
        # 所以后一轮看到的是前一轮已 WORKING 的状态。
        for index in range(limit):

            item = await listener.claim_next(
                owner=f"{owner_root}:worker:{index}"
            )

            if item is None:
                break

            claimed.append(item)

        if not claimed:
            # 没有 ready task：把最新快照带回父图，由 listener 收尾
            # （全部终态 -> 清空；仍有未完成 -> 记录原因并结束本轮）。
            try:
                snapshot = await listener.snapshot() or snapshot
            except Exception:  # noqa: BLE001 - 上面已经确认板可读
                pass

            # worker 在子图里完成的任务，父图是在**这一波回来之后**才
            # 看到板上的终态的：对比上一轮 state 与当前快照，补发
            # subagent.task.updated / completed。
            _emit_subagent_progress(
                tasks,
                snapshot,
                plan_id=plan_id or "",
                revision=_revision(plan),
            )

            span.set_output_summary({
                "dispatched": 0,
                "remaining": len(snapshot),
            })
            return Command(
                goto=TASK_LISTENER_NODE,
                update={
                    "current_task": {},
                    "tasks": list(snapshot),
                },
            )

        # 最后一次 claim 的快照包含前面几次 claim 的 WORKING 状态。
        latest = claimed[-1].tasks

        if latest is not None:
            snapshot = latest

        # 上一波 worker 的完成态（WORKING -> COMPLETED）先发。
        _emit_subagent_progress(
            tasks,
            snapshot,
            plan_id=plan_id or "",
            revision=_revision(plan),
        )

        # 本波预 claim：每个 worker 一条 created（开泳道）+ 一条 claimed
        # （往泳道里放任务）。两条都带稳定的 worker_id。
        revision = _revision(plan)
        pending_count = _pending_count(snapshot)

        for index, item in enumerate(claimed):

            worker_id = f"{owner_root}:worker:{index}"
            task_id = str(item.task.get("id") or "")

            publish_subagent_created(
                plan_id=plan_id or "",
                revision=revision,
                worker_id=worker_id,
                worker_index=index,
                task_id=task_id,
                task_count=len(snapshot),
                pending_count=pending_count,
                context_id=context_id,
            )

            publish_subagent_claimed(
                plan_id=plan_id or "",
                revision=revision,
                worker_id=worker_id,
                task=item.task,
                current_task_id=task_id or None,
            )

        sends = [
            Send(
                SUBAGENT_WORKER_NODE,
                _worker_payload(
                    state,
                    context,
                    index=index,
                    claimed=item,
                    tasks=snapshot,
                ),
            )
            for index, item in enumerate(claimed)
        ]

        span.set_output_summary({
            "dispatched": len(sends),
            "claimed": [
                str(item.task.get("id") or "") for item in claimed
            ],
            "remaining": len(snapshot),
        })

        return Command(
            goto=sends,
            update={
                "current_task": {},
                "tasks": list(snapshot),
                "subagent_mode": "on",
                "subagent_workers": limit,
            },
        )


# ============================================================
# internals
# ============================================================


def _emit_subagent_progress(
        previous: list[dict[str, Any]],
        current: list[dict[str, Any]],
        *,
        plan_id: str,
        revision: int,
) -> None:
    """
    对比上一轮 state 快照与任务板最新快照，补发子任务状态变化。

    为什么在**父图**补发：worker 是独立 `ainvoke` 的子图，它的 custom
    writer 不是父运行那条流。worker 完成时只会写任务板与子图 checkpoint；
    父图在这一波 join 之后才看得到终态，因此由这里统一发事件。

    ⚠️ SUBMITTED -> WORKING 不在这里发
    --------------------------------
    那是 claim 的语义，由 `publish_subagent_claimed` 发一条更完整的
    事件（带 task 快照与 worker_id）。两处都发会让前端收到重复的
    「任务开始」。

    终态（COMPLETED / FAILED / CANCELED / REJECTED）额外发一条
    `subagent.completed`，让泳道收口。
    """

    previous_by_id = {
        str(task.get("id") or ""): task
        for task in previous
        if str(task.get("id") or "")
    }

    for task in current:

        tid = str(task.get("id") or "")

        if not tid:
            continue

        before = previous_by_id.get(tid)

        if before is None:
            continue

        previous_state = status_state(before)
        current_state = status_state(task)

        if previous_state == current_state:
            continue

        if previous_state in SUBMITTED and current_state in WORKING:
            continue

        # claim 在完成时会被清空，worker_id 要从**上一轮**的 claim 里取。
        worker_id = _claim_owner(before) or _claim_owner(task)

        publish_subagent_task_updated(
            plan_id=plan_id,
            revision=revision,
            worker_id=worker_id,
            task=task,
            previous_state=previous_state,
        )

        if current_state in TERMINAL:
            publish_subagent_completed(
                plan_id=plan_id,
                revision=revision,
                worker_id=worker_id,
                task=task,
            )


def _pending_count(tasks: list[dict[str, Any]]) -> int:
    """还没到终态的任务数（created 事件里的团队进度）。"""

    return sum(1 for task in tasks if status_state(task) not in TERMINAL)


def _claim_owner(task: Mapping[str, Any]) -> str:
    metadata = task.get("metadata")

    if not isinstance(metadata, Mapping):
        return ""

    claim = metadata.get("claim")

    if not isinstance(claim, Mapping):
        return ""

    return str(claim.get("owner") or "")


def _revision(plan: Any) -> int:
    if not isinstance(plan, Mapping):
        return 0

    try:
        return int(plan.get("revision") or 0)
    except (TypeError, ValueError):
        return 0


def _worker_payload(
        state: ReadingState,
        context: ReactContext,
        *,
        index: int,
        claimed: ClaimedTask,
        tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    一个 worker 子图的初始 state。

    这是一个**新的子图 run**：messages 从空开始，任务以
    `pending_task` 交给 listener 收养。不要把父图 messages 复制进来
    （design_planner/task.md §7.12.2）。
    """

    plan = state.get("plan")

    return {
        "messages": [],
        "pending_task": claimed.task,
        "tasks": list(tasks),
        "plan": dict(plan) if isinstance(plan, Mapping) else {},
        "conversation_id": _context_id(state, context),
        "user_id": str(state.get("user_id") or ""),
        "worker_id": f"{context.pipeline_run_id}:worker:{index}",
        # 一任务一 worker；下一批由本节点重新扇出。
        "single_task": True,
        "context_policy": _worker_policy(state),
    }


def _worker_policy(state: ReadingState) -> dict[str, Any]:
    """worker 的上下文策略：基线 + 少量预算覆盖（见 _WORKER_POLICY）。"""

    policy = dict(_WORKER_POLICY)
    parent = state.get("context_policy")

    if isinstance(parent, Mapping):

        for key in _INHERITED_POLICY_KEYS:

            if key in parent:
                policy[key] = parent[key]

    return policy


def _worker_limit(state: ReadingState) -> int:
    """一次扇出的 worker 上限。至少 1，最多不设硬上限（由 ready 数约束）。"""

    try:
        value = int(state.get("subagent_workers") or 1)
    except (TypeError, ValueError):
        value = 1

    return max(1, value)


def _tasks(state: ReadingState) -> list[dict[str, Any]]:
    raw = state.get("tasks")

    if not isinstance(raw, (list, tuple)):
        return []

    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _plan_id(plan: Any) -> str | None:
    if not isinstance(plan, Mapping):
        return None

    value = str(plan.get("plan_id") or "").strip()

    return value or None


def _context_id(state: ReadingState, context: ReactContext) -> str:
    """任务板索引键 / 记忆 scope 用的会话 id（与 listener 同一条取法）。"""

    value = str(state.get("conversation_id") or "").strip()

    if value:
        return value

    if context.session_id is not None:
        return str(context.session_id)

    return str(context.pipeline_run_id)
