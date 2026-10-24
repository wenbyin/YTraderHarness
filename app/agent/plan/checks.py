"""任务图的公共判定：待完成任务、依赖环。

这两个判定同时服务三处：

    task_listener   `has_unfinished` 决定「还有没有活干」
    create_team     发布前再判一次；没有待完成任务就跳过
    planner         产出 A2A task 之后再判一次依赖环（防御性）

⚠️ 为什么单独一个模块
--------------------
`task_listener` 是图节点，它的核心判定（状态集合 / ready / 环）都是
**纯函数**。把这些纯函数放在节点模块或工具模块里，会逼着另一个调用方
导入一个带 LangGraph 依赖的模块；放这里之后三处共享同一份规则，
「什么算待完成」「什么算环」只有一处定义。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from .listener import (
    TERMINAL,
    blocked_by,
    is_ready,
    status_state,
    task_id,
)

__all__ = [
    "find_dependency_cycle",
    "has_parallel_ready",
    "has_pending_tasks",
    "pending_tasks",
    "ready_tasks",
]


def pending_tasks(
        tasks: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """还没进终态的任务（顺序保持输入顺序）。"""

    return [
        task for task in tasks
        if status_state(task) not in TERMINAL
    ]


def has_pending_tasks(
        tasks: Sequence[Mapping[str, Any]],
) -> bool:
    """还有没有待完成任务。与 task_listener 的判定同源。"""

    return bool(pending_tasks(tasks))


def ready_tasks(
        tasks: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """
    当前可以开始的全部任务（保持输入顺序）。

    复用 `listener.is_ready`：SUBMITTED、没有 claim、依赖都已 COMPLETED。
    与 `select_ready` 的区别是这里返回**全部** ready，而不是第一个。
    """

    by_id = {
        task_id(task): task
        for task in tasks
        if task_id(task)
    }

    return [
        task for task in tasks
        if is_ready(task, by_id=by_id)
    ]


def has_parallel_ready(
        tasks: Sequence[Mapping[str, Any]],
) -> bool:
    """
    当前是否有可并行执行的流程（至少两个 ready 任务）。

    这是「要不要开子代理团队」的判据：只有一个 ready 任务时，并行没有
    收益，直接顺序执行；有两个及以上，才值得让 main agent 做一次决策。
    """

    return len(ready_tasks(tasks)) >= 2


def find_dependency_cycle(
        tasks: Sequence[Mapping[str, Any]],
) -> list[str] | None:
    """
    在 A2A canonical task 列表里找一条依赖环。

    依赖关系读 `metadata.blocked_by`（A2A task id 列表）。指向列表外
    的 id 不算环（那是悬空依赖，由 ready 判定把它卡住）；自依赖按环处理。

    返回环上的节点序列（首尾相同，例如 `["t1", "t2", "t1"]`）；
    没有环返回 None。

    为什么不用 `contracts.validate_plan` 的 Kahn
    ----------------------------------------
    那里校验的是 planner 的 **TaskDraft**（local id），而这里校验的是
    转换后的 **A2A canonical dict**（id 形如 `<plan_id>:t1`）。两者
    时机与形状都不同；用同一个算法、两份输入，比把 draft 的校验结果
    当成本轮任务的保证更可靠（任务可能在 state 里被后续节点改过）。
    """

    items: dict[str, Mapping[str, Any]] = {}

    for task in tasks:

        tid = task_id(task)

        if tid:
            items[tid] = task

    if not items:
        return None

    indegree: dict[str, int] = {tid: 0 for tid in items}
    dependents: dict[str, list[str]] = {tid: [] for tid in items}

    for tid, task in items.items():

        for dep in blocked_by(task):

            if dep == tid:
                return [tid, tid]

            if dep not in items:
                # 悬空依赖不是环；它会让 is_ready 一直 False，
                # 由 listener 的死锁判定处理。
                continue

            indegree[tid] += 1
            dependents[dep].append(tid)

    queue = deque(
        tid for tid, degree in indegree.items() if degree == 0
    )
    ordered: list[str] = []

    while queue:

        tid = queue.popleft()
        ordered.append(tid)

        for child in dependents[tid]:

            indegree[child] -= 1

            if indegree[child] == 0:
                queue.append(child)

    if len(ordered) == len(items):
        return None

    remaining = set(items) - set(ordered)

    # 沿着 blocked_by 走一条真实环，而不是把剩余集合整个返回。
    start = next(iter(remaining))
    path: list[str] = []
    current = start

    while current not in path:

        path.append(current)

        next_deps = [
            dep for dep in blocked_by(items[current])
            if dep in remaining
        ]

        if not next_deps:
            break

        current = next_deps[0]

    if current in path:
        index = path.index(current)
        return [*path[index:], current]

    # 理论上不会走到这里（Kahn 剩余节点里一定有环）。兜底返回成员，
    # 至少让调用方能拒绝发布，而不是静默放行。
    return sorted(remaining)
