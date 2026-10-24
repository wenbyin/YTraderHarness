"""子代理事件的**生产者**（写 custom stream，不直接写 Redis）。

四条事件，scope 都是「某个 worker 的某个任务」：

    subagent.created        一个 worker 实例创建（父图预 claim 成功、
                            Send 之前；每 worker 一条）
    subagent.claimed        父图把一个已 claim 的任务交给该 worker
    subagent.task.updated   子任务状态变化（WORKING -> COMPLETED/FAILED）
    subagent.completed      某个 worker 执行完一个任务

⚠️ 为什么由**父图节点**发，而不是 worker 子图里发
---------------------------------------------
worker 现在是编译进父图的**子图节点**，它的 custom 事件确实会进父运行
的事件流（这点与合并前不同）。即便如此，`subagent.*` 仍由
`subagent_dispatch` 统一发，理由变成「**单一来源**」而不是「收不到」：

    - claim / completed 的权威判断要对比**任务板**与上一轮 state，
      只有父图的 dispatch 同时握有这两样；
    - 子图在完成时写任务板，不再自己发一条 task 事件 —— 否则同一状态
      会出现两个生产者（子图 + 父图），顺序与 id 都会漂。

因此事件的生产者是：

    subagent_dispatch     subagent.created / claimed（每波预 claim 时）
                          subagent.task.updated / completed
                          （收波时对比上一轮 state 与任务板快照，补发
                            worker 在子图里完成的那一步）

worker 子图内部只写**观测行**（node_span；recorder 是共享的）与工具
事件（tool.* 会经父 projector 投影），不写 subagent.*。
"""

from __future__ import annotations

from typing import Any

from app.common.event import SUBAGENT_STREAM_KIND, RunEventType

from .stream import write_custom

__all__ = [
    "publish_subagent_claimed",
    "publish_subagent_completed",
    "publish_subagent_created",
    "publish_subagent_task_updated",
]


def publish_subagent_created(
        *,
        plan_id: str,
        revision: int,
        worker_id: str,
        worker_index: int,
        task_id: str,
        task_count: int = 0,
        pending_count: int = 0,
        context_id: str = "",
        node: str = "subagent_dispatch",
) -> None:
    """
    一个 worker 实例创建（父图预 claim 成功、即将 Send 出去）。

    它是**每 worker 一条**（不是每团队一条）：前端据此先开一条泳道
    （`worker_id` 稳定），随后的 `subagent.claimed` 再往里放任务。
    一次派发 N 个 worker 就有 N 条 created + N 条 claimed。

    `task_count` / `pending_count` 是团队维度的上下文，重复出现是为了
    让前端在只收到某一条时也能画出进度（不必依赖先收到别的帧）。
    """

    write_custom({
        "kind": SUBAGENT_STREAM_KIND,
        "event": RunEventType.SUBAGENT_CREATED.value,
        "node": node,
        "plan_id": plan_id,
        "revision": revision,
        "worker_id": worker_id,
        "worker_index": worker_index,
        "task_id": task_id,
        "task_count": task_count,
        "pending_count": pending_count,
        "context_id": context_id,
    })


def publish_subagent_claimed(
        *,
        plan_id: str,
        revision: int,
        worker_id: str,
        task: dict[str, Any],
        current_task_id: str | None = None,
        node: str = "subagent_dispatch",
) -> None:
    """预 claim：`task` 是 claim 之后的快照（含 owner / attempt / lease）。"""

    write_custom({
        "kind": SUBAGENT_STREAM_KIND,
        "event": RunEventType.SUBAGENT_CLAIMED.value,
        "node": node,
        "plan_id": plan_id,
        "revision": revision,
        "worker_id": worker_id,
        "task": task,
        "task_id": str(task.get("id") or ""),
        "current_task_id": current_task_id,
    })


def publish_subagent_task_updated(
        *,
        plan_id: str,
        revision: int,
        worker_id: str,
        task: dict[str, Any],
        previous_state: str = "",
        current_task_id: str | None = None,
        node: str = "subagent_dispatch",
) -> None:
    """子任务状态变化。与 `task.updated` 同形，多 `worker_id` / `previous_state`。"""

    write_custom({
        "kind": SUBAGENT_STREAM_KIND,
        "event": RunEventType.SUBAGENT_TASK_UPDATED.value,
        "node": node,
        "plan_id": plan_id,
        "revision": revision,
        "worker_id": worker_id,
        "task": task,
        "task_id": str(task.get("id") or ""),
        "previous_state": previous_state,
        "current_task_id": current_task_id,
    })


def publish_subagent_completed(
        *,
        plan_id: str,
        revision: int,
        worker_id: str,
        task: dict[str, Any],
        node: str = "subagent_dispatch",
) -> None:
    """
    某个 worker 执行完一个任务。

    `status` 取 A2A 状态（COMPLETED / FAILED / ...）；失败也算完成 ——
    前端需要把这条泳道收掉，而不是让它永远转圈。
    """

    status = task.get("status")
    state = (
        str(status.get("state") or "")
        if isinstance(status, dict)
        else str(status or "")
    )

    write_custom({
        "kind": SUBAGENT_STREAM_KIND,
        "event": RunEventType.SUBAGENT_COMPLETED.value,
        "node": node,
        "plan_id": plan_id,
        "revision": revision,
        "worker_id": worker_id,
        "task": task,
        "task_id": str(task.get("id") or ""),
        "status": state,
    })
