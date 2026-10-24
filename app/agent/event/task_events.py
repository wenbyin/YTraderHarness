"""任务板事件的**生产者**（写 custom stream，不直接写 Redis）。

节点只写：

    get_stream_writer()({
        "kind": TASK_STREAM_KIND,
        "event": "plan.published" | "task.updated" | "plan.cleared",
        ...
    })

RunEventProjector 把它投影成 RunEventType.PLAN_PUBLISHED /
TASK_UPDATED / PLAN_CLEARED，再经 publish_run_event 进 Redis / SSE。

为什么走 custom stream
--------------------
与工具事件同一条纪律：节点不直接 publish_run_event，否则同一次运行会
出现两个事件来源（节点直发 + projector 投影），id / 顺序都会漂。
"""

from __future__ import annotations

from typing import Any

from app.common.event import TASK_STREAM_KIND, RunEventType

from .stream import write_custom

__all__ = [
    "publish_plan_cleared",
    "publish_plan_published",
    "publish_task_updated",
]


def publish_plan_published(
        *,
        plan_id: str,
        revision: int,
        goal: str,
        tasks: list[dict[str, Any]],
        current_task_id: str | None,
        publish_id: str | None = None,
        node: str = "planner",
) -> None:
    """planner 产出计划：前端用它一次性渲染 DAG。

    ⚠️ `current_task_id` 发布时应为 None：此时还没有任何任务被 claim，
    把它指到第一个任务会让前端把「排队中」画成「正在执行」。首次
    claim 后由 `task.updated`（顺序执行）或 `subagent.claimed`（并行）
    写 current。

    ⚠️ `publish_id` 是给订阅方的去重键
    ---------------------------------
    计划在 planner 子图的 task_generator 里发布，而 checkpoint 在
    super-step 之后提交：崩溃 / 恢复时同一份计划可能被重发。用
    `f"{plan_id}:{revision}"` 作为稳定键，消费方可以按它去重。
    """

    _write({
        "kind": TASK_STREAM_KIND,
        "event": RunEventType.PLAN_PUBLISHED.value,
        "node": node,
        "plan_id": plan_id,
        "revision": revision,
        "publish_id": publish_id,
        "goal": goal,
        "tasks": tasks,
        "current_task_id": current_task_id,
        "task_count": len(tasks),
    })


def publish_task_updated(
        *,
        plan_id: str,
        revision: int,
        task: dict[str, Any],
        current_task_id: str | None,
        node: str = "task_listener",
) -> None:
    """某个任务状态变化（claim / 完成）：前端按 task.id 合并。"""

    _write({
        "kind": TASK_STREAM_KIND,
        "event": RunEventType.TASK_UPDATED.value,
        "node": node,
        "plan_id": plan_id,
        "revision": revision,
        "task": task,
        "current_task_id": current_task_id,
    })


def publish_plan_cleared(
        *,
        plan_id: str,
        revision: int,
        reason: str,
        node: str = "task_listener",
) -> None:
    """任务列表清空：前端收起或标记任务面板。"""

    _write({
        "kind": TASK_STREAM_KIND,
        "event": RunEventType.PLAN_CLEARED.value,
        "node": node,
        "plan_id": plan_id,
        "revision": revision,
        "reason": reason,
    })


def _write(envelope: dict[str, Any]) -> None:
    """写 custom stream（实现见 event/stream.py，与子代理事件共用）。"""

    write_custom(envelope)
