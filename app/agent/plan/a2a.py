"""TaskDraft -> A2A Task 的同义转换。

这里**不改写** planner 的 `message`：整块 dict 进
`history[0].parts[0].data`。转换器只做结构补齐：

    local id          -> f"{plan_id}:{local_id}"
    message           -> Message(role=USER, parts=[data part])
    blocked_by        -> A2A task id 列表
    -                 -> blocked 反向边
    -                 -> status=SUBMITTED / timestamp / claim={}

⚠️ 不在这里做 I/O
----------------
不写 Redis、不写 graph state、不发事件。planner 节点先调用这里拿到
`Task` 对象，再序列化成 canonical dict；后续发布层单独接。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from a2a.helpers.proto_helpers import new_data_part
from a2a.types import Message, Role, Task, TaskState, TaskStatus
from a2a.utils.proto_utils import (
    normalize_large_integers_to_strings,
    validate_proto_required_fields,
)
from google.protobuf.json_format import MessageToDict

from .contracts import TaskDraft

__all__ = ["task_payload", "task_to_dict", "to_a2a_tasks"]


def to_a2a_tasks(
        tasks: Sequence[TaskDraft],
        *,
        context_id: str,
        plan_id: str,
        revision: int,
        source_run_id: str | None = None,
        existing: Mapping[str, Task] | None = None,
        now: datetime | None = None,
) -> list[Task]:
    """
    把**已校验、已拓扑排序**的 TaskDraft 列表转成 A2A Task。

    Parameters
    ----------
    tasks
        `validate_plan(...).tasks` 的顺序（依赖在前）。
    context_id
        逻辑会话 id；A2A 的 `contextId`。
    plan_id
        一次计划会话的稳定标识；与 local id 拼成 Task.id。
    revision
        计划版本；写进 metadata，不参与 id（重规划沿用同一 plan_id）。
    existing
        重规划时已有的 Task（按 A2A id 索引）。第一版只做“保留终态
        任务”的挂点；当前创建路径不依赖它。
    now
        创建时间；默认 UTC now。重试时调用方应传原任务的 timestamp。
    """

    if not plan_id:
        raise ValueError("plan_id 不能为空")

    if not context_id:
        raise ValueError("context_id 不能为空")

    local_ids = [task.id for task in tasks]

    if len(set(local_ids)) != len(local_ids):
        raise ValueError("tasks 里存在重复 local id")

    task_ids = {local: _task_id(plan_id, local) for local in local_ids}

    # 反向边：blocked_by 是入边，blocked 是出边。
    blocked: dict[str, list[str]] = {tid: [] for tid in task_ids.values()}

    for task in tasks:
        tid = task_ids[task.id]
        for dep in task.blocked_by:
            dep_id = task_ids.get(dep)

            if dep_id is None:
                raise ValueError(
                    f"任务 {task.id!r} 的 blocked_by 引用了未知 id {dep!r}"
                )

            blocked[dep_id].append(tid)

    created = now or datetime.now(timezone.utc)

    if created.tzinfo is None:
        raise ValueError("now 必须是带时区的 datetime")

    return [
        _one_task(
            task,
            task_id=task_ids[task.id],
            context_id=context_id,
            plan_id=plan_id,
            revision=revision,
            blocked_by=[task_ids[dep] for dep in task.blocked_by],
            blocked=blocked[task_ids[task.id]],
            source_run_id=source_run_id,
            created=created,
        )
        for task in tasks
    ]


def task_to_dict(task: Task) -> dict[str, Any]:
    """A2A Task -> canonical JSON dict（camelCase + 大写枚举）。"""

    return MessageToDict(task)


def task_payload(task: Task) -> Any:
    """取任务发起消息里的第一个 data part（planner 的 message 原样）。"""

    for message in task.history:
        for part in message.parts:
            if part.WhichOneof("content") == "data":
                return MessageToDict(part)["data"]

    raise ValueError(f"task {task.id!r} 没有 data part")


# ============================================================
# internals
# ============================================================


def _task_id(plan_id: str, local_id: str) -> str:
    return f"{plan_id}:{local_id}"


def _one_task(
        draft: TaskDraft,
        *,
        task_id: str,
        context_id: str,
        plan_id: str,
        revision: int,
        blocked_by: list[str],
        blocked: list[str],
        source_run_id: str | None,
        created: datetime,
) -> Task:
    message = normalize_large_integers_to_strings(draft.message)

    history = Message(
        message_id=f"{task_id}:message:1",
        task_id=task_id,
        context_id=context_id,
        role=Role.ROLE_USER,
        parts=[
            new_data_part(message, media_type="application/json"),
        ],
    )

    metadata: dict[str, Any] = {
        **draft.metadata,
        "skill_key": draft.skill_key,
        "plan_id": plan_id,
        "plan_revision": revision,
        "plan_task_id": draft.id,
        "blocked_by": blocked_by,
        "blocked": blocked,
        "block_policy": draft.block_policy,
        "capabilities": draft.capabilities,
        # 运行期状态：planner 无论写了什么，都以空 claim 创建。
        "claim": {},
    }

    if source_run_id:
        metadata["source_run_id"] = source_run_id

    task = Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(
            state=TaskState.TASK_STATE_SUBMITTED,
            timestamp=created,
        ),
        history=[history],
        metadata=metadata,
    )

    validate_proto_required_fields(task)

    return task
