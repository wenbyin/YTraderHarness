"""
实体 -> 对外契约的组装。

纯函数、不碰数据库 —— 与其余模块的 _assembly.py 同一分工。

本模块存在的理由是**同一个 Task 在两个接口里必须长得一样**：
列表页返回 TaskSummary、详情页返回 TaskDetail，后者继承前者。若两处
各写一遍字段映射，迟早出现"列表里的 status 与详情里的 status 含义
不同"这种事，而那种不一致没有任何测试会抓到。
"""

from __future__ import annotations

from app.common.task.models import Task as TaskEntity

from ..domain.schemas import TaskDetail, TaskSummary

__all__ = [
    "build_task_detail",
    "build_task_summary",
]


def build_task_summary(
        task: TaskEntity,
) -> TaskSummary:

    return TaskSummary(
        task_id=task.task_id,
        handler=task.handler,
        status=task.status,
        attempt=task.attempt,
        max_attempts=task.max_attempts,
        generation=task.generation,
        execute_at=task.execute_at,
        parent_task_id=task.parent_task_id,
        root_task_id=task.root_task_id,
        idempotency_key=task.idempotency_key,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


def build_task_detail(
        task: TaskEntity,
) -> TaskDetail:
    """详情比列表多的那几项：params、causation_id、error、租约。

    params 用 params_dict（原始 dict 视图）而不是实体上那个可能还是
    RawTaskParams 的对象：它只是在库里"还没被解释过"的形态，直接吐给
    前端会多出一层没有意义的包装。
    """

    return TaskDetail(
        task_id=task.task_id,
        handler=task.handler,
        status=task.status,
        attempt=task.attempt,
        max_attempts=task.max_attempts,
        generation=task.generation,
        execute_at=task.execute_at,
        parent_task_id=task.parent_task_id,
        root_task_id=task.root_task_id,
        idempotency_key=task.idempotency_key,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        params=task.params_dict,
        causation_id=task.causation_id,
        error=task.error,
        lease_expires_at=task.lease_expires_at,
        leased_by=task.leased_by,
    )
