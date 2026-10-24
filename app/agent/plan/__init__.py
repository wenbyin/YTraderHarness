"""Planner 任务草案、校验与 A2A 转换。

这一层做四件事：

    contracts.py   planner 输出的 TaskDraft / PlanDraft + DAG 校验
    a2a.py         TaskDraft -> A2A Task（以及 canonical dict）
    listener.py    state / Redis 两种任务源的 claim 规则
    publish.py     把 A2A task 发布到 Redis 任务板（执行态权威）

它不读写 graph state，也不调模型；planner 节点负责把模型输出交给这里。
"""

from .a2a import task_payload, task_to_dict, to_a2a_tasks
from .checks import (
    find_dependency_cycle,
    has_parallel_ready,
    has_pending_tasks,
    pending_tasks,
    ready_tasks,
)
from .contracts import (
    DEFAULT_BLOCK_POLICY,
    DEFAULT_SKILL_KEY,
    PlanDraft,
    PlanValidationError,
    TaskDraft,
    ValidatedPlan,
    normalize_payload,
    validate_plan,
)
from .listener import (
    ClaimedTask,
    RedisTaskSubscriber,
    StateTaskSubscriber,
    TaskListener,
    TaskSubscriber,
    build_task_subscriber,
)
from .publish import (
    PublishResult,
    RedisTaskBoard,
    TaskBoard,
    build_task_board,
    publish_plan,
)

__all__ = [
    "DEFAULT_BLOCK_POLICY",
    "DEFAULT_SKILL_KEY",
    "PlanDraft",
    "PlanValidationError",
    "ClaimedTask",
    "PublishResult",
    "RedisTaskBoard",
    "RedisTaskSubscriber",
    "StateTaskSubscriber",
    "TaskBoard",
    "TaskDraft",
    "TaskListener",
    "TaskSubscriber",
    "ValidatedPlan",
    "build_task_board",
    "build_task_subscriber",
    "find_dependency_cycle",
    "has_parallel_ready",
    "has_pending_tasks",
    "normalize_payload",
    "pending_tasks",
    "publish_plan",
    "ready_tasks",
    "task_payload",
    "task_to_dict",
    "to_a2a_tasks",
    "validate_plan",
]
