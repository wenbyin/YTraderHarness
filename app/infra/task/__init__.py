"""
Task Infrastructure 的**运行时**：注册表、执行者、投递器、投递保证。

依赖方向（spec §13）：

    app/common/task   <-   app/services/task   <-   app/infra/task

本包只认 app/common/task 里的端口（TaskStore / TaskDispatcher /
TaskHandlerRegistry）与 app/services/task 里的实现，**不 import 任何
业务模块**：

    Task Core  ->  HITL / Run / Memory      ✗ 禁止

业务 Handler 由组合根（app/main.py）在启动时通过
TaskHandlerRegistry.register(...) 挂进来。谁都可以 import 谁的那个地方，
只有组合根 —— 因此不存在循环依赖，也不必让 TaskService 认识所有业务服务。

四个组件与它们的分工：

    registry.py    handler 名字 -> 调谁、传什么（显式登记，不做动态 import）
    worker.py      执行**一条**任务（事务边界、重试判定、结果写回）
    dispatcher.py  投递 + 兜底扫描（哪条该跑）
    outbox.py      投递保证（哪条从没被叫过）
    retry.py       退避与"该不该再试"
"""

from .dispatcher import DbPollingTaskDispatcher
from .outbox import OutboxRelay
from .registry import TaskHandlerRegistryImpl
from .retry import RetryPolicy
from .worker import DEFAULT_LEASE_SECONDS, TaskRunOutcome, Worker

__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "DbPollingTaskDispatcher",
    "OutboxRelay",
    "RetryPolicy",
    "TaskHandlerRegistryImpl",
    "TaskRunOutcome",
    "Worker",
]
