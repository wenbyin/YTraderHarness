"""
统一的异步 Task 抽象（Async Task Infrastructure）。

系统中的所有后台异步执行都必须收敛到这里，不再为 HITL / Run / Memory /
Retry 分别实现异步机制。数据流：

    业务 Service
        -> 构造 Pydantic Task
        -> TaskService.submit()
        -> tasks + task_outbox（同一事务）
        -> TaskDispatcher
        -> Worker
        -> TaskHandlerRegistry
        -> ServiceFactory
        -> Service.method(params)

业务代码只依赖 TaskService 抽象，**不感知** Redis / MQ / Worker /
调度器 —— 那些都在 app/infra/task 下，本包一行都不 import。

本包是**纯叶子**
--------------
只依赖 pydantic 与 app.common.exceptions。不 import app.services.*，
也不 import app.infra.* —— 见 app/common/exceptions.py 里同样的约定。
"""

from .exceptions import (
    TaskError,
    TaskParamsError,
    TaskPermanentError,
    TaskStateError,
    UnknownHandlerError,
)
from .models import RawTaskParams, Task, TaskContext
from .protocols import (
    HandlerDefinition,
    TaskDispatcher,
    TaskHandlerRegistry,
    TaskSubmitter,
)
from .types import TaskStatus

__all__ = [
    "HandlerDefinition",
    "RawTaskParams",
    "Task",
    "TaskContext",
    "TaskDispatcher",
    "TaskError",
    "TaskHandlerRegistry",
    "TaskParamsError",
    "TaskPermanentError",
    "TaskStateError",
    "TaskStatus",
    "TaskSubmitter",
    "UnknownHandlerError",
]
