"""
TaskService：业务创建异步任务的唯一入口。

契约 -> 实现的对应没有登记表（理由与 repository/__init__.py 一样）：
工厂靠 resolve_impl 反射发现，因此实现必须在这里被 import 到。
"""

from .task_service import TaskService
from .task_service_impl import TaskServiceImpl

__all__ = [
    "TaskService",
    "TaskServiceImpl",
]
