"""
Task 模块的持久化契约与实现。

契约 -> 实现的对应**没有任何登记表**：工厂靠 common/introspect.py 的
resolve_impl 反射发现（契约的子类，或同包同名的 X_impl.py）。

⚠️ 因此本文件必须把实现也 import 进来
------------------------------------
resolve_impl 的第一条路是"已经在内存里的具体子类"。只导出契约的话，
实现模块可能从未被 import 过，反射就找不到它，报的是
"找不到具体实现" —— 而原因其实是"没人 import 过"。仓库里其余模块的
repository/__init__.py 都同时导出两者，就是这个原因。
"""

from .task import TaskStore
from .task_impl import TaskStoreImpl
from .task_outbox import TaskOutboxStore
from .task_outbox_impl import TaskOutboxStoreImpl

__all__ = [
    "TaskOutboxStore",
    "TaskOutboxStoreImpl",
    "TaskStore",
    "TaskStoreImpl",
]
