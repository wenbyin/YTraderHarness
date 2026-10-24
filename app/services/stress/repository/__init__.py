"""
Stress 模块的持久化契约与实现。

契约 -> 实现的对应没有登记表：工厂靠 common/introspect.py 的
resolve_impl 反射发现。因此本文件必须把实现也 import 进来，否则
「已经在内存里的子类」那条路找不到它。
"""

from .stress_run import StressRunStore
from .stress_run_impl import StressRunStoreImpl

__all__ = [
    "StressRunStore",
    "StressRunStoreImpl",
]
