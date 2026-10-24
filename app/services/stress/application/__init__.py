"""
StressRunService：压测运行的创建、查询与取消。

契约 -> 实现的对应没有登记表（理由与 repository/__init__.py 一样）：
工厂靠 resolve_impl 反射发现，因此实现必须在这里被 import 到。
"""

from .stress_run_service import StressRunService
from .stress_run_service_impl import StressRunServiceImpl

__all__ = [
    "StressRunService",
    "StressRunServiceImpl",
]
