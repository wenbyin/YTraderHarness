"""
Capability 模块的持久化契约与实现。

契约 -> 实现的对应**没有任何登记表**：
工厂靠 common/introspect.py 的 resolve_impl 反射发现
（契约的子类，或同包同名的 X_impl.py）。
"""

from .capability import CapabilityRepository
from .capability_impl import CapabilityRepositoryImpl

__all__ = [
    "CapabilityRepository",
    "CapabilityRepositoryImpl",
]
