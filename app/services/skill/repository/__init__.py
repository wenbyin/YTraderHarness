"""
Skill 模块的持久化契约与实现。

契约 -> 实现的对应**没有任何登记表**：工厂靠 common/introspect.py
的 resolve_impl 反射发现（契约的子类，或同包同名的 X_impl.py）。
"""

from .skill import SkillRepository
from .skill_capability import SkillCapabilityRepository
from .skill_capability_impl import (
    SkillCapabilityRepositoryImpl,
)
from .skill_impl import SkillRepositoryImpl

__all__ = [
    "SkillCapabilityRepository",
    "SkillCapabilityRepositoryImpl",
    "SkillRepository",
    "SkillRepositoryImpl",
]
