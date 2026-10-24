"""
Skill 模块的对外契约。

domain/schemas/
├── enum.py   SkillStatus
└── skill.py  SkillCreate / SkillUpdate / SkillSummary / SkillDetail /
              PagedSkillResponse / SkillCapabilityUpdate / CapabilityRef

全部结构逐字段对应 skill.yaml 的 components/schemas；字段名与枚举
取值都以那份 spec 为准。
"""

from .enum import SkillStatus
from .skill import (
    CapabilityRef,
    PagedSkillResponse,
    SkillCapabilityUpdate,
    SkillCreate,
    SkillDetail,
    SkillSummary,
    SkillUpdate,
)

__all__ = [
    "CapabilityRef",
    "PagedSkillResponse",
    "SkillCapabilityUpdate",
    "SkillCreate",
    "SkillDetail",
    "SkillStatus",
    "SkillSummary",
    "SkillUpdate",
]
