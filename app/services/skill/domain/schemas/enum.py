"""
Skill 模块的枚举。

取值与 skill.yaml、models.sql 严格一致。枚举只有这一处定义；
models/sqlalchemy 按 String 列存 `.value`，不并列定义第二份。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "SkillStatus",
]


class SkillStatus(str, Enum):
    """
    Skill 的生命周期状态。

    DRAFT
        编写中。主 agent 的 skill-list 不包含它，也不允许 spawn。

    ACTIVE
        可用。主 agent 的 skill-list 只列这一种。

    DISABLED
        退役。保留行以维持历史引用（后续 task board / run 记录会 ref
        本表），但不再被选中。常规退役用这个，不要物理删除。
    """

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
