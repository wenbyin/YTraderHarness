from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


class SkillCapability(Base):
    """
    Skill ↔ capability 的绑定。

    这张表**只决定工具可见性**，不是授权：

        skill_capability.capability_id
          -> capability.code
          -> tool_definition_classification
          -> tool_definition.tool_key（ACTIVE）

    双列主键：绑定本身没有身份，唯一的约束就是
    "同一个 (skill, capability) 只能出现一次"。

    删除语义刻意不对称
    ------------------
    skill 侧 CASCADE：删 skill 就该顺带解除绑定。
    capability 侧 RESTRICT：能力模块的删除检查只统计工具引用，
    这里若 CASCADE，force 删除一个能力会**无声**抹掉 skill 绑定，
    对应 skill 的工具可见性会悄悄缩小。RESTRICT 让这件事变成一次
    明确失败；capability 模块另有 count_skill_links 给出可读的 409。

    绑定按 capability.id，而不是 code —— 与
    tool_definition_classification 的方向一致。工具推导时才把 id
    换成 code。
    """

    __tablename__ = "skill_capability"

    skill_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("skill.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Skill ID。",
    )

    capability_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("capability.id", ondelete="RESTRICT"),
        primary_key=True,
        comment="能力 ID。",
    )

    __table_args__ = (
        Index(
            "idx_skill_capability_capability",
            "capability_id",
        ),
    )
