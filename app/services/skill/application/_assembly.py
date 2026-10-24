"""
实体 -> 对外契约的组装。

纯函数、不碰数据库。集中在这里是为了让"Skill 在列表页与详情页
长什么样"只有一份定义 —— 两处各拼一次，字段迟早会漂移。
"""

from __future__ import annotations

from app.services.capability.domain.entities import (
    Capability as CapabilityEntity,
)

from ..domain.entities import Skill as SkillEntity
from ..domain.schemas import (
    CapabilityRef,
    SkillDetail,
    SkillSummary,
)

__all__ = [
    "build_capability_ref",
    "build_skill_detail",
    "build_skill_summary",
]


def build_skill_summary(
        entity: SkillEntity,
) -> SkillSummary:

    return SkillSummary(
        id=entity.skill_id,
        skill_key=entity.skill_key,
        name=entity.name,
        description=entity.description,
        tags=list(entity.tags),
        status=entity.status,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def build_skill_detail(
        entity: SkillEntity,
        capabilities: list[CapabilityEntity],
) -> SkillDetail:

    return SkillDetail(
        id=entity.skill_id,
        skill_key=entity.skill_key,
        name=entity.name,
        description=entity.description,
        tags=list(entity.tags),
        status=entity.status,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        examples=list(entity.examples),
        instructions=entity.instructions,
        capabilities=[
            build_capability_ref(c) for c in capabilities
        ],
    )


def build_capability_ref(
        entity: CapabilityEntity,
) -> CapabilityRef:

    return CapabilityRef(
        id=entity.capability_id,
        code=entity.code,
        name=entity.name,
    )
