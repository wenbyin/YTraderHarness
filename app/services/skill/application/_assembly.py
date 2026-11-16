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
        # 只读容量度量。不来自请求，直接取实体上服务层算好的值。
        instructions_tokens=entity.instructions_tokens,
        prompt_tokens=entity.prompt_tokens,
        token_encoding=entity.token_encoding,
        token_computed_at=entity.token_computed_at,
    )


def build_skill_detail(
        entity: SkillEntity,
        capabilities: list[CapabilityEntity],
) -> SkillDetail:

    # 从摘要展开，而不是再抄一遍 12 个字段 —— 容量字段以后再加时，
    # 列表页与详情页不会一个有一个没有。
    return SkillDetail(
        **build_skill_summary(entity).model_dump(),
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
