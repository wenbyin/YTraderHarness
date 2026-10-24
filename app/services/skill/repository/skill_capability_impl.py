"""
PostgreSQL implementation of SkillCapabilityRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select

from app.services.capability.domain.entities import Capability
from app.services.capability.models.sqlalchemy.capability import (
    Capability as CapabilityModel,
)

from ..models.sqlalchemy.skill_capability import (
    SkillCapability as LinkModel,
)
from .skill_capability import SkillCapabilityRepository


class SkillCapabilityRepositoryImpl(
    SkillCapabilityRepository
):

    async def list_for_skill(
            self,
            skill_id: UUID,
    ) -> list[Capability]:

        # 一次 join 取回全部能力。另一侧是 capability 表（跨模块），
        # 详情页要的不只是 id —— code / name / 排序都在那张表里。
        result = await self._session.execute(
            select(CapabilityModel)
            .join(
                LinkModel,
                LinkModel.capability_id
                == CapabilityModel.capability_id,
            )
            .where(LinkModel.skill_id == skill_id)
            .order_by(
                CapabilityModel.sort_order,
                CapabilityModel.code,
            )
        )

        return [
            Capability.model_validate(model)
            for model in result.scalars()
        ]

    async def replace_for_skill(
            self,
            skill_id: UUID,
            capability_ids: list[UUID],
    ) -> None:

        await self._session.execute(
            delete(LinkModel).where(
                LinkModel.skill_id == skill_id
            )
        )

        # 去重：调用方可能传了重复 id，而 (skill_id, capability_id)
        # 是主键，重复插入会直接违反约束。
        for capability_id in dict.fromkeys(capability_ids):
            self._session.add(
                LinkModel(
                    skill_id=skill_id,
                    capability_id=capability_id,
                )
            )

        await self._session.flush()
