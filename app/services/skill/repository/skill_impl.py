"""
PostgreSQL implementation of SkillRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, or_, select

from ..domain.entities import Skill
from ..domain.schemas.enum import SkillStatus
from ..models.sqlalchemy.skill import Skill as SkillModel
from ..models.sqlalchemy.skill_capability import (
    SkillCapability as SkillCapabilityModel,
)
from .skill import SkillRepository


class SkillRepositoryImpl(SkillRepository):

    async def add(
            self,
            skill: Skill,
    ) -> Skill:

        model = SkillModel(
            skill_id=skill.skill_id,
            skill_key=skill.skill_key,
            name=skill.name,
            description=skill.description,
            tags=list(skill.tags),
            examples=list(skill.examples),
            instructions=skill.instructions,
            status=skill.status.value,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    async def update(
            self,
            skill: Skill,
    ) -> Skill:

        model = await self._session.get(
            SkillModel,
            skill.skill_id,
        )

        if model is None:
            raise ValueError(
                f"Skill 不存在：{skill.skill_id}"
            )

        # skill_key 刻意不改：后续表按它引用本行。
        model.name = skill.name
        model.description = skill.description
        model.tags = list(skill.tags)
        model.examples = list(skill.examples)
        model.instructions = skill.instructions
        model.status = skill.status.value
        model.updated_at = skill.updated_at

        await self._session.flush()

        return self._to_entity(model)

    async def delete(
            self,
            skill_id: UUID,
    ) -> None:

        await self._session.execute(
            delete(SkillModel).where(
                SkillModel.skill_id == skill_id
            )
        )

        await self._session.flush()

    async def get(
            self,
            skill_id: UUID,
    ) -> Skill | None:

        model = await self._session.get(SkillModel, skill_id)

        return None if model is None else self._to_entity(model)

    async def get_by_key(
            self,
            skill_key: str,
    ) -> Skill | None:

        result = await self._session.execute(
            select(SkillModel).where(
                SkillModel.skill_key == skill_key
            )
        )

        model = result.scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def list_paged(
            self,
            *,
            status: SkillStatus | None = None,
            keyword: str | None = None,
            capability_id: UUID | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[Skill], int]:

        conditions = []

        if status is not None:
            conditions.append(
                SkillModel.status == status.value
            )

        if keyword:
            pattern = f"%{keyword}%"
            conditions.append(
                or_(
                    SkillModel.skill_key.ilike(pattern),
                    SkillModel.name.ilike(pattern),
                    SkillModel.description.ilike(pattern),
                )
            )

        if capability_id is not None:
            # EXISTS 而不是 join：一个 skill 绑多个能力时，
            # join 会让同一行出现多次，分页和 total 都会错。
            conditions.append(
                select(SkillCapabilityModel.skill_id)
                .where(
                    SkillCapabilityModel.skill_id
                    == SkillModel.skill_id,
                    SkillCapabilityModel.capability_id
                    == capability_id,
                )
                .exists()
            )

        count_statement = select(
            func.count()
        ).select_from(SkillModel)

        page_statement = select(SkillModel)

        if conditions:
            count_statement = count_statement.where(*conditions)
            page_statement = page_statement.where(*conditions)

        total = (
            await self._session.execute(count_statement)
        ).scalar_one()

        result = await self._session.execute(
            page_statement.order_by(
                SkillModel.skill_key,
            ).offset(offset).limit(limit)
        )

        return (
            [self._to_entity(m) for m in result.scalars()],
            total,
        )

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: SkillModel,
    ) -> Skill:

        return Skill.model_validate(model)
