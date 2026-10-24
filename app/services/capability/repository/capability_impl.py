"""
PostgreSQL implementation of CapabilityRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, or_, select

from app.services.skill.models.sqlalchemy.skill_capability import (
    SkillCapability as SkillLinkModel,
)
from app.services.tool.models.sqlalchemy.tool_classification import (
    ToolDefinitionClassification as LinkModel,
)

from ..domain.entities import Capability
from ..models.sqlalchemy.capability import Capability as CapabilityModel
from .capability import CapabilityRepository


class CapabilityRepositoryImpl(CapabilityRepository):

    # ========================================================
    # Taxonomy
    # ========================================================

    async def add(
            self,
            capability: Capability,
    ) -> Capability:

        model = CapabilityModel(
            capability_id=capability.capability_id,
            code=capability.code,
            name=capability.name,
            description=capability.description,
            parent_id=capability.parent_id,
            sort_order=capability.sort_order,
            is_active=capability.is_active,
            created_at=capability.created_at,
            updated_at=capability.updated_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    async def update(
            self,
            capability: Capability,
    ) -> Capability:

        model = await self._session.get(
            CapabilityModel,
            capability.capability_id,
        )

        if model is None:
            raise ValueError(
                "能力不存在："
                f"{capability.capability_id}"
            )

        # code 刻意不改：它是稳定标识，关联与外部引用都按它。
        model.name = capability.name
        model.description = capability.description
        model.parent_id = capability.parent_id
        model.sort_order = capability.sort_order
        model.is_active = capability.is_active
        model.updated_at = capability.updated_at

        await self._session.flush()

        return self._to_entity(model)

    async def delete(
            self,
            capability_id: UUID,
    ) -> None:

        # 批量 DELETE 而不是 session.delete(model)：ORM 级联要求先把
        # definition_links 集合加载出来，而它是惰性关系 —— 在 async
        # session 上触发懒加载会抛 MissingGreenlet。关联行交给数据库的
        # ON DELETE CASCADE，一条 SQL 收尾，也没有隐式 IO。
        await self._session.execute(
            delete(CapabilityModel).where(
                CapabilityModel.capability_id == capability_id
            )
        )

        await self._session.flush()

    async def count_children(
            self,
            capability_id: UUID,
    ) -> int:

        result = await self._session.execute(
            select(func.count())
            .select_from(CapabilityModel)
            .where(
                CapabilityModel.parent_id == capability_id
            )
        )

        return result.scalar_one()

    async def get(
            self,
            capability_id: UUID,
    ) -> Capability | None:

        model = await self._session.get(
            CapabilityModel, capability_id
        )

        return None if model is None else self._to_entity(model)

    async def get_by_code(
            self,
            code: str,
    ) -> Capability | None:

        result = await self._session.execute(
            select(CapabilityModel).where(
                CapabilityModel.code == code
            )
        )

        model = result.scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def list(
            self,
            *,
            parent_id: UUID | None = None,
            keyword: str | None = None,
            is_active: bool | None = None,
    ) -> list[Capability]:

        statement = select(CapabilityModel)
        conditions = []

        if parent_id is not None:
            conditions.append(
                CapabilityModel.parent_id == parent_id
            )

        if keyword:
            pattern = f"%{keyword}%"
            conditions.append(
                or_(
                    CapabilityModel.code.ilike(pattern),
                    CapabilityModel.name.ilike(pattern),
                )
            )

        if is_active is not None:
            conditions.append(
                CapabilityModel.is_active == is_active
            )

        if conditions:
            statement = statement.where(*conditions)

        result = await self._session.execute(
            statement.order_by(
                CapabilityModel.sort_order,
                CapabilityModel.code,
            )
        )

        return [self._to_entity(m) for m in result.scalars()]

    # ========================================================
    # 反查引用
    # ========================================================

    async def count_links(
            self,
            capability_ids: list[UUID],
    ) -> int:

        if not capability_ids:
            return 0

        result = await self._session.execute(
            select(func.count())
            .select_from(LinkModel)
            .where(
                LinkModel.classification_id.in_(
                    capability_ids
                )
            )
        )

        return result.scalar_one()

    async def count_skill_links(
            self,
            capability_ids: list[UUID],
    ) -> int:

        if not capability_ids:
            return 0

        result = await self._session.execute(
            select(func.count())
            .select_from(SkillLinkModel)
            .where(
                SkillLinkModel.capability_id.in_(
                    capability_ids
                )
            )
        )

        return result.scalar_one()

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: CapabilityModel,
    ) -> Capability:

        return Capability.model_validate(model)
