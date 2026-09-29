"""
PostgreSQL implementation of ToolClassificationRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, or_, select

from ..domain.entities import ToolClassification
from ..models.sqlalchemy.tool_classification import (
    ToolClassification as ToolClassificationModel,
)
from ..models.sqlalchemy.tool_classification import (
    ToolDefinitionClassification as LinkModel,
)
from .tool_classification import ToolClassificationRepository


class ToolClassificationRepositoryImpl(
    ToolClassificationRepository
):

    # ========================================================
    # Taxonomy
    # ========================================================

    async def add(
            self,
            classification: ToolClassification,
    ) -> ToolClassification:

        model = ToolClassificationModel(
            classification_id=classification.classification_id,
            code=classification.code,
            name=classification.name,
            description=classification.description,
            parent_id=classification.parent_id,
            sort_order=classification.sort_order,
            is_active=classification.is_active,
            created_at=classification.created_at,
            updated_at=classification.updated_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    async def update(
            self,
            classification: ToolClassification,
    ) -> ToolClassification:

        model = await self._session.get(
            ToolClassificationModel,
            classification.classification_id,
        )

        if model is None:
            raise ValueError(
                "分类不存在："
                f"{classification.classification_id}"
            )

        # code 刻意不改：它是稳定标识，关联与外部引用都按它。
        model.name = classification.name
        model.description = classification.description
        model.parent_id = classification.parent_id
        model.sort_order = classification.sort_order
        model.is_active = classification.is_active
        model.updated_at = classification.updated_at

        await self._session.flush()

        return self._to_entity(model)

    async def get(
            self,
            classification_id: UUID,
    ) -> ToolClassification | None:

        model = await self._session.get(
            ToolClassificationModel, classification_id
        )

        return None if model is None else self._to_entity(model)

    async def get_by_code(
            self,
            code: str,
    ) -> ToolClassification | None:

        result = await self._session.execute(
            select(ToolClassificationModel).where(
                ToolClassificationModel.code == code
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
    ) -> list[ToolClassification]:

        statement = select(ToolClassificationModel)
        conditions = []

        if parent_id is not None:
            conditions.append(
                ToolClassificationModel.parent_id == parent_id
            )

        if keyword:
            pattern = f"%{keyword}%"
            conditions.append(
                or_(
                    ToolClassificationModel.code.ilike(pattern),
                    ToolClassificationModel.name.ilike(pattern),
                )
            )

        if is_active is not None:
            conditions.append(
                ToolClassificationModel.is_active == is_active
            )

        if conditions:
            statement = statement.where(*conditions)

        result = await self._session.execute(
            statement.order_by(
                ToolClassificationModel.sort_order,
                ToolClassificationModel.code,
            )
        )

        return [self._to_entity(m) for m in result.scalars()]

    # ========================================================
    # Tool ↔ Classification
    # ========================================================

    async def list_for_tool(
            self,
            tool_id: UUID,
    ) -> list[ToolClassification]:

        mapping = await self.list_for_tools([tool_id])

        return mapping.get(tool_id, [])

    async def list_for_tools(
            self,
            tool_ids: list[UUID],
    ) -> dict[UUID, list[ToolClassification]]:

        if not tool_ids:
            return {}

        # 一次 join 取回全部关联，避免列表页的 N+1。
        result = await self._session.execute(
            select(
                LinkModel.tool_id,
                ToolClassificationModel,
            )
            .join(
                ToolClassificationModel,
                ToolClassificationModel.classification_id
                == LinkModel.classification_id,
            )
            .where(LinkModel.tool_id.in_(tool_ids))
            .order_by(
                ToolClassificationModel.sort_order,
                ToolClassificationModel.code,
            )
        )

        mapping: dict[UUID, list[ToolClassification]] = {}

        for tool_id, model in result.all():
            mapping.setdefault(tool_id, []).append(
                self._to_entity(model)
            )

        return mapping

    async def replace_tool_classifications(
            self,
            tool_id: UUID,
            classification_ids: list[UUID],
    ) -> None:

        await self._session.execute(
            delete(LinkModel).where(LinkModel.tool_id == tool_id)
        )

        # 去重：调用方可能传了重复 id，而 (tool_id, classification_id)
        # 是主键，重复插入会直接违反约束。
        for classification_id in dict.fromkeys(
                classification_ids
        ):
            self._session.add(
                LinkModel(
                    tool_id=tool_id,
                    classification_id=classification_id,
                )
            )

        await self._session.flush()

    async def count_links(
            self,
            classification_ids: list[UUID],
    ) -> int:

        if not classification_ids:
            return 0

        result = await self._session.execute(
            select(func.count())
            .select_from(LinkModel)
            .where(
                LinkModel.classification_id.in_(
                    classification_ids
                )
            )
        )

        return result.scalar_one()

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: ToolClassificationModel,
    ) -> ToolClassification:

        return ToolClassification.model_validate(model)
