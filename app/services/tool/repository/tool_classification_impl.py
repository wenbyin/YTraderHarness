"""
PostgreSQL implementation of ToolClassificationRepository.

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

from ..models.sqlalchemy.tool_classification import (
    ToolDefinitionClassification as LinkModel,
)
from .tool_classification import ToolClassificationRepository


class ToolClassificationRepositoryImpl(
    ToolClassificationRepository
):

    async def list_for_tool(
            self,
            tool_id: UUID,
    ) -> list[Capability]:

        mapping = await self.list_for_tools([tool_id])

        return mapping.get(tool_id, [])

    async def list_for_tools(
            self,
            tool_ids: list[UUID],
    ) -> dict[UUID, list[Capability]]:

        if not tool_ids:
            return {}

        # 一次 join 取回全部关联，避免列表页的 N+1。
        # join 的另一侧是 capability 表 —— 关联表不自带能力的名字等
        # 信息，调用方（工具列表页）要的就是完整能力对象，因此这里
        # 必须跨模块 join，而不是只回一串 id。
        result = await self._session.execute(
            select(
                LinkModel.tool_id,
                CapabilityModel,
            )
            .join(
                CapabilityModel,
                CapabilityModel.capability_id
                == LinkModel.classification_id,
            )
            .where(LinkModel.tool_id.in_(tool_ids))
            .order_by(
                CapabilityModel.sort_order,
                CapabilityModel.code,
            )
        )

        mapping: dict[UUID, list[Capability]] = {}

        for tool_id, model in result.all():
            mapping.setdefault(tool_id, []).append(
                Capability.model_validate(model)
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
