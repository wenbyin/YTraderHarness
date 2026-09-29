"""
PostgreSQL implementation of ToolDocumentationRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select

from ..domain.entities import (
    ParameterDocumentation,
    ToolDocumentation,
)
from ..models.sqlalchemy.tool_documentation import (
    ToolDocumentation as ToolDocumentationModel,
)
from ..models.sqlalchemy.tool_parameter_documentation import (
    ToolParameterDocumentation as ParameterDocumentationModel,
)
from .tool_documentation import ToolDocumentationRepository


class ToolDocumentationRepositoryImpl(
    ToolDocumentationRepository
):

    # ========================================================
    # Write
    # ========================================================

    async def upsert(
            self,
            tool_id: UUID,
            *,
            description_override: str | None,
            return_description_override: str | None,
            updated_by: str | None,
            now: datetime,
    ) -> ToolDocumentation:

        model = await self._session.get(
            ToolDocumentationModel, tool_id
        )

        if model is None:
            model = ToolDocumentationModel(
                tool_id=tool_id,
                description_override=description_override,
                return_description_override=(
                    return_description_override
                ),
                updated_by=updated_by,
                updated_at=now,
            )
            self._session.add(model)

        else:
            model.description_override = description_override
            model.return_description_override = (
                return_description_override
            )
            model.updated_by = updated_by
            model.updated_at = now

        await self._session.flush()

        parameters = await self.list_parameter_documentations(
            tool_id
        )

        return self._to_entity(model, parameters)

    async def replace_parameter_documentations(
            self,
            tool_id: UUID,
            entries: list[tuple[str, str | None]],
            *,
            updated_by: str | None,
            now: datetime,
    ) -> None:
        """
        先删后插，整段替换。

        为 None 的覆盖**照样入库**（而不是跳过）：
        调用方可能就是想显式表达"这个参数我撤回到代码里的说明了"。
        若这里跳过、get() 那边却按行返回，就会出现
        "写完读回来少了几条"的错觉。两边保持一致更重要。

        代价是可能存下全 null 的行，无害 —— 它们不影响任何读取结果。
        """

        await self._session.execute(
            delete(ParameterDocumentationModel).where(
                ParameterDocumentationModel.tool_id == tool_id
            )
        )

        for path, override in entries:
            self._session.add(
                ParameterDocumentationModel(
                    parameter_documentation_id=uuid4(),
                    tool_id=tool_id,
                    parameter_path=path,
                    description_override=override,
                    updated_by=updated_by,
                    updated_at=now,
                )
            )

        await self._session.flush()

    # ========================================================
    # Read
    # ========================================================

    async def get(
            self,
            tool_id: UUID,
    ) -> ToolDocumentation | None:

        model = await self._session.get(
            ToolDocumentationModel, tool_id
        )

        if model is None:
            return None

        parameters = await self.list_parameter_documentations(
            tool_id
        )

        return self._to_entity(model, parameters)

    async def list_parameter_documentations(
            self,
            tool_id: UUID,
    ) -> list[ParameterDocumentation]:

        result = await self._session.execute(
            select(ParameterDocumentationModel)
            .where(ParameterDocumentationModel.tool_id == tool_id)
            .order_by(ParameterDocumentationModel.parameter_path)
        )

        return [
            ParameterDocumentation.model_validate(m)
            for m in result.scalars()
        ]

    async def map_description_overrides(
            self,
            tool_ids: list[UUID],
    ) -> dict[UUID, str | None]:

        if not tool_ids:
            return {}

        result = await self._session.execute(
            select(
                ToolDocumentationModel.tool_id,
                ToolDocumentationModel.description_override,
            ).where(
                ToolDocumentationModel.tool_id.in_(tool_ids)
            )
        )

        return {
            tool_id: override
            for tool_id, override in result.all()
        }

    async def count_for_tools(
            self,
            tool_ids: list[UUID],
    ) -> int:

        if not tool_ids:
            return 0

        tool_level = await self._session.execute(
            select(func.count())
            .select_from(ToolDocumentationModel)
            .where(ToolDocumentationModel.tool_id.in_(tool_ids))
        )

        parameter_level = await self._session.execute(
            select(func.count())
            .select_from(ParameterDocumentationModel)
            .where(ParameterDocumentationModel.tool_id.in_(tool_ids))
        )

        return (
            tool_level.scalar_one()
            + parameter_level.scalar_one()
        )

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: ToolDocumentationModel,
            parameters: list[ParameterDocumentation],
    ) -> ToolDocumentation:

        return ToolDocumentation(
            tool_id=model.tool_id,
            description_override=model.description_override,
            return_description_override=(
                model.return_description_override
            ),
            updated_by=model.updated_by,
            updated_at=model.updated_at,
            parameter_documentations=parameters,
        )
