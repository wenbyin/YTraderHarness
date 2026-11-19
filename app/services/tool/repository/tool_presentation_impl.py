"""
PostgreSQL implementation of ToolPresentationRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select

from ..domain.entities import ToolPresentation
from ..models.sqlalchemy.tool_presentation import (
    ToolPresentation as ToolPresentationModel,
)
from .tool_presentation import ToolPresentationRepository


class ToolPresentationRepositoryImpl(
    ToolPresentationRepository
):

    # ========================================================
    # Write
    # ========================================================

    async def upsert(
            self,
            tool_key: str,
            *,
            renderer: str,
            display_name: str | None,
            summary: str | None,
            show_input: bool,
            show_output: bool,
            props: dict[str, Any],
            updated_by: str | None,
            now: datetime,
    ) -> ToolPresentation:

        model = await self._session.get(
            ToolPresentationModel, tool_key
        )

        if model is None:
            model = ToolPresentationModel(
                tool_key=tool_key,
                renderer=renderer,
                display_name=display_name,
                summary=summary,
                show_input=show_input,
                show_output=show_output,
                props=props,
                updated_by=updated_by,
                updated_at=now,
            )
            self._session.add(model)

        else:
            # PUT 语义：整份覆盖。缺字段由服务层补默认值后传进来。
            model.renderer = renderer
            model.display_name = display_name
            model.summary = summary
            model.show_input = show_input
            model.show_output = show_output
            model.props = props
            model.updated_by = updated_by
            model.updated_at = now

        await self._session.flush()

        return self._to_entity(model)

    async def delete(
            self,
            tool_key: str,
    ) -> bool:

        result = await self._session.execute(
            delete(ToolPresentationModel).where(
                ToolPresentationModel.tool_key == tool_key
            )
        )

        await self._session.flush()

        return bool(result.rowcount)

    # ========================================================
    # Read
    # ========================================================

    async def get(
            self,
            tool_key: str,
    ) -> ToolPresentation | None:

        model = await self._session.get(
            ToolPresentationModel, tool_key
        )

        if model is None:
            return None

        return self._to_entity(model)

    async def list_all(self) -> list[ToolPresentation]:

        result = await self._session.execute(
            select(ToolPresentationModel).order_by(
                ToolPresentationModel.tool_key
            )
        )

        return [
            self._to_entity(model) for model in result.scalars()
        ]

    async def map_for_keys(
            self,
            tool_keys: list[str],
    ) -> dict[str, ToolPresentation]:

        # 空列表不能落成 `IN ()` —— 与其它仓储同一条守卫。
        if not tool_keys:
            return {}

        result = await self._session.execute(
            select(ToolPresentationModel).where(
                ToolPresentationModel.tool_key.in_(tool_keys)
            )
        )

        return {
            model.tool_key: self._to_entity(model)
            for model in result.scalars()
        }

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: ToolPresentationModel,
    ) -> ToolPresentation:

        return ToolPresentation.model_validate(model)
