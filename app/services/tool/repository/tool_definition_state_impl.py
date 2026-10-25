"""
PostgreSQL implementation of ToolDefinitionStateRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select

from ..domain.entities import ToolDefinitionState
from ..models.sqlalchemy.tool_definition_state import (
    ToolDefinitionState as ToolDefinitionStateModel,
)
from .tool_definition_state import ToolDefinitionStateRepository


class ToolDefinitionStateRepositoryImpl(
    ToolDefinitionStateRepository
):

    # ========================================================
    # Write
    # ========================================================

    async def disable(
            self,
            tool_key: str,
            *,
            disabled_by: str | None,
            now: datetime,
    ) -> ToolDefinitionState:

        model = await self._session.get(
            ToolDefinitionStateModel, tool_key
        )

        if model is None:
            model = ToolDefinitionStateModel(
                tool_key=tool_key,
                disabled_by=disabled_by,
                disabled_at=now,
            )
            self._session.add(model)

        else:
            # 幂等：重复禁用刷新审计字段，而不是报错。
            model.disabled_by = disabled_by
            model.disabled_at = now

        await self._session.flush()

        return self._to_entity(model)

    async def enable(
            self,
            tool_key: str,
    ) -> bool:

        result = await self._session.execute(
            delete(ToolDefinitionStateModel).where(
                ToolDefinitionStateModel.tool_key == tool_key
            )
        )

        await self._session.flush()

        return bool(result.rowcount)

    # ========================================================
    # Read
    # ========================================================

    async def list_disabled_keys(self) -> set[str]:

        result = await self._session.execute(
            select(ToolDefinitionStateModel.tool_key)
        )

        return set(result.scalars())

    async def disabled_keys_for(
            self,
            tool_keys: list[str],
    ) -> set[str]:

        # 空列表不能落成 `IN ()`：SQL 里那是语法错误，而"这一页没有
        # 工具"是完全正常的输入（与 tool_definition 里 IN 的守卫同一条）。
        if not tool_keys:
            return set()

        result = await self._session.execute(
            select(ToolDefinitionStateModel.tool_key).where(
                ToolDefinitionStateModel.tool_key.in_(tool_keys)
            )
        )

        return set(result.scalars())

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: ToolDefinitionStateModel,
    ) -> ToolDefinitionState:

        return ToolDefinitionState.model_validate(model)
