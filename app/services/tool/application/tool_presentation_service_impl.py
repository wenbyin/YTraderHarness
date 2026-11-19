"""
ToolPresentationService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext

from ..domain.schemas import (
    ToolPresentation,
    ToolPresentationState,
)
from ..repository.tool_definition import ToolDefinitionRepository
from ..repository.tool_presentation import (
    ToolPresentationRepository,
)
from ._assembly import build_presentation
from .tool_presentation_service import ToolPresentationService

__all__ = ["ToolPresentationServiceImpl"]


class ToolPresentationServiceImpl(ToolPresentationService):

    def __init__(
            self,
            *,
            tool_definition_repository: ToolDefinitionRepository,
            tool_presentation_repository: (
                ToolPresentationRepository
            ),
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._definitions = tool_definition_repository
        self._presentations = tool_presentation_repository

    async def set_presentation(
            self,
            tool_id: UUID,
            request: ToolPresentation,
            *,
            updated_by: str | None = None,
    ) -> ToolPresentationState:

        definition = await self._definition(tool_id)

        stored = await self._presentations.upsert(
            definition.tool_key,
            renderer=request.renderer.value,
            display_name=request.display_name,
            summary=request.summary,
            show_input=request.show_input,
            show_output=request.show_output,
            props=dict(request.props),
            updated_by=updated_by,
            now=datetime.now(timezone.utc),
        )

        return ToolPresentationState(
            tool_key=definition.tool_key,
            presented=True,
            presentation=build_presentation(stored),
        )

    async def clear_presentation(
            self,
            tool_id: UUID,
            *,
            updated_by: str | None = None,
    ) -> ToolPresentationState:

        definition = await self._definition(tool_id)

        # 没有行也成功：关闭展示是幂等的，重复点不该报错。
        await self._presentations.delete(definition.tool_key)

        return ToolPresentationState(
            tool_key=definition.tool_key,
            presented=False,
            presentation=None,
        )

    # ========================================================
    # internals
    # ========================================================

    async def _definition(self, tool_id: UUID):
        """解析 tool_id -> 定义。不存在就 404，不让孤儿行落库。"""

        definition = await self._definitions.get(tool_id)

        if definition is None:
            raise NotFoundError(f"Tool 定义不存在：{tool_id}")

        return definition
