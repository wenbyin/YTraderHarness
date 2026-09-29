"""
ToolQueryService 的默认实现。

事务边界由调用方（ServiceFactory.scope() 持有的 session）决定，
本服务内部不 commit / rollback。
"""

from __future__ import annotations

from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext

from ..domain.schemas import (
    PagedToolSummaryResponse,
    SourceState,
    SourceType,
    ToolDefinitionDetail,
)
from ..repository.tool_classification import (
    ToolClassificationRepository,
)
from ..repository.tool_definition import ToolDefinitionRepository
from ..repository.tool_documentation import (
    ToolDocumentationRepository,
)
from ._assembly import build_definition_detail, build_summary
from .tool_query_service import ToolQueryService


class ToolQueryServiceImpl(ToolQueryService):

    def __init__(
            self,
            *,
            tool_definition_repository: ToolDefinitionRepository,
            tool_documentation_repository: (
                ToolDocumentationRepository
            ),
            tool_classification_repository: (
                ToolClassificationRepository
            ),
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._definitions = tool_definition_repository
        self._documentations = tool_documentation_repository
        self._classifications = tool_classification_repository

    # ========================================================
    # 列表
    # ========================================================

    async def list_tools(
            self,
            *,
            keyword: str | None = None,
            source_type: SourceType | None = None,
            classification_code: str | None = None,
            source_state: SourceState | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> PagedToolSummaryResponse:

        offset = (page - 1) * page_size

        definitions, total = await self._definitions.list_paged(
            keyword=keyword,
            source_type=source_type,
            classification_code=classification_code,
            source_state=source_state,
            offset=offset,
            limit=page_size,
        )

        tool_ids = [d.tool_id for d in definitions]

        # 两个批量查询，而不是在循环里逐个取 ——
        # 一页 20 条时那是 40 次多余的往返（经典的 N+1）。
        overrides = (
            await self._documentations.map_description_overrides(
                tool_ids
            )
        )

        classification_map = (
            await self._classifications.list_for_tools(tool_ids)
        )

        items = [
            build_summary(
                definition,
                description_override=overrides.get(
                    definition.tool_id
                ),
                classifications=classification_map.get(
                    definition.tool_id, []
                ),
            )
            for definition in definitions
        ]

        return PagedToolSummaryResponse(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
        )

    # ========================================================
    # 详情
    # ========================================================

    async def get_tool(
            self,
            tool_id: UUID,
    ) -> ToolDefinitionDetail:

        definition = await self._definitions.get(tool_id)

        if definition is None:
            raise NotFoundError(f"Tool 定义不存在：{tool_id}")

        stored = await self._documentations.get(tool_id)

        classifications = (
            await self._classifications.list_for_tool(tool_id)
        )

        return build_definition_detail(
            definition,
            stored,
            classifications,
        )
