"""
ToolQueryService 的默认实现。

事务边界由调用方（ServiceFactory.scope() 持有的 session）决定，
本服务内部不 commit / rollback。
"""

from __future__ import annotations

from typing import Any
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

    # ========================================================
    # 解析（见契约里的说明）
    # ========================================================

    async def tool_keys_for_capabilities(
            self,
            codes: list[str],
    ) -> list[str]:

        definitions = (
            await self._definitions.list_active_by_classification_codes(
                codes
            )
        )

        # 仓储已经按 tool_key 排序，但**组装层不该依赖别人的排序**
        # （与 _assembly.build_chain 同一条规矩）：这里少去重一次，
        # 将来任何一条换掉的查询都会静默把重复的 key 交给模型。
        return list(dict.fromkeys(d.tool_key for d in definitions))

    async def tool_ids_for_keys(
            self,
            tool_keys: list[str],
    ) -> dict[str, UUID]:

        if not tool_keys:
            return {}

        definitions = await self._definitions.list_by_keys(tool_keys)

        return {d.tool_key: d.tool_id for d in definitions}

    async def tool_documentation_overrides(
            self,
            tool_keys: list[str],
    ) -> dict[str, dict[str, Any]]:

        if not tool_keys:
            return {}

        # 键 -> id 的桥只有这一条（仓储按 tool_id 存，模型只知道 key）。
        # 自己再写一次 join 就是第二份实现，而两份迟早会对同一个 key
        # 给出不同的 id。
        id_of = await self.tool_ids_for_keys(tool_keys)

        if not id_of:
            return {}

        tool_ids = list(id_of.values())

        # 两次批量查询，而不是逐工具取 —— 逐个取是经典的 N+1，
        # 而这条路跑在**每一轮对话**上（见 list_tools 里同一条说明）。
        descriptions = await self._documentations.map_description_overrides(
            tool_ids
        )

        parameters = await self._documentations.map_parameter_overrides(
            tool_ids
        )

        overrides: dict[str, dict[str, Any]] = {}

        for tool_key, tool_id in id_of.items():

            description = descriptions.get(tool_id)

            # 空串与 None 都算"没写"。这里用真值判据而不是 `is not None`
            # 是刻意的：与读接口 `_assembly.build_summary` 的
            # `description_override or source_raw` 必须是同一个口径，
            # 否则同一份覆盖在列表页生效、在模型那里不生效。
            params = {
                path: value
                for path, value in (parameters.get(tool_id) or {}).items()
                if value
            }

            if not description and not params:
                continue

            entry: dict[str, Any] = {"description": description or None}

            if params:
                entry["parameters"] = params

            overrides[tool_key] = entry

        return overrides
