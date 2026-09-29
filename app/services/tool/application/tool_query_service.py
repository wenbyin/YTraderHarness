"""
Tool 定义的查询契约。

对应 tool.yaml：
    GET /tools            列表
    GET /tools/{tool_id}  详情

只读，无副作用。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    PagedToolSummaryResponse,
    SourceState,
    SourceType,
    ToolDefinitionDetail,
)

__all__ = ["ToolQueryService"]


class ToolQueryService(BaseService):

    @abstractmethod
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
        """
        分页列出工具定义。

        classification_code 按分类的**稳定标识**筛（而不是 id）：
        分类 id 会在重建分类树时变，code 不会，而调用方
        （前端 / Agent 配置）记住的是 code。

        description 回的是**生效值**（人工覆盖优先），
        与详情页保持一致。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_tool(
            self,
            tool_id: UUID,
    ) -> ToolDefinitionDetail:
        """
        取一个工具的全部信息：事实 + 说明 + 分类。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError
