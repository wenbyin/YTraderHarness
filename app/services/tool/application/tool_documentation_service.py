"""
Tool 说明文字的读写契约。

对应 tool.yaml：
    GET /tools/{tool_id}/documentation
    PUT /tools/{tool_id}/documentation

这是**唯一**能改 Tool 说明的入口 —— definition 归代码，
同步不会覆盖这里写下的内容。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    ToolDocumentation,
    ToolDocumentationUpdate,
)

__all__ = ["ToolDocumentationService"]


class ToolDocumentationService(BaseService):

    @abstractmethod
    async def get_documentation(
            self,
            tool_id: UUID,
    ) -> ToolDocumentation:
        """
        取说明：代码来源 + 人工覆盖 + 生效值。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_documentation(
            self,
            tool_id: UUID,
            request: ToolDocumentationUpdate,
            *,
            updated_by: str | None = None,
    ) -> ToolDocumentation:
        """
        整体替换说明覆盖。

        语义是替换：request 里没列出的参数，其覆盖会被清掉
        （回到代码里的原始说明）。这样前端可以直接把整个表单
        提交上来，不必自己算增删 —— 与分类关联是同一套取舍。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError
