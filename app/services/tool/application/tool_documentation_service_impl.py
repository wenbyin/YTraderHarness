"""
ToolDocumentationService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext

from ..domain.schemas import (
    ToolDocumentation,
    ToolDocumentationUpdate,
)
from ..repository.tool_definition import ToolDefinitionRepository
from ..repository.tool_documentation import (
    ToolDocumentationRepository,
)
from ._assembly import build_documentation
from .tool_documentation_service import ToolDocumentationService


class ToolDocumentationServiceImpl(ToolDocumentationService):

    def __init__(
            self,
            *,
            tool_definition_repository: ToolDefinitionRepository,
            tool_documentation_repository: (
                ToolDocumentationRepository
            ),
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._definitions = tool_definition_repository
        self._documentations = tool_documentation_repository

    # ========================================================
    # 读
    # ========================================================

    async def get_documentation(
            self,
            tool_id: UUID,
    ) -> ToolDocumentation:

        definition = await self._require_definition(tool_id)

        stored = await self._documentations.get(tool_id)

        return build_documentation(definition, stored)

    # ========================================================
    # 写
    # ========================================================

    async def replace_documentation(
            self,
            tool_id: UUID,
            request: ToolDocumentationUpdate,
            *,
            updated_by: str | None = None,
    ) -> ToolDocumentation:

        definition = await self._require_definition(tool_id)

        now = datetime.now(timezone.utc)

        await self._documentations.upsert(
            tool_id,
            description_override=request.description_override,
            return_description_override=(
                request.return_description_override
            ),
            updated_by=updated_by,
            now=now,
        )

        await self._documentations.replace_parameter_documentations(
            tool_id,
            [
                (item.parameter_path, item.description_override)
                for item in request.parameter_descriptions
            ],
            updated_by=updated_by,
            now=now,
        )

        # 写完读回来再组装，而不是拿 request 拼一个返回 ——
        # 那样返回的是"我以为写进去的"，不是"库里实际是什么"。
        # 两者不一致时（例如路径写错），只有这样才发现得了。
        stored = await self._documentations.get(tool_id)

        return build_documentation(definition, stored)

    # ========================================================
    # internals
    # ========================================================

    async def _require_definition(self, tool_id: UUID):
        """
        先确认 Tool 存在。

        没有这一步的话，PUT 会往一个不存在的 tool_id 上写说明行，
        而外键约束会以一句数据库层的报错暴露 —— 那是 500，
        实际应该是 404。
        """

        definition = await self._definitions.get(tool_id)

        if definition is None:
            raise NotFoundError(f"Tool 定义不存在：{tool_id}")

        return definition
