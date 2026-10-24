"""
MCP provider 查询的实现。

三类事实的来处
--------------
配置      app.common.mcp_config.get_mcp_settings()
运行      app.infra.agent.mcp_client.get_mcp_provider_manager()
已同步    ToolDefinitionRepository.count_by_source_refs()

刻意**不缓存**：这个接口是管理面用的，一次 group_by 的成本远低于
「配置改了但页面还显示旧值」的排查成本。
"""

from __future__ import annotations

from app.common.mcp_config import get_mcp_settings
from app.common.service.base import ServiceContext
from app.infra.agent.mcp_client import (
    get_mcp_provider_manager,
    tool_name,
)

from ..domain.schemas import (
    MCPProviderListResponse,
    MCPProviderSummary,
)
from ..repository import ToolDefinitionRepository
from .mcp_provider_service import MCPProviderService

__all__ = ["MCPProviderServiceImpl"]


class MCPProviderServiceImpl(MCPProviderService):

    def __init__(
            self,
            tool_definition_repository: ToolDefinitionRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._definitions = tool_definition_repository

    async def list_providers(self) -> MCPProviderListResponse:

        settings = get_mcp_settings()
        manager = get_mcp_provider_manager()

        names = list(settings.providers)

        counts = (
            await self._definitions.count_by_source_refs(names)
            if names else {}
        )

        exposed = manager.tools_by_provider()

        items: list[MCPProviderSummary] = []

        for name, config in settings.providers.items():

            connected = manager.is_connected(name)

            admitted = 0

            if connected:

                admitted = sum(
                    1
                    for tool in exposed.get(name, [])
                    if config.tools.admits(tool_name(tool))
                )

            bucket = counts.get(
                name, {"active": 0, "missing": 0}
            )

            items.append(
                MCPProviderSummary(
                    name=name,
                    enabled=settings.enabled and config.enabled,
                    transport=config.transport,
                    connected=connected,
                    exposed_tool_count=admitted,
                    tools_mode=config.tools.mode,
                    tool_count=(
                        bucket["active"] + bucket["missing"]
                    ),
                    active_tool_count=bucket["active"],
                    missing_tool_count=bucket["missing"],
                )
            )

        return MCPProviderListResponse(items=items)
