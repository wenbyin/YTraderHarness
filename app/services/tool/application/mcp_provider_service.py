"""
MCP provider 的查询契约。

对应 mcp_provider.yaml：
    GET /mcp/providers

只读，无副作用。provider 定义在 `mcp.yaml`（部署配置），
本服务把配置、运行状态与库里的同步计数合成一份给前端。
"""

from __future__ import annotations

from abc import abstractmethod

from app.common.service.base import BaseService

from ..domain.schemas import MCPProviderListResponse

__all__ = ["MCPProviderService"]


class MCPProviderService(BaseService):

    @abstractmethod
    async def list_providers(self) -> MCPProviderListResponse:
        """
        列出全部配置里的 provider。

        包括**已停用**的：前端需要它们来渲染「有这个 provider、
        但当前关闭」的状态；直接过滤掉会让配置看起来变了。

        每个 provider 同时给三类事实：
            配置     enabled / transport / tools_mode
            运行     connected / exposed_tool_count
            库       active_tool_count / missing_tool_count / tool_count

        三类分别来自 mcp.yaml、进程内管理器、tool_definition。
        它们对不上是正常的（例如刚改配置还没同步），所以不合并成
        一个「总数」—— 那会把「配置里有、库里没有」这种状态藏起来。
        """

        raise NotImplementedError
