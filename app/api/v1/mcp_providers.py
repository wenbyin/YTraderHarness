"""
MCP provider 查询接口。

对应 mcp_provider.yaml 的 tag：
    MCP Providers

路径主语是 provider（外部 server），不是工具 —— 所以从 `/tools`
里分出来。前端用它渲染「有哪些来源可以过滤」；工具本身仍走
`GET /tools?source_type=mcp&source_ref=<provider>`。

⚠️ 这里只读。增删改 provider 要改 `mcp.yaml` 并重启进程：
连接是启动期建立的，热改配置不会自动重连（见
app/bootstrap/mcp.py）。把写接口做出来反而会造成「保存了但没
生效」的错觉。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.tool.application.mcp_provider_service import (
    MCPProviderService,
)
from app.services.tool.domain.schemas import (
    MCPProviderListResponse,
)

router = APIRouter()


@router.get(
    "/mcp/providers",
    response_model=MCPProviderListResponse,
    tags=["MCP Providers"],
)
async def list_mcp_providers(
        service: MCPProviderService = Service[MCPProviderService],
):
    """
    列出配置里的 MCP provider 及其运行 / 同步状态。

    三类事实分开给：
        enabled / connected      配置与运行状态
        exposed_tool_count       当前注册进进程的工具数
        active/missing/tool_count  库里已同步的行

    前端按 provider 过滤工具时用 `name` 作为 `source_ref`。
    """

    try:
        return await service.list_providers()

    except ValueError as exc:
        raise to_http_error(exc) from exc
