"""
MCP provider 的对外契约。

对应 mcp_provider.yaml：
    GET /mcp/providers

provider 本身住在 `mcp.yaml`（部署配置），不在数据库里。这里的
`tool_count` 等字段查的是 `tool_definition` 里 `source_ref` 等于
provider 名的行 —— 也就是**已同步**的工具。两边对不上的常见原因是
「刚改完配置还没点同步」，前端可以据此提示。
"""

from __future__ import annotations

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "MCPProviderListResponse",
    "MCPProviderSummary",
]


class MCPProviderSummary(APIModel):
    """一个 MCP provider 的配置 + 运行状态 + 已同步工具数。"""

    name: str = Field(
        description="provider 标识；同时是工具 source_ref 与 tool_key 前缀。",
    )

    enabled: bool = Field(
        description="配置里是否启用（总开关关闭时全部为 false）。",
    )

    transport: str = Field(
        description="连接方式。目前只有 stdio。",
    )

    connected: bool = Field(
        description=(
            "本进程当前是否已建立会话。"
            "false 表示 provider 连不上或尚未启动。"
        ),
    )

    exposed_tool_count: int = Field(
        default=0,
        description=(
            "通过 allowlist 过滤、并已注册进进程内注册表的工具数。"
            "connect 失败时为 0。"
        ),
    )

    tools_mode: str = Field(
        description="工具过滤模式：all / allowlist / denylist。",
    )

    tool_count: int = Field(
        default=0,
        description="tool_definition 里该 provider 的全部行数。",
    )

    active_tool_count: int = Field(
        default=0,
        description="其中 ACTIVE 的行数。",
    )

    missing_tool_count: int = Field(
        default=0,
        description="其中 MISSING 的行数（上次同步没扫到）。",
    )


class MCPProviderListResponse(APIModel):
    """provider 列表。"""

    items: list[MCPProviderSummary] = Field(
        default_factory=list,
        description="按配置里的声明顺序。",
    )
