"""
MCP provider 的启动期装配。

为什么放在 bootstrap 而不是 registry
------------------------------------
MCP 工具要同时落到两个地方：

    ToolRegistry    进程内注册表 —— select_specs / 暴露 / 执行都读它
    tool_definition 数据库快照 —— 管理接口与分类用它（走 sync 接口）

注册需要「客户端管理器」（infra）与「ToolSpec」（services），把两边
接起来是**组合根**的职责。放在 registry 里会让 services 反向依赖
infra 的连接栈（见 `app/infra/agent/mcp_client.py` 的说明），放在
infra 里又会反向依赖 services 的注册表 —— 两头都不对。

失败策略
--------
provider 连不上只记 ERROR，**不阻断应用启动**。外部进程是环境状况：
没装、路径变了、终端没开，都不该让整个 API 起不来。代价是那一轮
对话看不到它的工具 —— 启动日志里必须能看见原因，所以这里用
`logger.exception` 而不是 warning。
"""

from __future__ import annotations

import logging

from app.common.mcp_config import MCPSettings, get_mcp_settings
from app.infra.agent.mcp_client import (
    MCPProviderManager,
    build_structured_tool,
    get_mcp_provider_manager,
    tool_description,
    tool_name,
)
from app.services.tool.registry import ToolSpec, get_tool_registry

__all__ = [
    "initialize_mcp_providers",
    "register_mcp_tools",
    "shutdown_mcp_providers",
]

logger = logging.getLogger(__name__)


async def initialize_mcp_providers() -> int:
    """
    连接配置里的全部 provider，并把放行的工具注册进进程级注册表。

    返回注册的工具数。没配 `mcp.yaml` 时返回 0，不做任何连接。
    """

    settings = get_mcp_settings()

    if not settings.enabled:
        logger.info("MCP 未启用（没有 mcp.yaml 或 enabled=false）。")
        return 0

    manager = get_mcp_provider_manager()

    await manager.start(settings)

    return register_mcp_tools(manager=manager, settings=settings)


def register_mcp_tools(
        *,
        manager: MCPProviderManager | None = None,
        settings: MCPSettings | None = None,
) -> int:
    """
    把管理器里**已连接**的 provider 工具注册进 ToolRegistry。

    为什么注册而不是只放 DB
    ----------------------
    `select_specs` 只读进程级注册表，且要求 spec 有可执行对象
    （descriptor）。只写 `tool_definition` 的结果是「库里查得到、
    模型拿不到」—— 那正是 README 里记录的旧 bug 形状。

    放行过滤在这里再做一次：`manager.tools_by_provider()` 给的是
    server 暴露的全集，而 `mcp.yaml` 的 allowlist 决定我们认谁。
    扫描器那条路自己也会过滤，两处用同一个 `admits()`。
    """

    manager = manager or get_mcp_provider_manager()
    settings = settings or get_mcp_settings()

    registry = get_tool_registry()
    registered = 0
    skipped = 0

    for provider, tools in manager.tools_by_provider().items():

        config = settings.providers.get(provider)

        if config is None:
            # 管理器里有、配置里没了 —— 只可能来自热改配置后没重连。
            logger.warning(
                "MCP provider %s 已连接但不在配置里，跳过注册。",
                provider,
            )
            continue

        for tool in tools:

            name = tool_name(tool)

            if not name:
                continue

            if not config.tools.admits(name):
                skipped += 1
                continue

            registry.register(
                ToolSpec(
                    tool_key=f"{provider}.{name}",
                    name=name,
                    # 无 owner：执行入口就是这个 langchain 工具本身，
                    # 与模块级 langchain 工具走同一条执行分支
                    # （tool_executor._base_call 的 spec.tool 分支）。
                    tool=build_structured_tool(
                        provider, tool, manager,
                    ),
                    description=tool_description(tool),
                    source_ref=provider,
                )
            )

            registered += 1

    logger.info(
        "MCP 工具注册完成：%d 个，未放行跳过 %d 个。",
        registered, skipped,
    )

    return registered


async def shutdown_mcp_providers() -> None:
    """关闭全部常驻 MCP 会话。"""

    manager = get_mcp_provider_manager()

    await manager.stop()
