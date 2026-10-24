"""
MCP 来源的扫描器。

与另外两个扫描器的根本区别
--------------------------
`ServiceMethodScanner` / `LangchainToolScanner` 扫的是**代码**：遍历
包、触发 import，工具在编译期就存在于仓库里。MCP 的工具是
`tools/list` 返回的**协议事实** —— 仓库里没有它们，也不该有。
所以这里不发包遍历，只按 `mcp.yaml` 的连接配置去问 server。

一个 provider 连不上，跳过它并留 warning，其它 provider 照常产出。
「provider 暂时不可用」与「工具被删了」必须区分：前者不该让整个
同步失败，也不该让库里的行立刻变成 MISSING —— 后者由 sync 的
对账逻辑处理，这里只负责**本次能扫到什么**。

过滤在扫描阶段做
----------------
`tools.mode = allowlist` 且名字不在 `allow` 里的工具**根本不进**
清单。因此它们不会被 sync 写进 `tool_definition`，前端看不到，
模型更拿不到 —— 「未放行」与「不存在」对外是同一件事，不需要
再维护一个 enabled 状态。
"""

from __future__ import annotations

import logging
from typing import Any

from app.common.mcp_config import (
    MCPSettings,
    get_mcp_settings,
)
from app.infra.agent.mcp_client import (
    list_provider_tools,
    tool_description,
    tool_input_schema,
    tool_name,
)
from ..domain.schemas.enum import SourceType
from .schema import extract_parameter_descriptions
from .spec import ToolManifestData

__all__ = ["MCPToolScanner"]

logger = logging.getLogger(__name__)


class MCPToolScanner:
    """
    按配置连接 MCP provider，产出 `SourceType.MCP` 的事实快照。

    实现 `ToolScanner` 协议（`scan() -> list[ToolManifestData]`）。
    同步是协议的要求；一次性的 MCP 会话在独立线程里跑，见
    `mcp_client.list_provider_tools`。
    """

    source_type = SourceType.MCP

    def __init__(
            self,
            settings: MCPSettings | None = None,
    ) -> None:
        #: 注入点：测试给一份内存里的设置，生产走缓存加载。
        self._settings = settings

    def scan(self) -> list[ToolManifestData]:

        settings = (
            self._settings
            if self._settings is not None
            else get_mcp_settings()
        )

        if not settings.enabled:
            return []

        manifests: list[ToolManifestData] = []

        for name, provider in settings.active_providers().items():

            try:
                tools = list_provider_tools(provider)

            except Exception:
                # 环境状况，不是代码错误：server 没装、终端没开、
                # 启动超时都走这条。留一条带堆栈的 warning，然后
                # 让其它 provider 继续 —— 同步接口不该被一台机器上
                # 某个外部进程拖着一起失败。
                logger.warning(
                    "MCP provider %s 扫描失败，本次跳过它的工具。",
                    name,
                    exc_info=True,
                )
                continue

            seen: set[str] = set()
            admitted = 0

            for tool in tools:

                tool_name_ = tool_name(tool)

                if not tool_name_:
                    continue

                if tool_name_ in seen:
                    logger.warning(
                        "MCP provider %s 重复暴露工具 %r，只保留第一个。",
                        name, tool_name_,
                    )
                    continue

                seen.add(tool_name_)

                if not provider.tools.admits(tool_name_):
                    continue

                admitted += 1
                manifests.append(
                    self._to_manifest(name, tool)
                )

            logger.info(
                "MCP provider %s：%d 个工具，放行 %d。",
                name, len(tools), admitted,
            )

        return manifests

    @staticmethod
    def _to_manifest(
            provider: str,
            tool: Any,
    ) -> ToolManifestData:
        """
        一个 MCP tool -> 一份事实快照。

        `tool_key` 用 `<provider>.<tool>`，与本地工具的
        `<source_ref>.<name>` 同一形状；`entrypoint` 用
        `mcp:<provider>:<tool>`，明确标注它不是本进程里的
        Python 入口 —— 排查时一眼能看出该去哪个 server 找。
        """

        name = tool_name(tool)
        description = tool_description(tool)
        input_schema = tool_input_schema(tool)

        output_schema = getattr(tool, "output_schema", None)

        if not isinstance(output_schema, dict):
            output_schema = None

        parameters = extract_parameter_descriptions(input_schema)

        # 与本地扫描器的 `_docstring` 同一口径：什么都没有就存 null，
        # 而不是一个 {raw: None, parameters: {}} 的空壳 —— 后者会参与
        # manifest_hash，让「没变」的信息看起来像变了。
        docstring: dict[str, Any] | None = (
            {"raw": description, "parameters": parameters}
            if description or parameters
            else None
        )

        if docstring is not None and description:
            docstring["description"] = description

        return ToolManifestData(
            tool_key=f"{provider}.{name}",
            name=name,
            source_type=SourceType.MCP,
            source_ref=provider,
            entrypoint=f"mcp:{provider}:{name}",
            input_schema=input_schema,
            output_schema=output_schema,
            source_docstring=docstring,
        )
