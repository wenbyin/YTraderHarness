"""
发现器：编排多个扫描器，产出去重后的事实快照。

它自己**不做扫描** —— 那是 scanner.py 的事。这里只负责：

    跑哪些扫描器（按 source_type 过滤）
    按 source_ref 收窄
    跨来源去重与冲突检测

这样新增一种来源（MCP / OpenAPI）不用动本文件，加个扫描器就行。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..domain.schemas.enum import SourceType
from .registry import ToolRegistry, get_tool_registry
from .scanner import (
    LangchainToolScanner,
    ServiceMethodScanner,
    ToolScanner,
)
from .spec import ToolManifestData

__all__ = ["ToolDiscoverer"]

logger = logging.getLogger(__name__)


class ToolDiscoverer:
    """
    把若干扫描器的结果合成一份工具清单。

    无状态 —— 每次 discover() 都重新扫一遍，因此"改了代码再同步"
    一定能看到变化。
    """

    def __init__(self, scanners: Sequence[ToolScanner]) -> None:
        self._scanners = list(scanners)

    @classmethod
    def default(
            cls,
            registry: ToolRegistry | None = None,
    ) -> "ToolDiscoverer":
        """
        装配本进程里的全部来源。

        前两个扫描器都是 LOCAL_METHOD —— 区别在**取数方式与执行方式**，
        不在来源：一个来自 @tool 自注册（执行时要由工厂建 self），
        一个来自模块级的 langchain BaseTool（执行时直接调它自己）。

        第三个是 MCP：工具在外部 server 里，按 `mcp.yaml` 的连接配置
        走 `tools/list`，不扫代码。没有配置文件时它返回空清单。

        新增来源（OpenAPI）时在这里加一行即可。
        """

        # 惰性 import：`mcp_scanner` 会拖进 MCP 客户端（含可选的 mcp
        # SDK）。只用本地工具、或根本没装 SDK 的环境不该在 import
        # 注册表时就付这份成本 —— 与 scanner 里延迟取 BaseTool 同一取舍。
        from .mcp_scanner import MCPToolScanner

        return cls([
            ServiceMethodScanner(
                registry if registry is not None
                else get_tool_registry()
            ),
            LangchainToolScanner(),
            MCPToolScanner(),
        ])

    # ========================================================
    # 扫描
    # ========================================================

    def discover(
            self,
            *,
            source_types: list[SourceType] | None = None,
            source_refs: list[str] | None = None,
    ) -> list[ToolManifestData]:
        """
        跑一遍扫描器并合并结果。

        source_types 为空 = **全部来源**（不是"默认只要 local_method"——
        那是只有一种来源时的历史写法，现在有了第二种，"不指定"就该
        是"都要"）。

        两个过滤参数都是**收窄扫描范围**用的，不是"过滤结果"：
        被排除的扫描器根本不会被跑（MCP 那种要发网络请求的，
        不跑才是有意义的省）。
        """

        wanted_types = set(source_types) if source_types else None
        wanted_refs = set(source_refs) if source_refs else None

        merged: dict[str, ToolManifestData] = {}

        for scanner in self._scanners:

            if (
                wanted_types is not None
                and getattr(scanner, "source_type", None)
                not in wanted_types
            ):
                continue

            for manifest in scanner.scan():

                if (
                    wanted_refs is not None
                    and manifest.source_ref not in wanted_refs
                ):
                    continue

                existing = merged.get(manifest.tool_key)

                if existing is not None:
                    # 两条来源抢同一个 key —— 静默取其一比报错危险，
                    # 与 ToolRegistry.register 同一套取舍。
                    raise ValueError(
                        f"tool_key {manifest.tool_key!r} 被两条来源"
                        f"同时声明：{existing.entrypoint}（"
                        f"{existing.source_type.value}）与 "
                        f"{manifest.entrypoint}（"
                        f"{manifest.source_type.value}）。"
                        "请给其中一个换 key。"
                    )

                merged[manifest.tool_key] = manifest

        return sorted(
            merged.values(), key=lambda m: m.tool_key
        )
