"""
ytrader-market 的 MCP 接入冒烟。

跑法（服务先起来：D:\\workspace\\ytrader-market 下 `bin/marketd.exe -config config.yaml`）：

    python -m app.agent.mcp_market_smoke

为什么单独一个冒烟
------------------
MCP 的单元测试全在"纯逻辑"层（过滤、结果转换、配置解析），**不连
真实 server**。而这次接入的是一条新传输路径（streamable_http）——
协议对不对、超时够不够、工具名与 schema 长什么样，只有真连一次
才知道。所以这里刻意复用线上同一套代码（`mcp_client` 的管理器），
不另写一段 httpx：

    连上 -> tools/list -> tools/call -> 转换结果

验的是这几件事：

    1. provider 能连上（SDK 的 HTTP 客户端与对端协议对得上）
    2. allowlist 生效：写类工具（sync_data / sync_schema）进不来
    3. query_market_data 真能取到数据，且 diagnostics 能说明来源
    4. 结果经 `to_tool_payload` 之后是可解析的 JSON 文本

不依赖数据库、模型、FastAPI —— 只要 mcp SDK 装上就能跑，
所以它在 CI 里可以被跳过，在部署后可以当健康检查。
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from app.common.mcp_config import (
    MCPSettings,
    MCPProviderSettings,
    get_mcp_settings,
)
from app.infra.agent.mcp_client import (
    get_mcp_provider_manager,
    tool_input_schema,
    tool_name,
)

#: mcp.yaml 里的 provider 名。改配置时同步改这里，别用通配。
PROVIDER = "ytrader_market"

#: 对端即使放开了写接口，这些也不该出现在我们的注册表里。
FORBIDDEN = ("sync_data", "sync_schema", "get_task", "get_platform_stats")


def _as_json(payload: Any) -> Any:
    """
    把工具结果尽量解析成结构。

    ytrader-market 的每个工具都把 JSON 以**文本块**返回（MCP 里最
    常见的一种做法，`structuredContent` 可选）。这里解析成功就用
    结构，失败就原样返回 —— 冒烟的目的是看清内容，不是做严格解析。
    """

    if not isinstance(payload, str):
        return payload

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def _ok(step: str, message: str = "") -> None:
    print(f"  [ok]   {step}{(' — ' + message) if message else ''}")


def _fail(step: str, message: str) -> int:
    print(f"  [fail] {step} — {message}")
    return 1


def _raw_schema(tool: Any) -> dict[str, Any] | None:
    """工具的原始 inputSchema（SDK 的 snake_case 与协议的 camelCase 都认）。"""

    raw = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
    )

    return raw if isinstance(raw, dict) else None


def _has_null_properties(tool: Any) -> bool:
    """
    server 是否把无参工具的 properties 写成了 null。

    这一条要盯**原始** schema，不能用 `tool_input_schema()`：
    后者会兜底规范化（客户端止损），看起来永远是对的。
    """

    schema = _raw_schema(tool)

    return (
        schema is not None
        and "properties" in schema
        and schema["properties"] is None
    )


def _pick_provider() -> MCPProviderSettings | None:
    settings = get_mcp_settings()

    provider = settings.providers.get(PROVIDER)

    if provider is None:
        print(f"mcp.yaml 里没有 provider {PROVIDER!r}，"
              f"先按 mcp.yaml.template 补一段再跑。")
        return None

    if not provider.usable():
        print(f"provider {PROVIDER!r} 不可用：{provider.unusable_reason()}")
        return None

    return provider


async def _run(provider: MCPProviderSettings) -> int:
    manager = get_mcp_provider_manager()

    # 只连这一个 provider：tdx 是子进程，冒烟没必要为它付启动成本，
    # 也不该因为它的路径没配好把这次验证拖成失败。
    narrowed = MCPSettings(enabled=True, providers={PROVIDER: provider})

    await manager.start(narrowed)

    try:
        if not manager.is_connected(PROVIDER):
            return _fail(
                "连接", "provider 未连上（看上面的 ERROR 堆栈："
                        "多半是服务没起，或 url / transport 写错）",
            )

        _ok("连接", provider.url)

        tools = manager.tools_by_provider().get(PROVIDER) or []
        names = [tool_name(t) for t in tools]
        _ok("tools/list", f"{len(names)} 个：{', '.join(sorted(names))}")

        if not tools:
            return _fail("tools/list", "一个工具都没有")

        leaked = [n for n in FORBIDDEN if provider.tools.admits(n)]
        if leaked:
            return _fail("allowlist", f"写类工具没被拦住：{leaked}")

        _ok("allowlist", f"{len(FORBIDDEN)} 个写类/运维类工具均不可见")

        # 这一条是为一次真实故障加的：list_periods 曾返回
        # "properties": null，OpenAI 兼容接口在**请求体校验**阶段就 400，
        # 整轮对话失败，日志里看着像模型的问题。客户端现在会兜底规范化，
        # 但那只是止损 —— 冒烟要盯住 server 本身，别再退化回去。
        null_props = [
            tool_name(tool) for tool in tools
            if _has_null_properties(tool)
        ]

        if null_props:
            return _fail(
                "inputSchema",
                f"这些工具的 properties 是 null，模型 API 会 400：{null_props}",
            )

        _ok("inputSchema", "所有工具的 properties 都是对象")

        schema = next(
            (tool_input_schema(t) for t in tools
             if tool_name(t) == "query_market_data"),
            {},
        )
        required = schema.get("required") or []
        _ok("必填字段", f"query_market_data → {required}")

        # 只连上还不够：工具要真的进进程级注册表，模型才拿得到。
        # 这一步用的是 bootstrap 里线上那套注册（allowlist 二次过滤）。
        from app.bootstrap.mcp import register_mcp_tools
        from app.services.tool.registry import get_tool_registry

        registered = register_mcp_tools(
            manager=manager, settings=narrowed,
        )
        keys = sorted(
            key for key in get_tool_registry().keys()
            if key.startswith(f"{PROVIDER}.")
        )

        if registered != len(keys) or not keys:
            return _fail(
                "注册表", f"注册 {registered} 个，实际拿到 {len(keys)} 个",
            )

        _ok("注册表", f"{registered} 个 tool_key，例如 {keys[0]}")

        datasets = _as_json(
            await manager.call_tool(PROVIDER, "list_datasets", {})
        )
        if not isinstance(datasets, dict):
            return _fail("list_datasets", f"返回不是 JSON 对象：{datasets!r}")

        count = datasets.get("count")
        _ok("list_datasets", f"{count} 个数据集")

        result = _as_json(
            await manager.call_tool(
                PROVIDER,
                "query_market_data",
                {
                    "dataset_key": "cn_stock_bar_d1",
                    "instruments": ["600519.SH"],
                    "start_time": "2026-09-01",
                    "end_time": "2026-10-31",
                    "adjustment": "qfq",
                    "limit": 5,
                },
            )
        )

        if not isinstance(result, dict):
            return _fail("query_market_data", f"返回不是 JSON 对象：{result!r}")

        rows = result.get("bars") or []
        diagnostics = result.get("diagnostics") or {}

        _ok(
            "query_market_data",
            f"rows={result.get('rowCount')} source={result.get('source')} "
            f"complete={result.get('complete')} "
            f"providerCalls={diagnostics.get('providerCalls')}",
        )

        if not rows:
            # 空结果不算失败：本地没同步过这段区间时是正常现象，
            # 但要把它说清楚，别让人误以为接入坏了。
            print(
                "         （没返回 bar：该区间本地无数据且本次没回源。"
                "先跑一次同步，或确认 market.tushare 可用。）"
            )
        else:
            _ok("首行", json.dumps(rows[0], ensure_ascii=False))

        return 0

    finally:
        await manager.stop()


def main() -> int:
    provider = _pick_provider()

    if provider is None:
        return 1

    print(f"provider={PROVIDER} transport={provider.transport} url={provider.url}")

    try:
        return asyncio.run(_run(provider))

    except ImportError as exc:
        print(f"缺少依赖：{exc}（pip install -r requirements.txt）")
        return 1

    except Exception as exc:  # noqa: BLE001 - 冒烟的职责就是把异常打清楚
        print(f"[fail] 未预期异常：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
