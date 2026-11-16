"""
ytrader-core 的 MCP 接入冒烟。

跑法（core 侧不需要先起服务，stdio 会拉起子进程）：

    python -m app.agent.mcp_core_smoke

为什么单独一个冒烟
------------------
与 `mcp_market_smoke` 同一立场：MCP 的单元测试只覆盖纯逻辑（过滤、
结果转换、配置解析），**不连真实 server**。而这次接的是一条不同的
实现 —— ytrader-core 的 MCP 是手写 JSON-RPC（不依赖 mcp SDK），
协议帧、UTF-8 编码、isError 语义都得真连一次才知道对不对。

验的是这几件事：

    1. provider 能连上（SDK 客户端 vs 手写 server 的协议对得上）
    2. allowlist 生效：清单恰好是放行的那 10 个，没多没少
    3. 工具真进了进程级注册表（模型才拿得到）
    4. 真调得动：catalog_get / symbol_resolve 返回可解析 JSON
    5. 业务失败走的是 `isError`（客户端抛 MCPToolCallError），
       而不是把异常当结果回灌给模型

不依赖本项目数据库、模型、FastAPI —— 只要求 mcp SDK 装上、core
仓库在 D:\\workspace\\YTraderCore。core 自己的库连不上时，工具会
返回 isError，这里会如实报出来而不是假装通过。
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
    MCPToolCallError,
    get_mcp_provider_manager,
    tool_input_schema,
    tool_name,
)

#: mcp.yaml 里的 provider 名。改配置时同步改这里，别用通配。
PROVIDER = "ytrader_core"

#: 期望暴露的工具（与 mcp.yaml 的 allow 对齐 —— 两边漂移了要立刻发现）。
EXPECTED = {
    "catalog_get",
    "symbol_resolve",
    "bars_get",
    "indicator_preview",
    "strategy_get",
    "strategy_validate",
    "run_report",
    "strategy_apply",
    "strategy_publish",
    "run_backtest",
}

#: 这套 provider 不该出现的东西：别的 server 的工具名（说明串台了）、
#: 以及 core 明确没有的能力。
FORBIDDEN = ("execute_trade", "sync_data", "query_market_data")


def _as_json(payload: Any) -> Any:
    """工具结果尽量解析成结构（core 的每个工具都把 JSON 以文本块返回）。"""

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


def _has_null_properties(tool: Any) -> bool:
    """
    server 是否把 properties 写成了 null。

    要盯**原始** schema（`tool_input_schema` 会兜底规范化，看起来
    永远是对的）。这条坑在 ytrader-market 上真出现过：OpenAI 兼容
    接口在请求体校验阶段就 400，日志里却像模型的问题。
    """

    raw = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
    )

    return (
        isinstance(raw, dict)
        and "properties" in raw
        and raw["properties"] is None
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

    # 只连这一个 provider：tdx 是子进程、market 是远端，冒烟没必要为它们
    # 付启动成本，也不该因为它们没起把这次验证拖成失败。
    narrowed = MCPSettings(enabled=True, providers={PROVIDER: provider})

    await manager.start(narrowed)

    try:
        if not manager.is_connected(PROVIDER):
            return _fail(
                "连接",
                "provider 未连上（看上面的 ERROR 堆栈：多半是 core 仓库路径/"
                "venv 不对，或 `-m app.mcp` 的 cwd 没配）",
            )

        _ok("连接", f"{provider.command} {' '.join(provider.args)}")

        tools = manager.tools_by_provider().get(PROVIDER) or []
        names = {tool_name(t) for t in tools}

        _ok("tools/list", f"{len(names)} 个：{', '.join(sorted(names))}")

        if names != EXPECTED:
            return _fail(
                "tools/list",
                f"与 allowlist 不一致：多 {sorted(names - EXPECTED)} / "
                f"少 {sorted(EXPECTED - names)}。"
                f"core 改了工具集就同步改 EXPECTED 与 mcp.yaml。",
            )

        leaked = [n for n in FORBIDDEN if n in names]
        if leaked:
            return _fail("allowlist", f"不该出现的工具：{leaked}")

        _ok("allowlist", f"{len(FORBIDDEN)} 个别家工具均不可见")

        null_props = [tool_name(t) for t in tools if _has_null_properties(t)]
        if null_props:
            return _fail(
                "inputSchema",
                f"这些工具的 properties 是 null，模型 API 会 400：{null_props}",
            )

        _ok("inputSchema", "所有工具的 properties 都是对象")

        # 只连上还不够：工具要真的进进程级注册表，模型才拿得到。
        # 这一步用的是 bootstrap 里线上那套注册（allowlist 二次过滤）。
        from app.bootstrap.mcp import register_mcp_tools
        from app.services.tool.registry import get_tool_registry

        registered = register_mcp_tools(manager=manager, settings=narrowed)

        keys = sorted(
            key for key in get_tool_registry().keys()
            if key.startswith(f"{PROVIDER}.")
        )

        if registered != len(keys) or not keys:
            return _fail(
                "注册表", f"注册 {registered} 个，实际拿到 {len(keys)} 个",
            )

        _ok("注册表", f"{registered} 个 tool_key，例如 {keys[0]}")

        # 走到这里只证明"注册表里有这个名字"。真正要模型用得动，还得
        # 过一遍 langchain 的 args_schema 校验：core 的 inputSchema 写得
        # 很紧（enum / additionalProperties:false / required），schema
        # 转不成 args_schema 的话，注册成功但一调就炸。
        spec = get_tool_registry().get(f"{PROVIDER}.symbol_resolve")

        if spec is None or spec.tool is None:
            return _fail("ainvoke", "注册表里拿不到 symbol_resolve 的可执行对象")

        through_tool = _as_json(
            await spec.tool.ainvoke({"codes": ["600000"]})
        )

        codes = [i.get("symbol") for i in (through_tool.get("items") or [])]

        if codes != ["600000.SH"]:
            return _fail("ainvoke", f"经 StructuredTool 调用结果不对：{through_tool!r}")

        _ok("ainvoke", f"经 StructuredTool 调通：{codes}")

        catalog = _as_json(
            await manager.call_tool(PROVIDER, "catalog_get", {"kind": "market"})
        )

        if not isinstance(catalog, dict):
            return _fail("catalog_get", f"返回不是 JSON 对象：{catalog!r}")

        periods = catalog.get("periods") or []
        fields = catalog.get("fields") or []

        if not periods or not fields:
            return _fail(
                "catalog_get", f"periods={periods} fields={fields}",
            )

        _ok(
            "catalog_get",
            f"{len(periods)} 个周期，{len(fields)} 个字段，"
            f"首个字段写法 {fields[0].get('in_condition')}",
        )

        resolved = _as_json(
            await manager.call_tool(
                PROVIDER, "symbol_resolve", {"codes": ["600000", "sh600519"]},
            )
        )

        symbols = [
            item.get("symbol") for item in (resolved.get("items") or [])
        ]

        if symbols != ["600000.SH", "600519.SH"]:
            return _fail("symbol_resolve", f"规范化结果不对：{resolved!r}")

        _ok("symbol_resolve", f"{symbols}")

        # 业务失败必须是 isError（客户端抛异常），而不是一条"成功"的
        # 错误内容 —— 执行守卫靠"抛没抛"判定 FAILED。
        try:
            await manager.call_tool(
                PROVIDER, "catalog_get", {"kind": "nope"},
            )

        except MCPToolCallError as exc:
            _ok("isError", f"拼错的 kind 被拦下：{str(exc)[:60]}")

        else:
            return _fail(
                "isError",
                "拼错的 kind 返回了成功，说明 isError 没接住（业务失败会被"
                "当成正常结果回灌给模型）",
            )

        return 0

    finally:
        await manager.stop()


def main() -> int:

    provider = _pick_provider()

    if provider is None:
        return 1

    print(
        f"provider={PROVIDER} transport={provider.transport} "
        f"cwd={provider.cwd}"
    )

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
