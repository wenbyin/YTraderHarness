"""
MCP provider 的纯逻辑测试。

不连真实 server：扫描器、结果转换、provider 查询都只依赖
「一个带 name / description / input_schema 的对象」，用 SimpleNamespace
就够。真正拉起 `tdx-link-mcp` 的验证在手动冒烟里做（见 README_TOOL）。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.common.mcp_config import (
    MCPProviderSettings,
    MCPSettings,
    MCPToolFilter,
)
from app.infra.agent.mcp_client import (
    to_tool_payload,
    tool_input_schema,
    tool_name,
)
from app.services.tool.application import mcp_provider_service_impl
from app.services.tool.application.mcp_provider_service_impl import (
    MCPProviderServiceImpl,
)
from app.services.tool.registry import mcp_scanner
from app.services.tool.registry.mcp_scanner import MCPToolScanner


def _tool(
        name: str,
        description: str | None = "d",
        schema: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema=schema or {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "array",
                    "description": "股票代码",
                },
            },
        },
        output_schema=None,
    )


# ============================================================
# 过滤
# ============================================================


def test_allowlist_is_default_deny() -> None:
    flt = MCPToolFilter(mode="allowlist")

    assert not flt.admits("execute_trade")

    flt.allow.append("get_market_info")

    assert flt.admits("get_market_info")
    assert not flt.admits("execute_trade")


def test_deny_wins_in_every_mode() -> None:
    flt = MCPToolFilter(
        mode="all", deny=["execute_trade"],
    )

    assert flt.admits("get_market_info")
    assert not flt.admits("execute_trade")


# ============================================================
# 扫描
# ============================================================


def test_scanner_emits_only_admitted_tools(
        monkeypatch,
) -> None:
    settings = MCPSettings(
        providers={
            "tdx": MCPProviderSettings(
                command="tdx-link-mcp",
                tools=MCPToolFilter(
                    mode="allowlist",
                    allow=["get_market_info"],
                ),
            ),
        },
    )

    monkeypatch.setattr(
        mcp_scanner,
        "list_provider_tools",
        lambda provider: [
            _tool("get_market_info"),
            _tool("execute_trade"),
        ],
    )

    manifests = MCPToolScanner(settings).scan()

    assert [m.tool_key for m in manifests] == [
        "tdx.get_market_info"
    ]

    manifest = manifests[0]

    assert manifest.source_type.value == "mcp"
    assert manifest.source_ref == "tdx"
    assert manifest.entrypoint == "mcp:tdx:get_market_info"
    assert manifest.input_schema["properties"]["codes"]
    # 参数说明按 JSON Pointer 抽出，管理接口的覆盖层用同一套路径。
    assert manifest.source_docstring["parameters"] == {
        "/codes": "股票代码",
    }


def test_scanner_survives_unreachable_provider(
        monkeypatch,
) -> None:
    settings = MCPSettings(
        providers={
            "down": MCPProviderSettings(command="missing"),
            "tdx": MCPProviderSettings(
                command="tdx-link-mcp",
                tools=MCPToolFilter(mode="all"),
            ),
        },
    )

    def fake_list(provider):
        if provider.command == "missing":
            raise RuntimeError("连不上")
        return [_tool("get_market_info")]

    monkeypatch.setattr(
        mcp_scanner, "list_provider_tools", fake_list,
    )

    manifests = MCPToolScanner(settings).scan()

    assert [m.tool_key for m in manifests] == [
        "tdx.get_market_info"
    ]


def test_scanner_is_off_without_config() -> None:
    assert MCPToolScanner(
        MCPSettings(enabled=False)
    ).scan() == []


# ============================================================
# 结果转换
# ============================================================


def test_structured_content_wins() -> None:
    result = SimpleNamespace(
        is_error=False,
        structured_content={"rows": [1]},
        content=[],
    )

    assert to_tool_payload(result) == {"rows": [1]}


def test_text_blocks_collapse() -> None:
    single = SimpleNamespace(
        is_error=False,
        structured_content=None,
        content=[SimpleNamespace(text="ok")],
    )

    assert to_tool_payload(single) == "ok"

    many = SimpleNamespace(
        is_error=False,
        structured_content=None,
        content=[
            SimpleNamespace(text="a"),
            SimpleNamespace(text="b"),
        ],
    )

    assert to_tool_payload(many) == ["a", "b"]


def test_tool_field_name_compatibility() -> None:
    old_sdk = SimpleNamespace(name="x", inputSchema={"type": "object"})

    assert tool_name(old_sdk) == "x"
    assert tool_input_schema(old_sdk) == {"type": "object"}


# ============================================================
# provider 查询
# ============================================================


class _FakeRepo:

    def __init__(self, counts=None) -> None:
        self._counts = counts or {}

    async def count_by_source_refs(self, source_refs):
        return {
            ref: self._counts.get(
                ref, {"active": 0, "missing": 0}
            )
            for ref in source_refs
        }


class _FakeManager:

    def __init__(self, connected: bool = True) -> None:
        self._connected = connected

    def is_connected(self, name: str) -> bool:
        return self._connected and name == "tdx"

    def tools_by_provider(self):
        if not self._connected:
            return {}

        return {
            "tdx": [
                _tool("get_market_info"),
                _tool("execute_trade"),
            ],
        }


async def test_provider_list_merges_config_runtime_and_db(
        monkeypatch,
) -> None:
    settings = MCPSettings(
        providers={
            "tdx": MCPProviderSettings(
                command="tdx-link-mcp",
                tools=MCPToolFilter(
                    mode="allowlist",
                    allow=["get_market_info"],
                ),
            ),
        },
    )

    monkeypatch.setattr(
        mcp_provider_service_impl,
        "get_mcp_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        mcp_provider_service_impl,
        "get_mcp_provider_manager",
        lambda: _FakeManager(connected=True),
    )

    service = MCPProviderServiceImpl(
        _FakeRepo({"tdx": {"active": 8, "missing": 2}})
    )

    response = await service.list_providers()

    item = response.items[0]

    assert item.name == "tdx"
    assert item.enabled is True
    assert item.connected is True
    # 两个工具里只放行一个。
    assert item.exposed_tool_count == 1
    assert item.tool_count == 10
    assert item.active_tool_count == 8
    assert item.missing_tool_count == 2


async def test_disabled_global_switch(monkeypatch) -> None:
    settings = MCPSettings(
        enabled=False,
        providers={"tdx": MCPProviderSettings(command="x")},
    )

    monkeypatch.setattr(
        mcp_provider_service_impl,
        "get_mcp_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        mcp_provider_service_impl,
        "get_mcp_provider_manager",
        lambda: _FakeManager(connected=False),
    )

    item = (
        await MCPProviderServiceImpl(_FakeRepo()).list_providers()
    ).items[0]

    assert item.enabled is False
    assert item.connected is False
    assert item.exposed_tool_count == 0
