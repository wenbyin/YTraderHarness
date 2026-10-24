"""
按来源收窄同步时，对账也必须收窄。

MCP 接入后这条路径会变得常见（前端只想刷新一个 provider）。
如果 `_retire_absent` 拿全表比对，一次 `source_types=["mcp"]` 会把
所有本地工具判成「这次没扫到」并转 MISSING —— 它们只是没参与本轮
同步，不是消失了。
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.services.tool.application.tool_discovery_service_impl import (
    ToolDiscoveryServiceImpl,
)
from app.services.tool.domain.schemas import (
    SourceType,
    ToolSyncRequest,
)


def _definition(key: str, source_type: SourceType, ref: str):
    return SimpleNamespace(
        tool_key=key,
        source_type=source_type,
        source_ref=ref,
        tool_id=uuid4(),
    )


class _FakeDefinitions:

    def __init__(self, rows) -> None:
        self._rows = rows
        self.marked: list = []

    async def list_active(self):
        return list(self._rows)

    async def mark_missing(self, tool_ids, *, now):
        self.marked.extend(tool_ids)
        return len(tool_ids)

    async def count_for_tools(self, tool_ids):  # pragma: no cover
        return 0


class _EmptyDiscoverer:

    def discover(self, **kwargs):
        return []


def _service(rows) -> tuple[ToolDiscoveryServiceImpl, _FakeDefinitions]:
    definitions = _FakeDefinitions(rows)

    service = ToolDiscoveryServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=SimpleNamespace(),
        discoverer=_EmptyDiscoverer(),
    )

    return service, definitions


async def test_sync_by_source_type_only_retires_that_type() -> None:
    local = _definition(
        "document.list", SourceType.LOCAL_METHOD, "document",
    )
    mcp = _definition(
        "tdx.get_market_info", SourceType.MCP, "tdx",
    )

    service, definitions = _service([local, mcp])

    result = await service.sync(
        ToolSyncRequest(source_types=["mcp"])
    )

    assert result.missing == 1
    assert definitions.marked == [mcp.tool_id]


async def test_sync_by_source_ref_only_retires_that_ref() -> None:
    tdx = _definition(
        "tdx.get_market_info", SourceType.MCP, "tdx",
    )
    other = _definition(
        "other.get_market_info", SourceType.MCP, "other",
    )

    service, definitions = _service([tdx, other])

    await service.sync(
        ToolSyncRequest(source_refs=["tdx"])
    )

    assert definitions.marked == [tdx.tool_id]


async def test_full_sync_keeps_previous_behaviour() -> None:
    local = _definition(
        "document.list", SourceType.LOCAL_METHOD, "document",
    )
    mcp = _definition(
        "tdx.get_market_info", SourceType.MCP, "tdx",
    )

    service, definitions = _service([local, mcp])

    result = await service.sync(ToolSyncRequest())

    assert result.missing == 2
    assert set(definitions.marked) == {
        local.tool_id, mcp.tool_id,
    }
