"""
remove_missing=true 必须能删掉**上一轮**就已经是 MISSING 的行。

回归背景（真实事故，2026-11-03）
------------------------------
`_retire_absent` 原来只拿 `list_active()` 做反向比对，而 MISSING 的行
不在 ACTIVE 里 —— 于是这些行既不会被重新标记、也不会被删除：

    前端点「执行同步（remove_missing=true）」
    → ToolSyncResult: scanned 44 / unchanged 44 / missing 0
    → 刷新，缺失的工具仍然在列表里

用户看到的症状是"删除按钮没生效"，代码里的原因是"要删的那批行根本
没被查出来"。下面三个用例分别锁住：能删、按范围删、以及默认模式
仍然不重复计数。
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.services.tool.application.tool_discovery_service_impl import (
    ToolDiscoveryServiceImpl,
)
from app.services.tool.domain.schemas import (
    SourceState,
    SourceType,
    ToolSyncRequest,
)


def _definition(
        key: str,
        source_type: SourceType,
        ref: str,
        state: SourceState = SourceState.ACTIVE,
):
    return SimpleNamespace(
        tool_key=key,
        source_type=source_type,
        source_ref=ref,
        source_state=state,
        tool_id=uuid4(),
    )


class _FakeDefinitions:

    def __init__(self, rows) -> None:
        self._rows = rows
        self.marked: list = []
        self.deleted: list = []

    async def list_active(self):
        return [
            r for r in self._rows
            if r.source_state == SourceState.ACTIVE
        ]

    async def list_missing(self):
        return [
            r for r in self._rows
            if r.source_state == SourceState.MISSING
        ]

    async def mark_missing(self, tool_ids, *, now):
        self.marked.extend(tool_ids)
        return len(tool_ids)

    async def delete_many(self, tool_ids):
        self.deleted.extend(tool_ids)
        return len(tool_ids)


class _FakeDocumentations:

    def __init__(self, per_tool: int = 0) -> None:
        self.per_tool = per_tool
        self.counted: list = []

    async def count_for_tools(self, tool_ids):
        self.counted.extend(tool_ids)
        return len(tool_ids) * self.per_tool


class _EmptyDiscoverer:

    def discover(self, **kwargs):
        return []


def _service(rows, *, docs_per_tool: int = 0):
    definitions = _FakeDefinitions(rows)
    documentations = _FakeDocumentations(docs_per_tool)

    service = ToolDiscoveryServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=documentations,
        discoverer=_EmptyDiscoverer(),
    )

    return service, definitions, documentations


async def test_remove_missing_deletes_previously_missing_rows() -> None:
    """上一轮标成 MISSING 的行，本轮 remove_missing=true 要真的删掉。"""

    gone = _definition(
        "ytrader_market.list_boards",
        SourceType.MCP,
        "ytrader_market",
        SourceState.MISSING,
    )
    also_gone = _definition(
        "ytrader_market.get_board_members",
        SourceType.MCP,
        "ytrader_market",
        SourceState.MISSING,
    )

    service, definitions, _ = _service([gone, also_gone])

    result = await service.sync(
        ToolSyncRequest(remove_missing=True)
    )

    # 扫描结果是空的：这两行在代码里都已经不存在了。
    assert result.scanned == 0
    assert result.missing == 2
    assert set(definitions.deleted) == {
        gone.tool_id, also_gone.tool_id,
    }


async def test_remove_missing_counts_orphaned_documents() -> None:
    """被删掉的说明行数要如实报出来（CASCADE 是静默的）。"""

    gone = _definition(
        "ytrader_market.list_boards",
        SourceType.MCP,
        "ytrader_market",
        SourceState.MISSING,
    )

    service, definitions, documentations = _service(
        [gone], docs_per_tool=2
    )

    result = await service.sync(
        ToolSyncRequest(remove_missing=True)
    )

    assert documentations.counted == [gone.tool_id]
    assert result.orphaned_documentations == 2
    assert definitions.deleted == [gone.tool_id]


async def test_remove_missing_still_respects_source_scope() -> None:
    """删是不可逆的，范围收窄在这里同样必须生效。"""

    tdx_gone = _definition(
        "tdx.list_experts",
        SourceType.MCP,
        "tdx",
        SourceState.MISSING,
    )
    other_gone = _definition(
        "other.list_experts",
        SourceType.MCP,
        "other",
        SourceState.MISSING,
    )

    service, definitions, _ = _service([tdx_gone, other_gone])

    await service.sync(
        ToolSyncRequest(remove_missing=True, source_refs=["tdx"])
    )

    assert definitions.deleted == [tdx_gone.tool_id]


async def test_default_sync_does_not_recount_missing_rows() -> None:
    """remove_missing=false 时不重复报、也不删除（保持原行为）。"""

    gone = _definition(
        "ytrader_market.list_boards",
        SourceType.MCP,
        "ytrader_market",
        SourceState.MISSING,
    )

    service, definitions, _ = _service([gone])

    result = await service.sync(ToolSyncRequest())

    assert result.missing == 0
    assert definitions.marked == []
    assert definitions.deleted == []
