"""
运行期禁用：过滤点、缓存与失效时机。

这三条是这次改动的要害：

    1. 禁用只发生在 tool_keys_for_capabilities（主 agent / planner /
       skill 共用的唯一 DB -> key 解析点），下游 select_specs 不查库；
    2. 快照进程内只加载一次，除非写路径 invalidate；
    3. invalidate 挂在 post_commit 上，回滚不能把快照清掉。
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.common.service.base import ServiceContext
from app.services.tool.application.tool_definition_state_service_impl import (
    ToolDefinitionStateServiceImpl,
)
from app.services.tool.application.tool_query_service_impl import (
    ToolQueryServiceImpl,
)
from app.services.tool.registry.state import get_disabled_tool_cache


@pytest.fixture(autouse=True)
def _reset_disabled_cache():
    """快照是进程级单例，测试之间必须清干净。"""

    get_disabled_tool_cache().clear()
    yield
    get_disabled_tool_cache().clear()


class _FakeDefinitions:
    def __init__(self, rows=None, definition=None):
        self.rows = list(rows or [])
        self.definition = definition

    async def list_active_by_classification_codes(self, codes):
        return list(self.rows)

    async def get(self, tool_id):
        if self.definition is None:
            return None
        return (
            self.definition
            if tool_id == self.definition.tool_id
            else None
        )


class _FakeStates:
    def __init__(self, disabled=None):
        self.disabled = set(disabled or ())
        self.loads = 0

    async def list_disabled_keys(self):
        self.loads += 1
        return set(self.disabled)

    async def disabled_keys_for(self, tool_keys):
        return self.disabled & set(tool_keys)

    async def disable(self, tool_key, *, disabled_by, now):
        self.disabled.add(tool_key)
        return None

    async def enable(self, tool_key):
        existed = tool_key in self.disabled
        self.disabled.discard(tool_key)
        return existed


def _query_service(definitions, states) -> ToolQueryServiceImpl:
    return ToolQueryServiceImpl(
        tool_definition_repository=definitions,
        tool_definition_state_repository=states,
        tool_documentation_repository=SimpleNamespace(),
        tool_classification_repository=SimpleNamespace(),
    )


def _row(tool_key: str):
    return SimpleNamespace(tool_key=tool_key)


async def test_only_explicitly_disabled_keys_are_filtered() -> None:
    definitions = _FakeDefinitions(
        [_row("document.list"), _row("document.count")]
    )
    states = _FakeStates({"document.count"})

    keys = await _query_service(
        definitions, states
    ).tool_keys_for_capabilities(["document"])

    assert keys == ["document.list"]


async def test_snapshot_is_loaded_once_across_runs() -> None:
    definitions = _FakeDefinitions([_row("document.list")])
    states = _FakeStates()

    service = _query_service(definitions, states)

    await service.tool_keys_for_capabilities(["document"])
    await service.tool_keys_for_capabilities(["document"])
    await service.tool_keys_for_capabilities(["document"])

    assert states.loads == 1


async def test_stale_snapshot_is_not_written_back_after_invalidation() -> None:
    """
    加载与 PATCH 并发：第一次读到的旧值不能写回快照。

    模拟：第一次读期间写路径提交并 invalidate，读到的是旧快照；
    服务应重读，最终返回/缓存的是新值。
    """

    definitions = _FakeDefinitions(
        [_row("document.list"), _row("document.count")]
    )

    cache = get_disabled_tool_cache()

    class _RacingStates(_FakeStates):
        def __init__(self):
            super().__init__({"document.list"})
            self.calls = 0

        async def list_disabled_keys(self):
            self.calls += 1

            if self.calls == 1:
                stale = set(self.disabled)
                # 这次读与写回之间，另一个请求提交了新的禁用状态。
                self.disabled = {"document.count"}
                cache.invalidate()
                return stale

            return set(self.disabled)

    states = _RacingStates()

    keys = await _query_service(
        definitions, states
    ).tool_keys_for_capabilities(["document"])

    assert keys == ["document.list"]
    assert states.calls == 2
    assert cache.disabled_keys() == frozenset({"document.count"})


async def test_enable_removes_the_row_and_invalidates_after_commit() -> None:
    tool_id = uuid4()
    definition = SimpleNamespace(
        tool_id=tool_id, tool_key="document.list",
    )

    definitions = _FakeDefinitions(definition=definition)
    states = _FakeStates({"document.list"})

    cache = get_disabled_tool_cache()
    cache.replace({"document.list"})

    ctx = ServiceContext()

    service = ToolDefinitionStateServiceImpl(
        tool_definition_repository=definitions,
        tool_definition_state_repository=states,
        ctx=ctx,
    )

    result = await service.set_enabled(tool_id, True)

    assert result.tool_key == "document.list"
    assert result.enabled is True
    assert states.disabled == set()

    # 事务还没提交：快照必须还在 —— 回滚时不能"提前"失效。
    assert cache.loaded

    for hook in ctx.post_commit:
        await hook()

    assert not cache.loaded


async def test_disable_writes_the_row() -> None:
    tool_id = uuid4()
    definition = SimpleNamespace(
        tool_id=tool_id, tool_key="document.count",
    )

    definitions = _FakeDefinitions(definition=definition)
    states = _FakeStates()

    service = ToolDefinitionStateServiceImpl(
        tool_definition_repository=definitions,
        tool_definition_state_repository=states,
        ctx=ServiceContext(),
    )

    result = await service.set_enabled(tool_id, False)

    assert result.enabled is False
    assert states.disabled == {"document.count"}
