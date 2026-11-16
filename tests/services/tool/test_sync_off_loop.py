"""
同步的扫描阶段必须在事件循环之外跑。

`ToolDiscoverer.discover()` 是同步的：本地扫描器要 import 包，
MCP 扫描器要等 `tools/list`（`list_provider_tools` 内部起了线程，
但**等它结果**的那一侧仍然阻塞）。直接在 async 路径里调用会把
循环按住整段扫描时间 —— 线上表现为 loop watchdog 报"事件循环被
卡住"，同时所有请求、SSE 心跳、Redis 读超时定时器一起停摆。

这条用例钉住的是线程归属，不是耗时：扫描慢没问题，占住循环不行。
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from app.services.tool.application.tool_discovery_service_impl import (
    ToolDiscoveryServiceImpl,
)


class _NoopDefinitions:

    async def list_active(self):
        return []

    async def mark_missing(self, tool_ids, *, now):  # pragma: no cover
        return 0

    async def count_for_tools(self, tool_ids):  # pragma: no cover
        return 0


class _ThreadProbeDiscoverer:

    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def discover(self, **kwargs):
        self.thread_ids.append(threading.get_ident())
        return []


async def test_discovery_runs_off_the_event_loop() -> None:
    discoverer = _ThreadProbeDiscoverer()

    service = ToolDiscoveryServiceImpl(
        tool_definition_repository=_NoopDefinitions(),
        tool_documentation_repository=SimpleNamespace(),
        discoverer=discoverer,
    )

    await service.sync()

    # 这里（协程恢复的位置）就是事件循环线程。
    assert discoverer.thread_ids
    assert threading.get_ident() not in discoverer.thread_ids
