"""
展示描述符的查找表：`tool_key` 与 wire name **双向**。

为什么需要两份键
---------------
同一次调用在两个地方用不同的名字：

    AIMessage.tool_call.name        wire name（`ytrader_market_search_instruments`）
    ToolMessage.name                tool_key（`ytrader_market.search_instruments`）
    投影层的 custom stream           tool_key

只建一种键，另一条路就查不到 presenter，症状是「实时藏了、历史又出现」
或者反过来。映射规则与投影层出站给模型的名字是同一个函数（`wire_name`），
不要各自拼一遍。

⚠️ 纯函数
--------
入参是一批仓储实体/描述符，不碰数据库 —— 调用方决定在哪段事务里查。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.services.tool.registry.selection import wire_name

__all__ = ["build_presentation_lookup"]


def build_presentation_lookup(
        rows: Iterable[Any],
) -> dict[str, Any]:
    """`tool_key -> 描述符` 与 `wire_name -> 描述符` 两张键并进一个表。"""

    lookup: dict[str, Any] = {}

    for row in rows:
        lookup[row.tool_key] = row
        lookup[wire_name(row.tool_key)] = row

    return lookup
