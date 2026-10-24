"""
开发用 mock 工具：固定五个、只读、无副作用。

为什么单独造一套
--------------
数据库里的能力 -> 工具关联需要种子数据；Phase 0 不想让前端联调依赖
"某条 capability 恰好挂了某个 tool_definition"。这五个工具不查库、不写
文件、不访问外部服务，只返回可预测的 mock 数据，因此可以稳定地验证：

    分类 -> capability_needs -> mock capability resolver
         -> tool_keys -> bind_tools -> mock model 返回 tool_calls
         -> tools 节点 fanout -> 工具执行 -> agent 下一轮

`app.mock.tools` **不在** `TOOL_SOURCE_PACKAGES` 里：它只在 --mock 启动时
由 app/mock/runtime.py 主动 scan 一次，生产注册表不会出现这五条。

`TOOL_SOURCE_REF = "mock"` 让 tool_key 变成 `mock.echo` 这种稳定形式；
provider 侧的 wire name 仍由 select_specs / wire_name 转换，不在这里处理。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

__all__ = [
    "TOOL_SOURCE_REF",
    "calc",
    "current_time",
    "echo",
    "lookup",
    "ping",
]

#: 五个 mock 工具的 source_ref；tool_key = mock.<函数名>。
TOOL_SOURCE_REF = "mock"


@tool
async def echo(text: str = "ping") -> dict[str, Any]:
    """
    回显一段文本。用于验证工具调用链路与参数传递。

    参数 text 是任意字符串。
    """

    return {"tool": "mock.echo", "text": text}


@tool
async def lookup(query: str = "mock query", limit: int = 3) -> dict[str, Any]:
    """
    模拟一次只读检索。不访问数据库、不访问网络。

    参数 query 是查询文本；limit 是返回条数（1~10）。
    """

    if limit < 1 or limit > 10:
        raise ValueError("limit 必须在 1 ~ 10 之间。")

    return {
        "tool": "mock.lookup",
        "query": query,
        "limit": limit,
        "items": [
            {
                "id": f"mock-{index}",
                "title": f"mock result {index}",
                "snippet": f"关于 {query} 的模拟结果 {index}",
            }
            for index in range(1, limit + 1)
        ],
    }


@tool
async def calc(expression: str = "1+1") -> dict[str, Any]:
    """
    模拟一次计算。**不执行任意代码**，只回显表达式与一个固定结果。

    参数 expression 是用户想计算的表达式。
    """

    return {
        "tool": "mock.calc",
        "expression": expression,
        "result": 42,
        "note": "mock 工具：没有真正求值",
    }


@tool
async def current_time(timezone_name: str = "Asia/Shanghai") -> dict[str, Any]:
    """
    返回当前 UTC 时间与请求的时区名。用于验证无参数/字符串参数工具。

    参数 timezone_name 是 IANA 时区名；mock 不做时区换算。
    """

    now = datetime.now(timezone.utc)

    return {
        "tool": "mock.current_time",
        "timezone": timezone_name,
        "iso8601_utc": now.isoformat(timespec="seconds"),
        "note": "mock 工具：只给 UTC，不做时区换算",
    }


@tool
async def ping() -> dict[str, Any]:
    """无参数探针：返回 pong。用于验证零参数工具的绑定与执行。"""

    return {"tool": "mock.ping", "pong": True}
