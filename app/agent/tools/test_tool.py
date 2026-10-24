"""
实验工具：只睡眠，不做任何业务操作。

用途
----
验证 timeout / retry 处理器：

    sleep_for(seconds=30)   默认睡 30 秒，用来触发 timeout

它被放在 `app.agent.tools` 下，因此会被 LangchainToolScanner 自动
扫到，`tool_key` 推导为 `agent.sleep_for`。

⚠️ 不要把默认值改小
-------------------
它就是给"睡眠 30 秒"这个实验用的。要测更短的时间就显式传参数；
改默认值会让实验结论与文档对不上。

⚠️ 不产生副作用
-------------
只 `await asyncio.sleep(...)`，没有外部调用。因此它适合用来观察：
    - timeout 取消是否真的生效
    - retry 是否按配置重新执行
    - 执行守卫把超时记成 UNKNOWN
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

__all__ = ["sleep_for"]


#: 允许的最大睡眠秒数。防止模型真的睡一小时把运行卡死。
MAX_SECONDS = 300.0


@tool
async def sleep_for(seconds: float = 30.0) -> dict:
    """
    实验工具：睡眠指定秒数后返回。默认 30 秒，不做任何业务操作。

    ⚠️ 只在用户明确要求验证超时 / 重试时调用；普通对话不要调用它。
    参数 seconds 的取值范围是 0 ~ 300 秒。
    """

    if seconds < 0 or seconds > MAX_SECONDS:
        raise ValueError(
            f"seconds 必须在 0 ~ {MAX_SECONDS:.0f} 之间，收到 {seconds}。"
        )

    await asyncio.sleep(seconds)

    return {
        "slept_seconds": seconds,
        "note": "实验工具：没有产生任何业务副作用",
    }
