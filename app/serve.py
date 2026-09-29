"""
服务启动逻辑（唯一的一份）。

为什么不能直接用 uvicorn.run() 或 uvicorn CLI
--------------------------------------------
Windows 上 psycopg 的异步模式**只能**跑在 SelectorEventLoop 上，
而 uvicorn 从 0.36 起改用 loop_factory，**完全绕过
asyncio.set_event_loop_policy()**：

    # uvicorn/loops/asyncio.py
    def asyncio_loop_factory(use_subprocess: bool = False):
        if sys.platform == "win32" and not use_subprocess:
            return asyncio.ProactorEventLoop     # <- 默认走这条
        return asyncio.SelectorEventLoop         # <- 只有 reload/workers 才走这条

所以以下做法在 Windows 上**全都无效**：

    1. `uvicorn app.main:app`
       —— CLI 先建循环再 import app，且它根本不看 policy

    2. `uvicorn.run(..., loop="asyncio")`
       —— 正是上面那段代码，Windows 非 subprocess 时返回 Proactor

    3. 在 app.main 模块级 set_event_loop_policy(...)
       —— 循环可能已经建好了，policy 改不动它

唯一可靠的做法是自己把 loop_factory 传给 asyncio.run（见 _run）。

所有入口都走本模块，避免再出现"某个入口忘了处理"：

    run_server.py        命令行
    app/main.py          直接 python app/main.py / PyCharm 里点运行
    app.serve            被上面两者调用

生产上跑在 Linux 没有这个问题（Linux 默认就是 selector），
但既然一份代码能同时覆盖，就不必分平台写两套入口。
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn

__all__ = ["loop_factory", "serve"]


def loop_factory() -> type[asyncio.AbstractEventLoop]:
    """
    事件循环类型。

    Windows 必须用 SelectorEventLoop（psycopg 不支持 Proactor）。
    Linux/macOS 默认就是 selector，显式指出来只是保持一致。
    """

    return asyncio.SelectorEventLoop


def serve(
        *,
        host: str = "localhost",
        port: int = 8080,
        reload: bool = False,
) -> None:

    config = uvicorn.Config(
        "app.main:app",
        host=host,
        port=port,
        loop="asyncio",
        reload=False,
    )

    if reload:
        # reload 要起 supervisor 子进程，子进程里的循环类型不受控，
        # checkpointer 会连不上。Windows 上建议别开。
        print(
            "警告：--reload 会走 uvicorn 的 supervisor 路径，"
            "Windows 上事件循环类型不受控，checkpointer 可能连不上。",
            file=sys.stderr,
        )
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=True,
            loop="asyncio",
        )
        return

    server = uvicorn.Server(config)

    # 关键：不用 server.run()（它会用 uvicorn 自己的 loop_factory，
    # Windows 上拿到 ProactorEventLoop），而是自己指定。
    asyncio.run(
        server.serve(),
        loop_factory=loop_factory(),
    )
