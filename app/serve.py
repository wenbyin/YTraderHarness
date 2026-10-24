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
import logging
import os
import sys

import uvicorn

__all__ = ["loop_factory", "serve"]


#: 日志级别。默认 INFO —— 见 `_configure_logging` 的说明。
LOG_LEVEL_ENV = "YTRADER_LOG_LEVEL"

#: 日志格式。用不着花哨：时间、级别、logger 名、正文。
LOG_FORMAT = (
    "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
)


def _configure_logging() -> None:
    """
    配置日志 —— **在此之前整个应用没有日志配置**。

    ⚠️ 这件事值得单独说，因为它一直静默失效
    --------------------------------------
    此前全仓库没有任何 `basicConfig` / `dictConfig` / `setLevel`。于是：

        root logger  level = WARNING，handlers = 0

    在这个状态下，Python 只会通过那条 `lastResort` 兜底把 **WARNING 及
    以上**打到 stderr（实测确认）。也就是说：

        logger.warning(...)   ✅ 出得来
        logger.info(...)      ❌ **全部被丢掉**

    后果是**全仓库写好的 `logger.info` 一句都没打印过** —— 包括那些
    "开始/进度/收尾/为什么跳过"的关键行。而它们恰恰是排查问题时最需要的：
    一个任务跑了几百秒，控制台却一个字都没有，只能靠猜。

    （实测踩过：一个评测任务跑了 527 秒，我让人去看日志，而实际上那里
    什么都不会有 —— 不是没写，是级别不对。）

    默认给 INFO
    -----------
    给 INFO 是因为那些 `logger.info` 本来就是**为了被看见**才写的。级别
    可以通过 `YTRADER_LOG_LEVEL` 调（`WARNING` 能回到从前的安静状态，
    `DEBUG` 更吵）。

    措辞上刻意**不设 `force=True`**：那会把 uvicorn 或调用方（PyCharm、
    测试）已经装好的 handler 拆掉。这里只补一个新的，谁先装谁生效不了
    就按 `basicConfig` 的既有语义来。
    """

    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()

    level = getattr(logging, level_name, None)

    if not isinstance(level, int):
        # 拼错了不该让服务起不来 —— 回落到 INFO 并说出来。
        level = logging.INFO
        print(
            f"⚠️ {LOG_LEVEL_ENV}={level_name!r} 不是合法级别，"
            "回落到 INFO",
            file=sys.stderr,
        )

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        stream=sys.stderr,
    )


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
        mock: bool | None = None,
) -> None:
    """
    启动服务。

    ``mock``：
        None  跟随 app/config/mock_model.yaml 的 enabled
        True  强制启用进程内 mock（--mock）
        False 强制关闭（--no-mock）

    用环境变量而不是函数参数穿透：``uvicorn.Config("app.main:app")``
    会在自己的导入路径里构造 app，lifespan 拿不到 CLI 参数；而环境变量
    在子进程 / reload 路径上也会继承。
    """

    if mock is not None:
        os.environ["MOCK_MODEL_ENABLED"] = "1" if mock else "0"

    # 记忆链路在 mock 模式下**默认开启**：L0/L1 的后台抽取与预召回
    # 都要真实跑一遍，才能验证任务链路。想把 agent 主链路和记忆任务
    # 分开压测时，显式设成 0：
    #
    #     YTRADER_MEMORY_CONTEXT=0 YTRADER_MEMORY_EXTRACTION=0
    #
    # 两个开关在消费侧的默认值都是 1（见 agent_builder_node 与
    # agent_run_dispatch），这里不再覆盖成 0。

    # ⚠️ 必须在建 uvicorn 之前 —— 见 `_configure_logging` 的说明。
    # 放在 `serve()` 里而不是模块级：测试与冒烟脚本 import `app.main`
    # 不该顺带把它们的日志level改掉。
    _configure_logging()

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
