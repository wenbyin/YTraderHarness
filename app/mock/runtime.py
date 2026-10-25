"""
Mock 模型服务的进程级生命周期。

组合根（``app/main.py`` 的 lifespan）只调用两个函数：

    start_mock_server_if_enabled() -> MockModelServer | None
    stop_mock_server(server)

这里额外做两件与"能不能连上"有关的接线：

1. 把 ``MOCK_MODEL_BASE_URL`` 写进环境变量，供 ``providers.yaml`` 的
   ``mock.base_url: "${MOCK_MODEL_BASE_URL:-...}"`` 展开。这样 app 配置里的
   端口与 LLM 客户端实际请求的地址不会各写一份。
2. 把回环地址加进 ``NO_PROXY``。Windows 的系统代理会劫持 httpx 对
   127.0.0.1 的请求（OpenAI SDK 走 httpx），症状是 mock 明明在监听、
   客户端却连不上或走代理超时。
"""

from __future__ import annotations

import logging
import os
import threading

from app.mock.model_server import MockModelServer
from app.mock.settings import is_mock_enabled, load_mock_config

__all__ = [
    "start_mock_server_if_enabled",
    "stop_mock_server",
]

logger = logging.getLogger(__name__)

_server: MockModelServer | None = None
_lock = threading.Lock()


def start_mock_server_if_enabled() -> MockModelServer | None:
    """
    启用时启动进程内 mock；否则返回 None。

    启动失败会抛异常 —— 显式要求 mock 却悄悄退回真实模型，是比启动失败
    更糟的结果（前端以为在测 mock，实际在花钱调云端）。
    """

    global _server

    if not is_mock_enabled():
        return None

    with _lock:
        if _server is not None:
            return _server

        config = load_mock_config()
        client_host = (
            "127.0.0.1"
            if config.host in {"0.0.0.0", "::", ""}
            else config.host
        )

        os.environ["MOCK_MODEL_BASE_URL"] = (
            f"http://{client_host}:{config.port}/v1"
        )
        _ensure_no_proxy()

        server = MockModelServer(config)
        server.start()

        # mock 工具只在 --mock 时进注册表：它们不在 TOOL_SOURCE_PACKAGES
        # 里，因此生产启动不会出现这五条。scan 是增量的，后面
        # register_react_graph 的默认 scan 不会把它们清掉。
        _register_mock_tools()

        _server = server

        logger.warning(
            "Mock model server 已启用（进程内守护线程）："
            "http://%s:%d/v1。仅用于开发 / 前端联调，生产环境不要打开。",
            config.host,
            config.port,
        )

        return server


def stop_mock_server(server: MockModelServer | None = None) -> None:
    """停止 mock；重复调用无副作用。"""

    global _server

    target = server if server is not None else _server

    if target is None:
        return

    try:
        target.stop()

    except Exception:
        logger.exception("停止 mock model server 失败。")

    finally:
        if _server is target:
            _server = None


def _register_mock_tools() -> None:
    """把 app/mock/tools.py 里的五个固定工具登记进进程级 ToolRegistry。"""

    from app.services.tool.registry.registry import get_tool_registry

    # scan 接收的是**包**名并遍历它的子模块；app.mock.tools 是模块，
    # 传它本身会安静返回空。传 app.mock 包，tools.py 会在遍历中被导入。
    specs = get_tool_registry().scan(packages=("app.mock",))
    mock_keys = sorted(
        spec.tool_key
        for spec in specs
        if spec.tool_key.startswith("mock.")
    )

    logger.info("Mock tools 已登记：%s", "、".join(mock_keys) or "（无）")


def _ensure_no_proxy() -> None:
    """把回环地址追加到 NO_PROXY / no_proxy，不覆盖用户已有配置。"""

    for name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(name, "")
        entries = [
            item.strip()
            for item in current.split(",")
            if item.strip()
        ]

        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in entries:
                entries.append(host)

        os.environ[name] = ",".join(entries)
