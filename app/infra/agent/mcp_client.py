"""
MCP 客户端：连接、列工具、调工具。

这一层**不碰**本项目的 ToolSpec / ToolRegistry —— 它只回答两个问题：

    server 现在暴露了哪些工具？      list_provider_tools()
    调某个工具，结果是什么？         MCPProviderManager.call_tool()

把 MCP 工具包装成 langchain `StructuredTool` 并注册进进程级注册表，
是**组合根**的事（`app/bootstrap/mcp.py`）。分开的理由是依赖方向：
`app/services/tool/registry` 需要调用本模块做扫描，而注册又需要
registry —— 合在一起就是环。现在 scanner → 本模块 → 配置，
注册由 bootstrap 从外面接起来，两边都不反向依赖。

两条连接路径
------------
常驻 session（执行）
    应用启动时连上，全程复用。每次工具调用都新起一个 `tdx-link-mcp`
    进程太贵，而且 server 侧的会话状态（实时订阅、专家配置）会丢。

一次性 session（扫描）
    `list_provider_tools()` 在独立线程里起一个短命会话，用完即关。
    它服务于同步接口（`ToolScanner.scan()` 是同步的），必须在
    **另一个线程**里跑事件循环 —— 在 FastAPI 的请求循环里直接
    `asyncio.run` 会报「loop is already running」。

结果转换
--------
MCP 的 `CallToolResult` 是内容块列表，不是 JSON。这里把
`structured_content`（有则优先）或文本块收成一个 Python 值，
交给执行器落库 / 回灌模型。`is_error` 转成异常而不是返回值 ——
执行守卫要靠「抛没抛」区分 FAILED 与 SUCCESS。
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from app.common.mcp_config import (
    MCPProviderSettings,
    MCPSettings,
    get_mcp_settings,
)

__all__ = [
    "MCPProviderManager",
    "MCPToolCallError",
    "build_structured_tool",
    "get_mcp_provider_manager",
    "list_provider_tools",
    "reset_mcp_provider_manager",
    "tool_description",
    "tool_input_schema",
    "tool_name",
    "to_tool_payload",
]

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

except ImportError as exc:  # pragma: no cover - 依赖缺失时走这条
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR: Exception | None = exc
else:
    _MCP_IMPORT_ERROR = None


class MCPToolCallError(RuntimeError):
    """server 返回 `is_error=true`。"""

    def __init__(self, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


# ============================================================
# 工具描述符（兼容不同版本的字段命名）
# ============================================================
#
# mcp SDK 的 pydantic 模型字段是 snake_case（input_schema），而协议
# 上是 camelCase（inputSchema）。这里两种都认，避免被一次 SDK 升级
# 打断 —— 反正取的是同一个值。


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def tool_description(tool: Any) -> str | None:
    description = getattr(tool, "description", None)

    return str(description) if description else None


def tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
    )

    return dict(schema) if isinstance(schema, dict) else {
        "type": "object",
        "properties": {},
    }


# ============================================================
# 结果转换
# ============================================================


def _is_error(result: Any) -> bool:
    value = getattr(result, "is_error", None)

    if value is None:
        value = getattr(result, "isError", None)

    return bool(value)


def to_tool_payload(result: Any) -> Any:
    """
    `CallToolResult` -> 可 JSON 序列化的值。

    `structured_content` 优先：它是 server 明确给出的结构化结果，
    比从文本块里猜可靠。没有则按内容块收：单个文本直接给字符串，
    多个给列表（非文本块保留 `model_dump`）。
    """

    structured = getattr(result, "structured_content", None)

    if structured is None:
        structured = getattr(result, "structuredContent", None)

    if structured is not None:
        return structured

    blocks: list[Any] = []

    for block in getattr(result, "content", None) or []:

        text = getattr(block, "text", None)

        if text is not None:
            blocks.append(text)
            continue

        dump = getattr(block, "model_dump", None)

        if callable(dump):
            blocks.append(dump(mode="json"))
        else:
            blocks.append(str(block))

    if not blocks:
        return None

    return blocks[0] if len(blocks) == 1 else blocks


def _error_text(result: Any) -> str:
    payload = to_tool_payload(result)

    if isinstance(payload, str):
        return payload

    return str(payload)


# ============================================================
# 常驻会话
# ============================================================


@dataclass
class _ProviderRuntime:
    name: str
    settings: MCPProviderSettings
    session: Any
    tools: list[Any]
    stack: AsyncExitStack
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _require_sdk() -> None:
    if _MCP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "未安装 mcp SDK，MCP provider 不可用："
            f"{_MCP_IMPORT_ERROR}"
        )


class MCPProviderManager:
    """
    进程级 provider 会话表。

    `start()` 幂等：重复调用只补连尚未连上的 provider。
    单个 provider 失败**不影响**其它 provider，也不阻断应用启动 ——
    外部进程起不来是环境状况，不是代码错误（与职责提示词预热同一
    取舍：记 ERROR 日志，但不拦启动）。
    """

    def __init__(self) -> None:
        self._providers: dict[str, _ProviderRuntime] = {}

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------

    async def start(
            self,
            settings: MCPSettings | None = None,
    ) -> None:
        """连接全部可用 provider。已连上的跳过。"""

        # ⚠️ 不在这里 `_require_sdk()`：缺 SDK 是环境状况，应该落到
        # 下面每个 provider 的 except 里（记 ERROR、继续启动），
        # 而不是把整个 lifespan 打崩。真正连接时才需要 SDK。
        settings = settings or get_mcp_settings()

        for name, provider in settings.active_providers().items():

            if name in self._providers:
                continue

            try:
                await self._connect(name, provider)

            except Exception:
                logger.exception(
                    "MCP provider %s 连接失败，本次启动不暴露它的工具。",
                    name,
                )

    async def _connect(
            self,
            name: str,
            settings: MCPProviderSettings,
    ) -> None:

        _require_sdk()

        params = StdioServerParameters(
            command=settings.command,
            args=list(settings.args),
            env=dict(settings.env) or None,
            cwd=settings.cwd,
        )

        stack = AsyncExitStack()

        try:
            read, write = await stack.enter_async_context(
                stdio_client(params)
            )

            session = await stack.enter_async_context(
                ClientSession(read, write)
            )

            await asyncio.wait_for(
                session.initialize(),
                timeout=settings.startup_timeout_seconds,
            )

            result = await asyncio.wait_for(
                session.list_tools(),
                timeout=settings.startup_timeout_seconds,
            )

        except BaseException:
            # 半开连接必须收掉，否则子进程会留着。
            await stack.aclose()
            raise

        tools = list(getattr(result, "tools", None) or [])

        self._providers[name] = _ProviderRuntime(
            name=name,
            settings=settings,
            session=session,
            tools=tools,
            stack=stack,
        )

        logger.info(
            "MCP provider %s 已连接：%d 个工具（放行 %d）。",
            name,
            len(tools),
            sum(
                1 for tool in tools
                if settings.tools.admits(tool_name(tool))
            ),
        )

    async def stop(self) -> None:
        """关闭全部会话。start() 之前调用是空操作。"""

        runtimes = list(self._providers.values())
        self._providers.clear()

        for runtime in reversed(runtimes):
            try:
                await runtime.stack.aclose()

            except Exception:
                logger.exception(
                    "MCP provider %s 关闭失败。", runtime.name,
                )

    # --------------------------------------------------------
    # 读取 / 调用
    # --------------------------------------------------------

    def is_connected(self, name: str) -> bool:
        return name in self._providers

    def tools_by_provider(self) -> dict[str, list[Any]]:
        return {
            name: list(runtime.tools)
            for name, runtime in self._providers.items()
        }

    async def call_tool(
            self,
            provider: str,
            tool: str,
            arguments: dict[str, Any] | None = None,
    ) -> Any:
        """
        调一个工具并转换结果。

        Raises:
            RuntimeError: provider 没连上。
            asyncio.TimeoutError: 超过 provider 的 call_timeout。
            MCPToolCallError: server 返回 is_error=true。
        """

        runtime = self._providers.get(provider)

        if runtime is None:
            raise RuntimeError(
                f"MCP provider {provider!r} 未连接，工具 {tool!r} "
                "无法执行。检查启动日志与 mcp.yaml。"
            )

        # 串行化同一 provider 的调用：stdio server 只有一条会话，
        # 而 tdx 侧的操作未必并发安全。不同 provider 之间互不阻塞。
        async with runtime.lock:
            result = await asyncio.wait_for(
                runtime.session.call_tool(
                    tool, arguments or {},
                ),
                timeout=runtime.settings.call_timeout_seconds,
            )

        if _is_error(result):
            raise MCPToolCallError(
                _error_text(result), to_tool_payload(result)
            )

        return to_tool_payload(result)


#: 进程级单例，与 ToolRegistry 同一形状。
_manager: MCPProviderManager | None = None


def get_mcp_provider_manager() -> MCPProviderManager:
    global _manager

    if _manager is None:
        _manager = MCPProviderManager()

    return _manager


def reset_mcp_provider_manager() -> None:
    """只给测试用。不清空会话 —— 先 await stop()。"""

    global _manager
    _manager = None


# ============================================================
# 一次性会话（扫描）
# ============================================================


async def _list_tools_once(
        settings: MCPProviderSettings,
) -> list[Any]:

    _require_sdk()

    params = StdioServerParameters(
        command=settings.command,
        args=list(settings.args),
        env=dict(settings.env) or None,
        cwd=settings.cwd,
    )

    async with AsyncExitStack() as stack:

        read, write = await stack.enter_async_context(
            stdio_client(params)
        )

        session = await stack.enter_async_context(
            ClientSession(read, write)
        )

        await asyncio.wait_for(
            session.initialize(),
            timeout=settings.startup_timeout_seconds,
        )

        result = await asyncio.wait_for(
            session.list_tools(),
            timeout=settings.startup_timeout_seconds,
        )

        return list(getattr(result, "tools", None) or [])


def list_provider_tools(
        settings: MCPProviderSettings,
) -> list[Any]:
    """
    同步地取一个 provider 的工具清单。

    必须在新线程里跑事件循环：调用方是 `ToolScanner.scan()`
    （同步协议），而它经常被 FastAPI 的 async 请求路径直接调用 ——
    当前线程已经有运行中的 loop，`asyncio.run` 会立刻报错。

    超时 = 2 × startup_timeout + 5 秒：`initialize` 与 `list_tools`
    各有一次 startup_timeout 的预算，外层只多留收尾余量。卡死的
    server 不该把同步接口（以及整个扫描）拖住。
    """

    def run() -> list[Any]:
        return asyncio.run(_list_tools_once(settings))

    pool = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="mcp-scan",
    )

    try:
        future = pool.submit(run)

        return future.result(
            timeout=settings.startup_timeout_seconds * 2 + 5,
        )

    finally:
        # ⚠️ 不能用 with：`ThreadPoolExecutor.__exit__` 是
        # shutdown(wait=True)，会把上面那个 timeout 变成摆设 ——
        # 卡死的 server 仍然能把调用方拖到天荒地老。
        pool.shutdown(wait=False, cancel_futures=True)


# ============================================================
# langchain 适配
# ============================================================


def build_structured_tool(
        provider: str,
        tool: Any,
        manager: MCPProviderManager,
):
    """
    把一个 MCP tool 包成 langchain `StructuredTool`。

    `args_schema` 直接给 MCP 的 JSON Schema —— 模型看到的参数与实际
    传给 server 的是同一份，不需要在中间再翻译一层。

    底层函数用 `**kwargs`：模型的参数由 StructuredTool 按 schema
    校验后原样传进来，这里不做二次解释。
    """

    from langchain_core.tools import StructuredTool

    name = tool_name(tool)

    async def invoke(**kwargs: Any) -> Any:
        return await manager.call_tool(provider, name, kwargs)

    return StructuredTool.from_function(
        coroutine=invoke,
        name=name,
        description=(
            tool_description(tool)
            or f"{provider} MCP 工具 {name}"
        ),
        args_schema=tool_input_schema(tool),
    )
