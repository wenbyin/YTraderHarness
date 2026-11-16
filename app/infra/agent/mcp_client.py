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

两类 provider
-------------
stdio             本地子进程（tdx-link-mcp）
streamable_http   远端 HTTP 端点（ytrader-market，POST /mcp）

差别只在 `_open_transport()`：拿到 (read, write) 之后，
initialize / list_tools / call_tool 与工具过滤完全共用一套。

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
from contextlib import AsyncExitStack, nullcontext
from copy import deepcopy
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
    streamablehttp_client = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR: Exception | None = exc
    _HTTP_IMPORT_ERROR: Exception | None = exc
else:
    _MCP_IMPORT_ERROR = None

    try:
        from mcp.client.streamable_http import streamablehttp_client

    except ImportError as exc:  # pragma: no cover - 只影响 http provider
        # 单独兜一层：这条 import 链在 Windows 上会牵到 pywin32
        # （mcp/os/win32/utilities.py），缺 pywintypes 时炸的是这里。
        # 把它和 stdio 分开，缺 pywin32 的环境至少还能用 tdx。
        streamablehttp_client = None  # type: ignore[assignment]
        _HTTP_IMPORT_ERROR = exc
    else:
        _HTTP_IMPORT_ERROR = None


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


def _drop_nulls(value: Any) -> Any:
    """递归去掉 None 值。

    只删 None，不删空 dict / 空 list / False / 0 —— 那几种是**有意义
    的 JSON 值**，删掉会改变 schema 语义。
    """

    if isinstance(value, dict):
        return {
            key: _drop_nulls(item)
            for key, item in value.items()
            if item is not None
        }

    if isinstance(value, list):
        return [_drop_nulls(item) for item in value]

    return value


def tool_input_schema(tool: Any) -> dict[str, Any]:
    """
    `inputSchema` -> 可交给模型 API 的 parameters。

    为什么客户端也要修一遍
    ----------------------
    MCP 只要求 inputSchema 是 JSON Schema，并没有要求它不含 null 子字段。
    但 OpenAI 兼容接口对 `function.parameters` 极其严格：

        Invalid schema for function 'x': null is not of type "object"

    而且这个 400 发生在**请求体校验**阶段 —— 不是"这个工具不可用"，
    是整轮对话直接失败，看上去还像模型的问题。我们控制不了所有
    MCP server（ytrader-market 的 `list_periods` 就踩过：
    `"properties": null`），所以在这一层兜住，同时留 warning
    让 server 的 bug 仍然可见，而不是被悄悄抹平。
    """

    raw = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
    )

    if not isinstance(raw, dict):
        return {"type": "object", "properties": {}}

    schema = _drop_nulls(raw)
    changed = schema != raw

    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
        changed = True

    if schema.get("type") != "object":
        # 顶层只能是对象：parameters 不接受数组/字符串当根。
        schema["type"] = "object"
        changed = True

    if changed:
        logger.warning(
            "MCP 工具 %s 的 inputSchema 含非法空值（properties/type），"
            "已规范化后使用；建议修 server。", tool_name(tool),
        )

    return schema


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
    #: 只对 stdio 生效：一条子进程会话上的并发调用未必安全
    #: （tdx 侧会改订阅/专家状态）。streamable_http 是无状态的单个
    #: POST，并行调用是正常的用法，串行反而会把并发的行情查询排队。
    serialized: bool = True
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _require_sdk() -> None:
    if _MCP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "未安装 mcp SDK，MCP provider 不可用："
            f"{_MCP_IMPORT_ERROR}"
        )


async def _open_transport(
        stack: AsyncExitStack,
        settings: MCPProviderSettings,
) -> tuple[Any, Any]:
    """
    按 `transport` 打开连接，返回一对 (read, write) 流。

    两种 transport 的差别到此为止 —— 上面的 initialize / list_tools /
    call_tool 全部共用：MCP 协议是同一套，只是底下管道不同
    （子进程 stdin/stdout vs. HTTP POST + SSE）。

    流的生命周期挂在 `stack` 上，由调用方在失败时 `aclose()`
    —— 半开的连接会漏子进程或漏 HTTP 会话。
    """

    if settings.transport == "stdio":

        params = StdioServerParameters(
            command=settings.command,
            args=list(settings.args),
            env=dict(settings.env) or None,
            cwd=settings.cwd,
        )

        return await stack.enter_async_context(stdio_client(params))

    if settings.transport == "streamable_http":

        if streamablehttp_client is None:
            raise RuntimeError(
                "streamable_http 客户端不可用（缺 pywin32 或 SDK 版本过旧）："
                f"{_HTTP_IMPORT_ERROR}"
            )

        # httpx 的读超时必须跟着 call_timeout 走：SDK 默认 30s 会把一次
        # 正常但慢的回源查询掐断（行情源限频时可达分钟级），症状是
        # 「MCP 工具无故超时」而不是 server 报错，很难查。
        timeout = settings.call_timeout_seconds

        read, write, _session_id = await stack.enter_async_context(
            streamablehttp_client(
                settings.url,
                headers=dict(settings.headers) or None,
                timeout=timeout,
                sse_read_timeout=max(timeout, 60.0),
            )
        )

        return read, write

    # Literal 已经挡掉了未知取值；留一条兜底，避免加了枚举却忘了分支。
    raise RuntimeError(f"不支持的 MCP transport: {settings.transport}")


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

        stack = AsyncExitStack()

        try:
            read, write = await _open_transport(stack, settings)

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
            serialized=settings.transport == "stdio",
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

        # stdio 只有一条子进程会话，且 tdx 侧的操作未必并发安全 —— 串行化。
        # streamable_http 是无状态 POST，并发调用本来就是正常用法；用同一
        # 把锁会把「同时查 5 只票」排队成 5 倍延迟。
        guard = runtime.lock if runtime.serialized else nullcontext()

        async with guard:
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

    async with AsyncExitStack() as stack:

        read, write = await _open_transport(stack, settings)

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

#: `config` 是 LangChain 的保留参数名。`BaseTool.arun` 会执行
#: `tool_kwargs[config_param] = config`（见 langchain_core/tools/base.py
#: 的 `_get_runnable_config_param`），把模型传的同名值覆盖成
#: RunnableConfig；`StructuredTool._arun` 又把它当具名参数吸收，
#: 底层 `coroutine(**kwargs)` 里就再也看不到这个参数了。
#:
#: MCP 里恰好把业务参数命名为 `config` 的工具（strategy_apply /
#: strategy_validate(kind=strategy|template)）因此永远到不了 server，
#: 报错却是「必须给 config」，看上去像模型传错，实际是适配层冲突。
#:
#: 解法：不动 LangChain、不动 MCP server —— 注册时把顶层参数换个名字，
#: 调用前再映射回来。选 `config_payload` 而不是 `draft`，因为它对
#: `kind=template` 的模板配置同样成立，也不与任何已知 MCP 字段重名。
_CONFIG_PARAM = "config"
_CONFIG_ALIAS = "config_payload"


def rewrite_config_param(
        schema: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """
    把 inputSchema 里**顶层**名为 `config` 的参数改成 `config_payload`。

    只动顶层 properties 与 required。嵌套对象里的 `config` 是值的一部分
    （例如 `kind=template` 的 `{template_type, config}`），LangChain 不看，
    原样保留。

    返回 `(schema, changed)`；`changed=False` 时返回的就是传入的对象，
    调用方据此决定要不要追加描述提示。
    """

    properties = schema.get("properties")

    if (
            not isinstance(properties, dict)
            or _CONFIG_PARAM not in properties
            or _CONFIG_ALIAS in properties  # 别覆盖 server 已有的同名字段
    ):
        return schema, False

    result = deepcopy(schema)
    props = result["properties"]
    props[_CONFIG_ALIAS] = props.pop(_CONFIG_PARAM)

    required = result.get("required")
    if isinstance(required, list):
        result["required"] = [
            _CONFIG_ALIAS if item == _CONFIG_PARAM else item
            for item in required
        ]

    return result, True


def build_structured_tool(
        provider: str,
        tool: Any,
        manager: MCPProviderManager,
):
    """
    把一个 MCP tool 包成 langchain `StructuredTool`。

    `args_schema` 用 MCP 的 JSON Schema（先做一次 `config` 改名），
    模型看到的参数与最终发给 server 的参数只差这一个别名，由
    `invoke` 映射回来。其余工具零改动。

    ⚠️ 为什么还要给 `ainvoke` 套一层
    --------------------------------
    平台侧的 skill / 工具描述写的都是 `config`，模型偶尔会照旧传
    `config`。这个键在 LangChain 内部就被覆盖，**下游无法恢复**
    （值在 `BaseTool.arun` 里已经被换成 RunnableConfig），所以必须在
    进入 LangChain 之前改名。这样两种写法都接得住：按新 schema 传
    `config_payload`，或按旧文档传 `config`。

    只覆盖 `ainvoke`：执行器（`tool_executor.py`）走的就是这条异步路径。
    `StructuredTool` 是 pydantic 模型，实例属性赋值会被 `__setattr__`
    拦掉，而这里只是覆盖一个非数据描述符方法，直接落 `__dict__` 即可。
    """

    from langchain_core.tools import StructuredTool

    name = tool_name(tool)

    async def invoke(**kwargs: Any) -> Any:
        if _CONFIG_ALIAS in kwargs:
            kwargs[_CONFIG_PARAM] = kwargs.pop(_CONFIG_ALIAS)
        return await manager.call_tool(provider, name, kwargs)

    schema, renamed = rewrite_config_param(tool_input_schema(tool))

    description = tool_description(tool) or f"{provider} MCP 工具 {name}"

    if renamed:
        description += (
            "\n\n⚠️ 接入层字段名：原参数 `config` 在这里注册为 "
            f"`{_CONFIG_ALIAS}`（内容与类型不变，客户端会自动还原）。"
            f"请按工具 schema 填 `{_CONFIG_ALIAS}`；填 `config` 也能识别。"
        )

    built = StructuredTool.from_function(
        coroutine=invoke,
        name=name,
        description=description,
        args_schema=schema,
    )

    if renamed:
        original_ainvoke = built.ainvoke

        async def ainvoke_compat(
                input: Any,
                config: Any = None,
                **kwargs: Any,
        ) -> Any:
            if isinstance(input, dict):
                # 执行器在注入 tool_call_id 时会给完整 ToolCall 形状。
                if input.get("type") == "tool_call":
                    args = dict(input.get("args") or {})
                    if _CONFIG_PARAM in args:
                        args[_CONFIG_ALIAS] = args.pop(_CONFIG_PARAM)
                    input = {**input, "args": args}
                elif _CONFIG_PARAM in input:
                    input = dict(input)
                    input[_CONFIG_ALIAS] = input.pop(_CONFIG_PARAM)

            return await original_ainvoke(input, config, **kwargs)

        object.__setattr__(built, "ainvoke", ainvoke_compat)

    return built
