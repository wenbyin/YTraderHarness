"""
真正把工具跑起来的那一层。

它做三件事，顺序固定：

    1. 取这个工具的处理器链（按 tool_id 挂的，从库里读）
    2. 用 chain.compile_chain 把它折叠成嵌套调用，最里层是"执行工具"
    3. 驱动它，返回一个 ToolOutcome

# 工具怎么被执行

`@tool` 标记的是一个 service **方法**，所以要三步（见 decorator.py
的"执行入口还没接"）：

    service = factory.create(spec.contract)     造实例（每次新建）
    request = resolve_request_model(spec.func)  扁平参数 -> 请求模型
    await getattr(service, spec.method_name)(**{参数名: request})

`resolve_request_model` 返回 None 时（方法的参数不是单个 pydantic
模型），参数原样透传 —— 那种方法的 schema 是 `synthesize_schema`
合成的，模型给的扁平 kwargs 正好就是它要的。

# 为什么错误是 ToolOutcome 而不是异常

见 context.py 的说明：重试要靠"这次失败了吗"来判断，超时要与其它
失败区分开。用异常表达会逼着重试实现 `except Exception` 一把抓。

⚠️ `asyncio.CancelledError` 是唯一的例外，它**向上抛**。它继承
BaseException，而 context.py 说 `ToolOutcome.error` 可以是
BaseException —— 但这里刻意收窄：吞掉取消会让 agent 运行无法被中断，
而且调用方会把"取消"包成一条错误 ToolMessage 继续跑，与取消的语义
正好相反。

# 事务边界：一次工具调用 = 一段事务

每次执行都经 `factory.scope(spec.contract)` 造服务，作用域就是这一次
调用：正常返回即提交，抛异常即整体回滚。一个工具方法内部跨多个仓储
调用因此是**原子的**，不必由工具自己去猜边界。

⚠️ 为什么不能图省事用 `factory.create()`
--------------------------------------
`create()` 是基工厂上的方法，而基工厂是 session_factory 模式 ——
于是仓储被构造成 `_session=None`，第一次查库就是
`AttributeError: 'NoneType' object has no attribute execute`。

`BaseRepository` 为那种模式准备了 `new_session()`，但**没有任何仓储
实现过它**（全仓 19 个仓储一律直接用 `self._session`），所以那条分支
今天只产出"一碰库就炸"的对象。工具这条路是唯一走它的地方。

# 其它边界（已知，不是疏漏）

    - 一次调用只驱动一条链，工具之间不共享状态（除了各自的
      `invocation.context`）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Callable
from uuid import UUID

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from app.agent.tools.processor.base import BaseToolProcessor
from app.agent.tools.processor.context import (
    CALLER_CONTEXT_KEY,
    RUN_CONTEXT_KEY,
    SESSION_CONTEXT_KEY,
    TOOL_STARTED_CONTEXT_KEY,
    AsyncToolCall,
    ToolInvocation,
    ToolOutcome,
)
from app.common.factories import get_service_factory
from app.services.tool.application import ToolProcessorService
from app.services.tool.registry import (
    ToolRegistry,
    get_tool_registry,
)
from app.services.tool.registry.processor_registry import (
    ToolProcessorRegistry,
    get_processor_registry,
)
from app.services.tool.registry.schema import (
    inject_tool_args,
    injected_args,
    resolve_request_model,
)
from app.services.tool.registry.selection import select_specs
from app.services.tool.registry.spec import ToolSpec

from .chain import CompiledChain, build_chain, compile_chain

__all__ = ["ToolExecutor", "get_tool_executor"]

logger = logging.getLogger(__name__)


#: 起掉但还没跑完的 ASYNC 处理器任务。
#:
#: ⚠️ **必须持引用**。`asyncio` 只保证"被引用的"任务不被回收 ——
#: `asyncio.create_task(...)` 的返回值一旦被丢掉，任务可能在跑完之前
#: 就被 GC 掉，症状是"事件偶尔丢一条、没有任何报错"，极难复现。
_pending_tasks: set[asyncio.Task] = set()


class ToolExecutor:
    """
    驱动一条处理器链并执行工具。

    依赖都可注入（默认取进程级的那几个），因此冒烟脚本能用一个
    假 loader 把链的语义与数据库解耦开。
    """

    def __init__(
            self,
            *,
            registry: ToolRegistry | None = None,
            processor_registry: ToolProcessorRegistry | None = None,
            chain_loader: (
                Callable[[UUID], Awaitable[Sequence[Any]]] | None
            ) = None,
            factory: Any = None,
    ) -> None:

        self._registry = (
            registry if registry is not None else get_tool_registry()
        )

        self._processors = (
            processor_registry
            if processor_registry is not None
            else get_processor_registry()
        )

        #: `tool_id -> 挂载记录`。不传就按服务层读库（只读、不开事务）。
        self._chain_loader = chain_loader or _load_chain_rows

        #: service 工厂。默认**延迟取**（第一次执行时才调
        #: get_service_factory）：应用没跑过 lifespan 时它会抛
        #: RuntimeError，而那应该在"真的要用它"的时候才发生 ——
        #: 构造 executor 本身不该依赖应用已经启动。
        self._factory = factory

    # ========================================================
    # 执行
    # ========================================================

    async def execute(
            self,
            tool_key: str,
            arguments: dict[str, Any] | None,
            *,
            tool_id: UUID | None = None,
            tool_call_id: str | None = None,
            caller_identity: Mapping[str, Any] | None = None,
            runtime: Any = None,
    ) -> ToolOutcome:
        """
        跑一次工具调用。

        Parameters
        ----------
        tool_id
            这个工具在库里的 id，处理器链按它挂。**None = 没有链**
            （代码里有、但还没同步进库），此时直接执行工具 ——
            少一层链不该让工具变成不可用。
        tool_call_id
            模型给的调用 id，放进 `invocation.context` 让处理器能关联
            （事件、审计）。与 `runtime.tool_call_id` 是两回事：
            后者由调用方构造的 ToolRuntime 带下来。
        runtime
            传给处理器的 langgraph `ToolRuntime`。由调用方（react 图的
            tools 节点）构造 —— 处理器靠它拿 `tool_call_id` / `config`
            （里面有 thread_id）/ `store`，审批拦截器三个都要。

            **可以是 None**（冒烟脚本、或图外直接调 `.execute()` 的路
            径）。那时审批这类依赖审批通道的处理器会走自己的
            fail-closed 分支并打 warning —— 见 processor/interceptor/
            approval.py 的 `_channel_unavailable`。
        """

        # 走 select_specs 而不是 registry.get：**不可执行的工具绝不执行**
        # 这条要在执行这一层也成立，而不是只靠调用方判断。它挡掉的是
        # "在注册表里、但没有 ToolDescriptor"（MCP / OpenAPI 那类来源
        # 还没落地）的 key —— 那种工具 registry.get 拿得到，却绑不了、
        # 也调不通。
        #
        # 调用方（tools 节点）算过同一个允许集，所以正常情况下不会撞上；
        # 撞上说明两处的判定不一致，那是接线错误，要说清楚。
        selected = select_specs([tool_key], registry=self._registry)

        if not selected:
            return ToolOutcome(
                error=LookupError(
                    f"工具 {tool_key!r} 不是可执行的工具"
                    "（不在代码注册表里，或没有执行入口），无法执行。"
                )
            )

        spec = selected[0]

        invocation_context: dict[str, Any] = {}

        if tool_call_id is not None:
            invocation_context["tool_call_id"] = tool_call_id

        # 调用者身份交给处理器 —— 工具级的鉴权（checkauth 拦截器）读的
        # 就是它。**快照**：一次运行里不会变，由派发方在建运行时存下来
        # （见 agent_run.caller_identity）。
        #
        # 拿不到（None）时不放这个键，而不是放一个空 dict：处理器据此
        # 区分"没有身份"与"有身份但没角色/组"，而 checkauth 对前者的
        # 默认行为是拒绝（fail-closed）。
        if caller_identity is not None:
            invocation_context[CALLER_CONTEXT_KEY] = dict(caller_identity)

        # 会话 / 运行身份：从图的 context 抄进调用上下文。
        #
        # 为什么抄一份而不是让处理器自己读 runtime：
        #     图外直调（runtime=None）也要有同样的形状；
        #     执行守卫按 SESSION / RUN 计数时不该依赖 langgraph 的
        #     ToolRuntime 一定存在。
        #
        # `pipeline_run_id` 是**业务运行**的身份（不是 config 顶层那个
        # langgraph run_id），见 ReactContext 的说明。
        graph_context = getattr(runtime, "context", None)

        session_id = getattr(graph_context, "session_id", None)
        run_id = getattr(graph_context, "pipeline_run_id", None)

        if session_id is not None:
            invocation_context[SESSION_CONTEXT_KEY] = session_id

        if run_id is not None:
            invocation_context[RUN_CONTEXT_KEY] = run_id

        invocation = ToolInvocation(
            tool_key=tool_key,
            # 本次调用**私有**的副本：处理器就地改它，调用方手里的
            # 原始参数不受影响（见 context.py 的契约）。
            arguments=dict(arguments or {}),
            tool_id=tool_id,
            context=invocation_context,
        )

        chain = await self._load_chain(tool_id)

        if isinstance(chain, ToolOutcome):
            # 链自身不自洽（半条 AROUND、手改库……）。此时**不执行
            # 工具**：链上可能挂着审批或脱敏，绕过它去跑比失败更糟。
            return chain

        base = self._base_call(
            spec, invocation, tool_call_id, runtime=runtime,
        )

        # ASYNC 观察者不参与嵌套，按 hook 在链外起掉（见 chain.py）。
        for entry in chain.async_before:
            _spawn(
                entry.processor.process(
                    entry.hook,
                    invocation,
                    _noop,
                    config=entry.config,
                    runtime=runtime,
                ),
                label=entry.processor_key,
            )

        outcome = await compile_chain(
            chain, invocation, base, runtime=runtime
        )()

        for entry in chain.async_after:
            _spawn(
                entry.processor.process(
                    entry.hook,
                    invocation,
                    _fixed(outcome),
                    config=entry.config,
                    runtime=runtime,
                ),
                label=entry.processor_key,
            )

        return outcome

    # ========================================================
    # internals
    # ========================================================

    async def _load_chain(
            self,
            tool_id: UUID | None,
    ) -> CompiledChain | ToolOutcome:
        """
        `CompiledChain`，或一个失败结果（链读不出来 / 不自洽）。
        """

        if tool_id is None:
            # 没有库里的行 —— 空链。这不是错误（见 execute 的说明）。
            return CompiledChain()

        try:
            rows = await self._chain_loader(tool_id)

            return build_chain(
                rows,
                resolve_processor=self._processors.get,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:  # noqa: BLE001 - 见下

            # 读链失败与链不自洽走同一条路：**不执行工具**。这里的
            # 错误类型是开放的（数据库断了、配置写错了、实现删了），
            # 一律收成一个失败结果交给模型与观测，而不是让整轮对话
            # 因为一个工具的配置坏掉而 failed。
            logger.exception(
                "工具 %s 的处理器链不可用，本次不执行。", tool_id
            )

            return ToolOutcome(error=exc)

    def _base_call(
            self,
            spec: ToolSpec,
            invocation: ToolInvocation,
            tool_call_id: str | None,
            *,
            runtime: Any = None,
    ) -> AsyncToolCall:
        """
        最里层：造 service、包请求模型、await 方法。

        `runtime` 是 `tools` 节点构造的官方 `ToolRuntime`，只用于给
        声明了注入参数的工具（ToolRuntime / InjectedState …）填参；
        service 方法那条路不用它。

        ⚠️ 参数在**被调用时**才从 `invocation.arguments` 读 —— 处理器
        就地改的就是它，提前读会拿到改动前的值。
        """

        async def call() -> ToolOutcome:

            started = time.monotonic()

            try:
                # ⚠️ 两种工具，两条调用路
                # --------------------
                # 判据是 `spec.tool`，不是"有没有 owner" —— 它就是那个
                # 字段存在的理由（见 ToolSpec 的类 docstring）。
                #
                # 模块级 langchain 工具没有 owner 类，也就没有可以交给
                # 工厂去建的 self：它自己就是可执行的 Runnable，直接调。
                # 参数按它自己的 args_schema 收 —— 与模型看到的
                # input_schema 是同一份（见 scanner._args_schema_of），
                # 所以不需要再过一遍 resolve_request_model。
                if spec.tool is not None:

                    # 执行守卫靠这个标记区分「中断时还没开始」与
                    # 「已经开始之后才中断」—— 后者可能已产生副作用，
                    # 要记 UNKNOWN 而不是 INTERRUPT/CANCELLED。
                    invocation.context[TOOL_STARTED_CONTEXT_KEY] = True

                    # ⚠️ 注入参数在调用**之前**塞进 arguments
                    # --------------------------------------
                    # `runtime` / InjectedState / store / tool_call_id
                    # 是执行器给的，不是模型给的。`inject_tool_args`
                    # 同时会剥掉模型伪造的同名参数（模型看不到它们，
                    # 但恶意构造的 args 里可能有）。
                    #
                    # tool_call_id 用本次调用的那个，与 ToolMessage /
                    # 审批账本对齐。
                    tool_func = (
                        getattr(spec.tool, "func", None)
                        or getattr(spec.tool, "coroutine", None)
                    )

                    call_args = (
                        inject_tool_args(
                            tool_func,
                            invocation.arguments,
                            runtime=runtime,
                            tool_call_id=tool_call_id,
                        )
                        if tool_func is not None
                        else dict(invocation.arguments)
                    )

                    # ⚠️ 声明了 InjectedToolCallId 的工具必须收**完整
                    # ToolCall**（LangChain 从这个形状里取 id），只传
                    # args dict 会报 "must be invoked with a full model
                    # ToolCall"。
                    injected = (
                        injected_args(tool_func)
                        if tool_func is not None
                        else {}
                    )

                    tool_input: Any = call_args

                    if any(
                            item.kind == "tool_call_id"
                            for item in injected.values()
                    ):
                        tool_input = {
                            "name": spec.tool.name,
                            "args": call_args,
                            "id": tool_call_id or "",
                            "type": "tool_call",
                        }

                    result = await spec.tool.ainvoke(tool_input)

                    # ⚠️ 完整 ToolCall 输入会得到 ToolMessage 包装
                    # --------------------------------------------
                    # langchain 在 `InjectedToolCallId` 那条路上要求传完整
                    # ToolCall（见上），而它会把**工具返回值**包成
                    # `ToolMessage(content='{"questions": ...}', ...)`。
                    #
                    # 如果这里原样交给上游，tools 节点会把这个对象再
                    # json.dumps 一次，模型看到的是
                    # `content='...' name='...' tool_call_id='...'` 这种
                    # repr，而不是实际结果（问答工具实测）。
                    if (
                            isinstance(tool_input, Mapping)
                            and tool_input.get("type") == "tool_call"
                            and isinstance(result, ToolMessage)
                    ):
                        result = result.content

                else:
                    factory = self._factory or get_service_factory()

                    # 每次新建实例，并给它**一段事务**（见模块 docstring 的
                    # 事务边界那一节）。
                    #
                    # ⚠️ 必须是 `scope` 而不是 `create`
                    # ---------------------------------
                    # `create()` 是从**基工厂**造的，而基工厂是
                    # session_factory 模式（它没有请求级 session）—— 于是
                    # 仓储被构造成 `_session=None`，第一次查库就炸：
                    #
                    #     AttributeError: 'NoneType' object has no attribute execute
                    #
                    # 根因是 `BaseRepository` 那个 "session_factory 模式"
                    # **没有任何仓储实现过**：全仓 19 个仓储一律直接用
                    # `self._session`。所以工具这条路上唯一能给它们 session
                    # 的办法就是开一个 scope。工具一执行就查库，因此这一条
                    # 不是"以防万一"。
                    async with factory.scope(spec.contract) as service:

                        # ⚠️ 实例访问经描述符协议拿到的是**绑定方法**
                        # （decorator.__get__ 的 obj 非 None 分支）。
                        method = getattr(service, spec.method_name)

                        resolved = resolve_request_model(spec.func)

                        # 见上面 langchain 工具分支的同一句：真正调用
                        # 工具方法之前打标记。
                        invocation.context[TOOL_STARTED_CONTEXT_KEY] = True

                        if resolved is None:
                            result = await method(**invocation.arguments)
                        else:
                            parameter, model = resolved
                            result = await method(
                                **{parameter: model(**invocation.arguments)}
                            )

            except asyncio.CancelledError:
                # 见模块 docstring：取消不是"工具失败了"。
                raise

            except GraphBubbleUp:
                # ⚠️ `interrupt()` 必须穿过执行器
                # ---------------------------------
                # HITL 工具（ask_user_question）在**工具函数体内**调用
                # interrupt()，它抛的是 GraphInterrupt（GraphBubbleUp 的
                # 子类）。下面那个宽泛的 Exception 会把它当成"工具执行
                # 失败"包成 ToolOutcome —— 图和前端都看不到中断。
                #
                # 同理适用于其它 LangGraph 冒泡控制异常。
                raise

            except Exception as exc:  # noqa: BLE001

                logger.exception(
                    "工具 %s 执行失败（tool_call_id=%s）。",
                    spec.tool_key,
                    tool_call_id,
                )

                return ToolOutcome(
                    error=exc, duration_ms=_elapsed_ms(started)
                )

            return ToolOutcome(
                value=result, duration_ms=_elapsed_ms(started)
            )

        return call


async def _noop() -> ToolOutcome:
    """
    ASYNC BEFORE 处理器的 next_call。

    它不会被真的用到（观察者不驱动控制流），给一个"空结果"是为了
    万一有实现顺手 `await next_call()` 时不会炸在 None 上。
    """

    return ToolOutcome()


def _fixed(outcome: ToolOutcome) -> AsyncToolCall:
    """
    ASYNC AFTER 处理器的 next_call：把**已经算出来**的结果交给它。

    AFTER 的 `process()` 会先 `await next_call()` 再调 `after()`，
    所以这里不能再执行一次工具 —— 那会真的跑两遍。
    """

    async def call() -> ToolOutcome:
        return outcome

    return call


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def _spawn(coro: Awaitable[ToolOutcome], *, label: str) -> None:
    """
    起一个 ASYNC 处理器任务，不等它的返回值。

    返回值丢弃是 ASYNC 的定义（见 base.py），但**异常不丢**：
    取不到就会变成 asyncio 的 "Task exception was never retrieved"，
    在解释器退出时才冒出来、且与业务无关联。这里记一条 WARNING 并
    把异常带上。
    """

    task = asyncio.ensure_future(coro)

    _pending_tasks.add(task)

    def _done(finished: asyncio.Task) -> None:

        _pending_tasks.discard(finished)

        if finished.cancelled():
            return

        error = finished.exception()

        if error is not None:
            logger.warning(
                "ASYNC 处理器 %s 失败（返回值本来就会被丢弃，"
                "但异常必须看得见）：",
                label,
                exc_info=error,
            )

    task.add_done_callback(_done)


async def _load_chain_rows(tool_id: UUID) -> Sequence[Any]:
    """
    默认的读链方式：走服务层（只读，不开写事务）。

    复用 `ToolProcessorService.list_chain` 而不是直接拿仓储：链的
    组装（排序、把行与实现对应起来）已经在服务层做过一遍，这里再来
    一遍就是第二份实现。`enabled_only=True` 正是"执行时真正会跑的
    那几条"。
    """

    factory = get_service_factory()

    async with factory.scope(
            ToolProcessorService, begin=False
    ) as service:

        chain = await service.list_chain(tool_id, enabled_only=True)

    return chain.processors


#: 进程级单例。执行器无状态（除了那几个进程级依赖），每次新建没有意义。
_executor = ToolExecutor()


def get_tool_executor() -> ToolExecutor:
    """取进程级执行器。"""

    return _executor
