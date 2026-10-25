"""
默认 Provider Adapter：走 langchain，复用既有的 `app/common/llm.py`。

为什么是"复用"而不是"再写一套 provider 构造"
------------------------------------------
仓库里已经有一份**能用的** provider 客户端构造：`app/common/llm.py`
的 `get_chat_model()`。它承载了三件本模块不该重复的知识：

    provider 名 -> 客户端类    （init_chat_model 的别名表 + 不走它的
                                那几家直接构造，见 _DIRECT_PROVIDERS）
    连接信息与密钥              （providers.yaml 的 llm 段 + ${ENV} 展开）
    provider 特有参数            （temperature 的"不传就不传"这类取舍）

因此本模块的 Factory **不自己 import openai / langchain 包**，而是把
"建客户端"这件事转交给它（见 `_default_chat_model_factory`）。本模块
新增的只有 provider 之上的那一层：模型目录、重试、降级、超时、异常
归一化与观测。

两类 provider 都能服务
---------------------
langchain 的 `init_chat_model` 支持 27 家（openai / anthropic /
google / azure…），而 deepseek / kimi 这些 OpenAI 兼容家都能直接吃。
所以**一个** adapter 类就够覆盖它们；只有出现"非 langchain 客户端"
的 provider（自研网关、gRPC 服务）才需要新的 adapter。

⚠️ 它对客户端只做**鸭子调用**
---------------------------
`bind_tools` / `with_structured_output` / `ainvoke` / `usage_metadata`
都是运行时用 `getattr` 探的。因此本模块**不 import langchain**，
"只想读个模型配置"的场景不会被拖起 openai + httpx 一整串依赖。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Mapping
from typing import Any, Callable

from .errors import (
    ModelConfigurationError,
    ModelProviderError,
    normalize_error,
)
from .leaked_tool_call import LeakGate, strip_leaked_tool_calls
from .provider import DeltaCallback, ModelClient
from .redaction import redact_text
from .types import ModelDefinition, ModelRequest, ModelUsage, ProviderResult

__all__ = [
    "LangChainProviderAdapter",
    "default_chat_model_factory",
    "thinking_off_extra_body",
]

logger = logging.getLogger(__name__)


def default_chat_model_factory(**kwargs: Any) -> Any:
    """
    惰性转交给 `app/common/llm.py`。

    惰性 import 的理由与 `app/common/llm.py` 内部的惰性 import 一致：
    本函数只在**真要建客户端**时被调用，而"建客户端"从来不在 import
    阶段发生。
    """

    from app.common.llm import get_chat_model

    return get_chat_model(**kwargs)


class LangChainProviderAdapter:
    """
    用 langchain 的 chat 客户端服务一类 provider。

    参数
    ----
    chat_model_factory
        建客户端的可调用对象，签名与 `app.common.llm.get_chat_model`
        一致。默认就是它；测试或"换一条客户端来源"时替换它即可，
        不必改本类（依赖注入的接缝只留这一个）。
    """

    #: 适配器标识：model.yaml 里 `adapter: langchain`。
    name = "langchain"

    def __init__(
            self,
            *,
            chat_model_factory: Callable[..., Any] | None = None,
    ) -> None:

        self._factory = chat_model_factory or default_chat_model_factory

    # ========================================================
    # 建客户端
    # ========================================================

    def create_client(
            self,
            definition: ModelDefinition,
            *,
            credential: str | None,
            timeout: float,
            max_retries: int,
    ) -> Any:
        """
        建一个 chat 客户端。**同步、不联网**。

        ⚠️ `max_retries` 默认由 Factory 传 0
        -----------------------------------
        客户端内部重试与 Runtime 的 retry 叠加，是"3 × 3 = 9 次可计费
        调用"那类问题的来源（旧的 agent_node 注释里记着这件事）。
        现在 retry 只有一个所有者：**Runtime**。因此这里默认关掉
        客户端自己的重试。

        `config` 里的键全部作为 provider 参数透传（temperature /
        max_tokens / top_p …）。其中 `max_retries` 被**剔除**：它属于
        可靠性策略，不该混在模型参数里两边都生效。
        """

        config = {
            key: value
            for key, value in definition.config.items()
            if key != "max_retries"
        }

        try:
            return self._factory(
                provider=definition.provider,
                model=definition.model,
                api_key=credential,
                timeout=timeout,
                max_retries=max_retries,
                **config,
            )

        except Exception as exc:

            # 建不出来通常是**配置**问题（provider 名不认识、key 缺失、
            # 装少了包），而不是 provider 挂了 —— 归一化成可重试的
            # ModelProviderError 会让 Runtime 把同一个配置错重试三次，
            # 而真正的报错（"该设哪个环境变量"）会被埋掉。
            #
            # 因此这里抛 ModelConfigurationError：它是 FAIL 语义，而且
            # 换候选也没用（换一家 provider 同样缺配置）—— 早失败才对。
            raise _client_creation_error(definition, exc) from exc

    # ========================================================
    # 调用
    # ========================================================

    async def ainvoke(
            self,
            client: ModelClient,
            request: ModelRequest,
            *,
            timeout: float,
            on_delta: DeltaCallback | None = None,
            run_metadata: Mapping[str, Any] | None = None,
    ) -> ProviderResult:
        """
        发一次请求，把结果与用量读成统一形状。

        ⚠️ `timeout` 参数**默认实现不直接使用**（只在返回的 metadata
        里留痕）：唯一可靠的超时执行点是 Runtime 的 `asyncio.wait_for`
        与客户端**构造时**设置的超时。langchain 的 ainvoke 会把多余的
        kwargs 当成模型参数透传给服务端，把 `timeout=` 塞进去在某些
        provider 上会得到 400。留给需要它的适配器（比如支持 per-call
        deadline 的 SDK）。

        `on_delta` 非空时走流式与否在**这里**决定（判据见 `_can_stream`）：
        能流就逐块回调；不能流就退化成"一次性回调整段正文"。后者保证
        调用方**无论如何都拿得到正文**，只是不逐 token。
        """

        runnable = self._bind(client, request)

        #: 这次调用的身份（agent_type / model_role），挂到 langchain 的
        #: **运行元信息**上 —— 不是发给 provider 的参数（见 provider.py
        #: 对 `run_metadata` 的说明）。两条调用路径都要带，否则"流式的那
        #: 次有 role、非流式的那次没有"，下游按 role 过滤就会**按模型能力
        #: 时灵时不灵**：一个不支持流式的模型会让压缩结果照旧流进聊天窗。
        config = _run_config(run_metadata)

        # 打在调用**之前**：请求正文在失败时最需要看，而失败路径上
        # 没有任何"之后"。
        _log_request(client, request)

        try:
            if on_delta is not None and self._can_stream(client, request):
                message = await self._astream(
                    runnable, client, request, on_delta, config
                )

            else:
                message = await runnable.ainvoke(
                    list(request.messages), config=config
                )

                if on_delta is not None:
                    # 一次性回调也要过闸：这一段文本与流式那条路的产出是同一份
                    # 东西（见 leaked_tool_call），不能因为"反正只有一块"绕开。
                    text, _ = strip_leaked_tool_calls(_chunk_text(message))

                    if text:
                        await on_delta(text)

        except asyncio.CancelledError:
            # 取消**不是** provider 失败。归一化会把它变成可重试的
            # ModelProviderError，于是"上游超时取消"变成"provider 抽风，
            # 再试两次" —— 那是最糟的一种错误翻译（Runtime 的
            # _try_candidate 里有同一条注释）。
            #
            # ⚠️ 流式让这条路径**常见得多**：一个长回答有几百个 await
            # 点，每一处都可能被 wait_for 取消。这个 except 必须排在
            # 下面那个 BaseException 之前。
            raise

        except BaseException as exc:
            # 归一化发生在**最靠近 provider** 的地方：这一层的分类信息
            # 最全（异常对象、状态码都还在手上）。Runtime 那边还有一层
            # 兜底，但那只能给出可重试的 ModelProviderError。
            raise normalize_error(
                exc,
                provider=client.definition.provider,
                model=client.definition.model,
            ) from exc

        # ---- 出口的最后一道闸：模型把工具调用写成正文时拦掉 ----
        #
        # ⚠️ 在这一层拦，是因为**这条消息就是会被存下来的那一份**
        # --------------------------------------------------
        # 它接着会进图 state、进 checkpointer、进对话历史与 agent_message
        # 表。只拦推流（投影层那道闸）的话，用户界面上干净了，**库里那段
        # 垃圾还在**，而且下一轮会当作"我上一轮说过的话"再喂回模型。
        #
        # 两道闸用的是同一套规则（`leaked_tool_call`），分工是：这里管
        # "落到状态里的那一份"，投影层管"逐 token 推给用户的那一份" ——
        # 前者是整段文本、后者是分片，所以状态机不能共用（见那个模块的
        # docstring）。
        #
        # ⚠️ 思维链也要过（两条通道都清）
        # ----------------------------
        # 泄漏的成因是**思考模式**，而思考模式正是产出思维链的那条路 ——
        # 同一段标记因此完全可能落在 `reasoning_content` 里。它虽然不参与
        # 路由、不进记忆，但会随 `agent_message.reasoning_content` 落库、
        # 并推给前端渲染，所以在同一处一并清掉。
        #
        # 两个计数**分开**：正文命中才决定"这条回复是不是废的"（下面那个
        # raise 的判据保持原样，不因为思维链里的标记把一次正常回复判死）。
        hits = _strip_leaked_tool_calls(message)
        reasoning_hits = _strip_reasoning_leaks(message)

        if hits or reasoning_hits:

            logger.warning(
                "模型 %s 把工具调用写成了正文（正文命中 %d 处、思维链命中 %d "
                "处标记，已拦掉）：这是思考模式下的已知症状，见 model.yaml "
                "的 disable_thinking。",
                client.model_id,
                hits,
                reasoning_hits,
            )

        if hits and not _chunk_text(message).strip():
            # 整条回复除了标记没有别的：这**不是**一次成功的答复。
            # 抛（而不是返回空正文）与下面那条"流里一块都没有"同一个
            # 立场：空消息会以 completed 收尾，那种成功最难查。
            raise ModelProviderError(
                f"模型 {client.model_id} 把工具调用写成了正文"
                f"（命中 {hits} 处标记），这次回复没有任何可用内容。\n"
                "这是思考模式下的已知症状：确认 model.yaml 里该模型声明了 "
                "`disable_thinking: true`（见 "
                "app/agent/model/leaked_tool_call.py）。",
                provider=client.definition.provider,
                model=client.definition.model,
                details={"leaked_tool_call_markers": hits},
            )

        # 与上面那条配对。失败时不打这里 —— 那条路径由上面的"之前"与
        # observer 的 WARNING 覆盖。
        _log_response(client, message)

        metadata = _provider_metadata(message)

        if timeout:
            metadata.setdefault("attempt_timeout", timeout)

        return ProviderResult(
            message=message,
            usage=_extract_usage(message),
            provider=client.definition.provider,
            model=client.definition.model,
            metadata=metadata,
        )

    def _can_stream(
            self,
            client: ModelClient,
            request: ModelRequest,
    ) -> bool:
        """
        这次调用能不能走 provider 的**原生流**。两个判据：

            request.response_format is None
                `with_structured_output` 的流式各家实现不一致，有些
                根本不支持 —— 结构化输出一律退回 ainvoke
            capabilities.streaming
                逐模型声明（model.yaml 与 types.ModelCapabilities）

        ⚠️ streaming **只**在这里当"传输方式"的开关，**不进**
        `required_capabilities_for`。它是偏好而不是硬要求：把它当硬
        要求，会让本来能用的候选模型因为"不走流式"被从链上剔除 ——
        为了一个更好看的传输方式牺牲可用性，不划算。
        """

        if request.response_format is not None:
            return False

        capabilities = getattr(client.definition, "capabilities", None)

        return bool(getattr(capabilities, "streaming", False))

    async def _astream(
            self,
            runnable: Any,
            client: ModelClient,
            request: ModelRequest,
            on_delta: DeltaCallback,
            config: Mapping[str, Any] | None = None,
    ) -> Any:
        """
        消费 provider 的流：逐块抽正文 -> 回调 -> 用 `+` 累积。

        累积用 `AIMessageChunk.__add__`，它会把 `tool_call_chunks` 正确
        拼成 `tool_calls` —— 那是 route_after_agent 判断"要不要去 tools
        节点"的唯一依据，拼错就等于把工具调用丢掉。用 `+` 而不是
        `add_ai_message_chunks`，是为了不在本模块顶层 import langchain
        （见模块 docstring 的惰性纪律）。

        ⚠️ **先 await 回调，再拉下一块**
        ------------------------------
        await 提供了背压，于是回调的顺序与 provider 产出的顺序**严格
        一致**。若改成"先累积后回调"或把回调丢进后台任务，正文在消费
        端就会错序 —— 而错序的正文是无法从内容上发现的。

        代价是：一个慢的投递会占用这次 attempt 的超时预算（Runtime 的
        wait_for 覆盖整段消费）。这是刻意的取舍 —— 见
        app/common/stream/publisher.py 里"为什么是 await 而不是后台写"。
        """

        # 局部 import：本模块顶层刻意不拉 langchain（见模块 docstring）。
        from langchain_core.messages.utils import message_chunk_to_message

        aggregate: Any = None

        #: 逐块累积 provider 元信息：**后来的非空值覆盖先前的**。
        #:
        #: ⚠️ 不能直接用聚合后的 response_metadata
        #: ------------------------------------
        #: langchain 的 `AIMessageChunk.__add__` 对 response_metadata 里的
        #: **字符串是拼接**的（merge_dicts 的行为）。实测三块的
        #: model_name 会变成 'deepseek-v4deepseek-v4deepseek-v4'；
        #: 逐块都报 finish_reason 的 provider 会得到 'stopstop'。
        #:
        #: 那是为"合并若干片元信息"设计的，放到这里只会产出垃圾 ——
        #: 而且**不报错**，坏掉的是观测里那个字段，事后完全看不出是拼的。
        #:
        #: 自己合并成"最后说了算"：provider 的元信息是**状态**（这个模型
        #: 叫什么、为什么停下），不是可以相加的量。
        #:
        #: 已知取舍：嵌套 dict（如 token_usage）是整体覆盖而不是逐键合并。
        #: 各家要么只报一次、要么最后一块报累计值，覆盖是对的。
        #: 出口的闸（一个流一个实例）。增量分片**跨块**切断标记，所以这里
        #: 只能是有状态的（见 leaked_tool_call 的模块 docstring）。
        gate = LeakGate()

        async def _emit(text: str) -> None:

            cleaned = gate.feed(text)

            if cleaned:
                await on_delta(cleaned)

        metadata: dict[str, Any] = {}

        async for chunk in runnable.astream(list(request.messages), config=config):

            text = _chunk_text(chunk)

            if text:
                await _emit(text)

            chunk_metadata = getattr(chunk, "response_metadata", None)

            if isinstance(chunk_metadata, Mapping):
                metadata.update({
                    key: value
                    for key, value in chunk_metadata.items()
                    if value is not None
                })

            aggregate = chunk if aggregate is None else aggregate + chunk

        if aggregate is None:
            # 一块都没有：provider 没报错但也没产出。**抛**出去，而不是
            # 返回一个空消息 —— 空消息会以 completed 收尾，而模型其实
            # 什么都没说，那种"成功"最难查（与 tools.py 里"不返回假
            # ToolMessage"是同一条立场）。
            raise ModelProviderError(
                f"模型 {client.definition.model} 的流式响应"
                "没有产生任何 chunk。",
                provider=client.definition.provider,
                model=client.definition.model,
            )

        # 流结束：把闸里扣着的那点字符放出去。区段没开着时它只可能是"看着像
        # 标记前缀、其实不是"的几个字（比如正文正好以竖线结尾）—— 那是正常
        # 正文，不能吞（区段开着时 flush 会把整段丢掉，见 leak 模块）。
        tail = gate.flush()

        if tail:
            await on_delta(tail)

        # 累积出来的是 **chunk**，不是消息。图 state / checkpointer /
        # 历史处理器拿到的必须是真正的 AIMessage。
        message = message_chunk_to_message(aggregate)

        if metadata:
            # 覆盖掉被拼接出来的那一份（理由见上面 metadata 的说明）。
            message.response_metadata = metadata

        return message

    async def aclose(self, client: ModelClient) -> None:
        """
        释放客户端。

        langchain 的 chat 客户端持有 httpx 连接池，多数实现有 `aclose`；
        没有就跳过 —— 缺一个 close 方法不是错误，静默失败才是。
        """

        closer = getattr(client.client, "aclose", None)

        if closer is None:
            return

        try:
            await closer()

        except Exception:

            logger.warning(
                "关闭 provider 客户端 %s 时出错（忽略）：",
                client.model_id,
                exc_info=True,
            )

    # ========================================================
    # internals
    # ========================================================

    def _bind(
            self,
            client: ModelClient,
            request: ModelRequest,
    ) -> Any:
        """
        把请求的"工装"绑到客户端上（工具 / 结构化输出）。

        刻意保留与原 agent_node 相同的顺序（先 bind_tools 再
        with_structured_output）：两者都会返回新的 Runnable 包装，
        换个顺序在某些 provider 上会得到不同的请求体 —— 保持一致
        就不会引入一次静默的行为变化。

        ⚠️ 只有 `request.tools` 非空才 bind_tools：对不支持工具的模型
        调它会直接抛错，而"这次没有工具"是完全正常的一种请求。

        ⚠️ **同时**给 tools 与 response_format 时，tools 会被丢掉
        ------------------------------------------------------
        `bind_tools` 返回的是 RunnableBinding，而它上面的
        `with_structured_output` 经 `__getattr__` 落到**底层模型**的方法上，
        新建一个绑定并把外层那个（带着工具的）丢弃。实测最终请求里只剩
        结构化输出那个 schema 工具。

        今天不会撞上：分类调用只带 response_format、ReAct 主循环只带 tools。
        但"两个都带"在语义上本来也说不通（一个要模型自己选工具、一个要它
        只输出 JSON），真需要时应显式决定，而不是靠这里的顺序碰巧生效。
        """

        runnable = client.client

        # "关掉思考模式"那段 extra_body 由模型定义声明，**两条分支都要传** ——
        # 见 thinking_off_extra_body 里两种症状的说明。
        #
        # ⚠️ 这里曾经只传给 `with_structured_output`，于是带工具的调用照旧跑在
        # 思考模式下。而那种情况的症状**不是报错**，是模型把工具调用写成
        # DSML 标记当正文吐出来（图读不到 tool_calls → 工具一次没跑 → 那段
        # 标记被当回答发给用户）。所以漏传它没有任何迹象，直到有人看见那段
        # 怪文本。当时 thinking_off_extra_body 的判据也只看 response_format。
        extra = thinking_off_extra_body(client.definition, request)

        if request.tools and hasattr(runnable, "bind_tools"):
            runnable = runnable.bind_tools(list(request.tools), **extra)

        if request.response_format is not None and hasattr(
                runnable, "with_structured_output"
        ):
            # 有些端点（deepseek）默认跑在思考模式下，而思考模式拒绝
            # 结构化输出发出的强制 tool_choice —— 那一段 extra_body 由模型
            # 定义声明，**只在这一次调用上**生效。
            runnable = runnable.with_structured_output(
                request.response_format,
                **extra,
            )

        # ⚠️ 第三条分支：纯文本调用
        # -------------------------
        # 既不带 tools 也不带 response_format 时，上面两条一条都占不到 ——
        # 于是 `extra` 算出来没人用、**直接丢掉**。而这类调用恰恰是
        # `thinking_off_extra_body` 里那段"一律关"存在的**唯一理由**：
        # 2026-10-13 那次泄漏（run 51b31758）就是纯文本调用，模型把工具调用
        # 写成了 DSML 标记当正文吐出来，而判据一条都不占、思考照旧开着。
        #
        # 症状是**静默的**：声明 `disable_thinking: true` 的模型在这一类调用
        # 上仍然跑在思考模式下 —— 配置读起来像装好了，实际没拦住。所以
        # `bind` 这里不做 `hasattr` 兜底（另外两条要兜，因为不是所有模型都有
        # `bind_tools` / `with_structured_output`；`bind` 在 Runnable 基类上，
        # 恒在）—— 真缺了就该炸，而不是安静地不生效。
        if extra and not request.tools and request.response_format is None:
            runnable = runnable.bind(**extra)

        return runnable


# ============================================================
# internals
# ============================================================


def _log_request(
        client: ModelClient,
        request: ModelRequest,
) -> None:
    """
    DEBUG 级：这次请求的**正文**（提示词全文 + 工具 + 结构化输出要求）。

    ⚠️ 为什么默认看不见
    ------------------
    `observer.LoggingModelObserver` 有一条明确的立场：日志只打结论、不打
    正文 —— "日志不该是'模型说了什么'的第二份存储，日志既没有访问控制
    也没有保留策略"。这一条不推翻那个立场：它把正文压在 **DEBUG**，
    默认级别下不出现，要看的人显式调级别。

    ⚠️ 正文必须先过脱敏
    ------------------
    `redaction.py` 的规矩是"任何要落日志的东西都先过这里"。提示词里
    出现密钥通常是**意外**（用户粘贴、异常消息回显），不是有人故意写进去，
    所以不能靠"注意别打"。

    ⚠️ 不截断
    --------
    截断会让"模型到底看到了什么"这个唯一目的失效。代价是长对话的日志
    行很长 —— 这是打开 DEBUG 的人自己选的。

    级别判断放在这里而不是调用处：`isEnabledFor` 挡在两处拼接之前，
    否则每次调用都要先把整段对话渲染成字符串再丢掉。
    """

    if not logger.isEnabledFor(logging.DEBUG):
        return

    tools = getattr(request, "tools", ()) or ()
    schema = getattr(request, "response_format", None)

    logger.debug(
        "LLM 请求 -> %s:%s | %d 条消息 | 工具 %s | 结构化输出 %s\n%s",
        client.definition.provider,
        client.definition.model,
        len(request.messages),
        [getattr(t, "name", type(t).__name__) for t in tools] or "无",
        getattr(schema, "__name__", schema) or "无",
        _render_messages(request.messages),
    )


def _log_response(
        client: ModelClient,
        message: Any,
) -> None:
    """DEBUG 级：这次响应的正文（含工具调用）。与 `_log_request` 配套。"""

    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "LLM 响应 <- %s:%s\n%s",
        client.definition.provider,
        client.definition.model,
        _render_message(message),
    )


def _render_messages(messages: Any) -> str:
    """把一串 langchain 消息渲染成多行文本。"""

    return "\n".join(_render_message(m) for m in messages)


def _render_message(message: Any) -> str:
    """
    一条消息渲染成 `[role] 正文`，工具调用另起一行。

    内容可能是 str（常态），也可能是**块列表**（多模态 / 新式 content
    blocks）—— 后者原样 json 出来，省得在这里实现一套分块语义。
    渲染不出来时退回 `repr`：日志的职责是"看得见"，不是"格式完美"。
    """

    try:
        role = getattr(message, "type", type(message).__name__)

        content = getattr(message, "content", "")

        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)

        lines = [f"[{role}] {content}"]

        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            lines.append(
                "  tool_calls="
                + json.dumps(tool_calls, ensure_ascii=False, default=str)
            )

        return redact_text("\n".join(lines))

    except Exception:  # noqa: BLE001 — 渲染失败不该影响这次调用

        return f"<无法渲染 {type(message).__name__}: {redact_text(str(message))}>"


def thinking_off_extra_body(
        definition: ModelDefinition,
        request: ModelRequest,
) -> dict[str, Any]:
    """
    这次调用要额外并进请求体的参数 —— 目前只有"关掉思考模式"这一种。

    判定顺序（**关优先**）：

        1. 模型 / 调用方显式 disable_thinking  -> 关
        2. response_format 非空（JSON schema） -> 关（强制 tool_choice / 截断）
        3. request.enable_thinking             -> 开（覆盖"带 tools"）
        4. 带 tools                            -> 关（防 DSML 泄漏）
        5. 其余（纯文本）                      -> 不关，走 provider 默认

    ReAct 主循环把 `enable_thinking=True`，于是"带工具 -> 关思考"不再
    适用于它；但结构化输出（分类等）仍然关。

    两个判据**缺一不可**，合起来才是"按模型 × 这次要什么"：

        模型声明了 structured_output_extra_body    （model.yaml 逐模型声明）
        且 这次调用**与思考模式冲突**                 （见下，三种情形）
           或 这个模型**一律关**                       （disable_thinking）

    ⚠️ 第二条的第二个分支是 2026-10-13 加的，理由见下面"第三种冲突"那段的
    下半部分：按**调用形状**猜会漏，而漏的形状恰好是"看起来最无害"的纯文本
    调用。返回 `{}` 而不是 `None`：调用方直接 `**` 展开，空字典天然就是
    "什么都不加"，调用处因此没有分支。

    为什么要这一层
    -------------
    有些端点默认跑在思考模式下。三种**症状完全不同**的冲突：

        结构化输出        **直接 400**
            `with_structured_output(method="function_calling")` 会发强制
            tool_choice（langchain 对 `ChatDeepSeek` 的默认 method 就是这个，
            普通 `ChatOpenAI` 是 `json_schema`）。实测 deepseek：

                400 Thinking mode does not support this tool_choice

        带工具            **不报错，但工具调用漏成正文**
            `bind_tools` 发的是 `tool_choice="auto"`，撞不上那条 400；可实测
            deepseek-v4-flash 在思考模式下会把调用写成它自家的 DSML 标记，
            当普通正文吐出来：

                <|DSML| calls><|DSML| invoke name="memory_search"> …

            于是 langchain 在 content 里读不到 tool_calls，图判定"模型没要求
            调工具"，直接把那段标记当回答交给用户 —— **一次工具都没跑，也没
            有任何报错**。它看起来只是"模型答得怪"，比 400 难查得多。

        调用方显式要求  **输出被截断**（`ModelRequest.disable_thinking`）
            纯文本调用（格式写在提示词里、既不带 tools 也不带 response_format）
            拿不到前两条判据，于是它也在思考模式下跑。而 `max_tokens` 是
            **思考与答案共用**的：推理吃掉一大截，答案写到一半就被截 ——
            实测 L1 抽取报的是 `Unterminated string starting at …`，
            看起来像模型乱写，其实是**没写完**（那一轮重试救回来了，但
            每次重试都要重付一次模型调用）。

        ⚠️ 上面这三条是"按**这次调用长什么样**"判的，而它**漏过**一次
        ---------------------------------------------------------
        2026-10-13 实测（run 51b31758）：一次**纯文本**调用 —— 既不带 tools
        也不带 response_format、调用方也没要求关 —— 于是三条判据一条都不占，
        模型照旧跑在思考模式下，把工具调用写成 DSML 标记当正文吐出来（149 条
        delta 与 completed 全文都是标记，`finish_reason` 是 `stop`，运行表记
        `completed`：**没有任何一处报错**）。

        为什么"纯文本 + 没有工具"这一轮会想调工具：同一会话前面几轮真的调过，
        工具名在对话历史里，模型于是"接着调"。

        **结论**：按调用形状猜，永远有下一种漏法（这次是"看起来最无害"的那种
        形状）。所以多了一个**模型级**开关 `disable_thinking`：声明它的模型
        一律关，与这次调用长什么样无关（判据见 ModelDefinition 的说明）。

        ⚠️ 今天那个开关**没人开着**：主循环那个模型（flash）撤掉了它 ——
        因为思维链要留给用户看（`reasoning_content` 落库 + 推给前端），
        一律关就等于那条链路永远没有数据。于是策略回到**形状判据**：

            带工具 / 带结构化输出   关（这两类会撞 400 或 DSML）
            纯文本调用              开（用户能看到推理）

        代价是那个纯文本形状的泄漏可能回来，兜底在**出口的闸**
        （`leaked_tool_call` 的两道，正文与思维链都过），不是"应该不会发生"。
        真要去掉这一类风险，加回 `disable_thinking: true` 即可 —— 但那是
        拿"看不到推理"换的，见 model.yaml 里那段取舍。

    ⚠️ 判据的演化（两次都是被实测推着加的）
    --------------------------------
    1. 原来只覆盖结构化输出，注释写着"带工具的调用理论上也可能撞上同类限制，
       但没观察到，所以不预先加"。后来观察到了，症状是上面那段 DSML。
    2. 再后来纯文本的结构化抽取也被截断了（第三种），于是加了显式开关。

    两次都修正了同一个取舍：当时担心"加一个没验证过的参数会换来一条没人认识
    的 400"，而实际代价是**静默失效**或**间歇性截断** —— 都比一条 400 难查。

    ⚠️ 不要顺手把它扩到"所有调用"
    --------------------------
    关掉思考会让推理 / 摘要这类调用失去推理链（qwen 上还实测到输出 token 明显
    变少、不报错只变差，见 model.yaml）。所以判据只认"这次调用的输出形状由我们
    给定"，不认"这次调用是给人看的"。

    ⚠️ 没有声明的模型置了也没用
    ------------------------
    `structured_output_extra_body` 是**逐模型**声明且各 provider 写法不同的私有
    参数（deepseek 是 `thinking: {type: disabled}`，qwen 是
    `enable_thinking: false`）。模型没声明时这里返回 `{}` —— 静默不生效、不报错，
    所以给新模型配 `disable_thinking` 时要确认它声明过这一段。

    ⚠️ 不要在 `config` 里也写 extra_body
    ----------------------------------
    构造期的 `extra_body` 会进 `payload` 的默认值，而调用期这个是**浅覆盖**
    （`{**default_params, **kwargs}`）不是深合并 —— 两处都写时构造期那些键会
    无声消失。一处来源，就是这里。
    """

    extra_body = definition.structured_output_extra_body

    if not extra_body:
        return {}

    # 关优先：模型级 / 调用级显式关闭
    if definition.disable_thinking or request.disable_thinking:
        return _thinking_off_extra(extra_body)

    # 结构化输出必须关：思考模式不接受强制 tool_choice，且思考与正文
    # 共用 max_tokens，容易把 JSON 截断。
    if request.response_format is not None:
        return _thinking_off_extra(extra_body)

    # ReAct 主循环显式要求保留思考：**覆盖**"带 tools"这一条。
    if request.enable_thinking:
        return {}

    # 带普通工具：默认关，防止 DSML 泄漏。
    if request.tools:
        return _thinking_off_extra(extra_body)

    # 纯文本：不关，走 provider 默认（通常是思考模式）。
    return {}


def _thinking_off_extra(extra_body: Mapping[str, Any]) -> dict[str, Any]:
    """模型定义的关思考参数 -> 请求体片段（深拷贝）。"""

    # deepcopy 而不是 dict(...)：这是**嵌套**dict，浅拷贝只复制最外层 ——
    # 里层的 {"thinking": {...}} 仍与模型定义共享。而定义是进程级注册表里的
    # 那一份，谁就地改一下（langchain 合并 payload 时、或某个适配器顺手改），
    # 就把**所有**后续调用改掉了，且没有任何地方会报错。
    return {"extra_body": copy.deepcopy(dict(extra_body))}


def _client_creation_error(
        definition: ModelDefinition,
        exc: BaseException,
) -> ModelConfigurationError:
    """
    把"建客户端失败"包成一条能直接照做修的报错。

    `get_chat_model` 自己的报错已经不错（缺 key 时会说该设哪个环境
    变量），这里只补两件它看不到的事：是**哪个模型定义**要建，以及
    这次用的**凭据引用名**（不打印值）。
    """

    return ModelConfigurationError(
        f"无法为模型 {definition.id!r} 创建 provider 客户端：\n"
        f"    {type(exc).__name__}: {exc}\n"
        "常见原因：provider 没在 providers.yaml 的 llm 段里配置、API Key "
        "环境变量为空、或对应的 langchain provider 包没有安装。",
        provider=definition.provider,
        model=definition.model,
        details={
            "model_id": definition.id,
            "credential_ref": definition.credential_ref,
            "provider_error_type": type(exc).__name__,
        },
    )


def _chunk_text(chunk: Any) -> str:
    """
    这一块 chunk 里**用户可见的正文增量**。

    只认 `.text`。langchain 1.x 里它是**属性**（更早的版本是方法），
    两种都兜住。

    ⚠️ 为什么不能拿 `content` 顶替
    ----------------------------
    `.text` 天然排除 reasoning / thinking 块与 tool_call 参数；而
    `content` 在推理模型上是**块的列表**：

        [{"type": "reasoning", "reasoning": "..."},
         {"type": "text",      "text": "答案"}]

    直接把 content 流出去，等于把模型的草稿纸当正文推给用户；而
    tool-call 参数是 `{"ci` / `ty": "SF"}` 这种碎片，推进正文通道
    会把消费端的缓冲拼成一团乱码。两者都**不是** message.delta。

    返回空串 = "这一块没有正文可发"（纯 reasoning 块、纯 tool_call 块、
    心跳块），调用方据此跳过，而不是发一条空 delta。

    ⚠️ 不 strip：首尾空格是**真实的 token**，去掉会让消费端拼出来的
    正文与权威全文（message.completed 里那份）对不上。
    """

    attr = getattr(chunk, "text", None)

    # ⚠️ 顺序要紧：**先判 str，再判可调用**。
    #
    # langchain 1.x 的 `.text` 是属性，返回的是 `TextAccessor` —— 一个
    # **str 子类**，同时**也可调用**（1.x 为了兼容旧的 `msg.text()` 写法
    # 留的壳，调用那个形式会发 DeprecationWarning）。
    #
    # 所以 `callable(attr)` 在 1.x 上**永远为真**。先判可调用就等于对
    # 每一个 token 都走一次已弃用的路径：现在刷一屏弃用告警，等 1.x
    # 摘掉那个兼容壳时直接报错。
    if isinstance(attr, str):
        return attr

    # 真正的方法形式（更早的 langchain，`.text` 就是个方法）。
    if callable(attr):
        text = attr()
        return text if isinstance(text, str) else ""

    # 没有 `.text`（非 langchain 的鸭子对象）时兜底 —— 但**只在 content
    # 是字符串**时用。是列表就放弃：那样无法排除里面混着 reasoning 块。
    content = getattr(chunk, "content", "")

    return content if isinstance(content, str) else ""


def _extract_usage(message: Any) -> ModelUsage | None:
    """
    从响应里读 token 用量。

    两个位置都试，因为各家填的地方不同：

        usage_metadata       langchain 归一化后的位置（首选）
        response_metadata    provider 原始结构（openai 兼容端点常在这里
                             放 token_usage）

    都读不到就返回 None（= 没报），**不编 0** —— 见 ModelUsage 与
    ModelPricing.estimate 的说明。
    """

    usage = _as_mapping(getattr(message, "usage_metadata", None))

    parsed = ModelUsage.from_metadata(usage)

    if parsed is not None:
        return parsed

    response_metadata = _as_mapping(
        getattr(message, "response_metadata", None)
    )

    if response_metadata is None:
        return None

    for key in ("token_usage", "usage", "usage_metadata"):

        candidate = _as_mapping(response_metadata.get(key))

        parsed = ModelUsage.from_metadata(candidate)

        if parsed is not None:
            return parsed

    return None


def _strip_leaked_tool_calls(message: Any) -> int:
    """
    **就地**清掉消息正文里的"泄漏的工具调用"标记，返回命中处数。

    ⚠️ 为什么可以就地改
    ----------------
    `message` 是刚刚由 provider 的响应构造出来的（`message_chunk_to_message`
    或 `runnable.ainvoke` 的返回），还没有交给任何别的消费者；而它接下来要进
    图 state / checkpointer / 对话历史 / agent_message 表。留一份"原始版本"
    没有任何读者 —— 那段原始文本里除了标记什么都没有。

    只认 `content` 是**字符串**、或内容块列表里那些 `text` 块。别的形状不动
    （认不出就不猜）。真正防的是"标记混进可见正文"这一种，而实测泄漏就是这个
    形状。
    """

    content = getattr(message, "content", None)

    if isinstance(content, str):

        cleaned, hits = strip_leaked_tool_calls(content)

        if hits:
            message.content = cleaned

        return hits

    if isinstance(content, list):

        total = 0

        for block in content:

            if not isinstance(block, dict):
                continue

            text = block.get("text")

            if not isinstance(text, str):
                continue

            cleaned, hits = strip_leaked_tool_calls(text)

            if hits:
                block["text"] = cleaned
                total += hits

        return total

    return 0


def _strip_reasoning_leaks(message: Any) -> int:
    """
    **就地**清掉**思维链**里的泄漏标记，返回命中处数。

    与 `_strip_leaked_tool_calls` 是同一套规则的两个容器：那个管 `content`
    （可见正文），这个管思维链。两者分开而不是合成一个函数，因为命中数
    的**用途不同** —— 正文命中决定"这次回复是不是废的"（调用方据此抛
    ModelProviderError），思维链命中只是一段要清的文本。

    覆盖两种存放形状（与 `message_reasoning` 的取值来源一致）：

        additional_kwargs["reasoning_content"]      DeepSeek / Ollama / …
        content 里 type=reasoning 块的 reasoning 键   Anthropic 式

    ⚠️ 就地改 `additional_kwargs` 的那个 dict 是可以的
    -------------------------------------------------
    与 `message` 本身同一处境（刚刚由 provider 响应构造、还没有别的读者），
    而且它是**本次调用**的产物，不共享。改的是 dict 内部的键，不是重绑
    属性 —— 不触发 pydantic 的赋值校验。
    """

    total = 0

    kwargs = getattr(message, "additional_kwargs", None)

    if isinstance(kwargs, dict):

        value = kwargs.get("reasoning_content")

        if isinstance(value, str):

            cleaned, hits = strip_leaked_tool_calls(value)

            if hits:
                kwargs["reasoning_content"] = cleaned
                total += hits

    content = getattr(message, "content", None)

    if isinstance(content, list):

        for block in content:

            if not isinstance(block, dict):
                continue

            # 只认推理块：文本块的 `text` 归上面那个函数管，两处都改会让
            # 同一处命中被计两次。
            if block.get("type") != "reasoning":
                continue

            for key in ("reasoning", "text"):

                value = block.get(key)

                if not isinstance(value, str):
                    continue

                cleaned, hits = strip_leaked_tool_calls(value)

                if hits:
                    block[key] = cleaned
                    total += hits

    return total


def _run_config(
        run_metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """
    `run_metadata` -> langchain 的 runnable config。

    没有元信息就返回 **None** 而不是空 dict：`{"metadata": {}}` 与"没给
    config"在 langchain 里不等价 —— 后者是"用继承来的 config"，前者是
    "给这次调用一份自己的（空）metadata"。返回空 dict 会把父级的
    metadata 挡掉，而 `langgraph_node` 正住在那里（run_event_projector
    全靠它过滤节点）。

    ⚠️ 它只进 `config`，**不进模型参数**：`ainvoke` 的位置参数/kwargs
    会被当成模型参数透传给端点，塞进去在某些 provider 上直接 400。
    """

    if not run_metadata:
        return None

    return {"metadata": dict(run_metadata)}


def _provider_metadata(message: Any) -> dict[str, Any]:
    """
    摘几条 provider 侧的元信息进观测。

    刻意**白名单**而不是"把 response_metadata 整个抄进去"：那里面
    常常带着完整的请求头、配额、甚至组织 id，而观测行是持久化的。
    只取对排查真的有用的几个字段。
    """

    response_metadata = _as_mapping(
        getattr(message, "response_metadata", None)
    )

    if response_metadata is None:
        return {}

    return {
        key: response_metadata[key]
        for key in (
            "finish_reason",
            "model_name",
            "model",
            "id",
            "system_fingerprint",
        )
        if key in response_metadata
    }


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """只接受 Mapping（`[]` / None / 别的对象一律当"没有"）。"""

    return value if isinstance(value, Mapping) else None
