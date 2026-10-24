"""
Model Runtime：一次模型调用的**可靠性**执行者。

它是 LLM Node 唯一接触的模型调用抽象：

    response = await runtime.ainvoke(request)

Runtime 负责（规格 §8）
----------------------
    Model 调用           调 provider 客户端
    Timeout              每次 provider attempt 的超时由这里统一执行
    Retry                **只**对可重试错误做有限次重试
    Fallback/降级        primary 失败 -> fallback 1 -> fallback 2
    Capability 相容      降级目标必须满足本次调用要求的能力
    异常归一化            任何 provider 异常都在这里变成 Model 层异常
    Usage / Latency / Cost 逐次 attempt 的观测

Runtime **不**负责（规格 §9、§10、§11、§19）
-------------------------------------------
    ❌ 重新执行 LLM Node
    ❌ 重新构建上下文（Context Builder）
    ❌ 执行工具
    ❌ 读写 Graph State
    ❌ HITL / 等待人工 / 判断工具是否高风险
    ❌ 调用别的 Agent 或图（`graph.invoke` / `agent.run` / `tool.invoke`）

一句话的分界线：

    需要理解 Agent / Graph / Business Context 的逻辑 → LLM Node
    只需要理解 Model / Provider / Invocation 的逻辑 → Model Runtime

⚠️ 一次 invoke ≠ 一次 provider 调用（规格 §9）
--------------------------------------------
    一次 LLM Node execution
        ↓
    一次 ModelRuntime.ainvoke        <- 调用方只见这一层
        ↓
    N 次 Provider Attempt            <- retry / fallback 发生在这里

因此 fallback 不会让节点重跑、上下文不会重建、工具不会重执行、
Graph State 不会重放。这条语义是整个设计的地基：`ainvoke` 内部无论
试了几次、换了几家，对外都只是**一次**调用。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    ErrorDisposition,
    ModelCandidateFailure,
    ModelCandidatesExhaustedError,
    ModelCapabilityError,
    ModelError,
    ModelInvalidRequestError,
    ModelProviderError,
    ModelResolutionError,
    ModelTimeoutError,
    normalize_error,
)
from .observer import (
    ModelCallContext,
    ModelCallObserver,
    ModelCallSummary,
    NullModelObserver,
    emit_attempt,
    emit_call,
)
from .provider import DeltaCallback, ModelClient
from .redaction import redact_mapping
from .trace import ModelTrace
from app.common.event import (
    EventType as TraceEventType,
    SpanKind,
    SpanLink,
)
from app.common.event.trace import LinkRelation
from .types import (
    ModelAttempt,
    ModelCapabilities,
    ModelDefinition,
    ModelProfile,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelRuntimePolicy,
    ModelUsage,
    new_invocation_id,
    required_capabilities_for,
)

__all__ = ["ModelRuntime"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CallState:
    """
    一次 `ainvoke` 的**工作集**。

    与 `ModelRuntime` 实例分开是刻意的：Runtime 实例**会被并发共享**
    （Registry 按 profile 缓存它，见 registry.py），因此任何"这次调用的
    东西"都不能挂在 `self` 上 —— 否则 A 调用的 attempt 会串进 B 调用的
    metadata（与 `ContextBuildState` 不挂 self 是同一条纪律）。
    """

    invocation_id: str
    started: float = field(default_factory=time.perf_counter)

    attempts: list[ModelAttempt] = field(default_factory=list)

    #: 因能力不相容被剔除的候选（模型 id），按 profile 顺序。
    incompatible: list[str] = field(default_factory=list)

    response: ModelResponse | None = None
    last_error: ModelError | None = None
    estimated_cost: float | None = None

    #: 这次调用已经有多少个正文增量**真的发出去了**（0 = 还没有向任何
    #: 消费端承诺过内容）。它是"还能不能重试 / 降级"的唯一判据 ——
    #: 见 _try_candidate 里的 commit 规则。
    deltas_emitted: int = 0

    #: 本次调用的 trace 上下文（None = 未开启）。
    trace: ModelTrace | None = None

    #: 当前 attempt 的 Model Span（`TraceSpan` 句柄），成功 / 失败时收尾。
    current_span: Any = None

    #: 上一个失败 attempt 的 span_id / 关系，用于 retry_of / fallback_from
    #: Link。只在同一个 _CallState 内有效（并发调用各有一份）。
    last_failed_span_id: str | None = None
    last_failed_relation: LinkRelation | None = None

    def record(self, attempt: ModelAttempt) -> None:

        self.attempts.append(attempt)

        if attempt.estimated_cost is None:
            return

        self.estimated_cost = (
            attempt.estimated_cost
            if self.estimated_cost is None
            else self.estimated_cost + attempt.estimated_cost
        )

    @property
    def latency_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    @property
    def usage(self) -> ModelUsage | None:
        """
        这次调用的用量。

        成功时是**成功那次**的用量；失败时取最后一次报过用量的
        attempt（有些 provider 在截断 / 内容过滤的失败响应里也会报
        用量，那部分是真花了钱的，丢掉会让成本对不上）。
        """

        if self.response is not None:
            return self.response.usage

        for attempt in reversed(self.attempts):

            if attempt.usage is not None:
                return attempt.usage

        return None


class ModelRuntime:
    """
    一个 **Model Profile 的运行实例**。

    它是**无状态**的：所有"这次调用的东西"都在 `ainvoke` 的局部
    `_CallState` 里，因此同一个 Runtime 可以被并发调用共享 —— 这也是
    Registry 能缓存它的前提。
    """

    def __init__(
            self,
            *,
            profile: ModelProfile,
            factory: Any,
            definition_lookup: Callable[[str], ModelDefinition],
            agent_type: str,
            model_role: str,
            context: ModelResolveContext | None = None,
            observer: ModelCallObserver | None = None,
    ) -> None:

        self._profile = profile
        self._factory = factory
        self._lookup = definition_lookup
        self._observer: ModelCallObserver = observer or NullModelObserver()
        self._agent_type = agent_type
        self._model_role = model_role
        self._context = context or ModelResolveContext()

    # ========================================================
    # 只读属性
    # ========================================================

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def profile_id(self) -> str:
        return self._profile.id

    @property
    def agent_type(self) -> str:
        return self._agent_type

    @property
    def model_role(self) -> str:
        return self._model_role

    @property
    def policy(self) -> ModelRuntimePolicy:
        return self._profile.policy

    # ========================================================
    # 候选
    # ========================================================

    def candidates(
            self,
            required: ModelCapabilities | None = None,
    ) -> tuple[ModelDefinition, ...]:
        """
        这次调用会用到哪些模型（按尝试顺序）。

        **能力不相容的 fallback 不会出现在这里** —— 规格 §8.3 的例子
        （需要 tool_calling 却降级到不支持工具调用的模型）在结构上
        就不成立：它根本没进候选链。

        primary 不相容时**抛 ModelCapabilityError** 而不是返回空链：
        那是配置错，不该表现为"没有可用的模型"。
        """

        state = _CallState(invocation_id="candidates")

        return self._candidate_chain(
            self._requirements(None, required),
            self.policy,
            state,
        )

    def describe(self) -> dict[str, Any]:
        """给目录接口 / 日志的一份摘要。"""

        return {
            "profile": self._profile.describe(),
            "agent_type": self._agent_type,
            "model_role": self._model_role,
            "context": {
                "credential_ref": self._context.credential_ref,
                "attributes": dict(self._context.attributes),
                "required_capabilities": (
                    self._context.required_capabilities.as_dict()
                    if self._context.required_capabilities
                    else None
                ),
            },
        }

    # ========================================================
    # 执行
    # ========================================================

    async def ainvoke(
            self,
            request: ModelRequest,
            *,
            required_capabilities: ModelCapabilities | None = None,
            policy: ModelRuntimePolicy | None = None,
            on_delta: DeltaCallback | None = None,
            trace: ModelTrace | None = None,
    ) -> ModelResponse:
        """
        执行一次模型调用（内部可能包含多次 provider attempt）。

        Parameters
        ----------
        request
            这次调用的输入。**同一个对象**会被用于所有 attempt ——
            retry 与 fallback 重发的是同一个请求，这是规格 §9 的
            "继续相同 request"。
        required_capabilities
            调用点临时追加的能力要求（与节点的 context、请求内容三者
            取并集）。给了它就能让"这一次需要 vision"不影响到
            Registry 缓存的 Runtime。
        policy
            本次覆盖 profile 的策略（少数调用需要更短超时/更多重试）。
        on_delta
            正文增量的回调（见 `provider.DeltaCallback`）。给了它就**尽量**
            走流式，逐块把正文回调出去，返回值仍然是一条拼好的消息。
            不保证逐 token —— provider 不支持流式时会退化成一次性回调
            整段正文。

            ⚠️ **一旦有增量发出去，这次调用就不再重试、不再降级** ——
            理由见 `_try_candidate` 里的 commit 规则。

        Raises
        ------
        ModelCapabilityError
            primary 不满足要求的能力（配置错，调用前就失败）。
        ModelInvalidRequestError
            请求本身不合法（provider 拒绝，或 messages 为空）。
        ModelTimeoutError
            整条链的总预算耗尽（不再尝试剩余候选）。
        ModelCandidatesExhaustedError
            所有候选都失败（**它是 ModelUnavailableError 的子类**，
            因此 `except ModelUnavailableError` 能接住）。
        ModelError
            任何其它模型层错误。**永远不会**是 provider SDK 的异常。
        """

        active_policy = policy or self._profile.policy

        state = _CallState(
            invocation_id=new_invocation_id(),
            trace=trace if trace is not None and trace.emitter.enabled else None,
        )

        # 包一层只为了记住"已经发过增量了"（见 _delta_sink）——
        # 那个计数是 commit 规则的唯一依据，而 commit 规则又决定了
        # "还能不能重试"，所以这一层不能省。
        sink = (
            self._delta_sink(state, on_delta)
            if on_delta is not None
            else None
        )

        call_context = ModelCallContext(
            invocation_id=state.invocation_id,
            agent_type=self._agent_type,
            model_role=self._model_role,
            profile_id=self._profile.id,
        )

        if not request.messages:
            # 调用前的输入校验：不发生任何 provider attempt，因此也不
            # 产生调用级观测（没有可观测的过程）。
            raise ModelInvalidRequestError(
                "ModelRequest.messages 为空。",
                details={"profile_id": self._profile.id},
            )

        try:
            return await self._run(
                request,
                active_policy,
                required_capabilities,
                state,
                call_context,
                sink,
            )

        finally:
            # ⚠️ 观测**绝不能**影响业务结果。
            #
            # 这里是在 finally 里，所以"构造 summary 时抛错"会把调用方
            # 真正该看到的模型异常**顶掉** —— 那是最坏的一种遮蔽。因此
            # summary 的构造也包起来（与 infra/graph 的 _safe_record 同
            # 一条立场）。
            try:
                summary = self._summary(state, request)

            except Exception:

                logger.exception(
                    "构造模型调用观测记录失败（忽略，业务结果不受影响）："
                    "profile=%s invocation=%s",
                    self._profile.id,
                    state.invocation_id,
                )

            else:
                emit_call(self._observer, summary)

    # ========================================================
    # internals —— 执行
    # ========================================================

    async def _run(
            self,
            request: ModelRequest,
            policy: ModelRuntimePolicy,
            required_capabilities: ModelCapabilities | None,
            state: _CallState,
            call_context: ModelCallContext,
            on_delta: DeltaCallback | None,
    ) -> ModelResponse:

        requirements = self._requirements(
            request, required_capabilities
        )

        chain = self._candidate_chain(requirements, policy, state)

        deadline = (
            state.started + policy.total_timeout
            if policy.total_timeout is not None
            else None
        )

        if state.incompatible:
            # 剔除过候选 = 这次调用的可靠性**已知打了折**，值得留痕：
            # "为什么没用上 fallback" 是事后最难猜的问题之一。
            logger.warning(
                "profile %s 有 %d 个候选因能力不相容被剔除：%s"
                "（本次要求：%s）",
                self._profile.id,
                len(state.incompatible),
                state.incompatible,
                requirements.describe(),
            )

        for index, definition in enumerate(chain):

            if index and not policy.fallback_enabled:
                # 候选链本身已按开关构建；这里是第二道保险。
                break

            response = await self._try_candidate(
                definition,
                candidate_index=index,
                request=request,
                policy=policy,
                deadline=deadline,
                state=state,
                call_context=call_context,
                on_delta=on_delta,
            )

            if response is not None:
                state.response = response
                return response

            # 这个候选放弃了。只有链上还有下一个时才叫"降级"。
            if (
                    state.trace is not None
                    and state.last_failed_span_id is not None
                    and index + 1 < len(chain)
            ):
                last_error = state.last_error
                state.trace.emitter.instant(
                    TraceEventType.MODEL_FALLBACK,
                    span_kind=SpanKind.MODEL,
                    span_name="model.invoke",
                    span_id=state.last_failed_span_id,
                    parent_span_id=state.trace.parent_span_id,
                    payload={
                        "failed_span_id": state.last_failed_span_id,
                        "requested_model": definition.id,
                        "fallback_model": chain[index + 1].id,
                        "reason": (
                            last_error.code
                            if last_error is not None
                            and last_error.code
                            else type(last_error).__name__
                            if last_error is not None
                            else None
                        ),
                    },
                )

        # 链走完 —— 记账的必须是**抛出去的那个**错误，而不是最后一次
        # attempt 的错误：观测要能对上调用方拿到的东西（"exhausted" 与
        # "最后一家的 unavailable" 不是同一件事）。
        terminal = self._exhausted(chain, state)
        state.last_error = terminal

        raise terminal

    def _start_model_span(
            self,
            state: _CallState,
            definition: ModelDefinition,
            attempt_no: int,
    ) -> Any:
        """
        每个 provider attempt 一个 Model Span。

        重试 / 降级换新 Span，并通过 links 指向前一个失败的 Span —— 这是
        提示词第 4 节第 7 条。
        """

        state.current_span = None

        trace = state.trace

        if trace is None:
            return None

        links: list[SpanLink] = []

        if state.last_failed_span_id and state.last_failed_relation is not None:
            links.append(
                SpanLink(
                    relation=state.last_failed_relation,
                    span_id=state.last_failed_span_id,
                )
            )

        span = trace.emitter.start(
            TraceEventType.MODEL_STARTED,
            span_kind=SpanKind.MODEL,
            span_name="model.invoke",
            end_type=TraceEventType.MODEL_COMPLETED,
            failed_type=TraceEventType.MODEL_FAILED,
            parent_span_id=trace.parent_span_id,
            attempt=attempt_no,
            links=links,
            payload={
                "model_role": self._model_role,
                "provider": definition.provider,
                "model_name": definition.model,
                # 当前 ainvoke 路径由 langgraph 的协议流式驱动（见
                # agent_node 的说明）；有没有 token 由投影层按实际 delta
                # 判断，这里声明的是"这次请求走了流式通道"。
                "streaming": True,
            },
        )

        state.current_span = span

        return span

    def _fail_model_span(
            self,
            state: _CallState,
            error: BaseException,
    ) -> None:
        span = state.current_span

        if span is not None:
            span.fail(error)
            state.last_failed_span_id = span.span_id

        state.current_span = None

    async def _try_candidate(
            self,
            definition: ModelDefinition,
            *,
            candidate_index: int,
            request: ModelRequest,
            policy: ModelRuntimePolicy,
            deadline: float | None,
            state: _CallState,
            call_context: ModelCallContext,
            on_delta: DeltaCallback | None = None,
    ) -> ModelResponse | None:
        """
        试一个候选模型，返回 None 表示"这个候选放弃了，换下一个"。

        只有两种情况会**抛**出去而不是返回 None：

            FAIL 语义的错误（请求不合法、能力不符）—— 换谁都没用
            总预算耗尽 —— 剩下的候选也没有时间了
        """

        try:
            # `aget` 而不是 `get`：建客户端的路径上挂着一串惰性 import
            # （`langchain_openai` -> `openai.types...`，实测按住事件循环
            # 7 秒），必须丢出循环 —— 详见 factory.aget 与
            # app/common/llm.py 的 warmup_chat_models。
            client = await self._factory.aget(
                definition, timeout=policy.timeout
            )

        except ModelError as error:

            # 客户端建不出来（provider 没注册、凭据缺失）。记一次
            # "没发生的 attempt"：链上"这家试过了"与"这家根本没起来"
            # 是两件事，观测里必须分得开。
            state.record(
                ModelAttempt(
                    provider=definition.provider,
                    model=definition.model,
                    candidate_index=candidate_index,
                    attempt=0,
                    latency_ms=0,
                    success=False,
                    error_type=type(error).__name__,
                    error_code=error.code,
                    error_message=error.message,
                )
            )

            state.last_error = error

            emit_attempt(
                self._observer, state.attempts[-1], context=call_context
            )

            # 客户端都没建起来：这不是一次 provider 请求，但它是"这家候选
            # 失败了"的事实。开一个 Span 记下来，attempt 仍从 1 开始。
            self._start_model_span(state, definition, 1)
            self._fail_model_span(state, error)

            if error.disposition is ErrorDisposition.FAIL:
                raise error

            return None

        for attempt_no in range(1, policy.max_attempts + 1):

            budget = self._budget(deadline)

            if budget is not None and budget <= 0:

                error = ModelTimeoutError(
                    "模型调用的总预算（ModelRuntimePolicy.total_timeout）"
                    "已经耗尽，不再尝试剩余候选。",
                    provider=definition.provider,
                    model=definition.model,
                    details={
                        "profile_id": self._profile.id,
                        "total_timeout": policy.total_timeout,
                        "attempted": [
                            attempt.model_id
                            for attempt in state.attempts
                        ],
                    },
                )

                state.last_error = error

                raise error

            attempt_timeout = (
                policy.timeout
                if budget is None
                else min(policy.timeout, budget)
            )

            started = time.perf_counter()

            span = self._start_model_span(state, definition, attempt_no)

            try:
                result = await self._dispatch(
                    client,
                    request,
                    attempt_timeout,
                    on_delta=on_delta,
                    trace_span_id=None if span is None else span.span_id,
                    trace_parent_span_id=(
                        state.trace.parent_span_id
                        if state.trace is not None
                        else None
                    ),
                )

            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                # 取消**不是** provider 失败：把 CancelledError 归成
                # ModelProviderError 会让"上游超时取消"变成"provider
                # 抽风，再试两次" —— 那是最糟的一种错误翻译。
                raise

            except BaseException as exc:  # noqa: BLE001

                error = normalize_error(
                    exc,
                    provider=definition.provider,
                    model=definition.model,
                )

                state.record(
                    ModelAttempt(
                        provider=definition.provider,
                        model=definition.model,
                        candidate_index=candidate_index,
                        attempt=attempt_no,
                        latency_ms=_ms(started),
                        success=False,
                        error_type=type(error).__name__,
                        error_code=error.code,
                        error_message=error.message,
                        timeout=attempt_timeout,
                    )
                )

                state.last_error = error

                emit_attempt(
                    self._observer, state.attempts[-1], context=call_context
                )

                self._fail_model_span(state, error)

                decision = self._decide(error, policy)

                if state.deltas_emitted:
                    # ⚠️ commit 规则：正文已经**发出去了**，消费端看到了
                    # 半截。重试或降级会让它再看到**第二份**正文，而我们
                    # 没有任何机制能撤回第一份（Redis Stream 没有撤回
                    # 原语，已渲染的界面也擦不掉）。
                    #
                    # 所以一旦有增量出去，这次 attempt 就是**终局**：
                    # 不管这个错误本身多"可重试"，都直接抛。
                    #
                    # 反过来说，连接失败 / 429 / 还没吐一个字就挂的情况
                    # deltas_emitted 仍是 0，正常重试与降级照旧 ——
                    # 那才是瞬时故障的大多数。
                    logger.warning(
                        "模型 %s 在发出 %d 个正文增量之后失败（%s）："
                        "不重试、不降级 —— 已经推给消费端的内容无法撤回。",
                        definition.id,
                        state.deltas_emitted,
                        error.code,
                    )

                    raise error

                if decision is ErrorDisposition.FAIL:
                    raise error

                if (
                        decision is ErrorDisposition.RETRY
                        and attempt_no < policy.max_attempts
                ):

                    delay = self._backoff(policy, attempt_no)

                    logger.debug(
                        "模型 %s 第 %d 次尝试失败（%s），%.2fs 后重试。",
                        definition.id,
                        attempt_no,
                        error.code,
                        delay,
                    )

                    # trace：重试决策本身也是一条事件（频率低，值得留）。
                    if span is not None and state.trace is not None:
                        state.trace.emitter.instant(
                            TraceEventType.MODEL_RETRY_SCHEDULED,
                            span_kind=SpanKind.MODEL,
                            span_name="model.invoke",
                            span_id=span.span_id,
                            parent_span_id=state.trace.parent_span_id,
                            attempt=attempt_no,
                            payload={
                                "failed_span_id": span.span_id,
                                "next_attempt": attempt_no + 1,
                                "delay_ms": int(delay * 1000),
                                "reason": error.code or type(error).__name__,
                            },
                        )
                        state.last_failed_relation = LinkRelation.RETRY_OF

                    if delay > 0:
                        await asyncio.sleep(delay)

                    continue

                logger.debug(
                    "候选模型 %s 放弃（%s，第 %d 次尝试）：换下一个候选。",
                    definition.id,
                    error.code,
                    attempt_no,
                )

                # 下一个候选的 attempt 会通过 links 指回这次失败。
                # 具体是"降级"还是"单纯放弃"由 _run 判断（它知道链上
                # 还有没有下一个）。
                state.last_failed_relation = LinkRelation.FALLBACK_FROM

                return None

            else:
                return self._succeed(
                    definition,
                    result,
                    request=request,
                    candidate_index=candidate_index,
                    policy=policy,
                    state=state,
                    call_context=call_context,
                    attempt_no=attempt_no,
                    attempt_timeout=attempt_timeout,
                    started=started,
                )

        # for 正常结束 = 重试次数用尽。上面的 return None 已经覆盖了
        # 每一个出口，这里只是为了让类型与阅读都完整。
        return None

    def _delta_sink(
            self,
            state: _CallState,
            on_delta: DeltaCallback,
    ) -> DeltaCallback:
        """
        把调用方的回调包一层，**只**为了记住"已经发过增量了"。

        这个计数是 commit 规则（见 `_try_candidate`）的唯一依据，因此
        必须挂在 `_CallState` 上：Runtime 实例是**并发共享**的（Registry
        按 profile 缓存它），挂在 self 上会让 A 调用的增量串进 B 调用的
        判断 —— 于是 B 明明一个字都没发，却被判成"不能重试"。

        顺带挡掉空串：空 delta 对消费端没有意义，但它会让
        `deltas_emitted` 涨，从而误触 commit 规则。
        """

        async def sink(text: str) -> None:

            if not text:
                return

            state.deltas_emitted += 1

            await on_delta(text)

        return sink

    async def _dispatch(
            self,
            client: ModelClient,
            request: ModelRequest,
            timeout: float,
            *,
            on_delta: DeltaCallback | None = None,
            trace_span_id: str | None = None,
            trace_parent_span_id: str | None = None,
    ) -> Any:
        """
        真正把请求发给 provider，并**强制**一个超时。

        ⚠️ 随调用一起下去的是**这次调用是谁**（agent_type / model_role）
        ------------------------------------------------------------
        它走 `run_metadata` 挂到 provider 调用框架的运行元信息上，不是
        发给 provider 的参数。理由不是观测，是**下游分不清调用**：
        langgraph 的 `messages` 事件只带 `langgraph_node`，而同一个节点里
        会跑好几次模型调用（主循环 + 上下文压缩），事件流那边只能靠
        节点名区分 —— 于是压缩结果会被当成助手正文推给用户（实测过，
        见 run_event_projector 的 message_roles）。

        超时执行点有两个，这里都要：

            客户端构造时的 timeout    SDK 自己会中断请求并释放连接
            wait_for                 兜底：SDK 的超时不覆盖 bind_tools /
                                     with_structured_output / 排队等
                                     事前工作，也不保证一定生效
                                      （有些兼容端点会忽略它）

        两个都留是刻意的：只有一个时，超时会在某些 provider 上"不生效"，
        而那种失败表现为"这次调用挂了 10 分钟"，最难查。

        ⚠️ 流式调用时 wait_for 覆盖的是**整段消费**（发请求 -> 最后一个
        chunk），而不只是"首字节之前"。所以一个慢的 on_delta 回调会吃掉
        attempt 的超时预算；而中途被取消时 `deltas_emitted` 已经 > 0，
        commit 规则会让这次调用直接失败、不重试。投递侧因此必须快
        （见 app/common/stream/publisher.py 里"为什么是 await"）。
        """

        run_metadata: dict[str, Any] = {
            "agent_type": self._agent_type,
            "model_role": self._model_role,
        }

        # Trace 关联：消息协议的每条事件都会带上这份 metadata，
        # 投影层据此把 message-start / 首个 delta 对到 Model Span 上
        # （model.first_token 的 ttft 计算）。
        if trace_span_id is not None:
            run_metadata["trace_span_id"] = trace_span_id
            run_metadata["trace_parent_span_id"] = trace_parent_span_id
            run_metadata["trace_streaming"] = True

        return await asyncio.wait_for(
            client.adapter.ainvoke(
                client,
                request,
                timeout=timeout,
                on_delta=on_delta,
                run_metadata=run_metadata,
            ),
            timeout=timeout,
        )

    def _succeed(
            self,
            definition: ModelDefinition,
            result: Any,
            *,
            request: ModelRequest,
            candidate_index: int,
            policy: ModelRuntimePolicy,
            state: _CallState,
            call_context: ModelCallContext,
            attempt_no: int,
            attempt_timeout: float,
            started: float,
    ) -> ModelResponse:

        usage = result.usage

        cost = (
            definition.pricing.estimate(usage)
            if definition.pricing is not None
            else None
        )

        state.record(
            ModelAttempt(
                provider=definition.provider,
                model=definition.model,
                candidate_index=candidate_index,
                attempt=attempt_no,
                latency_ms=_ms(started),
                success=True,
                usage=usage,
                estimated_cost=cost,
                timeout=attempt_timeout,
            )
        )

        emit_attempt(
            self._observer, state.attempts[-1], context=call_context
        )

        provider = result.provider or definition.provider
        model = result.model or definition.model

        # "重试过" = 整次调用里**曾经**有过第 2 次尝试（不限于最后成功
        # 的那一次）。原来那版（只看成功 attempt 的下标）在"primary 重试
        # 两次后换 backup、backup 一次成功"的场景里会说 retried=False ——
        # 那正是最需要看见"这次重试过"的场景。
        retried = any(
            attempt.attempt > 1 for attempt in state.attempts
        )

        degraded = (
            len(state.attempts) > 1
            or candidate_index > 0
            or bool(state.incompatible)
        )

        # ---- Trace: 收尾当前 Model Span ----
        span = state.current_span
        state.current_span = None

        if span is not None:
            metadata = result.metadata or {}
            provider_metadata = metadata.get("provider_metadata")
            provider_metadata = (
                provider_metadata
                if isinstance(provider_metadata, dict)
                else metadata
            )
            response_metadata = getattr(
                result.message, "response_metadata", None
            )
            response_metadata = (
                response_metadata
                if isinstance(response_metadata, dict)
                else {}
            )

            span.complete(
                payload={
                    "input_tokens": (
                        None if usage is None else usage.input_tokens
                    ),
                    "output_tokens": (
                        None if usage is None else usage.output_tokens
                    ),
                    "finish_reason": (
                        provider_metadata.get("finish_reason")
                        or response_metadata.get("finish_reason")
                    ),
                    "response_id": (
                        provider_metadata.get("id")
                        or getattr(result.message, "id", None)
                    ),
                }
            )

        return ModelResponse(
            message=result.message,
            provider=provider,
            model=model,
            usage=usage,
            latency_ms=state.latency_ms,
            attempts=len(state.attempts),
            degraded=degraded,
            metadata={
                "invocation_id": state.invocation_id,
                "profile_id": self._profile.id,
                "agent_type": self._agent_type,
                "model_role": self._model_role,
                "model_id": definition.id,
                "fallback_used": candidate_index > 0,
                "retried": retried,
                "provider_attempts": len(state.attempts),
                "incompatible_candidates": list(state.incompatible),
                "attempts": [
                    attempt.as_metadata() for attempt in state.attempts
                ],
                "request": request.describe(),
                "request_metadata": redact_mapping(dict(request.metadata)),
                "policy": policy.describe(),
                "estimated_cost": state.estimated_cost,
                "provider_metadata": redact_mapping(dict(result.metadata)),
            },
        )

    # ========================================================
    # internals —— 候选链与策略
    # ========================================================

    def _candidate_chain(
            self,
            requirements: ModelCapabilities,
            policy: ModelRuntimePolicy,
            state: _CallState,
    ) -> tuple[ModelDefinition, ...]:
        """
        primary + 相容的 fallbacks。

        ⚠️ primary 不相容 -> 抛
        ----------------------
        "主模型不够用"是**配置错**（profile 指错了模型，或模型的能力
        声明与实际不符），把它静默降级会让"我们一直在用一个做不到这件事
        的模型"长期无人发现。因此直接抛，且信息里列出缺哪几位能力。

        ⚠️ fallback 不相容 -> 剔除 + 记警告
        ---------------------------------
        这是常态：fallback 常常是更便宜 / 更小的模型，它做不到某些事
        完全正常。剔除它，并把它记进 `incompatible_candidates` 让观测
        看得见 —— 否则"为什么没降级"永远查不出来。
        """

        primary = self._definition(self._profile.primary)

        missing = primary.capabilities.missing(requirements)

        if missing:
            raise ModelCapabilityError(
                f"主模型 {primary.id!r} 不支持本次调用要求的能力："
                f"{list(missing)}。\n"
                f"它声明支持：{primary.capabilities.describe()}\n"
                f"本次要求：{requirements.describe()}\n"
                "两条路：换 profile 的 primary，或修正该模型的能力声明"
                "（model.yaml）。\n"
                "**不会**静默降级：那会让“一直用一个做不到这件事的模型”"
                "长期无人发现。",
                provider=primary.provider,
                model=primary.model,
                details={
                    "profile_id": self._profile.id,
                    "missing": list(missing),
                    "required": requirements.as_dict(),
                    "supported": primary.capabilities.as_dict(),
                },
            )

        chain = [primary]

        if not policy.fallback_enabled:
            return tuple(chain)

        for reference in self._profile.fallbacks:

            fallback = self._definition(reference)

            if fallback.id in {item.id for item in chain}:
                # profile 里已经禁止重复候选（见 types.py），这里防的是
                # **别名**指向同一个模型（primary="a:x"，fallback="x"）。
                logger.debug(
                    "profile %s 的候选 %s 与前面重复（别名），跳过。",
                    self._profile.id,
                    fallback.id,
                )
                continue

            missing = fallback.capabilities.missing(requirements)

            if missing:

                state.incompatible.append(fallback.id)

                logger.warning(
                    "候选 %s 因能力不相容被剔除（缺 %s；本次要求 %s），"
                    "不参与本次降级。",
                    fallback.id,
                    list(missing),
                    requirements.describe(),
                )

                continue

            chain.append(fallback)

        return tuple(chain)

    def _requirements(
            self,
            request: ModelRequest | None,
            extra: ModelCapabilities | None,
    ) -> ModelCapabilities:
        """
        本次调用的能力要求 = 三处取并集。

            节点给的 context.required_capabilities  （"我知道我要 vision"）
            request.required_capabilities            （"这次请求要工具"）
            调用点临时给的 extra                       （"这一次特别需要…"）
            request 内容推导                          （tools / response_format）
        """

        requirements = (
            self._context.required_capabilities
            or ModelCapabilities()
        )

        if request is not None:
            requirements = requirements.merged(
                request.required_capabilities
            ).merged(required_capabilities_for(request))

        return requirements.merged(extra)

    def _decide(
            self,
            error: ModelError,
            policy: ModelRuntimePolicy,
    ) -> ErrorDisposition:
        """
        这个错误该怎么处理。

        类的 disposition 是**默认**，policy.retryable_errors 是**收窄**：
        一个 disposition=RETRY 的错误，如果它的错误码不在策略的表里，
        就退化成 FALLBACK（换下一个候选，而不是原地重试）。

        这让"这次别重试"成为一个按 profile 配置的开关，也保证
        "不能对所有异常无脑 retry"（规格 §8.2）在代码上是**默认**行为
        而不是纪律要求。
        """

        disposition = error.disposition

        if disposition is ErrorDisposition.RETRY and not policy.allows_retry(
                error.code
        ):
            return ErrorDisposition.FALLBACK

        return disposition

    def _exhausted(
            self,
            chain: tuple[ModelDefinition, ...],
            state: _CallState,
    ) -> ModelError:
        """
        候选链走完了 —— 决定抛什么。

        单候选：抛**原始错误**。没有链可怪，包一层只会把"请求不合法"
        糊成"服务不可用"，而这两种错的修法完全不同。

        多候选：抛统一的 `ModelCandidatesExhaustedError`
        （`ModelUnavailableError` 的子类）。规格 §22 要求"所有 provider
        都失败时返回统一的 ModelUnavailableError，而不是泄漏
        provider-specific exception"；details 里带上逐家明细，信息不丢。
        """

        last = state.last_error

        if last is None:
            # 理论上到不了：能走到这里说明每个候选都返回了 None，
            # 而返回 None 的分支一定先写了 last_error。
            last = ModelProviderError(
                "所有候选模型都没有产生结果，且没有记录到具体错误。",
                details={"profile_id": self._profile.id},
            )

        if len(chain) == 1:

            # 单候选不等于"没有链上的信息"：候选可能**曾经**有多个，只是
            # 被能力门禁剔到只剩一个。那种情况下调用方最需要知道的恰恰是
            # "你的 fallback 为什么没被用上"，所以把它补进原始错误的
            # details —— 换错误类型就为了带这行信息并不划算（原始类型
            # 更精确，比如"超时"与"不可用"是两种修法）。
            if state.incompatible:

                last.details.setdefault(
                    "incompatible_candidates", list(state.incompatible)
                )
                last.details.setdefault("profile_id", self._profile.id)

            return last

        failures = [
            ModelCandidateFailure(
                f"{attempt.model_id} 第 {attempt.attempt} 次尝试失败："
                f"{attempt.error_type}({attempt.error_code})",
                provider=attempt.provider,
                model=attempt.model,
                details={
                    "attempt": attempt.attempt,
                    "candidate_index": attempt.candidate_index,
                    "error_code": attempt.error_code,
                },
            )
            for attempt in state.attempts
            if not attempt.success
        ]

        return ModelCandidatesExhaustedError(
            f"{len(chain)} 个候选模型全部失败"
            f"（{[item.id for item in chain]}）；"
            f"最后一次失败：{last}",
            provider=last.provider,
            model=last.model,
            details={
                "profile_id": self._profile.id,
                "candidates": [item.id for item in chain],
                "incompatible_candidates": list(state.incompatible),
                "provider_attempts": len(state.attempts),
                "failures": [
                    {
                        "model_id": failure.qualified_model,
                        "attempt": failure.details.get("attempt"),
                        "error_code": failure.details.get("error_code"),
                        "message": failure.message,
                    }
                    for failure in failures
                ],
                "last_error": last.as_metadata(),
            },
        )

    def _definition(self, reference: str) -> ModelDefinition:
        """把一个模型引用（id 或别名）解析成定义。"""

        definition = self._lookup(reference)

        if definition is not None:
            return definition

        raise ModelResolutionError(
            f"profile {self._profile.id!r} 引用的模型 {reference!r} "
            "不在注册表里。\n"
            "常见原因：model.yaml 里删了该模型定义却忘了改 profile，"
            "或者主/备模型写错了名字。",
            details={
                "profile_id": self._profile.id,
                "reference": reference,
            },
        )

    # ========================================================
    # internals —— 汇总与工具
    # ========================================================

    def _summary(
            self,
            state: _CallState,
            request: ModelRequest,
    ) -> ModelCallSummary:
        """
        构造整次调用的观测记录。

        ⚠️ 无论成功失败**都要发**：只记成功的调用会让"失败率"这个
        最重要的数字等于 0，而那正是最需要它的地方。
        """

        response = state.response

        error = state.last_error if response is None else None

        return ModelCallSummary(
            invocation_id=state.invocation_id,
            agent_type=self._agent_type,
            model_role=self._model_role,
            profile_id=self._profile.id,
            started_at=state.started,
            latency_ms=state.latency_ms,
            success=response is not None,
            degraded=response.degraded if response is not None else bool(
                state.attempts
            ),
            provider=(
                response.provider
                if response is not None
                else (error.provider if error else None)
            ),
            model=(
                response.model
                if response is not None
                else (error.model if error else None)
            ),
            attempts=tuple(state.attempts),
            usage=state.usage,
            estimated_cost=state.estimated_cost,
            incompatible_candidates=tuple(state.incompatible),
            error_type=type(error).__name__ if error else None,
            error_code=error.code if error else None,
            error_message=error.message if error else None,
            request_metadata=redact_mapping(dict(request.metadata)),
        )

    @staticmethod
    def _budget(deadline: float | None) -> float | None:
        """距离总预算耗尽还剩多少秒；没有 deadline 时是 None。"""

        if deadline is None:
            return None

        return deadline - time.perf_counter()

    @staticmethod
    def _backoff(
            policy: ModelRuntimePolicy,
            attempt_no: int,
    ) -> float:
        """
        第 `attempt_no` 次尝试失败后的等待时间（指数退避，带上限）。

        `retry_backoff=0`（默认）时恒为 0 —— 不退避。生产上遇到限流
        就该在 profile 里设它，而不是让"重试"变成"立刻再撞一次"。
        """

        if policy.retry_backoff <= 0:
            return 0.0

        delay = policy.retry_backoff * (2 ** (attempt_no - 1))

        return min(delay, policy.max_retry_backoff)


def _ms(started: float) -> int:
    """perf_counter 差值 -> 毫秒。"""

    return int((time.perf_counter() - started) * 1000)
