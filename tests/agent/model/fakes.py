"""
Model 模块测试用的假实现。

它们只实现**端口**（ProviderAdapter / CredentialResolver / Observer /
ContextBuilder / Recorder），不碰任何真 provider —— 因此整套测试不需要
密钥、不联网、不依赖模型服务，跑起来是毫秒级的。

⚠️ 为什么是"脚本化"而不是"打桩返回固定值"
---------------------------------------
规格 §22 要求的几个场景（retry、fallback、能力不相容）本质上是
**一连串不同的结果**：

    attempt 1 -> timeout
    attempt 2 -> success

所以假适配器的核心是"按模型 id 排队一串预设结果"，而不是一个固定的
返回。`Outcome` 就是这个队列的元素。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from app.agent.context.builder.base import AgentInvokeContextBuilder
from app.agent.context.builder.contracts import (
    ContextBuildState,
    ModelRequest as ContextModelRequest,
)
from app.agent.model import (
    DefaultModelRegistry,
    DeltaCallback,
    ModelCapabilities,
    ModelDefinition,
    ModelProfile,
    ModelRoleBinding,
    ProviderResult,
)
from app.agent.model.types import ModelUsage
from app.common.context import StageId

__all__ = [
    "FakeAdapter",
    "FakeCredentialResolver",
    "FakeRecorder",
    "Outcome",
    "RecordingObserver",
    "StubContextBuilder",
    "fake_exception",
    "make_definition",
    "make_registry",
    "make_run_context",
]


# ============================================================
# 假 provider 适配器
# ============================================================


@dataclass
class Outcome:
    """
    一次 provider attempt 的预设结果。

    下面几条只该给一个（都不给 = 返回一条默认消息）：

        message    成功，返回这条消息
        usage      成功时的用量
        error      失败，抛这个异常
        delay      先睡这么久（用来触发超时）

    流式相关的两条（`deltas` / `raise_after`）与 `error` 配合：

        deltas       调用方给了 on_delta 时，先逐块吐这些正文增量
        raise_after  吐完第 N 个增量之后抛 error —— 用来测 commit 规则
                     （"已经发出增量就不能再重试/降级"）。
                     None = 吐完所有增量再按 error 决定
    """

    message: Any = None
    usage: ModelUsage | None = None
    error: BaseException | None = None
    delay: float = 0.0

    deltas: tuple[str, ...] = ()
    raise_after: int | None = None


class FakeAdapter:
    """
    满足 `ProviderAdapter` 协议的假实现。

    记录四样东西，供断言：

        calls         每次 ainvoke 的 (model_id, timeout) —— 顺序即真实顺序
        created       每次 create_client 的参数（含**凭据**与 max_retries）
        closed        被关闭的客户端数
        run_metadata  每次调用带下去的**身份**（agent_type / model_role）——
                      Runtime 注入了什么，只有这里看得见
    """

    name = "fake"

    def __init__(
            self,
            script: Mapping[str, Sequence[Outcome]] | None = None,
            *,
            default: Outcome | None = None,
    ) -> None:

        self._script: dict[str, list[Outcome]] = {
            key: list(value) for key, value in (script or {}).items()
        }

        self._default = default or Outcome(message="ok")

        self.calls: list[tuple[str, float]] = []
        self.created: list[dict[str, Any]] = []
        self.closed: int = 0

        #: Runtime 每次调用传下来的身份（见 provider.py 的 run_metadata）。
        self.run_metadata: list[dict[str, Any]] = []

        #: 实际回调出去的正文增量，顺序即真实顺序。
        self.deltas: list[str] = []

        #: 交给 adapter 的请求对象（用来断言"重试/降级用的是同一个
        #: request" —— 规格 §9：继续相同 request，而不是重新构建）。
        self.requests: list[Any] = []

    # --------------------------------------------------------

    def create_client(
            self,
            definition: ModelDefinition,
            *,
            credential: str | None,
            timeout: float,
            max_retries: int,
    ) -> Any:

        self.created.append({
            "model_id": definition.id,
            "credential": credential,
            "timeout": timeout,
            "max_retries": max_retries,
        })

        # 客户端对象只要带 definition 就够了 —— 假适配器的 ainvoke
        # 需要知道"这次是哪个模型"。
        return SimpleNamespace(definition=definition)

    async def ainvoke(
            self,
            client: Any,
            request: Any,
            *,
            timeout: float,
            on_delta: DeltaCallback | None = None,
            run_metadata: Mapping[str, Any] | None = None,
    ) -> ProviderResult:

        model_id = client.definition.id

        self.calls.append((model_id, timeout))
        self.requests.append(request)
        self.run_metadata.append(dict(run_metadata or {}))

        outcome = self._next(model_id)

        if outcome.delay:
            # ⚠️ 真的睡：超时是 Runtime 用 asyncio.wait_for 执行的，
            # 不真睡就测不到它。delay 一律取得远大于测试用的 timeout。
            await asyncio.sleep(outcome.delay)

        # 先吐增量。`await on_delta(...)` 与真适配器一样是**顺序**回调 ——
        # commit 规则依赖那个顺序与计数（Runtime 的 _delta_sink 数的就是
        # 这里调了几次）。
        #
        # 注意 Runtime **总是**把 on_delta 传下来（没流式时是 None），
        # 所以这个参数必须收下，哪怕本次用不到。
        if on_delta is not None:

            for index, text in enumerate(outcome.deltas, start=1):

                self.deltas.append(text)

                await on_delta(text)

                if (
                        outcome.raise_after == index
                        and outcome.error is not None
                ):
                    raise outcome.error

        if outcome.error is not None:
            raise outcome.error

        return ProviderResult(
            message=outcome.message if outcome.message is not None else "ok",
            usage=outcome.usage,
            provider=client.definition.provider,
            model=client.definition.model,
            metadata={"finish_reason": "stop"},
        )

    async def aclose(self, client: Any) -> None:
        self.closed += 1

    # --------------------------------------------------------

    def _next(self, model_id: str) -> Outcome:

        queue = self._script.get(model_id)

        if queue:
            return queue.pop(0)

        return self._default

    def call_count(self, model_id: str | None = None) -> int:

        if model_id is None:
            return len(self.calls)

        return sum(1 for item, _ in self.calls if item == model_id)

    def called_models(self) -> list[str]:

        return [item for item, _ in self.calls]


# ============================================================
# 假凭据解析器
# ============================================================


class FakeCredentialResolver:
    """按 ref 返回预设秘密；未登记则返回 None（= 不覆盖）。"""

    def __init__(self, secrets: Mapping[str, str] | None = None) -> None:

        self.secrets = dict(secrets or {})
        self.refs: list[str] = []

    def resolve(self, ref: str) -> str | None:

        self.refs.append(ref)

        return self.secrets.get(ref)

    def __repr__(self) -> str:
        return (
            f"FakeCredentialResolver(refs={sorted(self.secrets)!r}, "
            "values=<hidden>)"
        )


# ============================================================
# 假观测写入者（infra/graph 的 GraphRunRecorder 端口）
# ============================================================


class FakeRecorder:
    """
    收集节点运行行。

    只实现 `upsert_node_run` —— 那是 `node_span` 唯一用到的接口
    （见 infra/graph/instrumentation.py 的 _safe_record）。
    """

    def __init__(self) -> None:

        self.updates: list[Any] = []

    async def upsert_node_run(self, update: Any) -> None:

        self.updates.append(update)

    # --------------------------------------------------------

    def by_node(self, node_id: str) -> list[Any]:

        return [item for item in self.updates if item.node_id == node_id]

    def last(self, node_id: str) -> Any:

        found = self.by_node(node_id)

        return found[-1] if found else None


# ============================================================
# 假观测者
# ============================================================


class RecordingObserver:
    """
    把 attempt 与整次调用的观测都收下来。

    "一次 invoke 内部试了几次"这件事只有它看得见 —— 断言它就等于
    断言规格 §9 的那条语义。
    """

    def __init__(self) -> None:

        self.attempts: list[Any] = []
        self.summaries: list[Any] = []

    def on_attempt(self, attempt: Any, *, context: Any) -> None:

        self.attempts.append((context, attempt))

    def on_call(self, summary: Any) -> None:

        self.summaries.append(summary)

    # --------------------------------------------------------

    def codes(self) -> list[str | None]:

        return [item.error_code for _, item in self.attempts]

    def models(self) -> list[str]:

        return [item.model_id for _, item in self.attempts]


# ============================================================
# 假 Context Builder
# ============================================================


class StubContextBuilder(AgentInvokeContextBuilder):
    """
    只做两件事的 builder：产出 resolution，产出 ModelRequest。

    它声明参与的阶段刻意只有入口与出口 —— 那正是
    `BaseContextBuilder.__init_subclass__` 要求的两个（任何声明了阶段的
    实现都必须包含它们）。

    每一次 `build` 都被计数：这是"fallback 不会重建上下文"的断言依据
    （规格 §9）。
    """

    version = "1.0.0"
    display_name = "测试用最小 builder"
    description = "只产出固定消息的 builder，用于节点/运行时测试。"

    enabled_stages = frozenset({
        StageId.CONTEXT_RESOLVER,
        StageId.PROMPT_ASSEMBLER,
    })

    #: 类级计数（实例不能有属性 —— BaseContextBuilder 封死了 __setattr__）。
    build_count: int = 0

    #: 产出多少条消息（不含进来的那些）。
    reply_with: str = "你好"

    #: 这次请求"可用"的工具。要触发能力门禁的用例把它设成非空
    #: （子类里覆写即可 —— 类属性，实例上不能有东西）。
    tools: tuple[Any, ...] = ()

    async def _stage_prompt_assembler(
            self,
            state: ContextBuildState,
    ) -> None:

        type(self).build_count += 1

        state.draft = ContextModelRequest(
            messages=(
                *state.request.messages,
                _human_message(type(self).reply_with),
            ),
            tools=tuple(type(self).tools),
        )


def _human_message(text: str) -> Any:
    """
    造一条"像消息"的东西。

    刻意**不用** langchain 的 HumanMessage：Model 模块的测试不该被
    langchain 绑住（模块本身对它只有调用约定，没有依赖）。真正需要
    AIMessage 的是图/路由那边 —— 那些测试自己 import。
    """

    return SimpleNamespace(content=text, type="human")


# ============================================================
# 装配帮手
# ============================================================


def make_definition(
        model: str,
        *,
        provider: str = "fake",
        capabilities: ModelCapabilities | None = None,
        config: Mapping[str, Any] | None = None,
        credential_ref: str | None = None,
        aliases: Sequence[str] = (),
        pricing: Any = None,
) -> ModelDefinition:

    return ModelDefinition(
        id=f"{provider}:{model}",
        provider=provider,
        model=model,
        capabilities=capabilities or ModelCapabilities(),
        config=dict(config or {}),
        credential_ref=credential_ref,
        aliases=tuple(aliases),
        pricing=pricing,
    )


def make_registry(
        *,
        models: Sequence[ModelDefinition],
        profiles: Sequence[ModelProfile],
        bindings: Sequence[ModelRoleBinding],
        adapter: FakeAdapter,
        credentials: Any = None,
        observer: Any = None,
) -> DefaultModelRegistry:
    """
    用内存目录装一个注册表（不读 model.yaml）。

    这正是"Registry 是注入依赖"的价值：测试不必有配置文件、不必有
    密钥，就能跑完整的 Registry -> Resolver -> Factory -> Runtime ->
    Provider 链（规格 §22 的第一条）。
    """

    registry = DefaultModelRegistry(
        credentials=credentials,
        observer=observer,
    )

    for provider in sorted({item.provider for item in models}):
        registry.register_provider(provider, adapter)

    for definition in models:
        registry.register_model(definition)

    for profile in profiles:
        registry.register_profile(profile)

    for binding in bindings:
        registry.register_binding(binding)

    return registry


def make_run_context(recorder: Any, *, thread_id: str = "thread-1") -> Any:
    """infra 的 RunContext（图注册时 context_factory 的入参）。"""

    from app.infra.graph import RunContext

    return RunContext(
        pipeline_run_id=uuid4(),
        thread_id=thread_id,
        trigger_source="test",
        recorder=recorder,
        # 工具执行账本的判重键第一段。本替身服务的用例不执行工具，
        # 但真装配下这一项必须有值。
        session_id=uuid4(),
    )


# ============================================================
# 假 provider 异常
# ============================================================


def fake_exception(
        name: str,
        *,
        base: type[BaseException] = Exception,
        message: str = "boom",
        **attributes: Any,
) -> BaseException:
    """
    造一个**名字对得上**的假 provider 异常。

    归一化是按类名 + 状态码做的（见 model/errors.py），因此"造一个叫
    `RateLimitError` 的异常"就足以驱动它 —— 这也正是那套设计的好处：
    测试**不需要**为了验证归一化去装 openai / anthropic。

    （`fake_exception("APITimeoutError", message=...)` 得到的对象在
    `normalize_error` 眼里与 httpx 那个同名异常完全一样。）
    """

    klass = type(
        name,
        (base,),
        {"__module__": "fake_provider_sdk", **attributes},
    )

    return klass(message)
