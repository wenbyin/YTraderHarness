"""
Context Builder 测试用的假实现。

它们都只实现**端口**（ModelRegistry 的最小面 / 上下文源 / 压缩器），
因此整套测试：不联网、不需要密钥、不需要数据库，跑起来是毫秒级。

⚠️ 为什么假 ModelRegistry 只需要两个方法
------------------------------------
Context Builder 对 ModelRegistry 的用法就两处（规格 §十七、§二十一）：

    get_runtime(agent_type=..., model_role=...)   摘要要发一次模型调用
    definition(profile.primary).context_window    预算要读模型窗口

因此假实现只需要这两件，而**不该**长出别的（长出别的就说明
ContextBuilder 越界去碰了模型层的其它职责）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.model import (
    ModelCapabilities,
    ModelDefinition,
    ModelProfile,
    ModelUsage,
)
from app.services.session.domain.schemas.enum import MessageRole

__all__ = [
    "FakeKnowledgeSource",
    "FakeMemoryRetrieval",
    "FakeMemorySource",
    "FakeModelRegistry",
    "FakeProfileSource",
    "FakeRuntime",
    "RecordingCompactor",
    "human",
    "install_memory_retrieval",
    "memory_preview",
    "message_ids",
]


# ============================================================
# 模型注册表（最小面）
# ============================================================


@dataclass
class FakeRuntime:
    """假的 ModelRuntime：记录请求，返回一条固定的摘要。"""

    profile: ModelProfile

    reply: str = "摘要内容"

    calls: list[Any] = field(default_factory=list)

    async def ainvoke(self, request: Any, **kwargs: Any) -> Any:
        """只做"收下请求、回一条消息"——retry / fallback 不在这里测。"""

        from app.agent.model import ModelResponse

        self.calls.append(request)

        return ModelResponse(
            message=AIMessage(content=self.reply),
            provider="fake",
            model=self.profile.primary,
            usage=ModelUsage(
                input_tokens=100, output_tokens=20, total_tokens=120
            ),
            attempts=1,
        )


class FakeModelRegistry:
    """
    `ModelRegistry` 协议里 Context Builder 用到的那两个方法。

    刻意**不**实现注册 / 解析：Context Builder 不该碰那些（规格 §十八）。
    """

    def __init__(
            self,
            *,
            primary: str = "fake:summarizer",
            context_window: int | None = 8192,
            reply: str = "摘要内容",
            resolved_model_role: str | None = None,
    ) -> None:

        self.primary = primary
        self.context_window = context_window
        self.reply = reply

        self.profile = ModelProfile(id="fake_profile", primary=primary)

        self.runtime = FakeRuntime(profile=self.profile, reply=reply)

        #: 被问过的 (agent_type, model_role) —— 断言"用了哪个角色"。
        self.resolved: list[tuple[str, str]] = []

        #: 只认这一个角色（None = 不校验）。
        self.resolved_model_role = resolved_model_role

        self.definition_calls: list[str] = []

    # --------------------------------------------------------

    def get_runtime(
            self,
            *,
            agent_type: str,
            model_role: str,
            context: Any = None,
    ) -> FakeRuntime:

        self.resolved.append((agent_type, model_role))

        if (
                self.resolved_model_role is not None
                and model_role != self.resolved_model_role
        ):
            raise AssertionError(
                f"Context Builder 用了 model_role={model_role!r}，"
                f"期望 {self.resolved_model_role!r}。"
            )

        return self.runtime

    def definition(self, reference: str) -> ModelDefinition:

        self.definition_calls.append(reference)

        return ModelDefinition(
            id=reference,
            provider="fake",
            model="summarizer",
            capabilities=ModelCapabilities(structured_output=True),
            context_window=self.context_window,
        )

    # --------------------------------------------------------

    @property
    def summary_calls(self) -> int:

        return len(self.runtime.calls)


# ============================================================
# 外部上下文源
# ============================================================


class FakeProfileSource:

    def __init__(self, profile: Mapping[str, Any] | None = None) -> None:

        self.profile = dict(profile or {})
        self.user_ids: list[str] = []

    async def load_profile(self, *, user_id: str) -> Mapping[str, Any] | None:

        self.user_ids.append(user_id)

        return dict(self.profile) or None


class FakeMemorySource:

    def __init__(
            self,
            items: Sequence[Mapping[str, Any]] = (),
            *,
            error: BaseException | None = None,
    ) -> None:

        self.items = [dict(item) for item in items]
        self.error = error
        self.queries: list[tuple[str, str | None, int]] = []

    async def retrieve(
            self,
            *,
            user_id: str,
            query: str | None,
            limit: int,
    ) -> Sequence[Mapping[str, Any]]:

        self.queries.append((user_id, query, limit))

        if self.error is not None:
            raise self.error

        return self.items[:limit]


class FakeKnowledgeSource:

    def __init__(
            self,
            items: Sequence[Mapping[str, Any]] = (),
            *,
            error: BaseException | None = None,
    ) -> None:

        self.items = [dict(item) for item in items]
        self.error = error
        self.queries: list[tuple[str, str | None, int]] = []

    async def retrieve(
            self,
            *,
            query: str,
            conversation_id: str | None,
            limit: int,
    ) -> Sequence[Mapping[str, Any]]:

        self.queries.append((query, conversation_id, limit))

        if self.error is not None:
            raise self.error

        return self.items[:limit]


# ============================================================
# 压缩器
# ============================================================


class RecordingCompactor:
    """
    假的摘要器：记录"要折哪几条"，返回固定文本。

    用它测的是**记账**（revision / cursor / 写回工作集），而"真的调了哪个
    模型"由 FakeModelRegistry 那边测（见 test_processors.py 里对真实
    `ModelRegistrySummaryCompactor` 的用例）。
    """

    def __init__(self, text: str = "摘要后的文本") -> None:

        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def compact(
            self,
            *,
            previous: Any,
            messages: Sequence[Any],
            max_output_tokens: int,
    ) -> Any:

        from app.agent.context.processor.summarization import SummaryDraft

        self.calls.append({
            "previous_revision": previous.revision if previous else None,
            "messages": list(messages),
            "max_output_tokens": max_output_tokens,
        })

        return SummaryDraft(
            text=self.text,
            provider="fake",
            model="fake:summarizer",
            usage=ModelUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
            folded_messages=len(messages),
        )


# ============================================================
# 记忆召回（阶段 5）
# ============================================================
#
# ⚠️ 这一层的输入是**服务**，不是 `deps.memory_source`
# -------------------------------------------------
# MemoryProcessor 走的是 `MemoryQueryService.retrieval_preview`（与
# `POST /memory/retrieval` 同一条实现）。`FakeMemorySource` 那张脸（实现
# `MemorySource` 端口）已经不是它的输入了 —— 那个端口今天全仓没有实现。


def memory_preview(
        *,
        persona: str = "偏好简洁",
        conversations: int = 0,
        facts: int = 0,
) -> Any:
    """
    一台假 preview，形状对齐 `MemoryRetrievalPrompt`。

    ⚠️ 长得像而不是真的构造那个 pydantic 模型
    --------------------------------------
    那个模型串着 persona / scene / conversation / fact 四张 DTO，构造一次
    要摆十几个字段，而 Processor 只读三项（每段的 `block` / `items`、以及
    `degraded`）。假对象把"这一层真正依赖什么"钉住：将来 Processor 多读
    一项，这里会**立刻**缺字段，而不是悄悄过。
    """

    sections: list[Any] = [
        SimpleNamespace(
            block="persona",
            # ⚠️ `version` 不是可有可无的：排版把它打在标题里
            # （`用户画像（v3）`，见 prompts/memory_retrieval.py 的
            # `render_blocks`），少了它排版会以 AttributeError 炸掉，
            # 而症状是"两层记忆段一起消失"（处理器 optional -> FAILED）。
            items=[SimpleNamespace(content=persona, filename="persona.md", version=1)],
        ),
        # 空的一段 —— 钉住"空的层不出段"（不是给一个空壳段）。
        SimpleNamespace(block="scenes", items=[], scenes_total=0),
    ]

    if conversations:
        sections.append(
            SimpleNamespace(
                block="conversations",
                items=[
                    SimpleNamespace(
                        role=MessageRole.USER,
                        text=f"原话{index}",
                        message_id=uuid4(),
                        # 排版会打时间（`_at(hit.created_at)`）。给 None 是
                        # **合法输入**：那一节整个省掉（见 prompts/
                        # memory_retrieval.py 的 `_stamp`），所以这里恰好也
                        # 钉住"取不到时间时不炸、只是不打它"。
                        created_at=None,
                    )
                    for index in range(conversations)
                ],
            )
        )

    if facts:
        sections.append(
            SimpleNamespace(
                block="facts",
                items=[
                    SimpleNamespace(
                        content=f"记忆{index}",
                        type="rule",
                        record_id=uuid4(),
                        # 与上面 L0 那条同理：两个时间给 None（那一节省掉）。
                        created_at=None,
                        updated_at=None,
                    )
                    for index in range(facts)
                ],
            )
        )

    return SimpleNamespace(
        sections=sections,
        degraded=[],
        prompt=SimpleNamespace(user=""),
    )


class FakeMemoryRetrieval:
    """假 `MemoryQueryService`：按剧本应答，并记下收到的请求。"""

    def __init__(
            self,
            preview: Any = None,
            error: BaseException | None = None,
    ) -> None:
        self.preview = preview if preview is not None else memory_preview()
        self.error = error
        self.requests: list[Any] = []

    async def retrieval_preview(self, request: Any) -> Any:
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return self.preview


def install_memory_retrieval(monkeypatch: Any, service: Any) -> None:
    """
    把记忆处理器取服务的那一处（模块级 `get_service_factory`）换成假的。

    打在**那个模块**上而不是全局：处理器是从它自己的命名空间取这个函数的，
    而别的处理器与应用级工厂不该被这个夹具影响。
    """

    from app.agent.context.processor.processors import memory as memory_module

    class _Factory:
        def scope(self, contract: type, *, begin: bool = True) -> Any:
            class _Scope:
                async def __aenter__(self) -> Any:
                    return service

                async def __aexit__(self, *exc: Any) -> bool:
                    return False

            return _Scope()

    monkeypatch.setattr(
        memory_module, "get_service_factory", lambda: _Factory()
    )


# ============================================================
# 消息帮手
# ============================================================


def human(text: str, *, id: str | None = None) -> HumanMessage:
    """造一条带 id 的 HumanMessage（id 对 summarized_until 很重要）。"""

    return HumanMessage(content=text, id=id)


def message_ids(messages: Sequence[Any]) -> list[str | None]:

    return [getattr(message, "id", None) for message in messages]
