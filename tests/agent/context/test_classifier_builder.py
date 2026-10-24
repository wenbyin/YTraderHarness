"""
分类上下文的输入形状。

分类模型要能读懂当前这句话的指代（"那再查一下昨天的"），因此需要
**历史摘要作为背景**；但摘要不是分类对象，分类对象永远是最后一条
用户输入。这里钉住三件事：

    1. state 里的 summary 被注入成「历史摘要」system 分节
    2. 当前用户输入仍然是最后一条 human 消息
    3. 分类阶段**不会**为了这轮摘要再调一次模型
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.agent.context.builder.classifier import (
    ClassifierProcessorContextBuilder,
)
from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.model import ModelRole
from app.agent.prompt.prompts.classification import PROMPT
from app.common.context import ContextBuilderType, StageId

from context_fakes import FakeModelRegistry

_SUMMARY = "用户此前询问了 AAPL 的行情，并偏好简洁回答。"
_CURRENT = "那再查一下昨天的"
_SYSTEM_PROMPT = "你是任务分诊器。"


def _request(
        state: dict,
        *,
        registry: FakeModelRegistry | None = None,
) -> ContextBuildRequest:

    return ContextBuildRequest(
        builder_type=ContextBuilderType.CLASSIFIER,
        thread_id="thread-1",
        messages=tuple(state.get("messages") or ()),
        run_id=uuid4(),
        graph_state=state,
        agent_type="react",
        model_role=ModelRole.CLASSIFICATION,
        dependencies=ContextDependencies(
            model_registry=registry or FakeModelRegistry()
        ),
    )


def _state(*, summary: dict | None = _SUMMARY) -> dict:

    state: dict = {
        "messages": (HumanMessage(content=_CURRENT, id="m2"),),
        "conversation_id": "c1",
        "user_id": "u1",
        "context_policy": {
            "system_prompt": _SYSTEM_PROMPT,
            "include_history": True,
            "min_recent_turns": 1,
            "max_recent_turns": 1,
            "max_history_tokens": 0,
            "include_summary": True,
            # 与 agent_builder_node._graph_state 一致：摘要只读。
            "allow_compaction": False,
        },
    }

    if summary is not None:
        state["summary"] = {
            "text": _SUMMARY,
            "revision": 2,
            "summarized_until_message_id": "m1",
        }

    return state


def test_classifier_builder_declares_readonly_summary_stage() -> None:
    builder = ClassifierProcessorContextBuilder()

    assert StageId.SUMMARY_LOADER in builder.enabled_stages


async def test_summary_is_injected_as_system_background() -> None:
    registry = FakeModelRegistry()

    result = await ClassifierProcessorContextBuilder().build_with_result(
        _request(_state(), registry=registry)
    )

    messages = result.model_request.messages

    assert messages[0].type == "system"
    assert _SYSTEM_PROMPT in messages[0].content
    assert '<ref:summary desc="历史摘要' in messages[0].content
    assert _SUMMARY in messages[0].content

    # 当前输入是最后一条 human 消息，摘要没有被当成用户输入。
    human = [
        message for message in messages
        if message.type == "human"
    ]

    assert [message.content for message in human] == [_CURRENT]

    # 只读注入：摘要阶段不应该调用摘要模型。
    assert registry.summary_calls == 0


async def test_no_summary_means_no_summary_section() -> None:
    result = await ClassifierProcessorContextBuilder().build_with_result(
        _request(_state(summary=None))
    )

    system = result.model_request.messages[0]

    assert system.type == "system"
    assert "ref:summary" not in system.content


def test_prompt_marks_summary_as_background_only() -> None:
    text = PROMPT.render(candidates=[])

    assert "历史摘要" in text
    assert "此前对话的概览" in text
    assert "分类对象**永远是最后一条用户输入**" in text
    assert "不要对摘要本身分类" in text
