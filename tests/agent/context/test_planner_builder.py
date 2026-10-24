"""Planner 的窄 Context Builder：resolver + summary/history + prompt。"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.builder.planner import (
    PlannerProcessorContextBuilder,
)
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.context.registry import create_context_builder
from app.agent.model import ModelRole
from app.common.context import ContextBuilderType, StageId

from context_fakes import FakeModelRegistry

__all__: list[str] = []


def test_planner_builder_is_discovered() -> None:
    builder = create_context_builder(ContextBuilderType.PLANNER)

    assert isinstance(builder, PlannerProcessorContextBuilder)
    assert builder.enabled_stages == frozenset({
        StageId.CONTEXT_RESOLVER,
        StageId.SUMMARY_LOADER,
        StageId.HISTORY_SELECTOR,
        StageId.CAPABILITY_LOADER,
        StageId.TOKEN_BUDGETER,
        StageId.PROMPT_ASSEMBLER,
    })
    assert builder.required_stages == frozenset({
        StageId.CONTEXT_RESOLVER,
        StageId.PROMPT_ASSEMBLER,
    })


@pytest.mark.asyncio
async def test_planner_builder_uses_prompt_and_history_only() -> None:
    messages = (HumanMessage("规划这个季度的对比", id="m1"),)

    graph_state = {
        "messages": messages,
        "context_policy": {
            "system_prompt": "你是 planner",
            "include_summary": False,
            "include_history": True,
            "include_user_profile": False,
            "include_memory": False,
            "include_retrieved_context": False,
            "include_capabilities": False,
            "enforce_token_budget": False,
            "allow_compaction": False,
            "validate": False,
        },
    }

    request = ContextBuildRequest(
        builder_type=ContextBuilderType.PLANNER,
        thread_id="thread-1",
        messages=messages,
        run_id=uuid4(),
        graph_state=graph_state,
        agent_type="planner",
        model_role=ModelRole.REASONING,
        dependencies=ContextDependencies(
            model_registry=FakeModelRegistry(),
        ),
    )

    result = await PlannerProcessorContextBuilder().build_with_result(
        request
    )
    model_request = result.model_request

    assert str(model_request.messages[0].content) == "你是 planner"
    assert any(
        "规划这个季度的对比" in str(message.content)
        for message in model_request.messages
    )
    assert model_request.tools == ()

    stages = {
        item["stage"] for item in result.processor_executions
    }
    assert stages <= {
        StageId.CONTEXT_RESOLVER.value,
        StageId.SUMMARY_LOADER.value,
        StageId.HISTORY_SELECTOR.value,
        StageId.CAPABILITY_LOADER.value,
        StageId.TOKEN_BUDGETER.value,
        StageId.PROMPT_ASSEMBLER.value,
    }


class _FakeToolSource:
    """只验证 CAPABILITY_LOADER 把 tool_keys 解析成工具。"""

    async def resolve(self, *, tool_keys, conversation_id=None, overrides=None):
        from langchain_core.tools import tool

        @tool
        def planner_demo(text: str) -> str:
            """一个测试工具。"""

            return text

        return [planner_demo]


@pytest.mark.asyncio
async def test_planner_builder_binds_tools_in_react_mode() -> None:
    messages = (HumanMessage("查一下技能目录", id="m1"),)

    graph_state = {
        "messages": messages,
        "context_policy": {
            "system_prompt": "你是 planner",
            "include_summary": False,
            "include_history": True,
            "include_user_profile": False,
            "include_memory": False,
            "include_retrieved_context": False,
            "include_capabilities": True,
            "tool_keys": ["skill.list_skills"],
            "enforce_token_budget": False,
            "allow_compaction": False,
            "validate": False,
        },
    }

    request = ContextBuildRequest(
        builder_type=ContextBuilderType.PLANNER,
        thread_id="thread-1",
        messages=messages,
        run_id=uuid4(),
        graph_state=graph_state,
        agent_type="planner",
        model_role=ModelRole.REASONING,
        dependencies=ContextDependencies(
            model_registry=FakeModelRegistry(),
            tool_source=_FakeToolSource(),
        ),
    )

    result = await PlannerProcessorContextBuilder().build_with_result(
        request
    )

    # CAPABILITY_LOADER 真的把 tool_keys 解析成了 ModelRequest.tools。
    # （processor_executions 在未开 trace 时为空，不能拿它当执行证据。）
    assert len(result.model_request.tools) == 1
    assert result.model_request.tools[0].name == "planner_demo"
