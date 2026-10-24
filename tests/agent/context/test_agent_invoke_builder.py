"""
`agent_invoke` Builder 的端到端行为（规格 §四、§九、§十九、§二十）。

跑的是**生产路径上那一个实现**（反射发现的那个），因此这里验证的是
真实装配：`create_context_builder(AGENT_INVOKE)` 拿到它、13 个阶段方法
委托给流水线、策略从 resolver 出来、快照落进请求 metadata。
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.agent.context.builder.agent_invoke import (
    AgentInvokeProcessorContextBuilder,
)
from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.context.processor.errors import ContextStateError
from app.agent.context.processor.pipeline import PIPELINE_STAGE_ORDER
from app.agent.context.registry import create_context_builder
from app.agent.model import ModelRole
from app.common.context import ContextBuilderType, StageId

from context_fakes import (
    FakeMemoryRetrieval,
    FakeModelRegistry,
    install_memory_retrieval,
    memory_preview,
)


def _request(state: dict, **overrides) -> ContextBuildRequest:

    kwargs = {
        "builder_type": ContextBuilderType.AGENT_INVOKE,
        "thread_id": "thread-1",
        "messages": tuple(state.get("messages") or ()),
        "run_id": uuid4(),
        "graph_state": state,
        "agent_type": "react",
        "model_role": ModelRole.REASONING,
        "dependencies": ContextDependencies(
            model_registry=FakeModelRegistry()
        ),
    }

    kwargs.update(overrides)

    return ContextBuildRequest(**kwargs)


def _state(**overrides) -> dict:

    state = {
        "messages": (HumanMessage(content="你好", id="m1"),),
        "user_id": "u1",
        "conversation_id": "c1",
        "context_policy": {"system_prompt": "你是助手"},
    }

    state.update(overrides)

    return state


# ============================================================
# 装配
# ============================================================


def test_builder_is_discovered_by_the_registry() -> None:
    """反射发现拿到的是这个实现（生产路径只有它一个）。"""

    builder = create_context_builder(ContextBuilderType.AGENT_INVOKE)

    assert isinstance(builder, AgentInvokeProcessorContextBuilder)
    assert builder.get_key() is ContextBuilderType.AGENT_INVOKE


def test_builder_declares_every_stage_and_two_required() -> None:
    """声明参与全部 13 个阶段，其中入口与出口是 required。"""

    builder = AgentInvokeProcessorContextBuilder()

    assert builder.enabled_stages == frozenset(PIPELINE_STAGE_ORDER)
    assert builder.required_stages == frozenset({
        StageId.CONTEXT_RESOLVER,   # 入口
        StageId.PROMPT_ASSEMBLER,   # 出口
    })


def test_pipeline_covers_every_stage_in_fixed_order() -> None:
    """流水线覆盖全部阶段，且顺序与固定模板**逐字一致**。"""

    # ⚠️ 一个阶段上可以有**多个**处理器（`memory_retriever` 上就是两个：
    # memory_profile=45 与 memory=50，见 processors/__init__.py 那张表）。
    # 所以先把阶段去重（保序），再与模板比 —— 直等比长度的话，第二个处理器
    # 一加进来这条断言就红，而它要守的其实是"每个阶段都有、顺序与模板一致"。
    stages = list(dict.fromkeys(
        processor.spec_of().stage
        for processor in AgentInvokeProcessorContextBuilder.pipeline().processors
    ))

    assert stages == list(PIPELINE_STAGE_ORDER)


def test_stage_policies_are_derivable_for_the_catalog_api() -> None:
    """既有契约的 `stage_policies()` 仍然可用（目录接口靠它）。"""

    builder = AgentInvokeProcessorContextBuilder()

    policies = builder.stage_policies()

    assert len(policies) == len(PIPELINE_STAGE_ORDER)
    assert policies[0].enabled and policies[0].required


# ============================================================
# 一次构建
# ============================================================


async def test_build_produces_a_model_request() -> None:
    """产出的是 Model 模块的 `ModelRequest`（唯一请求类型）。"""

    from app.agent.model import ModelRequest

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state())
    )

    assert isinstance(result.model_request, ModelRequest)
    assert result.model_request.messages
    assert result.model_request.metadata["context_builder"] == "agent_invoke"


async def test_build_is_callable_through_the_base_contract() -> None:
    """`build()` 仍然只返回 ModelRequest（既有调用方不受影响）。"""

    from app.agent.model import ModelRequest

    built = await AgentInvokeProcessorContextBuilder().build(_request(_state()))

    assert isinstance(built, ModelRequest)
    assert built.metadata["context_snapshot"]["builder_type"] == "agent_invoke"


async def test_snapshot_reports_every_processor() -> None:
    """快照里每个阶段都有 Processor 状态（规格 §廿六）。"""

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state())
    )

    # 断的是**阶段覆盖**，不是"一个阶段一条"：`memory_retriever` 上有两个
    # 处理器（memory_profile / memory）。
    assert {
        item["stage"] for item in result.processor_executions
    } == {stage.value for stage in PIPELINE_STAGE_ORDER}

    assert all(
        item["status"] in ("executed", "skipped", "failed")
        for item in result.processor_executions
    )


# ============================================================
# 策略：基线 ⊕ 覆盖（规格 §九 / §二十）
# ============================================================


async def test_policy_defaults_come_from_the_resolver() -> None:
    """没给覆盖项 -> 用基线策略。"""

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state())
    )

    policy = result.snapshot.policy

    assert policy["source"] == "default:agent_invoke"
    assert policy["include_memory"] is True
    assert policy["min_recent_turns"] == 3
    assert policy["max_recent_turns"] == 5


async def test_policy_overrides_from_state_win() -> None:
    """图 state 的 `context_policy` 覆盖基线。"""

    state = _state(context_policy={
        "system_prompt": "你是助手",
        "include_memory": False,
        "min_recent_turns": 2,
        "max_recent_turns": 2,
    })

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(state)
    )

    assert result.snapshot.policy["include_memory"] is False
    assert result.snapshot.policy["max_recent_turns"] == 2


async def test_invalid_policy_gate_type_fails_the_build() -> None:
    """
    `include_memory: "yes"` -> FAILED（规格 §九，不做隐式 bool 转换）。
    """

    state = _state(context_policy={"include_memory": "yes"})

    with pytest.raises(ContextStateError) as excinfo:
        await AgentInvokeProcessorContextBuilder().build(_request(state))

    assert "必须是 bool" in str(excinfo.value)
    # ⚠️ 报的是**先跑的那个**绑定该路径的处理器：`include_memory` 同时绑在
    # `memory_profile`(45) 与 `memory`(50) 上（见 processors/__init__.py 的
    # 表），而流水线在第一个门控失败处就停 —— 所以这里期望的是 45 那个。
    assert excinfo.value.processor == "memory_profile"


async def test_unknown_policy_key_is_preserved() -> None:
    """未知键保留在有效策略里（开放契约：自定义 Processor 可以绑它）。"""

    state = _state(context_policy={"my_custom_switch": True})

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(state)
    )

    assert result.snapshot.policy["my_custom_switch"] is True


# ============================================================
# 执行层级：INITIAL / CONTINUATION（规格 §四）
# ============================================================


async def test_phase_is_initial_for_a_fresh_turn() -> None:
    """只有一条用户消息 -> INITIAL。"""

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state())
    )

    assert result.snapshot.phase == "initial"


async def test_phase_is_continuation_after_a_tool_result() -> None:
    """
    最后一条用户消息之后有 ToolMessage -> CONTINUATION。

    这就是"第二次执行不会重复创建初始 Conversation"的判据：历史处理器
    在 CONTINUATION 时**只**用图 state 里的消息。
    """

    from langchain_core.messages import AIMessage, ToolMessage

    messages = (
        HumanMessage(content="查一下", id="u1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{
                "name": "lookup", "args": {}, "id": "c1", "type": "tool_call",
            }],
        ),
        ToolMessage(content="结果", tool_call_id="c1", id="t1"),
    )

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state(messages=messages))
    )

    assert result.snapshot.phase == "continuation"
    assert "t1" in result.snapshot.history_message_ids


async def test_history_only_ever_selects_from_state() -> None:
    """
    历史**只从 Graph State 里选** —— 它没有任何"取更多"的口子。

    ⚠️ 这条曾经是"CONTINUATION 不回库重放"（规格 §四B 的红线），做法是
    注入一个计数用的 `ConversationSource` 再看它有没有被问。
    `ConversationSource` 与语义历史检索都已删除（从落地起就没有生产
    实现，更早的内容现在归记忆模块的 L0 层），于是那条不变量变成了
    **平凡真理**：没有可走错的另一条路。

    守卫因此换到了记忆模块上 —— 见
    `test_processors.py::test_memory_replays_instead_of_recalling`
    （一个运行只召回一次）与本目录的 memory_smoke.py。
    """

    from langchain_core.messages import AIMessage, ToolMessage

    messages = (
        HumanMessage(content="查一下", id="u1"),
        AIMessage(
            content="", id="a1",
            tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}],
        ),
        ToolMessage(content="结果", tool_call_id="c1", id="t1"),
    )

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state(messages=messages))
    )

    # CONTINUATION：选出来的就是 state 里那三条（整轮不拆）
    assert set(result.selected_history_message_ids) == {"u1", "a1", "t1"}

    # 而历史处理器能看到的输入里**没有**任何"源"
    record = next(
        item for item in result.processor_executions
        if item["processor"] == "history"
    )

    assert set(record["detail"]["inputs"]) == {
        "messages", "current_message_id", "current_message", "summary",
    }


# ============================================================
# 失败路径
# ============================================================


async def test_empty_messages_fails_instead_of_building_an_empty_request() -> None:
    """
    没有任何消息 -> 构建**失败**（而不是产出一个空请求）。

    两条防线都会拦下它，取决于缺到哪一步：
        图 state 里没有 messages      -> `messages` 这条绑定由 Builder
                                        从请求补上（值为空），于是
                                        history 选不出东西
        选出来是空的                  -> validator 报"没有任何用户消息"

    两种都是 Fatal。这里只断言"失败"，因为**报错内容**取决于哪条防线
    先响 —— 而两条都对（见 validation.py 的说明）。
    """

    from app.agent.context.processor.errors import ContextBuildError

    state = _state(messages=())
    state.pop("messages")

    with pytest.raises(ContextBuildError) as excinfo:
        await AgentInvokeProcessorContextBuilder().build(_request(state))

    assert "消息" in str(excinfo.value)


async def test_capability_promise_without_source_fails() -> None:
    """策略点名要工具但没有任何解析入口 -> FAILED（承诺兑现不了）。"""

    state = _state(context_policy={"tool_keys": ["lookup"]})

    with pytest.raises(Exception) as excinfo:
        await AgentInvokeProcessorContextBuilder().build(_request(state))

    assert "ToolExposureSource" in str(excinfo.value)


async def test_validator_blocks_an_over_budget_context() -> None:
    """超限且裁不动 -> Fatal（规格 §二十八）。"""

    from app.agent.context.processor.errors import ContextBudgetError

    state = _state(
        messages=(HumanMessage(content="长" * 5000, id="m1"),),
        context_policy={
            # 每段都给 0、输出也 0、余量 0 -> 只有窗口能限制它
            "profile_budget": 0,
            "memory_budget": 0,
            "retrieval_budget": 0,
            "tool_budget": 0,
            "system_budget": 0,
            "max_output_tokens": 0,
            "reserve_tokens": 0,
            "max_history_tokens": None,
        },
    )

    deps = ContextDependencies(
        model_registry=FakeModelRegistry(context_window=100)
    )

    with pytest.raises(ContextBudgetError):
        await AgentInvokeProcessorContextBuilder().build(
            _request(state, dependencies=deps)
        )


async def test_build_with_result_reports_processor_outcomes(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_with_result` 给出富结果（规格 §十九）。"""

    # ⚠️ 记忆段要真的注入，得两样都给：一个召回决策（Agent Builder 那一轮
    # 写的），以及一个能应答的服务。处理器自己调的是
    # `MemoryQueryService.retrieval_preview` —— 不是 state 里现成的条目。
    install_memory_retrieval(
        monkeypatch,
        FakeMemoryRetrieval(preview=memory_preview(facts=1)),
    )

    state = _state(
        user_context={"profile": {"tier": "gold"}},
        context_policy={
            "system_prompt": "你是助手",
            "memory_retrieve": {
                "l0": True, "l1": True, "agent_id": "general",
            },
        },
    )

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(state)
    )

    assert result.context_hash
    assert result.selected_history_message_ids == ("m1",)
    assert result.injected_memory_ids
    assert "profile" in result.snapshot.injected_sections
    # 四段共用 `memory` 这个 section 名，所以 `injected_sections` 里它会
    # 出现多次 —— 快照的 `memory_ids` 正是靠这个名字把它们收在一起的。
    assert result.snapshot.injected_sections.count("memory") >= 1
    assert result.as_metadata()["context_hash"] == result.context_hash


@pytest.mark.asyncio
async def test_trace_stage_spans_cover_real_stages() -> None:
    """
    只有真正执行的 stage 才发 Span；父子关系指向 context.build。
    """

    from app.agent.context.trace import ContextTrace
    from app.common.event import TraceEmitter

    raw: list[dict] = []

    emitter = TraceEmitter(
        trace_id="trace-1",
        run_id="run-1",
        thread_id="session-1",
        writer=raw.append,
    )

    builder = create_context_builder(ContextBuilderType.AGENT_INVOKE)

    request = _request(
        _state(),
        trace=ContextTrace(emitter=emitter, parent_span_id="build-span"),
    )

    await builder.build(request)

    started = {
        item["event_type"]
        for item in raw
        if item["event_type"].endswith(".started")
    }

    assert "context.prompt_assemble.started" in started
    assert all(
        item["parent_span_id"] == "build-span"
        for item in raw
    )

    # 有 started 就一定有同 span_id 的 completed。
    completed_spans = {
        item["span_id"]
        for item in raw
        if item["event_type"].endswith(".completed")
    }
    started_spans = {
        item["span_id"]
        for item in raw
        if item["event_type"].endswith(".started")
    }
    assert started_spans <= completed_spans
