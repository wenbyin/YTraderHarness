"""
具体处理器的行为（规格 §廿九 的 Unit Test 1-12）。

历史选择是重点：它是"用户 Conversation 是唯一原始事实来源，而 LLM
Context 可以被裁剪"这条原则唯一的落点，也是工具调用 / 结果完整性最
容易出错的地方。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.context.builder.agent_invoke import (
    AgentInvokeProcessorContextBuilder,
)
from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.budget import (
    HeuristicTokenCounter,
    TokenBudget,
)
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.context.processor.history import TurnWindowHistorySelector
from app.agent.context.processor.policy import ContextPolicy
from app.agent.context.processor.summarization import (
    ModelRegistrySummaryCompactor,
)
from app.agent.model import ModelRole
from app.common.context import ContextBuilderType

from context_fakes import (
    FakeKnowledgeSource,
    FakeMemoryRetrieval,
    FakeModelRegistry,
    FakeProfileSource,
    install_memory_retrieval,
    memory_preview,
)


# ============================================================
# 帮手
# ============================================================


def _messages(*, turns: int, tool_rounds: int = 0) -> list:
    """造 `turns` 轮对话（每轮 Human + AI）。"""

    messages: list = []

    for index in range(turns):

        messages.append(HumanMessage(content=f"问题{index}", id=f"u{index}"))
        messages.append(AIMessage(content=f"回答{index}", id=f"a{index}"))

    return messages


def _tool_round(messages: list, *, index: int, call_id: str) -> list:
    """在一轮里插入一次工具往返（AI 要求调用 + Tool 结果）。"""

    messages.append(
        AIMessage(
            content="",
            id=f"tool-ai{index}",
            tool_calls=[{
                "name": "lookup",
                "args": {"q": index},
                "id": call_id,
                "type": "tool_call",
            }],
        )
    )
    messages.append(
        ToolMessage(content=f"结果{index}", tool_call_id=call_id, id=f"t{index}")
    )

    return messages


async def _build(
        *,
        state: dict,
        dependencies: ContextDependencies | None = None,
        builder=None,
):
    builder = builder or AgentInvokeProcessorContextBuilder()

    request = ContextBuildRequest(
        builder_type=ContextBuilderType.AGENT_INVOKE,
        thread_id="thread-1",
        messages=tuple(state.get("messages") or ()),
        run_id=uuid4(),
        graph_state=state,
        agent_type="react",
        model_role=ModelRole.REASONING,
        dependencies=dependencies,
    )

    return await builder.build_with_result(request)


def _registry(**kwargs) -> FakeModelRegistry:

    return FakeModelRegistry(**kwargs)


def _deps(**kwargs) -> ContextDependencies:

    kwargs.setdefault("model_registry", _registry())
    return ContextDependencies(**kwargs)


# ============================================================
# 1. 最近 8 轮
# ============================================================


async def test_window_turns_is_clamped_by_max_recent_turns() -> None:
    """
    窗口宽度 = clamp(未覆盖轮数, min_recent_turns, 上限)。

    关掉生成（allow_compaction=False）才能看到**没被摘要救过**的原始
    形状：12 轮全未覆盖，而上限是 `min(5, 3 + 12 // 10) = 4`。
    """

    state = {
        "messages": _messages(turns=12),
        "context_policy": {
            "system_prompt": "sys",
            "allow_compaction": False,
        },
    }

    result = await _build(state=state, dependencies=_deps())

    humans = [
        message for message in result.messages
        if isinstance(message, HumanMessage)
    ]

    assert len(humans) == 4
    assert humans[0].content == "问题8"  # 12 - 4 = 8
    assert humans[-1].content == "问题11"

    history = next(
        item for item in result.processor_executions
        if item["processor"] == "history"
    )

    assert history["detail"]["outcome"]["turn_limit"] == 4
    # 没有摘要，所以未覆盖的 8 轮**没有地方放** —— 这正是不变量要盯的事。
    assert history["detail"]["outcome"]["gap_turns"] == 8


async def test_window_bounds_are_configurable() -> None:
    """窗口的下限与上限都由策略给（不是硬编码）。"""

    state = {
        "messages": _messages(turns=12),
        "context_policy": {
            "min_recent_turns": 2,
            "max_recent_turns": 2,
            "allow_compaction": False,
        },
    }

    result = await _build(state=state, dependencies=_deps())

    humans = [
        message for message in result.messages
        if isinstance(message, HumanMessage)
    ]

    assert [item.content for item in humans] == ["问题10", "问题11"]


# ============================================================
# 2. History Token 超限
# ============================================================


def test_token_budget_trims_oldest_turns_first() -> None:
    """
    超预算时从**最早**的轮次开始整轮裁剪 —— 但不低于 `min_recent_turns`。

    所以预算能动的范围就是 `[min_recent_turns, 上限]` 这一段：12 轮时上限
    是 4，于是最多丢 1 轮。再往前丢就不是"省预算"，而是"把最近的现场
    删掉"（那件事由 compaction 压重叠来做，它有自己的界）。
    """

    messages = _messages(turns=12)

    selector = TurnWindowHistorySelector()
    counter = HeuristicTokenCounter()

    policy = ContextPolicy(
        min_recent_turns=3, max_recent_turns=5, max_history_tokens=10_000
    )

    full = selector.select(
        messages, policy=policy, token_counter=counter
    )

    assert full.turns == 4  # clamp(12, 3, min(5, 3 + 1)) = 4
    assert full.dropped_for_tokens == 0

    tight = selector.select(
        messages,
        policy=policy,
        token_counter=counter,
        history_budget=1,  # 极端预算
    )

    assert tight.dropped_for_tokens == 1  # 只能丢到下限为止
    assert tight.turns == 3
    # 留下来的永远是**尾部**
    assert tight.messages[-1].content == "回答11"


def test_current_turn_is_never_trimmed() -> None:
    """当前这一轮（最后一条 Human 起）永远保留（规格 §十五）。"""

    messages = _messages(turns=4)

    selection = TurnWindowHistorySelector().select(
        messages,
        policy=ContextPolicy(),
        token_counter=HeuristicTokenCounter(),
        history_budget=0,  # 极端预算
    )

    assert selection.messages
    assert selection.messages[-1].content == "回答3"
    assert any(
        isinstance(message, HumanMessage) and message.content == "问题3"
        for message in selection.messages
    )


def test_system_messages_are_not_trimmed() -> None:
    """System Prompt 不进入普通 history trim（规格 §十五）。"""

    from langchain_core.messages import SystemMessage

    messages = [
        SystemMessage(content="你是助手", id="s1"),
        *_messages(turns=5),
    ]

    selection = TurnWindowHistorySelector().select(
        messages,
        policy=ContextPolicy(min_recent_turns=1, max_recent_turns=1),
        token_counter=HeuristicTokenCounter(),
    )

    assert selection.messages[0].content == "你是助手"
    assert selection.skipped_summarized == 0


# ============================================================
# 3. Tool Call / Tool Result 完整性
# ============================================================


def test_tool_round_is_not_split() -> None:
    """整轮裁剪 -> 一次工具往返不会被拆开。"""

    messages = _messages(turns=3)
    _tool_round(messages, index=2, call_id="c2")

    selection = TurnWindowHistorySelector().select(
        messages,
        policy=ContextPolicy(min_recent_turns=1, max_recent_turns=1),
        token_counter=HeuristicTokenCounter(),
    )

    ids = [getattr(message, "id", None) for message in selection.messages]

    assert "tool-ai2" in ids
    assert "t2" in ids


def test_orphan_tool_call_is_dropped() -> None:
    """
    Assistant 的 tool_call 没有结果 -> 去掉那条 AI 消息。

    provider 对不成对的工具调用会 400，而报错指向 provider —— 因此在
    选择阶段就修掉。
    """

    messages = [
        HumanMessage(content="问题", id="u0"),
        AIMessage(
            content="",
            id="ai-orphan",
            tool_calls=[{
                "name": "lookup", "args": {}, "id": "c9", "type": "tool_call",
            }],
        ),
        # 故意不给 ToolMessage
        AIMessage(content="回答", id="a0"),
    ]

    selection = TurnWindowHistorySelector().select(
        messages,
        policy=ContextPolicy(),
        token_counter=HeuristicTokenCounter(),
    )

    ids = [getattr(message, "id", None) for message in selection.messages]

    assert "ai-orphan" not in ids
    assert selection.dropped_broken_tool_rounds == 1


def test_orphan_tool_result_is_dropped() -> None:
    """孤立的 ToolMessage（没有对应请求）也要去掉。"""

    messages = [
        HumanMessage(content="问题", id="u0"),
        ToolMessage(content="孤儿结果", tool_call_id="missing", id="t-orphan"),
        AIMessage(content="回答", id="a0"),
    ]

    selection = TurnWindowHistorySelector().select(
        messages,
        policy=ContextPolicy(),
        token_counter=HeuristicTokenCounter(),
    )

    ids = [getattr(message, "id", None) for message in selection.messages]

    assert "t-orphan" not in ids


async def test_tool_result_is_visible_to_the_second_build() -> None:
    """Tool Result 进入第一次之后的构建（规格 §四B）。"""

    messages = _messages(turns=1)
    _tool_round(messages, index=0, call_id="c0")

    state = {"messages": messages, "context_policy": {}}

    result = await _build(state=state, dependencies=_deps())

    assert result.snapshot.phase == "continuation"
    assert any(
        isinstance(message, ToolMessage) for message in result.messages
    )


# ============================================================
# 4. Summary 注入 + revision
# ============================================================


async def test_summary_is_injected() -> None:
    """摘要注入成一段 system 片段（规格 §十六）。"""

    state = {
        "messages": _messages(turns=2),
        "context_policy": {"system_prompt": "sys"},
        "summary": {
            "text": "之前聊过天气",
            "revision": 3,
            "summarized_until_message_id": "a0",
        },
    }

    result = await _build(state=state, dependencies=_deps())

    system = result.messages[0]

    assert "之前聊过天气" in system.content
    assert result.summary_revision == 3
    assert "summary" in result.snapshot.injected_sections


async def test_summarized_messages_are_excluded_from_history() -> None:
    """
    未被覆盖的部分正好等于窗口上限时，历史里**一条重复的都没有**。

    8 轮、摘要覆盖 5 轮 -> u=3，而 8 轮的上限是 `min(5, 3 + 0) = 3`，
    于是窗口正好是未覆盖的那 3 轮：零重叠、零缺口，摘要与历史首尾相接。
    """

    messages = _messages(turns=8)

    state = {
        "messages": messages,
        "context_policy": {},
        "summary": {
            "text": "前五轮摘要",
            "revision": 1,
            "summarized_until_message_id": "a4",
        },
    }

    result = await _build(state=state, dependencies=_deps())

    humans = [
        message.content for message in result.messages
        if isinstance(message, HumanMessage)
    ]

    assert humans == ["问题5", "问题6", "问题7"]

    history = next(
        item for item in result.processor_executions
        if item["processor"] == "history"
    )

    assert history["detail"]["outcome"]["overlap_turns"] == 0
    assert history["detail"]["outcome"]["gap_turns"] == 0


async def test_summary_absent_is_not_an_error() -> None:
    """没有摘要 -> 不报错（summary 是可选输入）。"""

    state = {"messages": _messages(turns=1), "context_policy": {}}

    result = await _build(state=state, dependencies=_deps())

    assert result.summary_revision is None
    assert "summary" not in result.snapshot.injected_sections


# ============================================================
# 5. Profile / Memory / Retrieval 注入
# ============================================================


async def test_profile_is_injected_from_state() -> None:

    state = {
        "messages": _messages(turns=1),
        "context_policy": {},
        "user_id": "u1",
        "user_context": {"profile": {"tier": "gold", "locale": "zh"}},
    }

    result = await _build(state=state, dependencies=_deps())

    assert "tier: gold" in result.messages[0].content
    assert "profile" in result.snapshot.injected_sections


async def test_profile_source_wins_over_state() -> None:
    """配了源 -> 源说了算（即使 state 里也有）。"""

    source = FakeProfileSource({"tier": "platinum"})

    state = {
        "messages": _messages(turns=1),
        "context_policy": {},
        "user_id": "u1",
        "user_context": {"profile": {"tier": "gold"}},
    }

    result = await _build(
        state=state,
        dependencies=_deps(profile_source=source),
    )

    assert source.user_ids == ["u1"]
    assert "platinum" in result.messages[0].content
    assert "gold" not in result.messages[0].content


# ⚠️ 这一层的输入不是"state 里准备好的记忆条目"
# --------------------------------------------
# 处理器自己去调 `MemoryQueryService.retrieval_preview`（与
# `POST /memory/retrieval` 同一条实现），所以注入的是**服务工厂**，不是
# `deps.memory_source`（那个端口今天没有实现，处理器也不再走它）。假实现
# 与夹具在 context_fakes.py 里（`FakeMemoryRetrieval` /
# `install_memory_retrieval` / `memory_preview`）。
#
# 决策走 `context_policy.memory_retrieve`（Agent Builder 那一轮写的）。
# 一次运行只召回一次：INITIAL 真召回并把渲染好的分节交给 state 保活，
# CONTINUATION 直接读回来 —— 所以 state 要么给决策（走召回），要么给
# `memory.blocks`（走复用）。


def _recall_decision(**over: Any) -> dict[str, Any]:
    decision: dict[str, Any] = {"l0": True, "l1": True, "agent_id": "general"}
    decision.update(over)
    return decision


async def test_memory_is_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """记忆注入（规格 §廿九 第 7 条）：决策说召回 -> 四层进 system。"""

    service = FakeMemoryRetrieval(
        preview=memory_preview(conversations=1, facts=1)
    )
    install_memory_retrieval(monkeypatch, service)

    state = {
        "messages": _messages(turns=1),
        "context_policy": {"memory_retrieve": _recall_decision()},
        "user_id": "u1",
        "conversation_id": str(uuid4()),
    }

    result = await _build(state=state, dependencies=_deps())

    system = result.messages[0].content

    assert "偏好简洁" in system
    assert "原话0" in system
    assert "记忆0" in system
    assert "## 用户画像" in system
    assert "## 相关原话" in system
    assert "## 相关记忆" in system

    # 空的那一层（场景）**不出段**，不是给一个空壳。
    assert "## 记忆场景索引" not in system

    # L0 的 role 渲染成 `user`，不是 `MessageRole.USER`（混入枚举的
    # `__format__` 给的是后者，那是既有缺陷，见 memory_retrieval.py）。
    assert "[user]" in system
    assert "MessageRole" not in system

    # 四段的 source_ids 都算在 memory 名下（快照的 memory_ids）。
    assert "persona.md" in result.injected_memory_ids

    assert service.requests and service.requests[0].target.user_id == "u1"
    assert service.requests[0].target.agent_id == "general"


async def test_memory_without_decision_injects_nothing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ⚠️ **没有决策就不召回** —— 这是"每轮都白跑一次向量检索"的唯一守卫。

    Agent Builder 在"没有用户消息"那条早返回里根本不写 `context_policy`，
    所以这一格合法地不存在。
    """

    service = FakeMemoryRetrieval(preview=memory_preview(facts=1))
    install_memory_retrieval(monkeypatch, service)

    state = {
        "messages": _messages(turns=1),
        "context_policy": {},
        "user_id": "u1",
    }

    result = await _build(state=state, dependencies=_deps())

    assert service.requests == []
    assert result.injected_memory_ids == ()


async def test_memory_failure_does_not_fail_the_build(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    记忆服务挂了 -> 可选失败（规格 §二十八：Memory 不可用属于 Optional）。

    ⚠️ 失败必须**看得见**：处理器自己**不吞**异常，靠 `optional=True` 让
    流水线记 FAILED + 一条 warning。吞掉的话快照里什么都看不见，而快照
    才是排查时看的地方。
    """

    service = FakeMemoryRetrieval(error=RuntimeError("记忆服务不可用"))
    install_memory_retrieval(monkeypatch, service)

    state = {
        "messages": _messages(turns=1),
        "context_policy": {"memory_retrieve": _recall_decision()},
        "user_id": "u1",
    }

    result = await _build(state=state, dependencies=_deps())

    assert result.messages  # 构建成功
    assert result.snapshot.warnings
    assert any(
        item["processor"] == "memory" and item["status"] == "failed"
        for item in result.processor_executions
    )


async def test_memory_gate_false_skips_the_service(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_memory=False -> 处理器跳过，服务**一次都不问**。"""

    service = FakeMemoryRetrieval()
    install_memory_retrieval(monkeypatch, service)

    state = {
        "messages": _messages(turns=1),
        "context_policy": {
            "include_memory": False,
            "memory_retrieve": _recall_decision(),
        },
        "user_id": "u1",
    }

    result = await _build(state=state, dependencies=_deps())

    assert service.requests == []
    assert result.injected_memory_ids == ()

    record = next(
        item for item in result.processor_executions
        if item["processor"] == "memory"
    )

    assert record["status"] == "skipped"
    assert record["when_value"] is False


async def test_memory_replays_instead_of_recalling(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CONTINUATION 用保活的分节，**不再问服务**（一个运行只召回一次）。

    ⚠️ 这件事没有别的守卫：写错的表现是每过一个工具往返多付一次向量检索
    —— 功能全对，只是变慢，而变慢不会有人发现。所以这里用一台"一被调用
    就报错"的服务：只要它被碰过，断言就会失败。
    """

    service = FakeMemoryRetrieval(
        error=AssertionError("CONTINUATION 不该调服务")
    )
    install_memory_retrieval(monkeypatch, service)

    # ⚠️ `_messages` 的 `tool_rounds` 参数**没被用**（helper 自己的遗留），
    # 所以这里显式插一次工具往返 —— `_phase` 认的就是"最后一条 Human 之后
    # 出现过 ToolMessage"。
    messages = _messages(turns=1)

    _tool_round(messages, index=0, call_id="c1")

    state = {
        "messages": messages,
        "context_policy": {"memory_retrieve": _recall_decision()},
        "user_id": "u1",
        "memory": {
            "blocks": [
                {
                    "label": "相关记忆",
                    "text": "偏好简洁",
                    "source_ids": ["m1"],
                }
            ]
        },
    }

    result = await _build(state=state, dependencies=_deps())

    assert service.requests == []
    assert "偏好简洁" in result.messages[0].content
    assert "## 相关记忆" in result.messages[0].content
    assert result.injected_memory_ids == ("m1",)


async def test_retrieval_is_injected_and_optional() -> None:
    """检索结果注入（规格 §廿九 第 8 条）。"""

    state = {
        "messages": _messages(turns=1),
        "context_policy": {},
        "retrieved_context": {
            "items": [
                {"id": "d1", "title": "文档", "text": "均线策略"}
            ]
        },
    }

    result = await _build(state=state, dependencies=_deps())

    assert "均线策略" in result.messages[0].content
    assert result.injected_context_ids == ("d1",)


async def test_retrieval_failure_is_optional() -> None:

    source = FakeKnowledgeSource(error=RuntimeError("检索服务不可用"))

    state = {
        "messages": _messages(turns=1),
        "context_policy": {},
    }

    result = await _build(
        state=state,
        dependencies=_deps(knowledge_source=source),
    )

    assert result.messages
    assert any(
        item["processor"] == "retrieval" and item["status"] == "failed"
        for item in result.processor_executions
    )


# ============================================================
# 6. Token Budget 计算
# ============================================================


def test_budget_allocates_history_from_window() -> None:
    """预算 = 窗口 - 各段 - 输出 - 余量（规格 §二十一）。"""

    policy = ContextPolicy(
        system_budget=100,
        profile_budget=50,
        memory_budget=50,
        retrieval_budget=100,
        tool_budget=200,
        max_output_tokens=300,
        reserve_tokens=100,
        max_history_tokens=None,
    )

    budget = TokenBudget.allocate(context_window=2000, policy=policy)

    assert budget.allocated == 100 + 50 + 50 + 100 + 200 + 300 + 100
    assert budget.history_budget == 2000 - budget.allocated


def test_budget_respects_max_history_tokens() -> None:
    """`max_history_tokens` 是更紧的一层上限，与窗口取小值。"""

    policy = ContextPolicy(max_history_tokens=500)

    budget = TokenBudget.allocate(context_window=100_000, policy=policy)

    assert budget.history_budget == 500


def test_budget_too_small_is_fatal() -> None:
    """窗口装不下固定占用 -> Fatal（没有任何裁剪能救）。"""

    from app.agent.context.processor.errors import ContextBudgetError

    policy = ContextPolicy(max_output_tokens=1000, reserve_tokens=1000)

    with pytest.raises(ContextBudgetError):
        TokenBudget.allocate(context_window=500, policy=policy)


async def test_budget_uses_model_registry_window() -> None:
    """窗口来自 ModelRegistry 的模型定义（规格 §二十一）。"""

    registry = _registry(context_window=32768)

    state = {"messages": _messages(turns=1), "context_policy": {}}

    result = await _build(
        state=state,
        dependencies=ContextDependencies(model_registry=registry),
    )

    assert result.snapshot.model_context_window == 32768
    assert result.snapshot.context_window_source == "model_registry"
    assert registry.definition_calls  # 真的问了 Registry


async def test_budget_falls_back_when_window_unknown() -> None:
    """模型没声明窗口 -> 回落 + warning（兜底必须是可见的）。"""

    registry = _registry(context_window=None)

    state = {
        "messages": _messages(turns=1),
        "context_policy": {"default_context_window": 32768},
    }

    result = await _build(
        state=state,
        dependencies=ContextDependencies(model_registry=registry),
    )

    assert result.snapshot.model_context_window == 32768
    assert result.snapshot.context_window_source == "policy_default"
    assert any("context_window" in item for item in result.snapshot.warnings)


def test_heuristic_counter_is_not_message_count() -> None:
    """"不要简单 len(messages)"：计数要按内容算。"""

    counter = HeuristicTokenCounter(chars_per_token=3.0)

    short = counter.count_messages([AIMessage(content="短")])
    long = counter.count_messages([AIMessage(content="长" * 300)])

    assert long > short * 10
    assert counter.count_tools([]) == 0


# ============================================================
# 7. Snapshot
# ============================================================


async def test_snapshot_explains_the_invocation() -> None:
    """快照能回答"为什么这一轮看到了这些"（规格 §廿七）。"""

    state = {
        "messages": _messages(turns=3),
        "context_policy": {"system_prompt": "sys", "include_memory": False},
        "user_id": "u1",
        "memory": {"items": [{"id": "m1", "text": "x"}]},
        "retrieved_context": {"items": [{"id": "d1", "text": "y"}]},
    }

    result = await _build(state=state, dependencies=_deps())

    snapshot = result.snapshot

    assert snapshot.conversation_id == "thread-1"
    assert snapshot.builder_type == "agent_invoke"
    assert snapshot.agent_type == "react"
    assert snapshot.phase == "initial"
    assert snapshot.history_message_ids
    assert snapshot.retrieved_context_ids == ("d1",)
    assert snapshot.memory_ids == ()          # gate 关掉了
    assert snapshot.policy["include_memory"] is False
    assert snapshot.budget["history_budget"] > 0
    assert snapshot.input_token_count > 0
    assert snapshot.context_hash
    assert snapshot.tool_names == ()

    # 快照必须 JSON 安全（它要进 JSONB 观测列）
    import json

    assert json.dumps(snapshot.as_metadata(), ensure_ascii=False)


async def test_snapshot_is_carried_in_model_request_metadata() -> None:
    """快照随请求 metadata 走（节点靠它接进图观测）。"""

    state = {"messages": _messages(turns=1), "context_policy": {}}

    result = await _build(state=state, dependencies=_deps())

    raw = result.model_request.metadata

    assert raw["context_snapshot"]["context_hash"] == result.context_hash
    assert raw["context_snapshot"]["history_message_ids"]
    assert raw["context_token_usage"] == {}  # 这次没有嵌套模型调用


async def test_context_hash_is_stable_for_same_input() -> None:
    """同样的输入 -> 同样的哈希（Replay / 去重的依据）。"""

    state = {"messages": _messages(turns=2), "context_policy": {"system_prompt": "s"}}

    first = await _build(state=state, dependencies=_deps())
    second = await _build(state=state, dependencies=_deps())

    assert first.context_hash == second.context_hash


# ============================================================
# 8. Summary 生成：走 ModelRegistry（唯一调模型的阶段）
# ============================================================


async def test_summarizer_uses_model_registry() -> None:
    """摘要用 ModelRegistry 的模型（规格 §十七、§廿九 第 11 条）。"""

    registry = _registry(reply="这是新的摘要")

    compactor = ModelRegistrySummaryCompactor(
        model_registry=registry,
        agent_type="react",
        system_prompt="你是摘要器。",
    )

    draft = await compactor.compact(
        previous=None,
        messages=_messages(turns=2),
        max_output_tokens=256,
    )

    assert draft.text == "这是新的摘要"
    assert registry.resolved == [("react", ModelRole.SUMMARIZATION)]
    assert registry.summary_calls == 1
    assert draft.usage is not None


async def test_summary_prompt_carries_previous_summary() -> None:
    """摘要的 prompt 里带上一版摘要（合并而不是另起一段）。"""

    from app.agent.context.processor.values import SummarySnapshot

    registry = _registry()

    compactor = ModelRegistrySummaryCompactor(
        model_registry=registry,
        agent_type="react",
        system_prompt="你是摘要器。",
    )

    request = compactor.build_request(
        previous=SummarySnapshot(text="旧摘要内容", revision=7),
        messages=_messages(turns=1),
        max_output_tokens=100,
    )

    payload = request.messages[1].content

    assert "旧摘要内容" in payload
    assert "revision 7" in payload
    assert "问题0" in payload


async def test_summary_is_generated_above_the_limit_and_bumps_revision() -> None:
    """未覆盖轮数超过窗口上限才生成；生成后 revision +1 且写了 cursor。"""

    state = {
        "messages": _messages(turns=12),
        "context_policy": {"max_history_tokens": 100_000},
    }

    result = await _build(state=state, dependencies=_deps())

    assert result.summary_revision == 1

    updates = (result.model_request.metadata or {}).get("context_state_updates")

    assert updates is not None
    assert updates["summary"]["revision"] == 1
    assert updates["summary"]["summarized_until_message_id"]
    assert updates["summary"]["text"]

    summary = next(
        item for item in result.processor_executions
        if item["processor"] == "summary"
    )

    outcome = summary["detail"]["outcome"]

    assert outcome["generated"] is True
    assert outcome["folded_messages"] > 0
    # 生成之后：摘要覆盖 11 轮（当前这一轮不折），窗口回到下限 3 轮。
    assert outcome["uncovered_turns"] == 12
    assert outcome["turn_limit"] == 4


async def test_summary_is_not_generated_below_the_limit() -> None:
    """未覆盖轮数没超过上限就**不**生成（"摘要不是每次请求都重新生成"）。"""

    state = {"messages": _messages(turns=2), "context_policy": {}}

    result = await _build(state=state, dependencies=_deps())

    assert result.summary_revision is None
    assert "context_state_updates" not in (result.model_request.metadata or {})

    summary = next(
        item for item in result.processor_executions
        if item["processor"] == "summary"
    )

    assert summary["detail"]["outcome"]["generated"] is False
    assert summary["detail"]["outcome"]["reason"] == "below_threshold"
    assert summary["status"] == "executed"


async def test_generation_is_off_when_compaction_is_disallowed() -> None:
    """allow_compaction=False -> 不生成摘要（显式决定，进快照）。"""

    state = {
        "messages": _messages(turns=12),
        "context_policy": {"allow_compaction": False},
    }

    result = await _build(state=state, dependencies=_deps())

    assert result.summary_revision is None

    summary = next(
        item for item in result.processor_executions
        if item["processor"] == "summary"
    )

    assert summary["detail"]["outcome"]["generated"] is False
    assert summary["detail"]["outcome"]["reason"] == "compaction_disabled"

    compaction = next(
        item for item in result.processor_executions
        if item["processor"] == "compaction"
    )

    assert compaction["status"] == "skipped"


async def test_compaction_cuts_the_overlap_when_over_budget() -> None:
    """
    超预算时 compaction 把窗口与摘要的重叠压成 0（纯切片，不调模型）。

    12 轮、默认策略 -> summary 生成，覆盖 11 轮，窗口保底 3 轮，于是有
    2 轮重叠（问题9、问题10）。给一个极紧的 history 预算，compaction
    应当把它们切掉，且**不**产生缺口。
    """

    state = {
        "messages": _messages(turns=12),
        "context_policy": {
            "min_recent_turns": 3,
            "max_recent_turns": 5,
            "max_history_tokens": 1,
        },
    }

    result = await _build(state=state, dependencies=_deps())

    compaction = next(
        item for item in result.processor_executions
        if item["processor"] == "compaction"
    )

    outcome = compaction["detail"]["outcome"]

    assert outcome["compacted"] is True
    assert outcome["reason"] == "overlap_cut"

    humans = [
        message.content for message in result.messages
        if isinstance(message, HumanMessage)
    ]

    # 压完只剩当前这一轮（摘要已经覆盖到问题10）。
    assert humans == ["问题11"]

    history = next(
        item for item in result.processor_executions
        if item["processor"] == "history"
    )

    assert history["detail"]["outcome"]["gap_turns"] == 0


async def test_summary_usage_is_recorded() -> None:
    """嵌套模型调用的用量进结果的 token_usage（规格 §十九）。"""

    state = {"messages": _messages(turns=12), "context_policy": {}}

    result = await _build(state=state, dependencies=_deps())

    assert result.token_usage["input_tokens"] == 100
    assert result.token_usage["output_tokens"] == 20
    assert result.token_usage["calls"] == 1


# ============================================================
# 9. 折叠节奏：0 -> 50 轮的完整轨迹
# ============================================================


def test_fold_cadence_over_fifty_turns() -> None:
    """
    逐轮推演 0→50：**缺口整列恒为 0**，且折叠点就是设计里那张表。

    这条用例钉的不是某个实现细节，而是三个阶段共同遵守的那条不变量
    （"摘要 ∪ 窗口 = 全部对话"，见 turns.py）。它是这个设计唯一的
    正确性判据 —— 任何一处把窗口算宽/算窄一轮，这里都会红。

    ⚠️ gap 要在**折完之后**看
    ------------------------
    触发那一轮、折叠之前，gap 恰好是 1 —— 那**就是触发信号本身**
    （"窗口往前一滑就要露出缺口了"）。折完它立刻回到 0。两者之间隔着
    同一次 build 里的两个阶段（summary 在 order 30、history 在 40），
    而模型调用在整条流水线之后，所以模型**看不到**那个瞬态缺口。

    折叠点：4, 7, 11, 15, 19, 24, 29, 34, 39, 44, 49
    间隔：  3, 3,  4,  4,  4,  5,  5,  5,  5,  5,  5
    （上限 `min(5, 3 + T // 10)` 随对话变长，间隔跟着从 3 拉到 5）
    """

    from app.agent.context.processor.turns import plan_window

    messages: list = []
    cursor: str | None = None
    folded_at: list[int] = []

    for turn in range(1, 51):

        messages.append(HumanMessage(content=f"问题{turn - 1}", id=f"u{turn - 1}"))
        messages.append(AIMessage(content=f"回答{turn - 1}", id=f"a{turn - 1}"))

        plan = plan_window(messages, cursor, min_turns=3, max_turns=5)

        if plan.should_summarize:

            folded_at.append(turn)

            # 折完的 cursor = 最后一个被折进去的轮的末条消息，然后重算
            # —— 这一步对应"summary 阶段写回 cursor、history 阶段据此
            # 重新划窗口"。
            cursor = f"a{plan.fold_end - 1}"

            plan = plan_window(messages, cursor, min_turns=3, max_turns=5)

        # 不变量：任何一轮都不允许出现"既不在摘要、也不在窗口"的轮。
        assert plan.gap_turns == 0, f"第 {turn} 轮出现缺口：{plan.as_metadata()}"
        assert plan.overlap_turns <= plan.min_turns - 1

    assert folded_at == [4, 7, 11, 15, 19, 24, 29, 34, 39, 44, 49]

    # 50 轮时的终局：摘要覆盖 1–48，窗口保留最后 3 轮（1 轮重叠）。
    final = plan_window(messages, cursor, min_turns=3, max_turns=5)

    assert final.covered_turns == 48
    assert final.uncovered_turns == 2
    assert final.window_turns == 3
    assert final.overlap_turns == 1
    assert final.gap_turns == 0
