"""
ReactGraph 与 ContextBuilder 的集成（规格 §廿三、§廿四、§廿九 Integration）。

跑的是**真的图**：`register_react_graph` + 编译 + InMemorySaver + 真的
`agent_node`，只有 provider 被换成了假适配器。

    User
     ↓
    ContextBuilder（LLM Node 的前置层）
     ↓
    ModelRegistry -> LLM
     ↓
    Tool
     ↓
    ContextBuilder（Tool Result 之后**再跑一次**）
     ↓
    LLM
     ↓
    Final Answer

⚠️ 工具节点是 monkey-patch 上去的
------------------------------
本文件测的是**上下文构建**，不是工具执行，所以把 tools 节点换成一个
只写 ToolMessage 的假节点。`graph.py` 把节点单列成 `NODE_FUNCTIONS`
字典，docstring 明写"是为了让冒烟脚本能 monkey-patch 掉某个节点" ——
本文件走的就是那条被认可的路。

⚠️ 假节点收到的是 **Send payload**，不是父状态
--------------------------------------------
tools 是**扇出**节点（一 tool_call 一分支，见 routing.route_after_agent）：
LangGraph 的 Send payload 会**取代** state，所以假节点读的是
`state["call"]` 而不是 `state["messages"]`。下面每个用例都只发一个
tool_call，因此"一个分支"与"一条 ToolMessage"是一一对应的。

（`extra_state` 里写的 `context_policy` 是分支的返回值，经 channel
合并回父状态。多分支时它们会互相覆盖 —— 那是这个假节点的局限，
不是被测行为；本文件的用例都只有一个调用。）
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from fakes import FakeAdapter, Outcome, make_definition, make_registry

from app.agent.graph.react import graph as graph_module
from app.agent.graph.react.catalog import AGENT_TYPE, MODEL_ROLE
from app.agent.graph.react.runtime import register_react_graph
from app.agent.model import (
    ModelCapabilities,
    ModelProfile,
    ModelRoleBinding,
    ModelRuntimePolicy,
)
from app.infra.graph import GraphRegistry



TOOL_CAPABLE = ModelCapabilities(tool_calling=True, structured_output=True)

TOOL_CALL = {
    "name": "lookup",
    "args": {"symbol": "AAPL"},
    "id": "call-1",
    "type": "tool_call",
}


# ============================================================
# 装配
# ============================================================


class _Recorder:
    """收集 node_span 写下来的观测行。"""

    def __init__(self) -> None:
        self.updates: list = []

    async def upsert_node_run(self, update) -> None:
        self.updates.append(update)

    def by_node(self, node_id: str) -> list:

        return [item for item in self.updates if item.node_id == node_id]

    def last(self, node_id: str):

        found = self.by_node(node_id)

        return found[-1] if found else None


class _RunContext:
    def __init__(self, recorder) -> None:
        from uuid import uuid4

        self.pipeline_run_id = uuid4()
        self.thread_id = "thread-1"
        self.trigger_source = "test"
        self.recorder = recorder
        # RunContext 契约的一部分（见 infra/graph/registry.py）。
        # 本文件不测鉴权，给 None 即可。
        self.caller_identity = None
        # 工具执行账本的判重键里有它（agent_session.session_id）。
        # ⚠️ 给 None 的话账本会拒绝执行任何工具 —— 本文件把 tools 节点
        # 换成了假节点，所以不走那条路；但真装配下它必须有值。
        self.session_id = uuid4()
        # 这一轮的职责（= 记忆 scope 的 agent 那一半）。本文件不摸记忆，
        # 给 None 即可 —— 但**必须有这个属性**：runtime 的 context_factory
        # 按契约直接取它（见 infra/graph/registry.py 的 duty_key）。
        self.duty_key = None


def _build_graph(adapter, *, tool_node) -> tuple:

    registry = make_registry(
        models=[make_definition("primary", capabilities=TOOL_CAPABLE)],
        profiles=[
            ModelProfile(
                id="reasoning_default",
                primary="fake:primary",
                policy=ModelRuntimePolicy(
                    timeout=5.0, max_retries=0, retry_backoff=0.0
                ),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type=AGENT_TYPE,
                model_role=MODEL_ROLE,
                profile="reasoning_default",
            )
        ],
        adapter=adapter,
    )

    recorder = _Recorder()

    definition = register_react_graph(
        registry=GraphRegistry(),
        checkpointer=InMemorySaver(),
        recorder=recorder,
        model_registry=registry,
        # builder 不传 -> 走 create_context_builder(AGENT_INVOKE)
        # （也就是生产路径上那个 Processor 版实现）
    )

    context = definition.context_factory(_RunContext(recorder))

    return definition, context, registry, recorder


@pytest.fixture
def tools_node(monkeypatch):
    """
    往图里挂一个假工具节点（见模块 docstring）。

    它把 ToolMessage 写回 messages；`extra_state` 用来测"state 变了 ->
    Processor gate 从 SKIPPED 变 EXECUTED"。
    """

    calls: list = []

    def make(*, extra_state: dict | None = None):

        async def fake_tools_node(state, runtime):

            calls.append(dict(state))

            # 扇出分支：这个节点一次只被要求处理一个调用。
            call = state["call"]

            result: dict = {
                "messages": [
                    ToolMessage(
                        content=f"结果:{call['name']}",
                        tool_call_id=call["id"],
                        id=f"tool-{call['id']}",
                    )
                ]
            }

            if extra_state:
                result.update(extra_state)

            return result

        monkeypatch.setitem(graph_module.NODE_FUNCTIONS, "tools", fake_tools_node)

        return calls

    return make


def _state(**overrides) -> dict:

    state = {
        "messages": [HumanMessage(content="查一下 AAPL", id="u1")],
        "user_id": "u1",
        "conversation_id": "c1",
        "context_policy": {"system_prompt": "你是交易助手"},
        "user_context": {"profile": {"tier": "gold"}},
        # 记忆处理器保活下来的分节（形状见 react/state.py 的 `memory` 字段）。
        # 它只在决策说"要召回"时才会被读到 —— 决策走
        # `context_policy.memory_retrieve`。
        "memory": {
            "blocks": [
                {
                    "label": "相关记忆",
                    "text": "偏好简洁",
                    "source_ids": ["mem-1"],
                }
            ]
        },
        "retrieved_context": {"items": []},
        "exposed_tools": [],
    }

    state.update(overrides)

    return state


async def _run(definition, context, state: dict, *, thread_id="thread-1"):

    return await definition.compiled.ainvoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
        context=context,
    )


# ============================================================
# 1. 完整循环
# ============================================================


async def test_react_loop_runs_through_context_builder_twice(tools_node) -> None:
    """
    一次工具往返：ContextBuilder 跑两次（第一次 + Tool Result 之后）。

        第一次 LLM -> 要求调工具
        Tool -> ToolMessage
        第二次 LLM -> 最终答复
    """

    tools_node()

    adapter = FakeAdapter(script={
        "fake:primary": [
            Outcome(message=AIMessage(
                content="",
                tool_calls=[TOOL_CALL],
                id="ai-1",
            )),
            Outcome(message=AIMessage(content="AAPL 现在 100", id="ai-2")),
        ]
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    final = await _run(definition, context, _state())

    # 最终答复
    assert final["messages"][-1].content == "AAPL 现在 100"

    # 模型被调了两次（各一次 provider attempt）
    assert adapter.called_models() == ["fake:primary", "fake:primary"]

    # 工具结果进了 state（工具没丢）
    tool_messages = [
        message for message in final["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].content == "结果:lookup"


async def test_second_llm_call_sees_the_tool_result(tools_node) -> None:
    """第二次 LLM 调用**用了** Tool Result（规格 §廿九 Integration）。"""

    tools_node()

    adapter = FakeAdapter(script={
        "fake:primary": [
            Outcome(message=AIMessage(content="", tool_calls=[TOOL_CALL], id="ai-1")),
            Outcome(message=AIMessage(content="done", id="ai-2")),
        ]
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    await _run(definition, context, _state())

    second_request = adapter.requests[-1]

    contents = [getattr(message, "content", "") for message in second_request.messages]

    assert any("结果:lookup" in str(item) for item in contents)

    # 而且那条 ToolMessage 在**第二次请求**里，不在第一次
    first_contents = [
        getattr(message, "content", "")
        for message in adapter.requests[0].messages
    ]

    assert not any("结果:lookup" in str(item) for item in first_contents)


async def test_context_builder_does_not_reload_the_conversation(tools_node) -> None:
    """
    第二次执行是 CONTINUATION —— 它基于当前 state 重算，**不**回库重放。

    （"不回库"在图里体现为：phase=continuation 时历史处理器只用
    state 里的消息；测试用一条计数源验证它一次都没被问到 —— 见
    test_agent_invoke_builder.py 的同名用例。这里断言的是可观测的结果：
    快照里的 phase 与选中的消息 id。）
    """

    tools_node()

    adapter = FakeAdapter(script={
        "fake:primary": [
            Outcome(message=AIMessage(content="", tool_calls=[TOOL_CALL], id="ai-1")),
            Outcome(message=AIMessage(content="done", id="ai-2")),
        ]
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    final = await _run(definition, context, _state())

    snapshot = final["context_result"]

    assert snapshot["phase"] == "continuation"
    assert snapshot["builder_type"] == "agent_invoke"

    # 选中的历史里**同时**有最初那条用户消息与工具结果（没有重建、没有丢）
    assert "u1" in snapshot["history_message_ids"]
    assert "tool-call-1" in snapshot["history_message_ids"]


async def test_every_llm_call_goes_through_the_context_builder(tools_node) -> None:
    """
    所有 LLM 调用都经过 ContextBuilder（规格 §廿四）。

    判据：**每一次** provider 调用之前，agent 节点都写了一轮
    context_snapshot（两次 provider 调用 -> 两轮 agent 观测，每轮都带
    `context_hash`）。
    """

    tools_node()

    adapter = FakeAdapter(script={
        "fake:primary": [
            Outcome(message=AIMessage(content="", tool_calls=[TOOL_CALL], id="ai-1")),
            Outcome(message=AIMessage(content="done", id="ai-2")),
        ]
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    await _run(definition, context, _state())

    agent_runs = [
        item for item in recorder.by_node("agent")
        if item.metrics and "context_hash" in item.metrics
    ]

    assert len(agent_runs) == 2

    for item in agent_runs:
        assert item.metrics["context_hash"]
        assert item.metrics["context_phase"] in ("initial", "continuation")
        assert item.metrics["model_provider"] == "fake"

    assert [item.metrics["context_phase"] for item in agent_runs] == [
        "initial",
        "continuation",
    ]


async def test_model_registry_is_the_only_model_entry(tools_node) -> None:
    """
    模型调用的唯一入口是 ModelRegistry。

    判据：唯一的 provider 适配器记到了**全部**调用。任何绕过 Registry
    的路径都不会出现在它的调用记录里（而那种路径不可能自己造出一个
    可用的客户端 —— 见 model/ 的 anti-pattern 清单）。
    """

    tools_node()

    adapter = FakeAdapter(script={
        "fake:primary": [
            Outcome(message=AIMessage(content="", tool_calls=[TOOL_CALL], id="ai-1")),
            Outcome(message=AIMessage(content="done", id="ai-2")),
        ]
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    await _run(definition, context, _state())

    # 两次 LLM 调用 = 两次 provider attempt，全部落在注册表里那个适配器上
    assert len(adapter.calls) == 2

    # 而 ContextBuilder 的预算/压缩也走的是**同一个**注册表
    # （fake:primary 没有 context_window -> 预算用策略兜底并记 warning）
    snapshot = (await _run(
        definition, context, _state(), thread_id="thread-2"
    ))["context_result"]

    assert snapshot["model_role"] == MODEL_ROLE
    assert snapshot["context_window_source"] in ("model_registry", "policy_default")


# ============================================================
# 2. Processor gate 每轮重新判定（规格 §廿九 Integration）
# ============================================================


async def test_gate_is_re_evaluated_and_can_flip_to_executed(tools_node) -> None:
    """
    State 变化后，同一个 Processor 可以从 SKIPPED 变成 EXECUTED。

    工具节点把 `context_policy.include_memory` 从 False 改成 True ——
    第二次构建时 memory 处理器因此真的跑了（gate 每轮按当前 state 判定，
    而不是在第一次构建时被缓存）。
    """

    tools_node(extra_state={
        "context_policy": {
            "system_prompt": "你是交易助手",
            "include_memory": True,
            # ⚠️ 光开 gate 不够了：记忆处理器还要知道"以谁的身份搜"才动手
            # （判据见 processors/memory.py 的 `process`）。标签由 agent_id
            # 缺失时那一格表达"上游认为不该召回"。
            #
            # 这里走的是 CONTINUATION 那条路 —— 复用 `memory` 通道里保活的
            # 分节，所以本测试**不需要**接一个真的记忆服务。
            "memory_retrieve": {
                "l0": False, "l1": False, "agent_id": "general",
            },
        }
    })

    adapter = FakeAdapter(script={
        "fake:primary": [
            Outcome(message=AIMessage(content="", tool_calls=[TOOL_CALL], id="ai-1")),
            Outcome(message=AIMessage(content="done", id="ai-2")),
        ]
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    final = await _run(
        definition,
        context,
        _state(context_policy={
            "system_prompt": "你是交易助手",
            "include_memory": False,
        }),
    )

    # 最后那一轮的快照里，memory 是 EXECUTED（state 被工具节点改了）
    snapshot = final["context_result"]

    statuses = {
        item["processor"]: item["status"] for item in snapshot["processors"]
    }

    assert statuses["memory"] == "executed"
    assert snapshot["memory_ids"] == ["mem-1"]

    # 而第一轮时它是 SKIPPED（同一张图、同一次运行内的两轮）
    first_round = [
        item for item in recorder.by_node("agent")
        if item.metrics and item.metrics.get("context_phase") == "initial"
    ]

    assert first_round
    assert first_round[0].metrics["context_memory_items"] == 0


async def test_context_result_is_written_back_to_state(tools_node) -> None:
    """快照回写进 state（规格 §廿七：可回看"这一轮看到了什么"）。"""

    tools_node()

    adapter = FakeAdapter(script={
        "fake:primary": [
            Outcome(message=AIMessage(content="直接回答", id="ai-1")),
        ]
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    final = await _run(definition, context, _state())

    snapshot = final["context_result"]

    assert snapshot["conversation_id"] == "c1"
    assert snapshot["context_hash"]
    assert snapshot["budget"]["history_budget"] > 0
    assert snapshot["input_token_count"] > 0


async def test_summary_is_written_back_only_when_compacted(tools_node) -> None:
    """
    只有 compaction 真的压过一轮，摘要才回写 state。

    （否则每一轮都会把同一份摘要重新写一遍 —— 见 state.py 里"其余通道
    是最后写入者赢"那条说明。）
    """

    tools_node()

    adapter = FakeAdapter(script={
        "fake:primary": [Outcome(message=AIMessage(content="ok", id="ai-1"))],
    })

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    final = await _run(definition, context, _state())

    # 没压过 -> 没有 summary 更新（state 里保持调用方给的原值）
    assert "summary" not in final or final["summary"] == {}


# ============================================================
# 3. 失败路径：不吞异常（规格 §廿八）
# ============================================================


async def test_invalid_gate_type_fails_the_run(tools_node) -> None:
    """`include_memory: "yes"` -> 本轮 Run 失败（不是静默执行）。"""

    tools_node()

    adapter = FakeAdapter()

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    from app.agent.context.processor.errors import ContextStateError

    with pytest.raises(ContextStateError):
        await _run(
            definition,
            context,
            _state(context_policy={"include_memory": "yes"}),
        )

    # 一次模型调用都没发生 —— 上下文没构建出来就不该调模型
    assert adapter.calls == []


async def test_missing_user_message_fails_the_run(tools_node) -> None:
    """没有用户消息 -> 失败（而不是让模型凭空答一句）。"""

    tools_node()

    adapter = FakeAdapter()

    definition, context, registry, recorder = _build_graph(adapter, tool_node=None)

    from app.agent.context.processor.errors import ContextBuildError

    with pytest.raises(ContextBuildError):
        await _run(
            definition,
            context,
            _state(messages=[], conversation_id="c1"),
        )

    assert adapter.calls == []
