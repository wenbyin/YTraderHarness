"""子代理子图：拓扑、routing、listener 与 builder 发现。"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Send

from app.agent.context.builder.subagent import (
    SubAgentProcessorContextBuilder,
)
from app.agent.context.registry import create_context_builder
from app.agent.graph.subagent import graph as graph_module
from app.agent.graph.subagent import nodes as nodes_module
from app.agent.graph.subagent.catalog import (
    LLM_NODE,
    TASK_LISTENER_NODE,
    TOOL_NODE,
)
from app.agent.graph.subagent.graph import build_subagent_graph
from app.agent.graph.subagent.nodes import llm_node, task_listener_node
from app.agent.graph.subagent.routing import (
    route_after_listener,
    route_after_llm,
)
from app.agent.plan.listener import StateTaskSubscriber
from app.common.context import ContextBuilderType

__all__: list[str] = []


def _task(
        tid: str,
        *,
        state: str = "TASK_STATE_SUBMITTED",
        claim: dict | None = None,
) -> dict:
    return {
        "id": tid,
        "status": {"state": state},
        "metadata": {
            "plan_id": "p1",
            "blocked_by": [],
            "claim": claim or {},
            "skill_key": "research",
        },
    }


class _Recorder:
    async def upsert_node_run(self, update) -> None:
        return None


def _context(buffer=None):
    return SimpleNamespace(
        pipeline_run_id=uuid4(),
        session_id=None,
        recorder=_Recorder(),
        trace_id=None,
        caller_identity=None,
        task_board=None,
        subagent_builder=buffer,
    )


def _runtime(buffer=None):
    return SimpleNamespace(context=_context(buffer))


# ============================================================
# 拓扑
# ============================================================


def test_graph_has_three_nodes_and_expected_edges() -> None:
    """
    子图只有三个节点：task_listener / llm_node / tool_node。
    tool_node 没有静态出边（Command.goto 决定下一跳）。
    """

    compiled = build_subagent_graph().compile()

    nodes = set(compiled.get_graph().nodes)

    assert {
        TASK_LISTENER_NODE,
        LLM_NODE,
        TOOL_NODE,
    } <= nodes

    edges = {
        (edge.source, edge.target)
        for edge in compiled.get_graph().edges
    }

    assert (TASK_LISTENER_NODE, LLM_NODE) in edges
    assert (TASK_LISTENER_NODE, END) in edges
    assert (LLM_NODE, TOOL_NODE) in edges
    assert (LLM_NODE, TASK_LISTENER_NODE) in edges

    # tool_node 的下一跳由 Command 决定，装配期**没有**指向业务节点的
    # 静态边（编译出的 `tool_node -> __end__` 是 langgraph 给无出边节点
    # 补的隐式边，不是装配期声明的路）。
    assert {
        (source, target)
        for source, target in edges
        if source == TOOL_NODE
    } == {(TOOL_NODE, END)}


def test_subagent_output_schema_is_empty() -> None:
    """
    子图作为父图节点时，output_schema 必须为空 —— 任何键都会按父图
    同名 channel 合并（messages 串进主对话 / tasks 并发写）。
    """

    from app.agent.graph.subagent.state import SubAgentOutput

    assert SubAgentOutput.__annotations__ == {}


def test_graph_module_registers_all_three_functions() -> None:
    assert set(graph_module.NODE_FUNCTIONS) == {
        TASK_LISTENER_NODE,
        LLM_NODE,
        TOOL_NODE,
    }


# ============================================================
# routing
# ============================================================


def test_route_after_listener_goes_to_llm_when_claimed() -> None:
    assert route_after_listener({"current_task": _task("t1")}) == LLM_NODE


def test_route_after_listener_ends_without_task() -> None:
    assert route_after_listener({"current_task": {}}) == END


def test_route_after_llm_sends_one_tool_branch_per_call() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "a", "args": {}, "id": "c1"},
            {"name": "b", "args": {}, "id": "c2"},
        ],
        id="m1",
    )

    state = {
        "messages": [message],
        "current_task": _task("t1"),
        "task_tool_keys": ["tool.a", "tool.b"],
    }

    destinations = route_after_llm(state)

    assert isinstance(destinations, list)
    assert len(destinations) == 2

    first = destinations[0]

    assert isinstance(first, Send)
    assert first.node == TOOL_NODE
    assert first.arg["call"]["id"] == "c1"
    assert first.arg["tool_keys"] == ["tool.a", "tool.b"]
    assert first.arg["return_to"] == LLM_NODE
    assert first.arg["is_controller"] is True
    assert first.arg["call_scope"] == "m1:c1"

    assert destinations[1].arg["is_controller"] is False


def test_route_after_llm_returns_to_listener_without_tool_calls() -> None:
    state = {
        "messages": [AIMessage("任务完成")],
        "current_task": _task("t1", state="TASK_STATE_WORKING"),
    }

    assert route_after_llm(state) == TASK_LISTENER_NODE


# ============================================================
# task_listener
# ============================================================


def _patch_state_source(monkeypatch) -> None:
    """强制 state source：单测不能连本机 Redis。"""

    def _subscriber(state, *, context_id, plan_id=None):
        return StateTaskSubscriber(state)

    monkeypatch.setattr(nodes_module, "build_task_subscriber", _subscriber)


@pytest.mark.asyncio
async def test_listener_claims_first_ready_task(monkeypatch) -> None:
    _patch_state_source(monkeypatch)

    state = {
        "messages": [HumanMessage("开始")],
        "tasks": [_task("t1"), _task("t2")],
        "current_task": {},
        "plan": {"plan_id": "p1", "revision": 1},
        "conversation_id": "c1",
    }

    update = await task_listener_node(state, _runtime())  # type: ignore[arg-type]

    assert update["task_id"] == "t1"
    assert update["attempt"] == 1
    assert update["skill_key"] == "research"
    assert update["current_task"]["status"]["state"] == "TASK_STATE_WORKING"
    assert update["task_tool_keys"] == []

    instruction = update["messages"][-1]

    assert isinstance(instruction, HumanMessage)
    assert instruction.id == "task:t1:instruction"
    assert instruction.additional_kwargs["kind"] == "task_instruction"


@pytest.mark.asyncio
async def test_listener_completes_previous_and_clears_transcript(
        monkeypatch,
) -> None:
    _patch_state_source(monkeypatch)

    working = _task(
        "t1",
        state="TASK_STATE_WORKING",
        claim={"owner": "run:worker", "token": "x"},
    )

    old = AIMessage("任务 1 的结果", id="a1")

    state = {
        "messages": [HumanMessage("开始", id="h1"), old],
        "tasks": [working, _task("t2")],
        "current_task": working,
        "plan": {"plan_id": "p1", "revision": 1},
        "conversation_id": "c1",
    }

    update = await task_listener_node(state, _runtime())  # type: ignore[arg-type]

    # 下一个任务被领到。
    assert update["task_id"] == "t2"

    # 本地任务列表里 t1 已 COMPLETED。
    by_id = {task["id"]: task for task in update["tasks"]}

    assert by_id["t1"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert by_id["t1"]["artifacts"][0]["parts"][0]["text"] == "任务 1 的结果"

    # 旧 transcript 被墓碑清掉，只留下新任务指令。
    messages = update["messages"]

    assert [m.id for m in messages if isinstance(m, RemoveMessage)] == [
        "h1", "a1",
    ]
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].id == "task:t2:instruction"


@pytest.mark.asyncio
async def test_listener_adopts_preclaimed_pending_task(monkeypatch) -> None:
    """
    dispatch 预 claim 的任务：收养它，不 claim、不完成。
    """

    calls: list = []

    def _subscriber(state, *, context_id, plan_id=None):
        calls.append(state)
        return StateTaskSubscriber(state)

    monkeypatch.setattr(nodes_module, "build_task_subscriber", _subscriber)

    pending = _task(
        "p1:t1",
        state="TASK_STATE_WORKING",
        claim={"owner": "run:worker:0", "token": "x", "attempt": 2},
    )

    state = {
        "messages": [],
        "pending_task": pending,
        "current_task": {},
        "tasks": [pending, _task("p1:t2")],
        "plan": {"plan_id": "p1", "revision": 1},
        "conversation_id": "c1",
        "worker_id": "run:worker:0",
        "single_task": True,
    }

    update = await task_listener_node(
        state, _runtime(),  # type: ignore[arg-type]
    )

    # 没有走 claim（预 claim 已经在父图完成）。
    assert calls == []

    assert update["pending_task"] == {}
    assert update["task_id"] == "p1:t1"
    assert update["attempt"] == 2
    assert update["current_task"]["status"]["state"] == "TASK_STATE_WORKING"
    assert update["messages"][-1].id == "task:p1:t1:instruction"


@pytest.mark.asyncio
async def test_listener_single_task_exits_after_completion(
        monkeypatch,
) -> None:
    """一任务一 worker：完成 current_task 后不 claim 下一个。"""

    _patch_state_source(monkeypatch)

    working = _task(
        "t1",
        state="TASK_STATE_WORKING",
        claim={"owner": "run:worker:0", "token": "x"},
    )

    state = {
        "messages": [AIMessage("任务 1 的结果", id="a1")],
        "current_task": working,
        "tasks": [working, _task("t2")],
        "plan": {"plan_id": "p1", "revision": 1},
        "conversation_id": "c1",
        "single_task": True,
    }

    update = await task_listener_node(
        state, _runtime(),  # type: ignore[arg-type]
    )

    assert update["current_task"] == {}
    assert update["task_id"] == ""

    by_id = {task["id"]: task for task in update["tasks"]}

    assert by_id["t1"]["status"]["state"] == "TASK_STATE_COMPLETED"
    # 没有抢下一个任务。
    assert by_id["t2"]["status"]["state"] == "TASK_STATE_SUBMITTED"
    assert "messages" not in update


@pytest.mark.asyncio
async def test_listener_ends_when_no_ready_task(monkeypatch) -> None:
    _patch_state_source(monkeypatch)

    state = {
        "messages": [],
        "tasks": [_task("t1", state="TASK_STATE_COMPLETED")],
        "current_task": {},
        "plan": {"plan_id": "p1", "revision": 1},
        "conversation_id": "c1",
    }

    update = await task_listener_node(state, _runtime())  # type: ignore[arg-type]

    assert update["current_task"] == {}
    assert update["task_id"] == ""
    assert route_after_listener({**state, **update}) == END


# ============================================================
# llm_node 的守卫
# ============================================================


@pytest.mark.asyncio
async def test_graph_runs_three_node_loop_and_ends(monkeypatch) -> None:
    """
    task_listener -> llm_node -> task_listener -> END。

    llm 用桩替换（本文件不测模型调用），验证三个节点确实串成一条
    有限回边：领一个任务、执行一次、结算后没有下一个任务就退出。
    """

    _patch_state_source(monkeypatch)

    calls: list[str] = []

    async def fake_llm(state, runtime, **kwargs):
        calls.append(state["task_id"])
        return {"messages": [AIMessage("任务完成", id="a1")]}

    monkeypatch.setattr(nodes_module, "run_agent_turn", fake_llm)

    compiled = build_subagent_graph().compile(
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "t1"}}

    await compiled.ainvoke(
        {
            "messages": [],
            "tasks": [_task("t1")],
            "current_task": {},
            "plan": {"plan_id": "p1", "revision": 1},
            "conversation_id": "c1",
        },
        config,
        # llm_node 只做 None 守卫；真正的模型调用被上面的桩替换。
        context=_context(buffer=object()),
    )

    assert calls == ["t1"]

    # 子图 output_schema 为空（作为父图节点时不回写父 state），所以看
    # checkpoint 而不是返回值。
    snapshot = await compiled.aget_state(config)

    assert snapshot.values["current_task"] == {}
    assert (
        snapshot.values["tasks"][0]["status"]["state"]
        == "TASK_STATE_COMPLETED"
    )


@pytest.mark.asyncio
async def test_llm_node_requires_current_task() -> None:
    with pytest.raises(RuntimeError, match="current_task"):
        await llm_node(
            {"messages": [], "current_task": {}},
            _runtime(),  # type: ignore[arg-type]
        )


# ============================================================
# builder
# ============================================================


def test_subagent_builder_is_discovered() -> None:
    builder = create_context_builder(ContextBuilderType.SUB_AGENT)

    assert isinstance(builder, SubAgentProcessorContextBuilder)
    assert len(builder.enabled_stages) == 13
    assert len(builder.required_stages) == 2
