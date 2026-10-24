"""Planner 子图：拓扑、隔离、路由与 Command 守门。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command, Send

from app.agent.graph.planner import graph as graph_module
from app.agent.graph.planner import nodes as planner_nodes
from app.agent.graph.planner.catalog import (
    PLANNER_LLM_NODE,
    TASK_GENERATOR_NODE,
    TOOL_NODE,
)
from app.agent.graph.planner.graph import build_planner_graph
from app.agent.graph.planner.nodes import _command_is_local
from app.agent.graph.planner.routing import route_after_planner_llm
from app.agent.graph.planner.state import PlannerInput, PlannerOutput

__all__: list[str] = []


# ============================================================
# 拓扑与 schema
# ============================================================


def test_graph_has_three_nodes_and_expected_edges() -> None:
    compiled = build_planner_graph().compile()

    nodes = set(compiled.get_graph().nodes)

    assert {PLANNER_LLM_NODE, TOOL_NODE, TASK_GENERATOR_NODE} <= nodes

    edges = {
        (edge.source, edge.target)
        for edge in compiled.get_graph().edges
    }

    assert (PLANNER_LLM_NODE, TOOL_NODE) in edges
    assert (PLANNER_LLM_NODE, TASK_GENERATOR_NODE) in edges
    assert (TASK_GENERATOR_NODE, END) in edges

    # planner_tool 的下一跳由 Command 决定；编译出的隐式 END 是
    # langgraph 给无出边节点补的，不是装配期声明的业务路。
    assert {
        (source, target)
        for source, target in edges
        if source == TOOL_NODE
    } == {(TOOL_NODE, END)}


def test_graph_module_registers_all_three_functions() -> None:
    assert set(graph_module.NODE_FUNCTIONS) == {
        PLANNER_LLM_NODE,
        TOOL_NODE,
        TASK_GENERATOR_NODE,
    }


def test_input_schema_does_not_accept_parent_context_policy() -> None:
    """父图 context_policy / current_task / tasks 不能灌进 planner。"""

    fields = set(PlannerInput.__annotations__)

    assert fields == {
        "messages",
        "conversation_id",
        "user_id",
        "summary",
        "plan",
    }
    assert "context_policy" not in fields
    assert "current_task" not in fields
    assert "tasks" not in fields
    assert "team_decision" not in fields


def test_output_schema_does_not_leak_messages() -> None:
    fields = set(PlannerOutput.__annotations__)

    assert fields == {
        "plan",
        "tasks",
        "current_task",
        "team_decision",
    }
    assert "messages" not in fields


# ============================================================
# routing
# ============================================================


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(context=None)


def test_route_goes_to_task_generator_without_tool_calls() -> None:
    state = {
        "messages": [AIMessage("探索完成", id="a1")],
        "context_policy": {"tool_keys": []},
    }

    assert route_after_planner_llm(state, _runtime()) == (  # type: ignore[arg-type]
        TASK_GENERATOR_NODE
    )


def test_route_sends_one_branch_per_tool_call() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "skill_list_skills", "args": {}, "id": "c1"},
            {"name": "capability_list_capabilities", "args": {}, "id": "c2"},
        ],
        id="m1",
    )

    state = {
        "messages": [message],
        "context_policy": {"tool_keys": ["skill.list_skills"]},
    }

    destinations = route_after_planner_llm(  # type: ignore[arg-type]
        state, _runtime()
    )

    assert isinstance(destinations, list)
    assert len(destinations) == 2
    assert all(isinstance(item, Send) for item in destinations)

    first = destinations[0]
    assert first.node == TOOL_NODE
    assert first.arg["return_to"] == PLANNER_LLM_NODE
    assert first.arg["is_controller"] is True
    assert first.arg["tool_keys"] == ["skill.list_skills"]
    assert first.arg["call_scope"] == "m1:c1"
    assert destinations[1].arg["is_controller"] is False


# ============================================================
# Command 守门
# ============================================================


def test_local_commands_are_allowed() -> None:
    assert _command_is_local(Command(goto=PLANNER_LLM_NODE))
    assert _command_is_local(
        Command(goto=[PLANNER_LLM_NODE, TASK_GENERATOR_NODE])
    )


def test_parent_and_unknown_commands_are_rejected() -> None:
    assert not _command_is_local(Command(goto="subagent_dispatch"))
    assert not _command_is_local(Command(goto="agent"))
    assert not _command_is_local(
        Command(graph=Command.PARENT, goto="agent")
    )


# ============================================================
# 子图运行与输出隔离
# ============================================================


@pytest.mark.asyncio
async def test_subgraph_runs_and_only_outputs_plan_channels(
        monkeypatch,
) -> None:
    async def fake_planner(state, runtime):
        return {"messages": [AIMessage("探索完成", id="a1")]}

    async def fake_generator(state, runtime):
        return {
            "plan": {"plan_id": "p1", "revision": 1},
            "tasks": [{"id": "p1:t1"}],
            "current_task": {},
            "team_decision": "",
        }

    monkeypatch.setitem(
        graph_module.NODE_FUNCTIONS, PLANNER_LLM_NODE, fake_planner
    )
    monkeypatch.setitem(
        graph_module.NODE_FUNCTIONS,
        TASK_GENERATOR_NODE,
        fake_generator,
    )

    compiled = build_planner_graph().compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "planner-t1"}}

    result = await compiled.ainvoke(
        {
            "messages": [HumanMessage("帮我规划")],
            "conversation_id": "c1",
            "plan": {},
        },
        config,
        context=SimpleNamespace(planner_builder=object()),
    )

    assert result["plan"]["plan_id"] == "p1"
    assert result["tasks"] == [{"id": "p1:t1"}]
    assert result["current_task"] == {}
    assert result["team_decision"] == ""
    assert "messages" not in result

    # 子图内部仍然有 transcript；它只是不合并回父图。
    snapshot = await compiled.aget_state(config)
    assert [
        str(message.content) for message in snapshot.values["messages"]
    ] == ["帮我规划", "探索完成"]


# ============================================================
# 真实节点 + 模型桩
# ============================================================


class _FakeRuntime:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def ainvoke(self, request, *, trace=None):
        return SimpleNamespace(
            message=AIMessage(
                content=json.dumps(self._payload, ensure_ascii=False)
            )
        )


class _FakeRegistry:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def get_runtime(self, **kwargs):
        return _FakeRuntime(self._payload)


class _FakePlannerBuilder:
    async def build(self, request):
        from app.agent.model import ModelRequest

        system = request.graph_state["context_policy"]["system_prompt"]

        return ModelRequest(
            messages=(
                SystemMessage(content=system),
                *request.messages,
            ),
            disable_thinking=True,
        )


@pytest.mark.asyncio
async def test_real_nodes_emit_plan_and_keep_policy_isolated(
        monkeypatch,
) -> None:
    captured_policies: list[dict] = []

    async def fake_run_agent_turn(state, runtime, **kwargs):
        captured_policies.append(dict(state["context_policy"]))
        return {"messages": [AIMessage("探索完成", id="a1")]}

    async def fake_tool_keys():
        return ["skill.list_skills", "capability.list_capabilities"], []

    async def skills():
        return (
            [{
                "skill_key": "research",
                "name": "研究",
                "description": "研究任务",
            }],
            [],
        )

    async def caps():
        return [], []

    events: list[dict] = []

    monkeypatch.setattr(planner_nodes, "run_agent_turn", fake_run_agent_turn)
    monkeypatch.setattr(planner_nodes, "_planner_tool_keys", fake_tool_keys)
    monkeypatch.setattr(planner_nodes, "_skill_catalog", skills)
    monkeypatch.setattr(planner_nodes, "_capability_catalog", caps)
    monkeypatch.setattr(
        planner_nodes, "node_trace", lambda runtime: (None, "span")
    )
    monkeypatch.setattr(
        planner_nodes,
        "publish_plan_published",
        lambda **kwargs: events.append(kwargs),
    )

    payload = {
        "goal": "测试目标",
        "tasks": [{
            "id": "t1",
            "message": {"description": "第一步", "params": {}},
            "metadata": {
                "skill_key": "research",
                "blocked_by": [],
                "capabilities": [],
            },
        }],
        "open_questions": [],
        "notes": [],
    }

    context = SimpleNamespace(
        planner_builder=_FakePlannerBuilder(),
        model_registry=_FakeRegistry(payload),
        pipeline_run_id=uuid4(),
        session_id=None,
        tool_source=None,
    )

    compiled = build_planner_graph().compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "planner-real-t1"}}

    result = await compiled.ainvoke(
        {
            "messages": [HumanMessage("帮我规划")],
            "conversation_id": "c1",
            "plan": {},
            # 父图的策略 / 任务通道：input_schema 不应把它们带进 planner；
            # 即使被带进来，planner_llm 也必须用自己重新算的策略覆盖。
            "context_policy": {
                "system_prompt": "主 agent 的职责提示词",
                "tool_keys": ["agent.create_team"],
            },
            "current_task": {"id": "old:t1"},
            "tasks": [{"id": "old:t1"}],
            "team_decision": "pending",
        },
        config,
        context=context,
    )

    assert captured_policies
    assert captured_policies[0]["system_prompt"] != "主 agent 的职责提示词"
    assert captured_policies[0]["tool_keys"] == [
        "skill.list_skills",
        "capability.list_capabilities",
    ]

    assert result["plan"]["goal"] == "测试目标"
    assert [task["id"].split(":")[1] for task in result["tasks"]] == ["t1"]
    assert result["current_task"] == {}
    assert result["team_decision"] == ""
    assert "messages" not in result

    # planner 自己的策略（含 tool_keys）留在子图 state，不覆盖父图。
    snapshot = await compiled.aget_state(config)
    policy = snapshot.values["context_policy"]

    assert policy["include_capabilities"] is True
    assert policy["tool_keys"] == [
        "skill.list_skills",
        "capability.list_capabilities",
    ]
    assert snapshot.values["planner_tools_resolved"] is True

    assert len(events) == 1
    assert events[0]["current_task_id"] is None
    assert events[0]["publish_id"] == (
        f"{result['plan']['plan_id']}:1"
    )


@pytest.mark.asyncio
async def test_parent_graph_merges_plan_but_not_planner_messages(
        monkeypatch,
) -> None:
    """父图拿到 plan/tasks；planner 的 transcript 不串进主对话。"""

    from app.agent.graph.react import graph as react_graph_module
    from app.agent.graph.react.catalog import (
        AGENT_BUILDER_NODE,
        AGENT_NODE,
        TASK_LISTENER_NODE,
    )

    async def fake_planner(state, runtime):
        return {"messages": [AIMessage("planner 内部探索", id="p-a1")]}

    async def fake_generator(state, runtime):
        return {
            "plan": {"plan_id": "p1", "revision": 1, "tasks": []},
            "tasks": [],
            "current_task": {},
            "team_decision": "",
        }

    async def fake_builder(state, runtime):
        return {"plan_status": "plan"}

    async def fake_listener(state, runtime):
        return {"tasks": [], "current_task": {}}

    async def fake_agent(state, runtime):
        return {"messages": [AIMessage("答复", id="a-final")]}

    monkeypatch.setitem(
        graph_module.NODE_FUNCTIONS, PLANNER_LLM_NODE, fake_planner
    )
    monkeypatch.setitem(
        graph_module.NODE_FUNCTIONS,
        TASK_GENERATOR_NODE,
        fake_generator,
    )
    monkeypatch.setitem(
        react_graph_module.NODE_FUNCTIONS, AGENT_BUILDER_NODE, fake_builder
    )
    monkeypatch.setitem(
        react_graph_module.NODE_FUNCTIONS,
        TASK_LISTENER_NODE,
        fake_listener,
    )
    monkeypatch.setitem(
        react_graph_module.NODE_FUNCTIONS, AGENT_NODE, fake_agent
    )

    compiled = react_graph_module.build_react_graph().compile(
        checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "parent-planner-t1"}}

    result = await compiled.ainvoke(
        {
            "messages": [HumanMessage("帮我规划")],
            "conversation_id": "c1",
            "plan": {},
            "tasks": [],
            "plan_status": "off",
        },
        config,
        context=SimpleNamespace(
            pipeline_run_id=uuid4(),
            session_id=None,
            planner_builder=object(),
            model_registry=None,
            tool_source=None,
        ),
    )

    assert result["plan"]["plan_id"] == "p1"
    assert [str(message.content) for message in result["messages"]] == [
        "帮我规划",
        "答复",
    ]
