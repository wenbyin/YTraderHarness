"""subagent_dispatch / subagent_worker：预 claim、Send 扇出、child thread。"""

from __future__ import annotations

import asyncio
import copy
import inspect
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph.subagent import nodes as subgraph_nodes
from app.agent.graph.subagent.graph import build_subagent_graph

from app.agent.graph.react import subagent_dispatch_node as dispatch_module
from app.agent.graph.react.catalog import (
    SUBAGENT_WORKER_NODE,
    TASK_LISTENER_NODE,
)
from app.agent.graph.react.subagent_dispatch_node import (
    subagent_dispatch_node,
)
from app.agent.plan.listener import (
    ClaimedTask,
    StateTaskSubscriber,
)

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
        },
    }


class _FakeBoard:
    """内存任务板：claim + snapshot + complete。"""

    def __init__(self, tasks: list[dict]) -> None:
        self.tasks = {task["id"]: copy.deepcopy(task) for task in tasks}

    def ordered(self) -> list[dict]:
        return [copy.deepcopy(self.tasks[key]) for key in sorted(self.tasks)]

    async def complete(self, *, task_id: str, task: dict) -> bool:
        if task_id not in self.tasks:
            return False

        self.tasks[task_id] = copy.deepcopy(dict(task))
        return True


class _FakeSubscriber:
    """Redis 形状的最小实现（dispatch 只要求不是 StateTaskSubscriber）。"""

    source = "redis"

    def __init__(self, board: _FakeBoard) -> None:
        self.board = board

    async def claim_next(self, *, owner: str) -> ClaimedTask | None:
        for task in self.board.ordered():
            if task["status"]["state"] != "TASK_STATE_SUBMITTED":
                continue

            if task["metadata"].get("claim"):
                continue

            task["status"]["state"] = "TASK_STATE_WORKING"
            task["metadata"]["claim"] = {
                "owner": owner,
                "token": "tok",
                "attempt": 1,
            }
            self.board.tasks[task["id"]] = task

            return ClaimedTask(
                task=copy.deepcopy(task),
                tasks=self.board.ordered(),
                source="redis",
            )

        return None

    async def snapshot(self) -> list[dict]:
        return self.board.ordered()


class _Recorder:
    async def upsert_node_run(self, update) -> None:
        return None


def _context(*, board=None, session_id=None):
    return SimpleNamespace(
        pipeline_run_id=uuid4(),
        session_id=session_id or uuid4(),
        recorder=_Recorder(),
        trace_id=None,
        caller_identity=None,
        task_board=board,
        # llm_node 只做 None 守卫；真正的模型调用在端到端用例里被替换。
        subagent_builder=object(),
    )


def _runtime(context):
    return SimpleNamespace(
        context=context,
        execution_info=SimpleNamespace(
            task_id="task",
            node_attempt=1,
            thread_id="parent-thread",
            run_id="run-1",
        ),
    )


def _state(*, tasks: list[dict], workers: int = 2) -> dict:
    return {
        "tasks": tasks,
        "current_task": {},
        "plan": {"plan_id": "p1", "revision": 1},
        "conversation_id": "c1",
        "user_id": "u1",
        "subagent_mode": "on",
        "subagent_workers": workers,
    }


def _patch_subscriber(monkeypatch, subscriber) -> None:
    monkeypatch.setattr(
        dispatch_module,
        "build_task_subscriber",
        lambda *args, **kwargs: subscriber,
    )


@pytest.fixture
def emitted(monkeypatch):
    """收集 dispatch 发出的子代理事件（生产端是 custom stream）。"""

    captured = {"created": [], "claimed": [], "updated": [], "completed": []}

    monkeypatch.setattr(
        dispatch_module,
        "publish_subagent_created",
        lambda **kwargs: captured["created"].append(kwargs),
    )
    monkeypatch.setattr(
        dispatch_module,
        "publish_subagent_claimed",
        lambda **kwargs: captured["claimed"].append(kwargs),
    )
    monkeypatch.setattr(
        dispatch_module,
        "publish_subagent_task_updated",
        lambda **kwargs: captured["updated"].append(kwargs),
    )
    monkeypatch.setattr(
        dispatch_module,
        "publish_subagent_completed",
        lambda **kwargs: captured["completed"].append(kwargs),
    )

    return captured


# ============================================================
# dispatch
# ============================================================


@pytest.mark.asyncio
async def test_dispatch_preclaims_and_fans_out(
        monkeypatch, emitted,
) -> None:
    board = _FakeBoard([_task("t1"), _task("t2"), _task("t3")])
    _patch_subscriber(monkeypatch, _FakeSubscriber(board))

    result = await subagent_dispatch_node(
        _state(tasks=board.ordered(), workers=2),
        _runtime(_context(board=board)),  # type: ignore[arg-type]
    )

    assert isinstance(result, Command)

    sends = result.goto

    assert isinstance(sends, list)
    assert len(sends) == 2
    assert [send.node for send in sends] == [SUBAGENT_WORKER_NODE] * 2
    assert [send.arg["pending_task"]["id"] for send in sends] == [
        "t1", "t2",
    ]
    assert all(send.arg["single_task"] is True for send in sends)
    assert [send.arg["worker_id"].split(":")[-1] for send in sends] == [
        "0", "1",
    ]

    by_id = {task["id"]: task for task in result.update["tasks"]}

    assert by_id["t1"]["status"]["state"] == "TASK_STATE_WORKING"
    assert by_id["t2"]["status"]["state"] == "TASK_STATE_WORKING"
    assert by_id["t3"]["status"]["state"] == "TASK_STATE_SUBMITTED"
    assert result.update["subagent_mode"] == "on"
    assert result.update["subagent_workers"] == 2

    # 每个 worker 一条 created（开泳道）+ 一条 claimed（放任务）。
    assert [
        item["worker_id"].split(":")[-1] for item in emitted["created"]
    ] == ["0", "1"]
    assert [
        item["task_id"] for item in emitted["created"]
    ] == ["t1", "t2"]
    assert [
        item["task"]["id"] for item in emitted["claimed"]
    ] == ["t1", "t2"]
    assert [
        item["worker_id"].split(":")[-1] for item in emitted["claimed"]
    ] == ["0", "1"]


@pytest.mark.asyncio
async def test_dispatch_next_wave_claims_remaining(monkeypatch) -> None:
    board = _FakeBoard([
        _task("t1", state="TASK_STATE_COMPLETED"),
        _task("t2", state="TASK_STATE_COMPLETED"),
        _task("t3"),
    ])
    _patch_subscriber(monkeypatch, _FakeSubscriber(board))

    result = await subagent_dispatch_node(
        _state(tasks=board.ordered(), workers=2),
        _runtime(_context(board=board)),  # type: ignore[arg-type]
    )

    sends = result.goto

    assert isinstance(sends, list)
    assert [send.arg["pending_task"]["id"] for send in sends] == ["t3"]


@pytest.mark.asyncio
async def test_dispatch_routes_to_listener_when_nothing_ready(
        monkeypatch,
) -> None:
    board = _FakeBoard([_task("t1", state="TASK_STATE_COMPLETED")])
    _patch_subscriber(monkeypatch, _FakeSubscriber(board))

    result = await subagent_dispatch_node(
        _state(tasks=board.ordered()),
        _runtime(_context(board=board)),  # type: ignore[arg-type]
    )

    assert isinstance(result, Command)
    assert result.goto == TASK_LISTENER_NODE
    assert result.update["current_task"] == {}


@pytest.mark.asyncio
async def test_dispatch_without_board_falls_back_to_sequential(
        monkeypatch,
) -> None:
    state = _state(tasks=[_task("t1"), _task("t2")])

    _patch_subscriber(
        monkeypatch,
        StateTaskSubscriber({"tasks": state["tasks"]}),
    )

    result = await subagent_dispatch_node(
        state,
        _runtime(_context()),  # type: ignore[arg-type]
    )

    assert isinstance(result, Command)
    assert result.goto == TASK_LISTENER_NODE
    assert result.update["subagent_mode"] == "off"
    assert result.update["subagent_workers"] == 0


# ============================================================
# 父图接线 + 端到端
# ============================================================


def test_parent_graph_wires_worker_as_subgraph() -> None:
    """
    worker 必须是**子图节点**，不是嵌套 ainvoke 的节点函数。

    interrupt 冒泡、resume、子图 custom 事件进入父运行流，全都依赖
    「子图由父 Pregel 管理」这一点。`SubAgentOutput`（空 output_schema）
    负责让它的状态不合并回父 state。
    """

    from app.agent.graph.react.catalog import (
        SUBAGENT_DISPATCH_NODE,
        SUBAGENT_WORKER_NODE,
    )
    from app.agent.graph.react.graph import build_react_graph

    graph = build_react_graph()

    assert (SUBAGENT_WORKER_NODE, SUBAGENT_DISPATCH_NODE) in graph.edges

    # 节点值是子图（不是普通 async 节点函数）。
    assert not inspect.iscoroutinefunction(
        graph.nodes[SUBAGENT_WORKER_NODE]
    )


@pytest.mark.asyncio
async def test_subgraph_node_does_not_merge_worker_messages(
        monkeypatch,
) -> None:
    """
    合并成节点之后，子图的 messages 不能串进父对话。

    `SubAgentOutput` 是空 schema —— 子图内部照样有完整 transcript，
    但作为节点返回给父图的是 `{}`。
    """

    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    from app.agent.graph.react.state import ReactState

    board = _FakeBoard([_task("t1"), _task("t2")])

    async def fake_llm(state, runtime, **kwargs):
        return {
            "messages": [
                AIMessage(
                    f"worker {state['task_id']} 内部答复",
                    id=f"worker-answer-{state['task_id']}",
                ),
            ]
        }

    monkeypatch.setattr(subgraph_nodes, "run_agent_turn", fake_llm)

    def _payload(task_id: str, index: int) -> dict:
        task = _task(
            task_id,
            state="TASK_STATE_WORKING",
            claim={
                "owner": f"run:worker:{index}",
                "token": "x",
                "attempt": 1,
            },
        )

        return {
            "messages": [],
            "pending_task": task,
            "tasks": board.ordered(),
            "plan": {"plan_id": "p1", "revision": 1},
            "conversation_id": "c1",
            "user_id": "u1",
            "worker_id": f"run:worker:{index}",
            "single_task": True,
        }

    async def fan(state):
        return Command(goto=[
            Send("subagent_worker", _payload("t1", 0)),
            Send("subagent_worker", _payload("t2", 1)),
        ])

    async def tally(state):
        return {}

    parent: StateGraph = StateGraph(ReactState)
    parent.add_node("fan", fan)
    parent.add_node(
        "subagent_worker",
        build_subagent_graph().compile(),
    )
    parent.add_node("tally", tally)
    parent.add_edge(START, "fan")
    parent.add_edge("subagent_worker", "tally")
    parent.add_edge("tally", END)

    compiled = parent.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "parent"}}

    result = await compiled.ainvoke(
        {"messages": [HumanMessage("用户问题", id="u1")]},
        config,
        context=_context(board=board),
    )

    assert [message.content for message in result["messages"]] == [
        "用户问题"
    ]

    # 两个并发 worker 的产出都在任务板上，不在父 messages 里。
    assert [
        task["status"]["state"] for task in board.ordered()
    ] == ["TASK_STATE_COMPLETED", "TASK_STATE_COMPLETED"]


@pytest.mark.asyncio
async def test_end_to_end_dispatch_worker_and_subgraph(
        monkeypatch, emitted,
) -> None:
    """
    真链路：dispatch 预 claim -> 并发跑子图 -> 写回任务板
            -> 下一轮 dispatch 发现没有 ready task -> 回 listener。

    只把模型调用换成桩；三节点拓扑、任务收养 / 完成、任务板写回都走
    真实实现。子图按 Send payload 直接 invoke（等价于父图的子图节点）。
    """

    board = _FakeBoard([_task("t1"), _task("t2")])
    _patch_subscriber(monkeypatch, _FakeSubscriber(board))

    async def fake_llm(state, runtime, **kwargs):
        return {
            "messages": [
                AIMessage(
                    f"完成 {state['task_id']}",
                    id=f"answer-{state['task_id']}",
                )
            ]
        }

    monkeypatch.setattr(subgraph_nodes, "run_agent_turn", fake_llm)

    subgraph = build_subagent_graph().compile(
        checkpointer=InMemorySaver()
    )
    context = _context(board=board)
    runtime = _runtime(context)

    first = await subagent_dispatch_node(
        _state(tasks=board.ordered(), workers=2),
        runtime,  # type: ignore[arg-type]
    )

    sends = first.goto

    assert isinstance(sends, list)
    assert len(sends) == 2

    await asyncio.gather(*[
        subgraph.ainvoke(
            send.arg,
            {"configurable": {"thread_id": f"worker-{index}"}},
            context=context,
        )
        for index, send in enumerate(sends)
    ])

    done = board.ordered()

    assert [task["status"]["state"] for task in done] == [
        "TASK_STATE_COMPLETED", "TASK_STATE_COMPLETED",
    ]
    assert [
        task["artifacts"][0]["parts"][0]["text"] for task in done
    ] == ["完成 t1", "完成 t2"]

    # 第二波：父图 state 里仍是第一波 claim 后的 WORKING 快照（真实
    # 情况就是 Command.update 写回去的那一份），任务板上已经是 COMPLETED。
    second = await subagent_dispatch_node(
        _state(tasks=first.update["tasks"], workers=2),
        runtime,  # type: ignore[arg-type]
    )

    assert isinstance(second, Command)
    assert second.goto == TASK_LISTENER_NODE

    # 收波时补发 worker 在子图里完成的那两步。
    assert [
        item["task"]["id"] for item in emitted["updated"]
    ] == ["t1", "t2"]
    assert [
        item["previous_state"] for item in emitted["updated"]
    ] == ["TASK_STATE_WORKING", "TASK_STATE_WORKING"]
    assert [
        item["task"]["id"] for item in emitted["completed"]
    ] == ["t1", "t2"]
    assert [
        item["worker_id"].split(":")[-1]
        for item in emitted["completed"]
    ] == ["0", "1"]
