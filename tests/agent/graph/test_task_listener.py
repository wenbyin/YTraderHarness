"""TaskSubscriber / TaskListener 的领取规则与拓扑。"""

from __future__ import annotations

import json

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from app.agent.graph.react import task_listener_node as listener_module
from app.agent.graph.react.graph import build_react_graph
from app.agent.graph.react.routing import route_after_agent
from app.agent.graph.react.task_listener_node import task_listener_node
from app.agent.plan.listener import (
    RedisTaskSubscriber,
    StateTaskSubscriber,
    complete_task,
    has_unfinished,
    select_ready,
)
from app.agent.tools.task_tool import CREATE_TEAM_TOOL_KEY

__all__: list[str] = []


def _task(
        tid: str,
        *,
        blocked_by: list[str] | None = None,
        state: str = "TASK_STATE_SUBMITTED",
        claim: dict | None = None,
) -> dict:
    return {
        "id": tid,
        "status": {"state": state},
        "metadata": {
            "plan_id": "p1",
            "blocked_by": blocked_by or [],
            "claim": claim or {},
        },
    }


def test_select_ready_picks_first_unblocked() -> None:
    tasks = [
        _task("t1"),
        _task("t2", blocked_by=["t1"]),
    ]

    ready = select_ready(tasks)

    assert ready is not None
    assert ready["id"] == "t1"


def test_select_ready_skips_claimed_and_blocked() -> None:
    tasks = [
        _task("t1", state="TASK_STATE_WORKING"),
        _task("t2", blocked_by=["t1"]),
        _task("t3"),
    ]

    ready = select_ready(tasks)

    assert ready is not None
    assert ready["id"] == "t3"


@pytest.mark.asyncio
async def test_state_subscriber_claims_and_marks_working() -> None:
    state = {
        "tasks": [
            _task("t1"),
            _task("t2", blocked_by=["t1"]),
        ]
    }

    claimed = await StateTaskSubscriber(state).claim_next(owner="run:main")

    assert claimed is not None
    assert claimed.source == "state"
    assert claimed.task["id"] == "t1"
    assert claimed.task["status"]["state"] == "TASK_STATE_WORKING"
    assert claimed.task["metadata"]["claim"]["owner"] == "run:main"

    assert claimed.tasks is not None
    by_id = {task["id"]: task for task in claimed.tasks}
    assert by_id["t1"]["status"]["state"] == "TASK_STATE_WORKING"
    assert by_id["t2"]["status"]["state"] == "TASK_STATE_SUBMITTED"


@pytest.mark.asyncio
async def test_state_subscriber_does_not_reclaim_working_task() -> None:
    state = {
        "tasks": [
            _task(
                "t1",
                state="TASK_STATE_WORKING",
                claim={"owner": "run:main", "token": "x"},
            )
        ]
    }

    assert await StateTaskSubscriber(state).claim_next(
        owner="run:main"
    ) is None


class _FakeRedis:
    """只实现 RedisTaskSubscriber 用到的那几个命令。"""

    def __init__(self, tasks: list[dict]) -> None:
        self._tasks = {task["id"]: task for task in tasks}

    async def smembers(self, key: str):
        return {tid.encode() for tid in self._tasks}

    async def hget(self, key: str, field: str):
        tid = key.split(":", 1)[1]
        task = self._tasks.get(tid)

        if task is None:
            return None

        return json.dumps(task, ensure_ascii=False).encode()

    async def get(self, key: str):
        return None

    async def eval(self, script, numkeys, key, claim_json, timestamp):
        tid = key.split(":", 1)[1]
        task = self._tasks.get(tid)

        if task is None:
            return None

        if task["status"]["state"] not in (
            "TASK_STATE_SUBMITTED", "submitted",
        ):
            return None

        task["status"]["state"] = "TASK_STATE_WORKING"
        task["status"]["timestamp"] = timestamp
        task["metadata"]["claim"] = json.loads(claim_json)
        task["metadata"]["claim"]["attempt"] = 1

        return json.dumps(task, ensure_ascii=False).encode()


@pytest.mark.asyncio
async def test_redis_subscriber_claims_via_cas() -> None:
    client = _FakeRedis([_task("t1")])

    claimed = await RedisTaskSubscriber(
        client, context_id="c1", plan_id="p1",
    ).claim_next(owner="run:worker")

    assert claimed is not None
    assert claimed.source == "redis"
    assert claimed.task["status"]["state"] == "TASK_STATE_WORKING"
    assert claimed.task["metadata"]["claim"]["owner"] == "run:worker"


def test_complete_task_marks_terminal_and_clears_claim() -> None:
    tasks = [_task("t1")]

    updated = complete_task(
        tasks,
        task_id_value="t1",
        summary="任务结果",
        context_id="c1",
    )

    assert updated[0]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert updated[0]["metadata"]["claim"] == {}
    assert updated[0]["metadata"]["result_summary"] == "任务结果"
    assert updated[0]["history"][-1]["role"] == "ROLE_AGENT"
    assert has_unfinished(updated) is False

    artifact = updated[0]["artifacts"][0]
    assert artifact["artifactId"] == "t1:result:1"
    assert artifact["name"] == "task_result"
    assert artifact["parts"][0]["text"] == "任务结果"


def test_artifact_keeps_full_result_while_history_keeps_summary() -> None:
    updated = complete_task(
        [_task("t1")],
        task_id_value="t1",
        summary="短摘要",
        result_text="完整成果正文" * 1000,
        context_id="c1",
    )

    task = updated[0]

    assert task["metadata"]["result_summary"] == "短摘要"
    assert task["artifacts"][0]["parts"][0]["text"] == (
        "完整成果正文" * 1000
    )
    assert "短摘要" in task["history"][-1]["parts"][0]["text"]
    assert "完整成果正文" not in task["history"][-1]["parts"][0]["text"]


class _Recorder:
    async def upsert_node_run(self, update) -> None:
        return None


def _runtime():
    return SimpleNamespace(
        context=SimpleNamespace(
            pipeline_run_id=uuid4(),
            session_id=None,
            recorder=_Recorder(),
            trace_id=None,
            caller_identity=None,
        )
    )


@pytest.mark.asyncio
async def test_listener_completes_current_and_claims_next(
        monkeypatch,
) -> None:
    updated_events: list[dict] = []
    cleared_events: list[dict] = []

    monkeypatch.setattr(
        listener_module,
        "publish_task_updated",
        lambda **kwargs: updated_events.append(kwargs),
    )
    monkeypatch.setattr(
        listener_module,
        "publish_plan_cleared",
        lambda **kwargs: cleared_events.append(kwargs),
    )

    working = _task(
        "t1",
        state="TASK_STATE_WORKING",
        claim={"owner": "run:main", "token": "x"},
    )

    state = {
        "messages": [AIMessage("任务 1 的结果")],
        "tasks": [working, _task("t2")],
        "current_task": working,
        "plan": {
            "plan_id": "p1",
            "revision": 1,
            "tasks": [
                {"id": "t1", "status": "pending"},
                {"id": "t2", "status": "pending"},
            ],
        },
        "conversation_id": "c1",
    }

    update = await task_listener_node(state, _runtime())  # type: ignore[arg-type]

    assert update["current_task"]["id"] == "t2"
    assert update["current_task"]["status"]["state"] == "TASK_STATE_WORKING"

    by_id = {task["id"]: task for task in update["tasks"]}
    assert by_id["t1"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert by_id["t1"]["metadata"]["claim"] == {}
    assert by_id["t2"]["status"]["state"] == "TASK_STATE_WORKING"

    plan_tasks = {task["id"]: task for task in update["plan"]["tasks"]}
    assert plan_tasks["t1"]["status"] == "done"

    assert [event["task"]["id"] for event in updated_events] == [
        "t1", "t2",
    ]
    assert updated_events[0]["task"]["status"]["state"] == (
        "TASK_STATE_COMPLETED"
    )
    assert updated_events[0]["task"]["artifacts"][0]["parts"][0]["text"] == (
        "任务 1 的结果"
    )
    assert updated_events[1]["current_task_id"] == "t2"
    assert cleared_events == []

    instruction = update["messages"][0]
    assert isinstance(instruction, HumanMessage)
    assert instruction.id == "task:t2:instruction"
    assert instruction.additional_kwargs["kind"] == "task_instruction"


@pytest.mark.asyncio
async def test_listener_starts_synthesis_when_all_done(
        monkeypatch,
) -> None:
    updated_events: list[dict] = []
    cleared_events: list[dict] = []

    monkeypatch.setattr(
        listener_module,
        "publish_task_updated",
        lambda **kwargs: updated_events.append(kwargs),
    )
    monkeypatch.setattr(
        listener_module,
        "publish_plan_cleared",
        lambda **kwargs: cleared_events.append(kwargs),
    )

    working = _task(
        "t1",
        state="TASK_STATE_WORKING",
        claim={"owner": "run:main", "token": "x"},
    )

    state = {
        "messages": [AIMessage("任务 1 的结果")],
        "tasks": [working],
        "current_task": working,
        "plan": {
            "plan_id": "p1",
            "revision": 1,
            "tasks": [{"id": "t1", "status": "pending"}],
        },
        "conversation_id": "c1",
        "context_policy": {
            "tool_keys": ["agent.create_team"],
            "include_capabilities": True,
        },
    }

    update = await task_listener_node(state, _runtime())  # type: ignore[arg-type]

    assert update["current_task"] == {}
    # 结果保留到汇总回合，不在这里清空。
    assert update["tasks"][0]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert update["plan"]["tasks"][0]["status"] == "done"
    assert update["hist_synthesis_keys"] == ["p1:1"]
    assert update["synthesis_pending"] is True
    assert update["context_policy"]["tool_keys"] == []
    assert update["context_policy"]["include_capabilities"] is False

    assert [event["task"]["status"]["state"] for event in updated_events] == [
        "TASK_STATE_COMPLETED",
    ]
    assert updated_events[0]["task"]["artifacts"][0]["parts"][0]["text"] == (
        "任务 1 的结果"
    )
    # 汇总还没完成，计划面板不能先关。
    assert cleared_events == []


@pytest.mark.asyncio
async def test_listener_clears_tasks_after_synthesis_key_seen(
        monkeypatch,
) -> None:
    cleared_events: list[dict] = []

    monkeypatch.setattr(
        listener_module,
        "publish_plan_cleared",
        lambda **kwargs: cleared_events.append(kwargs),
    )

    completed = _task("t1", state="TASK_STATE_COMPLETED")

    state = {
        "messages": [AIMessage("汇总完成")],
        "tasks": [completed],
        "current_task": {},
        "plan": {
            "plan_id": "p1",
            "revision": 1,
            "tasks": [{"id": "t1", "status": "done"}],
        },
        "conversation_id": "c1",
        "hist_synthesis_keys": ["p1:1"],
        "synthesis_pending": False,
    }

    update = await task_listener_node(state, _runtime())  # type: ignore[arg-type]

    assert update["tasks"] == []
    assert update["synthesis_pending"] is False
    assert update["hist_synthesis_keys"] == ["p1:1"]
    assert cleared_events[0]["reason"] == "completed"


@pytest.mark.asyncio
async def test_listener_keeps_results_while_synthesis_pending(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        listener_module, "publish_plan_cleared", lambda **kwargs: None
    )

    completed = _task("t1", state="TASK_STATE_COMPLETED")

    state = {
        "messages": [AIMessage("继续汇总")],
        "tasks": [completed],
        "current_task": {},
        "plan": {
            "plan_id": "p1",
            "revision": 1,
            "tasks": [{"id": "t1", "status": "done"}],
        },
        "conversation_id": "c1",
        "hist_synthesis_keys": ["p1:1"],
        "synthesis_pending": True,
    }

    update = await task_listener_node(state, _runtime())  # type: ignore[arg-type]

    assert update["tasks"][0]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert update["synthesis_pending"] is True


@pytest.mark.asyncio
async def test_agent_node_clears_synthesis_state_after_final_reply(
        monkeypatch,
) -> None:
    import app.agent.graph.react.agent_node as agent_node_module

    events: list[dict] = []

    async def fake_run(state, runtime, **kwargs):
        return {"messages": [AIMessage("汇总完成", id="a1")]}

    monkeypatch.setattr(agent_node_module, "run_agent_turn", fake_run)
    monkeypatch.setattr(
        agent_node_module,
        "publish_plan_cleared",
        lambda **kwargs: events.append(kwargs),
    )

    state = {
        "messages": [HumanMessage("原始请求")],
        "tasks": [_task("t1", state="TASK_STATE_COMPLETED")],
        "current_task": {},
        "synthesis_pending": True,
        "plan": {
            "plan_id": "p1",
            "revision": 1,
            "tasks": [{"id": "t1", "status": "done"}],
        },
    }

    update = await agent_node_module.agent_node(  # type: ignore[arg-type]
        state,
        SimpleNamespace(context=SimpleNamespace(builder=object())),
    )

    assert update["tasks"] == []
    assert update["current_task"] == {}
    assert update["synthesis_pending"] is False
    assert events[0]["reason"] == "completed"


def test_route_after_agent_goes_to_listener_while_tasks_remain() -> None:
    state = {
        "messages": [AIMessage("任务结果")],
        "tasks": [_task("t1")],
        "current_task": {},
    }

    assert route_after_agent(state, None) == "task_listener"  # type: ignore[arg-type]


def test_route_after_agent_ends_without_tasks_or_current_task() -> None:
    state = {
        "messages": [AIMessage("答复")],
        "tasks": [],
        "current_task": {},
    }

    assert route_after_agent(state, None) == END  # type: ignore[arg-type]


def test_graph_has_planner_listener_agent_path() -> None:
    compiled = build_react_graph().compile()
    edges = {
        (edge.source, edge.target)
        for edge in compiled.get_graph().edges
    }

    assert ("planner", "task_listener") in edges
    assert ("task_listener", "agent") in edges


# ============================================================
# claim 前的并行检测 / team decision
# ============================================================


def _plan() -> dict:
    return {
        "plan_id": "p1",
        "revision": 1,
        "goal": "并行任务",
        "tasks": [
            {"id": "t1", "status": "pending"},
            {"id": "t2", "status": "pending"},
            {"id": "t3", "status": "pending"},
        ],
    }


def _parallel_state(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content="做题", id="u1")],
        "tasks": [_task("t1"), _task("t2"), _task("t3")],
        "current_task": {},
        "plan": _plan(),
        "conversation_id": "c1",
        "context_policy": {"tool_keys": [CREATE_TEAM_TOOL_KEY]},
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_listener_defers_claim_when_parallel_ready(
        monkeypatch,
) -> None:
    events: list = []

    monkeypatch.setattr(
        listener_module,
        "publish_task_updated",
        lambda **kwargs: events.append(("task", kwargs)),
    )
    monkeypatch.setattr(
        listener_module,
        "publish_plan_cleared",
        lambda **kwargs: events.append(("cleared", kwargs)),
    )

    update = await task_listener_node(
        _parallel_state(), _runtime(),  # type: ignore[arg-type]
    )

    assert update["team_decision"] == "pending"
    assert update["current_task"] == {}
    assert "tasks" not in update
    # 决策提示由 ContextBuilder 的 TASK_RESOLVER 注入，listener 不写消息。
    assert "messages" not in update

    # 没有 claim，也没有任务事件。
    assert events == []


@pytest.mark.asyncio
async def test_listener_falls_back_to_claim_after_pending(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        listener_module, "publish_task_updated", lambda **kwargs: None,
    )
    monkeypatch.setattr(
        listener_module, "publish_plan_cleared", lambda **kwargs: None,
    )

    update = await task_listener_node(
        _parallel_state(team_decision="pending"),  # type: ignore[arg-type]
        _runtime(),  # type: ignore[arg-type]
    )

    assert update["team_decision"] == "declined"
    assert update["current_task"]["id"] == "t1"
    assert update["current_task"]["status"]["state"] == "TASK_STATE_WORKING"


@pytest.mark.asyncio
async def test_listener_claims_when_only_one_ready(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        listener_module, "publish_task_updated", lambda **kwargs: None,
    )
    monkeypatch.setattr(
        listener_module, "publish_plan_cleared", lambda **kwargs: None,
    )

    update = await task_listener_node(
        _parallel_state(
            tasks=[
                _task("t1"),
                _task("t2", blocked_by=["t1"]),
            ],
        ),  # type: ignore[arg-type]
        _runtime(),  # type: ignore[arg-type]
    )

    assert "team_decision" not in update
    assert update["current_task"]["id"] == "t1"


@pytest.mark.asyncio
async def test_listener_claims_when_create_team_not_exposed(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        listener_module, "publish_task_updated", lambda **kwargs: None,
    )
    monkeypatch.setattr(
        listener_module, "publish_plan_cleared", lambda **kwargs: None,
    )

    update = await task_listener_node(
        _parallel_state(
            context_policy={"tool_keys": ["agent.enter_plan_mode"]},
        ),  # type: ignore[arg-type]
        _runtime(),  # type: ignore[arg-type]
    )

    assert "team_decision" not in update
    assert update["current_task"]["id"] == "t1"


@pytest.mark.asyncio
async def test_listener_completes_then_defers_when_parallel_ready(
        monkeypatch,
) -> None:
    events: list = []

    monkeypatch.setattr(
        listener_module,
        "publish_task_updated",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        listener_module, "publish_plan_cleared", lambda **kwargs: None,
    )

    working = _task(
        "t1",
        state="TASK_STATE_WORKING",
        claim={"owner": "run:main", "token": "x"},
    )

    update = await task_listener_node(
        {
            "messages": [AIMessage("t1 的结果")],
            "tasks": [working, _task("t2"), _task("t3")],
            "current_task": working,
            "plan": _plan(),
            "conversation_id": "c1",
            "context_policy": {"tool_keys": [CREATE_TEAM_TOOL_KEY]},
        },  # type: ignore[arg-type]
        _runtime(),  # type: ignore[arg-type]
    )

    assert update["team_decision"] == "pending"
    assert update["current_task"] == {}

    by_id = {task["id"]: task for task in update["tasks"]}
    assert by_id["t1"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert by_id["t2"]["status"]["state"] == "TASK_STATE_SUBMITTED"

    # 完成事件不能因为「这一轮不 claim」而丢掉。
    assert [event["task"]["id"] for event in events] == ["t1"]
    assert events[0]["current_task_id"] is None
