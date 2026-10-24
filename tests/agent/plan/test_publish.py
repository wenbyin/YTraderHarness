"""任务板发布：键结构 / 幂等 / 降级，以及 planner 的下一跳开关。"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from app.agent.graph.react.graph import build_react_graph
from app.agent.graph.react.routing import (
    STOP_AFTER_PLANNER_ENV,
    STOP_AFTER_PUBLISH_ENV,
    _tool_sends,
    route_after_planner,
)
from app.agent.plan.publish import (
    DEFAULT_TTL_SECONDS,
    RedisTaskBoard,
    publish_plan,
    task_index_key,
    task_key,
)

__all__: list[str] = []


def _task(task_id: str) -> dict:
    return {
        "id": task_id,
        "contextId": "c1",
        "status": {"state": "TASK_STATE_SUBMITTED"},
        "metadata": {
            "plan_id": "p1",
            "blocked_by": [],
            "claim": {},
        },
    }


class _FakeRedis:
    """只实现发布用到的命令。"""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.expires: dict[str, int] = {}

    async def hexists(self, key: str, field: str) -> bool:
        return field in self.hashes.get(key, {})

    async def hset(self, key: str, mapping: dict | None = None, **_: object) -> int:
        self.hashes.setdefault(key, {}).update(
            {str(k): str(v) for k, v in (mapping or {}).items()}
        )
        return 1

    async def sadd(self, key: str, *members: str) -> int:
        bucket = self.sets.setdefault(key, set())
        before = len(bucket)
        bucket.update(str(member) for member in members)
        return len(bucket) - before

    async def expire(self, key: str, ttl: int) -> bool:
        self.expires[key] = int(ttl)
        return True


@pytest.mark.asyncio
async def test_publish_writes_task_rows_indexes_and_ready_marker() -> None:
    client = _FakeRedis()
    board = RedisTaskBoard(client)

    result = await board.publish(
        context_id="c1",
        plan_id="p1",
        revision=1,
        tasks=[_task("p1:t1"), _task("p1:t2")],
    )

    assert result.published == 2
    assert result.skipped == 0
    assert result.degraded is False

    row = client.hashes[task_key("p1:t1")]

    assert json.loads(row["task"])["id"] == "p1:t1"
    assert row["state"] == "TASK_STATE_SUBMITTED"
    assert row["context_id"] == "c1"
    assert row["plan_id"] == "p1"
    assert row["revision"] == "1"
    assert json.loads(row["claim"]) == {}
    assert row["updated_at"].endswith("Z")

    assert client.sets[task_index_key("c1")] == {"p1:t1", "p1:t2"}
    assert client.sets["agent_plan_tasks:p1:1"] == {"p1:t1", "p1:t2"}

    plan = client.hashes["agent_plan:p1:1"]

    assert plan["ready"] == "1"
    assert plan["task_count"] == "2"
    assert plan["context_id"] == "c1"

    assert set(client.expires) == {
        task_key("p1:t1"),
        task_key("p1:t2"),
        task_index_key("c1"),
        "agent_plan_tasks:p1:1",
        "agent_plan:p1:1",
    }
    assert all(ttl == DEFAULT_TTL_SECONDS for ttl in client.expires.values())


@pytest.mark.asyncio
async def test_publish_is_idempotent_and_keeps_runtime_state() -> None:
    client = _FakeRedis()
    board = RedisTaskBoard(client)

    await board.publish(
        context_id="c1",
        plan_id="p1",
        revision=1,
        tasks=[_task("p1:t1"), _task("p1:t2")],
    )

    # 模拟 worker 已经 claim 了 t1。
    client.hashes[task_key("p1:t1")]["state"] = "TASK_STATE_WORKING"

    again = await board.publish(
        context_id="c1",
        plan_id="p1",
        revision=1,
        tasks=[_task("p1:t1"), _task("p1:t2")],
    )

    assert again.published == 0
    assert again.skipped == 2
    assert client.hashes[task_key("p1:t1")]["state"] == "TASK_STATE_WORKING"


@pytest.mark.asyncio
async def test_publish_degrades_when_board_is_missing() -> None:
    result = await publish_plan(
        board=None,
        context_id="c1",
        plan_id="p1",
        revision=1,
        tasks=[_task("p1:t1")],
    )

    assert result.published == 0
    assert result.degraded is True
    assert "未装配" in result.reason


def test_route_after_planner_continues_by_default(monkeypatch) -> None:
    monkeypatch.delenv(STOP_AFTER_PLANNER_ENV, raising=False)
    monkeypatch.delenv(STOP_AFTER_PUBLISH_ENV, raising=False)

    assert route_after_planner({}) == "task_listener"  # type: ignore[arg-type]


def test_route_after_planner_can_stop(monkeypatch) -> None:
    monkeypatch.delenv(STOP_AFTER_PUBLISH_ENV, raising=False)
    monkeypatch.setenv(STOP_AFTER_PLANNER_ENV, "1")

    assert route_after_planner({}) == END  # type: ignore[arg-type]


def test_route_after_planner_legacy_stop_env(monkeypatch) -> None:
    monkeypatch.delenv(STOP_AFTER_PLANNER_ENV, raising=False)
    monkeypatch.setenv(STOP_AFTER_PUBLISH_ENV, "1")

    assert route_after_planner({}) == END  # type: ignore[arg-type]


def test_tool_sends_carry_tasks_in_payload() -> None:
    message = AIMessage(
        content="",
        id="ai-1",
        tool_calls=[{
            "name": "lookup",
            "args": {"symbol": "AAPL"},
            "id": "call-1",
            "type": "tool_call",
        }],
    )

    state = {
        "tasks": [{"id": "p1:t1"}],
        "context_policy": {"tool_keys": ["lookup"]},
    }

    sends = _tool_sends(state, message, plan_origin="tool")  # type: ignore[arg-type]

    assert len(sends) == 1
    assert sends[0].arg["tasks"] == [{"id": "p1:t1"}]


def test_graph_has_planner_end_path() -> None:
    compiled = build_react_graph().compile()
    edges = {
        (edge.source, edge.target)
        for edge in compiled.get_graph().edges
    }

    assert ("planner", "__end__") in edges
    assert ("planner", "task_listener") in edges
