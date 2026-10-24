"""Planner -> TaskDraft -> A2A Task 的纯转换与窄节点测试。"""

from __future__ import annotations

import json

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.graph.planner import nodes as planner_nodes
from app.agent.plan import (
    PlanDraft,
    PlanValidationError,
    normalize_payload,
    task_payload,
    task_to_dict,
    to_a2a_tasks,
    validate_plan,
)

__all__: list[str] = []


def _raw_plan() -> dict:
    return {
        "revision": 1,
        "goal": "测试目标",
        "tasks": [
            {
                "id": "t1",
                "message": {"description": "第一步", "params": {"n": "1"}},
                "metadata": {
                    "skill_key": "research",
                    "blocked_by": [],
                    "block_policy": "all_succeeded",
                    "capabilities": [],
                },
            },
            {
                "id": "t2",
                "message": {"description": "第二步", "params": {}},
                "metadata": {
                    "skill_key": "research",
                    "blocked_by": ["t1"],
                    "block_policy": "all_succeeded",
                    "capabilities": [],
                },
            },
        ],
        "open_questions": [],
        "notes": [],
    }


def _validated(raw: dict | None = None):
    plan = PlanDraft.model_validate(normalize_payload(raw or _raw_plan()))
    return plan, validate_plan(
        plan, known_skill_keys={"research"}, known_capabilities=None
    )


def test_topological_order_and_reverse_edges() -> None:
    _, validated = _validated()

    tasks = to_a2a_tasks(
        validated.tasks,
        context_id="session-1",
        plan_id="plan-1",
        revision=1,
    )

    assert [task.id for task in tasks] == ["plan-1:t1", "plan-1:t2"]

    t1 = task_to_dict(tasks[0])
    t2 = task_to_dict(tasks[1])

    assert t1["metadata"]["blocked_by"] == []
    assert t1["metadata"]["blocked"] == ["plan-1:t2"]
    assert t2["metadata"]["blocked_by"] == ["plan-1:t1"]
    assert t2["metadata"]["blocked"] == []


def test_message_is_preserved_as_data_part() -> None:
    _, validated = _validated()

    tasks = to_a2a_tasks(
        validated.tasks,
        context_id="session-1",
        plan_id="plan-1",
        revision=1,
    )

    payload = task_payload(tasks[0])

    assert payload["description"] == "第一步"
    assert payload["params"]["n"] == "1"


def test_cycle_is_rejected() -> None:
    raw = _raw_plan()
    raw["tasks"][0]["metadata"]["blocked_by"] = ["t2"]
    raw["tasks"][1]["metadata"]["blocked_by"] = ["t1"]

    plan = PlanDraft.model_validate(normalize_payload(raw))

    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_unknown_capability_is_dropped_with_warning() -> None:
    raw = _raw_plan()
    raw["tasks"][0]["metadata"]["capabilities"] = ["known", "unknown"]

    plan = PlanDraft.model_validate(normalize_payload(raw))
    validated = validate_plan(
        plan, known_capabilities={"known"}
    )

    assert validated.tasks[0].capabilities == ["known"]
    assert any("unknown" in note for note in validated.warnings)


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
    """只验证 planner 节点走 builder；真实窄流水线另有测试。"""

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
async def test_planner_plan_sets_tasks_without_current_task(
        monkeypatch,
) -> None:
    async def skills():
        return (
            [
                {
                    "skill_key": "research",
                    "name": "研究",
                    "description": "研究任务",
                }
            ],
            [],
        )

    async def caps():
        return [], []

    monkeypatch.setattr(planner_nodes, "_skill_catalog", skills)
    monkeypatch.setattr(planner_nodes, "_capability_catalog", caps)
    monkeypatch.setattr(
        planner_nodes, "node_trace", lambda runtime: (None, "span")
    )

    events: list[dict] = []
    monkeypatch.setattr(
        planner_nodes,
        "publish_plan_published",
        lambda **kwargs: events.append(kwargs),
    )

    context = SimpleNamespace(
        model_registry=_FakeRegistry(_raw_plan()),
        planner_builder=_FakePlannerBuilder(),
        pipeline_run_id=uuid4(),
        session_id=None,
    )
    runtime = SimpleNamespace(context=context)
    state = {
        "conversation_id": "session-1",
        "messages": [HumanMessage("帮我研究一下")],
    }

    update = await planner_nodes._generate(  # type: ignore[arg-type]
        state, runtime, context.planner_builder,
    )

    assert [item["id"].split(":")[1] for item in update["tasks"]] == [
        "t1", "t2",
    ]
    # 计划发布时一个任务都还没 claim：current 只能由 listener / dispatch
    # 在 claim 时写，否则前端会把排队中的 t1 当成正在执行。
    assert update["current_task"] == {}
    assert update["team_decision"] == ""
    assert update["plan"]["tasks"][0]["status"] == "pending"
    assert update["plan"]["revision"] == 1

    assert len(events) == 1
    assert events[0]["tasks"] == update["tasks"]
    assert events[0]["current_task_id"] is None
    assert events[0]["publish_id"] == (
        f"{update['plan']['plan_id']}:1"
    )
