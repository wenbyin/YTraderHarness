"""task_tool 的契约占位：名字 / key / 明确的 not_implemented。"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langgraph.types import Command

from app.agent.plan.publish import PublishResult
from app.agent.tools import task_tool

__all__: list[str] = []


def test_tool_keys_are_derived_from_names() -> None:
    assert task_tool.CREATE_TASK_TOOL_KEY == "agent.create_task"
    assert task_tool.CLAIM_TASK_TOOL_KEY == "agent.claim_task"
    assert task_tool.UPDATE_TASK_TOOL_KEY == "agent.update_task"
    assert task_tool.COMPLETE_TASK_TOOL_KEY == "agent.complete_task"
    assert task_tool.CREATE_TEAM_TOOL_KEY == "agent.create_team"

    assert task_tool.TASK_TOOL_KEYS == frozenset({
        "agent.create_task",
        "agent.claim_task",
        "agent.update_task",
        "agent.complete_task",
        "agent.create_team",
    })


@pytest.mark.asyncio
async def test_create_task_is_explicitly_not_implemented() -> None:
    result = await task_tool.create_task.ainvoke({
        "skill_key": "research",
        "description": "研究",
    })

    assert result["ok"] is False
    assert result["not_implemented"] is True
    assert result["tool"] == "agent.create_task"
    assert result["skill_key"] == "research"


@pytest.mark.asyncio
async def test_claim_and_update_do_not_pretend_success() -> None:
    claimed = await task_tool.claim_task.ainvoke({"task_id": "t1"})
    updated = await task_tool.update_task.ainvoke({
        "task_id": "t1",
        "status": "TASK_STATE_WORKING",
    })

    assert claimed["not_implemented"] is True
    assert updated["not_implemented"] is True
    assert updated["status"] == "TASK_STATE_WORKING"


@pytest.mark.asyncio
async def test_complete_is_still_a_placeholder() -> None:
    completed = await task_tool.complete_task.ainvoke({
        "task_id": "t1",
        "summary": "完成",
    })

    assert completed["not_implemented"] is True
    assert completed["task_id"] == "t1"


# ============================================================
# create_team：发布任务板
# ============================================================


def _task(
        task_id: str,
        *,
        state: str = "TASK_STATE_SUBMITTED",
        blocked_by: list[str] | None = None,
) -> dict:
    return {
        "id": task_id,
        "status": {"state": state},
        "metadata": {
            "plan_id": "p1",
            "blocked_by": blocked_by or [],
            "claim": {},
        },
    }


def _runtime(*, tasks: list, plan: dict | None = None, session_id=None):
    return SimpleNamespace(
        state={
            "tasks": tasks,
            "plan": plan or {"plan_id": "p1", "revision": 1},
        },
        context=SimpleNamespace(
            session_id=session_id or uuid4(),
            task_board=object(),
        ),
    )


@pytest.mark.asyncio
async def test_create_team_skips_without_tasks() -> None:
    # 直接 ainvoke 时没有 runtime 注入 -> state 为空。
    result = await task_tool.create_team.ainvoke({"worker_num": 2})

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "no_tasks"
    assert result["worker_num"] == 2


@pytest.mark.asyncio
async def test_create_team_skips_when_all_tasks_are_terminal() -> None:
    runtime = _runtime(tasks=[
        _task("p1:t1", state="TASK_STATE_COMPLETED"),
    ])

    result = await task_tool.create_team.coroutine(
        worker_num=2, runtime=runtime,
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "no_pending_tasks"


@pytest.mark.asyncio
async def test_create_team_refuses_dependency_cycle(monkeypatch) -> None:
    called: list = []

    async def fake_publish(**kwargs):
        called.append(kwargs)
        return PublishResult(published=0, skipped=0)

    monkeypatch.setattr(task_tool, "publish_plan", fake_publish)

    runtime = _runtime(tasks=[
        _task("p1:t1", blocked_by=["p1:t2"]),
        _task("p1:t2", blocked_by=["p1:t1"]),
    ])

    result = await task_tool.create_team.coroutine(
        worker_num=2, runtime=runtime,
    )

    assert result["ok"] is False
    assert result["reason"] == "dependency_cycle"
    assert result["cycle"][0] == result["cycle"][-1]
    assert called == []


@pytest.mark.asyncio
async def test_create_team_publishes_pending_tasks(monkeypatch) -> None:
    calls: list = []

    async def fake_publish(**kwargs):
        calls.append(kwargs)
        return PublishResult(published=2, skipped=0)

    monkeypatch.setattr(task_tool, "publish_plan", fake_publish)

    session_id = uuid4()
    tasks = [
        _task("p1:t1"),
        _task("p1:t2"),
        _task("p1:t3", state="TASK_STATE_COMPLETED"),
    ]

    result = await task_tool.create_team.coroutine(
        worker_num=5,
        runtime=_runtime(
            tasks=tasks,
            plan={"plan_id": "p1", "revision": 3},
            session_id=session_id,
        ),
    )

    # 审批通过后不再是普通 dict：返回 Command 跳到派发节点。
    assert isinstance(result, Command)
    assert result.goto == "subagent_dispatch"
    assert result.update["subagent_mode"] == "on"
    # worker 数受待完成任务数限制。
    assert result.update["subagent_workers"] == 2
    assert "已发布 2 个任务" in result.update["message"]

    assert len(calls) == 1
    assert calls[0]["context_id"] == str(session_id)
    assert calls[0]["plan_id"] == "p1"
    assert calls[0]["revision"] == 3
    assert calls[0]["tasks"] == tasks


@pytest.mark.asyncio
async def test_create_team_reports_board_unavailable(monkeypatch) -> None:
    async def fake_publish(**kwargs):
        return PublishResult(
            published=0,
            skipped=0,
            degraded=True,
            reason="Redis 未装配",
        )

    monkeypatch.setattr(task_tool, "publish_plan", fake_publish)

    result = await task_tool.create_team.coroutine(
        worker_num=1,
        runtime=_runtime(tasks=[_task("p1:t1")]),
    )

    assert result["ok"] is False
    assert result["degraded"] is True
    assert result["reason"] == "task_board_unavailable"
