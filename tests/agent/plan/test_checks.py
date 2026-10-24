"""任务图判定：待完成任务、依赖环。"""

from __future__ import annotations

from app.agent.plan.checks import (
    find_dependency_cycle,
    has_parallel_ready,
    has_pending_tasks,
    pending_tasks,
    ready_tasks,
)

__all__: list[str] = []


def _task(
        task_id: str,
        *,
        state: str = "TASK_STATE_SUBMITTED",
        blocked_by: list[str] | None = None,
) -> dict:
    return {
        "id": task_id,
        "status": {"state": state},
        "metadata": {"blocked_by": blocked_by or []},
    }


def test_pending_tasks_excludes_terminal_states() -> None:
    tasks = [
        _task("t1"),
        _task("t2", state="TASK_STATE_WORKING"),
        _task("t3", state="TASK_STATE_COMPLETED"),
        _task("t4", state="TASK_STATE_FAILED"),
    ]

    assert [task["id"] for task in pending_tasks(tasks)] == ["t1", "t2"]
    assert has_pending_tasks(tasks) is True
    assert has_pending_tasks(tasks[2:]) is False


def test_find_dependency_cycle_returns_the_cycle_path() -> None:
    tasks = [
        _task("t1", blocked_by=["t2"]),
        _task("t2", blocked_by=["t3"]),
        _task("t3", blocked_by=["t1"]),
    ]

    cycle = find_dependency_cycle(tasks)

    assert cycle is not None
    assert cycle[0] == cycle[-1]
    assert set(cycle[:-1]) == {"t1", "t2", "t3"}


def test_self_dependency_is_a_cycle() -> None:
    assert find_dependency_cycle([
        _task("t1", blocked_by=["t1"]),
    ]) == ["t1", "t1"]


def test_dangling_dependency_is_not_a_cycle() -> None:
    tasks = [
        _task("t1", blocked_by=["missing"]),
        _task("t2", blocked_by=["t1"]),
    ]

    assert find_dependency_cycle(tasks) is None


def test_dag_has_no_cycle() -> None:
    tasks = [
        _task("t1"),
        _task("t2", blocked_by=["t1"]),
        _task("t3", blocked_by=["t1", "t2"]),
    ]

    assert find_dependency_cycle(tasks) is None


# ============================================================
# 并行 ready
# ============================================================


def test_two_ready_tasks_are_parallel() -> None:
    tasks = [_task("t1"), _task("t2")]

    assert [task["id"] for task in ready_tasks(tasks)] == ["t1", "t2"]
    assert has_parallel_ready(tasks) is True


def test_single_ready_task_is_not_parallel() -> None:
    tasks = [
        _task("t1"),
        _task("t2", blocked_by=["t1"]),
    ]

    assert [task["id"] for task in ready_tasks(tasks)] == ["t1"]
    assert has_parallel_ready(tasks) is False


def test_working_tasks_are_not_ready() -> None:
    tasks = [
        _task("t1", state="TASK_STATE_WORKING"),
        _task("t2"),
    ]

    assert [task["id"] for task in ready_tasks(tasks)] == ["t2"]
    assert has_parallel_ready(tasks) is False


def test_completed_blockers_unlock_parallel_tasks() -> None:
    tasks = [
        _task("t1", state="TASK_STATE_COMPLETED"),
        _task("t2", blocked_by=["t1"]),
        _task("t3", blocked_by=["t1"]),
    ]

    assert [task["id"] for task in ready_tasks(tasks)] == ["t2", "t3"]
    assert has_parallel_ready(tasks) is True
