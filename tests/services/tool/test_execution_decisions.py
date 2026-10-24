"""
执行守卫的纯判定函数。

这是守卫的核心语义：三种状态各自独立计数、独立配额，不是三选一。
不需要数据库 —— `decide_existing` / `decide_new` 只吃状态计数与
配置，返回 decision。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.services.tool.application._execution import (
    decide_existing,
    decide_new,
    match_key_of,
    scope_key_of,
)
from app.services.tool.domain.entities import ToolExecution
from app.services.tool.domain.schemas.enum import (
    ExecutionDecision,
    ExecutionMatch,
    ExecutionScope,
    GuardAction,
    ToolExecutionStatus,
)
from app.services.tool.domain.schemas.execution import ExecutionStatusLimit


def _attempt(
        status: ToolExecutionStatus,
        *,
        result=None,
) -> ToolExecution:
    now = datetime.now(timezone.utc)

    return ToolExecution(
        execution_id=uuid4(),
        tool_key="agent.current_time",
        tool_id=None,
        session_id=uuid4(),
        run_id=uuid4(),
        user_id="u1",
        scope_type=ExecutionScope.SESSION,
        scope_key="s1",
        execution_type=ExecutionMatch.METHOD,
        param_hash="",
        match_key="m" * 64,
        status=status,
        counted=True,
        tool_call_id="call_1",
        result=result,
        error=None,
        created_at=now,
        updated_at=now,
    )


def _rule(
        limit: int | None,
        action: GuardAction = GuardAction.BLOCK,
) -> ExecutionStatusLimit:
    return ExecutionStatusLimit(limit=limit, action=action)


def _decide(
        *,
        counts: dict[ToolExecutionStatus, int] | None = None,
        success: ExecutionStatusLimit | None = None,
        error: ExecutionStatusLimit | None = None,
        unknown: ExecutionStatusLimit | None = None,
):
    return decide_new(
        counts=counts or {},
        success=success or _rule(3),
        error=error or _rule(3),
        unknown=unknown or _rule(1),
    )


# ============================================================
# decide_existing：同一次 tool_call_id 的 replay / resume
# ============================================================


def test_existing_success_replays_without_executing() -> None:
    decision, reason = decide_existing(
        _attempt(ToolExecutionStatus.SUCCESS)
    )

    assert decision is ExecutionDecision.REPLAY
    assert reason is None


@pytest.mark.parametrize("status", [
    ToolExecutionStatus.PENDING,
    ToolExecutionStatus.RUNNING,
])
def test_existing_in_flight_blocks(
        status: ToolExecutionStatus,
) -> None:
    decision, reason = decide_existing(_attempt(status))

    assert decision is ExecutionDecision.BLOCK
    assert reason


@pytest.mark.parametrize("status", [
    ToolExecutionStatus.UNKNOWN,
    ToolExecutionStatus.FAILED,
    ToolExecutionStatus.INTERRUPT,
    ToolExecutionStatus.REJECTED,
    ToolExecutionStatus.CANCELLED,
    ToolExecutionStatus.BLOCKED,
])
def test_existing_non_terminal_falls_through_to_limits(
        status: ToolExecutionStatus,
) -> None:
    """UNKNOWN / FAILED 由 status_counts 的配额判，不在这一层判。"""

    decision, _ = decide_existing(_attempt(status))

    assert decision is ExecutionDecision.EXECUTE


# ============================================================
# decide_new：三种状态独立配额
# ============================================================


def test_success_limit_blocks_the_next_call() -> None:
    """success.limit=3：第 1/2/3 次成功，第 4 次开始拦。"""

    decision, reason = _decide(
        counts={ToolExecutionStatus.SUCCESS: 2},
        success=_rule(3),
    )

    assert decision is ExecutionDecision.EXECUTE
    assert reason is None

    decision, reason = _decide(
        counts={ToolExecutionStatus.SUCCESS: 3},
        success=_rule(3),
    )

    assert decision is ExecutionDecision.BLOCK
    assert "成功" in (reason or "")
    assert "3" in (reason or "")


def test_in_flight_attempts_reserve_a_success_slot() -> None:
    """
    success.limit=1 时，两个并发分支只有一个能通过。

    第一个分支插入 PENDING；第二个分支看到 PENDING 也算一个
    success 名额，于是被拦下。
    """

    decision, reason = _decide(
        counts={ToolExecutionStatus.PENDING: 1},
        success=_rule(1),
        error=_rule(None),
    )

    assert decision is ExecutionDecision.BLOCK
    assert "成功" in (reason or "")


def test_success_limit_zero_is_a_kill_switch() -> None:
    decision, _ = _decide(
        counts={},
        success=_rule(0),
    )

    assert decision is ExecutionDecision.BLOCK


def test_success_limit_none_disables_success_control() -> None:
    decision, _ = _decide(
        counts={ToolExecutionStatus.SUCCESS: 99},
        success=_rule(None),
    )

    assert decision is ExecutionDecision.EXECUTE


def test_error_limit_blocks_the_next_error_after_the_limit() -> None:
    """
    error.limit=3：允许 3 次失败，第 4 次开始拦。

    这正是「想允许重试就设 3」的语义。
    """

    decision, _ = _decide(
        counts={ToolExecutionStatus.FAILED: 2},
        error=_rule(3),
    )

    assert decision is ExecutionDecision.EXECUTE

    decision, reason = _decide(
        counts={ToolExecutionStatus.FAILED: 3},
        error=_rule(3),
    )

    assert decision is ExecutionDecision.BLOCK
    assert "失败" in (reason or "")


def test_error_limit_none_allows_unlimited_retries() -> None:
    decision, _ = _decide(
        counts={ToolExecutionStatus.FAILED: 99},
        error=_rule(None),
    )

    assert decision is ExecutionDecision.EXECUTE


def test_unknown_limit_blocks_after_the_limit() -> None:
    """unknown.limit=1（默认）：出现一次 unknown 后就不再执行。"""

    decision, _ = _decide(
        counts={ToolExecutionStatus.UNKNOWN: 0},
        unknown=_rule(1),
    )

    assert decision is ExecutionDecision.EXECUTE

    decision, reason = _decide(
        counts={ToolExecutionStatus.UNKNOWN: 1},
        unknown=_rule(1),
    )

    assert decision is ExecutionDecision.BLOCK
    assert "unknown" in (reason or "")


def test_error_does_not_consume_the_success_quota() -> None:
    """
    退款失败了不应该占「成功一次」的名额。

    success.limit=1 + error.limit=None：失败多次之后仍然可以重试，
    直到成功一次才拦。
    """

    decision, _ = _decide(
        counts={ToolExecutionStatus.FAILED: 5},
        success=_rule(1),
        error=_rule(None),
    )

    assert decision is ExecutionDecision.EXECUTE

    decision, _ = _decide(
        counts={
            ToolExecutionStatus.FAILED: 5,
            ToolExecutionStatus.SUCCESS: 1,
        },
        success=_rule(1),
        error=_rule(None),
    )

    assert decision is ExecutionDecision.BLOCK


def test_allow_falls_through_to_the_next_status() -> None:
    """ALLOW 只是「这一条不拦」，判定继续往下走。"""

    decision, reason = _decide(
        counts={
            ToolExecutionStatus.SUCCESS: 3,
            ToolExecutionStatus.FAILED: 1,
        },
        success=_rule(3, GuardAction.ALLOW),
        error=_rule(1, GuardAction.BLOCK),
    )

    assert decision is ExecutionDecision.BLOCK
    assert "失败" in (reason or "")


def test_block_order_is_success_then_unknown_then_error() -> None:
    """多个状态同时超限时，按固定顺序给出第一个 BLOCK 的理由。"""

    decision, reason = _decide(
        counts={
            ToolExecutionStatus.SUCCESS: 1,
            ToolExecutionStatus.UNKNOWN: 1,
            ToolExecutionStatus.FAILED: 1,
        },
        success=_rule(1),
        unknown=_rule(1),
        error=_rule(1),
    )

    assert decision is ExecutionDecision.BLOCK
    assert "成功" in (reason or "")


def test_no_counts_executes() -> None:
    decision, reason = _decide()

    assert decision is ExecutionDecision.EXECUTE
    assert reason is None


# ============================================================
# match_key / scope_key
# ============================================================


def test_match_key_separates_method_and_invoke() -> None:
    method = match_key_of(
        tool_key="agent.current_time",
        scope=ExecutionScope.SESSION,
        scope_key="s1",
        execution=ExecutionMatch.METHOD,
        param_hash="",
    )
    invoke_a = match_key_of(
        tool_key="agent.current_time",
        scope=ExecutionScope.SESSION,
        scope_key="s1",
        execution=ExecutionMatch.INVOKE,
        param_hash="a" * 64,
    )
    invoke_b = match_key_of(
        tool_key="agent.current_time",
        scope=ExecutionScope.SESSION,
        scope_key="s1",
        execution=ExecutionMatch.INVOKE,
        param_hash="b" * 64,
    )

    assert method != invoke_a
    assert invoke_a != invoke_b


def test_match_key_is_deterministic() -> None:
    kwargs = dict(
        tool_key="agent.current_time",
        scope=ExecutionScope.USER,
        scope_key="u1",
        execution=ExecutionMatch.INVOKE,
        param_hash="a" * 64,
    )

    assert match_key_of(**kwargs) == match_key_of(**kwargs)


def test_scope_key_for_each_scope() -> None:
    session = uuid4()
    run = uuid4()

    assert scope_key_of(
        ExecutionScope.SESSION,
        session_id=session,
        run_id=run,
        user_id="u1",
    ) == str(session)
    assert scope_key_of(
        ExecutionScope.RUN,
        session_id=session,
        run_id=run,
        user_id="u1",
    ) == str(run)
    assert scope_key_of(
        ExecutionScope.USER,
        session_id=session,
        run_id=run,
        user_id="u1",
    ) == "u1"
    assert scope_key_of(
        ExecutionScope.GLOBAL,
        session_id=None,
        run_id=None,
        user_id=None,
    ) == "*"


@pytest.mark.parametrize("scope", [
    ExecutionScope.SESSION,
    ExecutionScope.RUN,
    ExecutionScope.USER,
])
def test_missing_identity_raises(scope: ExecutionScope) -> None:
    with pytest.raises(ValueError):
        scope_key_of(
            scope,
            session_id=None,
            run_id=None,
            user_id=None,
        )
