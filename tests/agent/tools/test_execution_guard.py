"""
执行守卫 AROUND 拦截器的行为。

这些用例把账本服务换成模块级桩（`_reserve` / `_settle`），因此
不需要 Postgres，但覆盖的是**真实拦截器**的中断 / 取消 / 拒绝 /
缓存 / 回放分支 —— 那些分支最容易把「工具没跑」记成「跑了」，
或者反过来。

策略矩阵本身（success / error / unknown 三种独立配额）在
`tests/services/tool/test_execution_decisions.py` 里验证，这里只验
拦截器把结果翻译成了正确的状态与 counted。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from langgraph.errors import GraphInterrupt

from app.agent.tools.processor.context import (
    SESSION_CONTEXT_KEY,
    TOOL_STARTED_CONTEXT_KEY,
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
    ToolTimeout,
)
from app.agent.tools.processor.interceptor import execution_guard as guard_module
from app.agent.tools.processor.interceptor.execution_guard import (
    ExecutionGuardConfig,
    ExecutionGuardProcessor,
)
from app.services.tool.domain.entities import ToolExecution
from app.services.tool.domain.schemas.enum import (
    ExecutionDecision,
    ExecutionMatch,
    ExecutionScope,
    ToolExecutionStatus,
)
from app.services.tool.domain.schemas.execution import (
    ExecutionGuardRequest,
    ExecutionReservation,
)


# ============================================================
# 工具
# ============================================================


def _attempt(
        status: ToolExecutionStatus = ToolExecutionStatus.PENDING,
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


def _invocation(
        *,
        started: bool = False,
        session: bool = True,
) -> ToolInvocation:
    context = {
        "tool_call_id": "call_1",
    }

    if session:
        context[SESSION_CONTEXT_KEY] = uuid4()

    if started:
        context[TOOL_STARTED_CONTEXT_KEY] = True

    return ToolInvocation(
        tool_key="agent.current_time",
        arguments={"timezone": "UTC"},
        tool_id=None,
        context=context,
    )


def _config(**overrides) -> dict:
    return ExecutionGuardConfig(**overrides).model_dump()


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def _next_call(outcome):
    calls = []

    async def call() -> ToolOutcome:
        calls.append(True)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return call, calls


# ============================================================
# 配置
# ============================================================


def test_default_policy_is_three_independent_limits() -> None:
    parsed = ExecutionGuardConfig()

    assert parsed.scope == "SESSION"
    assert parsed.execution == "METHOD"
    assert parsed.window_seconds is None

    assert parsed.success_limit == 3
    assert parsed.success_action == "BLOCK"

    assert parsed.error_limit == 3
    assert parsed.error_action == "BLOCK"

    assert parsed.unknown_limit == 1
    assert parsed.unknown_action == "BLOCK"


def test_old_nested_config_is_rejected() -> None:
    """旧版 success={limit, action} 不能被静默忽略。"""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExecutionGuardConfig(
            success={"limit": 1, "action": "BLOCK"}
        )


def test_config_schema_is_flat_for_the_frontend() -> None:
    """
    前端直接按 properties 生成控件，不解析嵌套模型。

    嵌套模型会输出 `$defs` + `$ref`，前端会显示成一个无法识别的
    对象、控件置空。这里锁住「全部是扁平的标量 / 枚举字段」。
    """

    schema = ExecutionGuardProcessor.get_config_schema()

    properties = schema["properties"]

    assert set(properties) == {
        "scope",
        "execution",
        "window_seconds",
        "success_limit",
        "success_action",
        "error_limit",
        "error_action",
        "unknown_limit",
        "unknown_action",
    }

    for name, definition in properties.items():
        assert "$ref" not in definition, name
        assert definition.get("type") != "object", name

    assert "$defs" not in schema


def test_user_and_global_allow_no_window() -> None:
    """
    USER / GLOBAL 不再强制 TTL：退款 / 下单这类 submit 工具需要
    「这个用户 / 所有人永久只成功一次」。
    """

    user = ExecutionGuardConfig(
        scope="USER", window_seconds=None
    )
    global_ = ExecutionGuardConfig(
        scope="GLOBAL", window_seconds=None
    )

    assert user.window_seconds is None
    assert global_.window_seconds is None

    # 显式窗口当然也合法。
    ExecutionGuardConfig(
        scope=ExecutionScope.GLOBAL, window_seconds=3600
    )


# ============================================================
# 身份缺失 -> fail-closed
# ============================================================


async def test_missing_session_is_denied(monkeypatch) -> None:
    processor = ExecutionGuardProcessor()
    next_call, calls = _next_call(ToolOutcome(value="ok"))

    outcome = await processor.around(
        _invocation(session=False),
        next_call,
        config={},
    )

    assert isinstance(outcome.error, ToolCallDenied)
    assert not calls


# ============================================================
# 预占结果
# ============================================================


async def test_block_does_not_call_the_tool(monkeypatch) -> None:
    processor = ExecutionGuardProcessor()
    next_call, calls = _next_call(ToolOutcome(value="ok"))
    settle = _Recorder()

    async def fake_reserve(request):
        assert isinstance(request, ExecutionGuardRequest)
        return ExecutionReservation(
            decision=ExecutionDecision.BLOCK,
            reason="额度已满",
        )

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)
    monkeypatch.setattr(guard_module, "_settle", settle)

    invocation = _invocation()

    outcome = await processor.around(
        invocation, next_call, config={}
    )

    assert isinstance(outcome.error, ToolCallDenied)
    assert "额度已满" in str(outcome.error)
    assert not calls
    assert not settle.calls
    assert invocation.context["execution_guard.decision"] == "BLOCK"


async def test_replay_returns_the_recorded_result(monkeypatch) -> None:
    processor = ExecutionGuardProcessor()
    next_call, calls = _next_call(ToolOutcome(value="should not run"))
    settle = _Recorder()

    async def fake_reserve(request):
        return ExecutionReservation(
            decision=ExecutionDecision.REPLAY,
            attempt=_attempt(
                ToolExecutionStatus.SUCCESS,
                result={"now": "2026-10-17"},
            ),
        )

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)
    monkeypatch.setattr(guard_module, "_settle", settle)

    outcome = await processor.around(
        _invocation(), next_call, config={}
    )

    assert outcome.value == {"now": "2026-10-17"}
    assert outcome.cached is True
    assert not calls
    assert not settle.calls


async def test_ledger_unavailable_is_fail_closed(monkeypatch) -> None:
    processor = ExecutionGuardProcessor()
    next_call, calls = _next_call(ToolOutcome(value="ok"))

    async def fake_reserve(request):
        raise RuntimeError("db down")

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)

    outcome = await processor.around(
        _invocation(), next_call, config={}
    )

    assert isinstance(outcome.error, ToolCallDenied)
    assert "账本不可用" in str(outcome.error)
    assert not calls


async def test_execute_without_attempt_is_denied(monkeypatch) -> None:
    processor = ExecutionGuardProcessor()
    next_call, calls = _next_call(ToolOutcome(value="ok"))

    async def fake_reserve(request):
        return ExecutionReservation(decision=ExecutionDecision.EXECUTE)

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)

    outcome = await processor.around(
        _invocation(), next_call, config={}
    )

    assert isinstance(outcome.error, ToolCallDenied)
    assert not calls


# ============================================================
# 执行结果 -> 状态 / counted
# ============================================================


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_counted"),
    [
        (ToolOutcome(value={"ok": 1}), ToolExecutionStatus.SUCCESS, True),
        (
            ToolOutcome(error=ToolCallDenied("no")),
            ToolExecutionStatus.REJECTED,
            False,
        ),
        (
            ToolOutcome(error=RuntimeError("boom")),
            ToolExecutionStatus.FAILED,
            True,
        ),
        (
            ToolOutcome(value="cached", cached=True),
            ToolExecutionStatus.SUCCESS,
            False,
        ),
    ],
)
async def test_outcome_is_classified(
        monkeypatch,
        outcome,
        expected_status,
        expected_counted,
) -> None:
    processor = ExecutionGuardProcessor()
    next_call, _ = _next_call(outcome)
    settle = _Recorder()

    async def fake_reserve(request):
        return ExecutionReservation(
            decision=ExecutionDecision.EXECUTE,
            attempt=_attempt(),
        )

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)
    monkeypatch.setattr(guard_module, "_settle", settle)

    returned = await processor.around(
        _invocation(), next_call, config={}
    )

    assert returned is outcome
    assert len(settle.calls) == 1

    args, kwargs = settle.calls[0]
    assert kwargs["status"] is expected_status
    assert kwargs["counted"] is expected_counted


# ============================================================
# 中断 / 取消
# ============================================================


async def test_interrupt_before_start_is_not_counted(monkeypatch) -> None:
    processor = ExecutionGuardProcessor()
    next_call, _ = _next_call(GraphInterrupt(()))
    settle = _Recorder()

    async def fake_reserve(request):
        return ExecutionReservation(
            decision=ExecutionDecision.EXECUTE,
            attempt=_attempt(),
        )

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)
    monkeypatch.setattr(guard_module, "_settle", settle)

    with pytest.raises(GraphInterrupt):
        await processor.around(
            _invocation(started=False), next_call, config={}
        )

    _, kwargs = settle.calls[0]
    assert kwargs["status"] is ToolExecutionStatus.INTERRUPT
    assert kwargs["counted"] is False


async def test_interrupt_after_start_is_unknown_and_counted(
        monkeypatch,
) -> None:
    processor = ExecutionGuardProcessor()
    next_call, _ = _next_call(GraphInterrupt(()))
    settle = _Recorder()

    async def fake_reserve(request):
        return ExecutionReservation(
            decision=ExecutionDecision.EXECUTE,
            attempt=_attempt(),
        )

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)
    monkeypatch.setattr(guard_module, "_settle", settle)

    with pytest.raises(GraphInterrupt):
        await processor.around(
            _invocation(started=True), next_call, config={}
        )

    _, kwargs = settle.calls[0]
    assert kwargs["status"] is ToolExecutionStatus.UNKNOWN
    assert kwargs["counted"] is True


async def test_cancelled_before_start_is_not_counted(monkeypatch) -> None:
    processor = ExecutionGuardProcessor()
    next_call, _ = _next_call(asyncio.CancelledError())
    settle = _Recorder()

    async def fake_reserve(request):
        return ExecutionReservation(
            decision=ExecutionDecision.EXECUTE,
            attempt=_attempt(),
        )

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)
    monkeypatch.setattr(guard_module, "_settle", settle)

    with pytest.raises(asyncio.CancelledError):
        await processor.around(
            _invocation(started=False), next_call, config={}
        )

    _, kwargs = settle.calls[0]
    assert kwargs["status"] is ToolExecutionStatus.CANCELLED
    assert kwargs["counted"] is False


async def test_cancelled_after_start_is_unknown_and_counted(
        monkeypatch,
) -> None:
    processor = ExecutionGuardProcessor()
    next_call, _ = _next_call(asyncio.CancelledError())
    settle = _Recorder()

    async def fake_reserve(request):
        return ExecutionReservation(
            decision=ExecutionDecision.EXECUTE,
            attempt=_attempt(),
        )

    monkeypatch.setattr(guard_module, "_reserve", fake_reserve)
    monkeypatch.setattr(guard_module, "_settle", settle)

    with pytest.raises(asyncio.CancelledError):
        await processor.around(
            _invocation(started=True), next_call, config={}
        )

    _, kwargs = settle.calls[0]
    assert kwargs["status"] is ToolExecutionStatus.UNKNOWN
    assert kwargs["counted"] is True


# ============================================================
# timeout 的分类
# ============================================================


def test_timeout_after_start_is_unknown_and_counted() -> None:
    """
    已经进到工具方法才超时：外部副作用可能已经发生。
    """

    status, counted = ExecutionGuardProcessor._classify(
        ToolOutcome(error=ToolTimeout(30.0)),
        started=True,
    )

    assert status is ToolExecutionStatus.UNKNOWN
    assert counted is True


def test_timeout_before_start_is_failed_and_not_counted() -> None:
    """
    还没进工具方法就超时：确认没有副作用。
    """

    status, counted = ExecutionGuardProcessor._classify(
        ToolOutcome(error=ToolTimeout(30.0)),
        started=False,
    )

    assert status is ToolExecutionStatus.FAILED
    assert counted is False
