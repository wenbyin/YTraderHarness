"""
timeout / retry 两个 AROUND 处理器。

拆开的意义：
    timeout 只管"单次等多久"
    retry   只管"失败了再试几次"
两者通过 next_call 的嵌套自由组合（retry 外层 / timeout 内层是默认）。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.agent.tools.processor.context import (
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
    ToolTimeout,
)
from app.agent.tools.processor.interceptor.retry import (
    RetryConfig,
    RetryProcessor,
)
from app.agent.tools.processor.interceptor.timeout import (
    TimeoutConfig,
    TimeoutProcessor,
)


def _invocation() -> ToolInvocation:
    return ToolInvocation(
        tool_key="agent.sleep_for",
        arguments={"seconds": 30},
    )


async def _next_returning(outcome: ToolOutcome):
    async def call() -> ToolOutcome:
        return outcome

    return call


def _next_sequence(outcomes: list[ToolOutcome]):
    """
    第 N 次调用返回第 N 个 outcome；用完后停在最后一个。

    同时暴露 calls 列表，方便断言"执行了几次"。
    """

    calls: list[int] = []

    async def call() -> ToolOutcome:
        index = len(calls)
        calls.append(index)

        return outcomes[min(index, len(outcomes) - 1)]

    return call, calls


# ============================================================
# timeout
# ============================================================


async def test_timeout_cancels_a_slow_call() -> None:
    processor = TimeoutProcessor()

    async def slow() -> ToolOutcome:
        await asyncio.sleep(1.0)
        return ToolOutcome(value="too late")

    started = time.monotonic()
    outcome = await processor.around(
        _invocation(), slow, config={"timeout_seconds": 0.01}
    )
    elapsed = time.monotonic() - started

    assert isinstance(outcome.error, ToolTimeout)
    assert outcome.ok is False
    assert elapsed < 0.5


async def test_timeout_passes_a_fast_call() -> None:
    processor = TimeoutProcessor()

    async def fast() -> ToolOutcome:
        return ToolOutcome(value={"ok": 1})

    outcome = await processor.around(
        _invocation(), fast, config={"timeout_seconds": 1.0}
    )

    assert outcome.ok
    assert outcome.value == {"ok": 1}


def test_timeout_seconds_must_be_positive() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TimeoutConfig(timeout_seconds=0)


def test_tool_timeout_is_a_timeout_error() -> None:
    """
    默认模式（retry_on_error=False）只重试 TimeoutError，
    ToolTimeout 是它的子类，因此能自然命中。
    """

    assert isinstance(ToolTimeout(1.0), TimeoutError)


# ============================================================
# retry
# ============================================================


async def test_retry_stops_after_success() -> None:
    processor = RetryProcessor()
    next_call, calls = _next_sequence([
        ToolOutcome(error=ToolTimeout(1.0)),
        ToolOutcome(error=ToolTimeout(1.0)),
        ToolOutcome(value={"ok": True}),
    ])

    outcome = await processor.around(
        _invocation(),
        next_call,
        config={
            "max_attempts": 3,
            "retry_on_error": False,
            "backoff_seconds": 0,
        },
    )

    assert outcome.ok
    assert outcome.value == {"ok": True}
    assert len(calls) == 3


async def test_retry_stops_at_max_attempts() -> None:
    processor = RetryProcessor()
    next_call, calls = _next_sequence([
        ToolOutcome(error=ToolTimeout(1.0)),
    ])

    outcome = await processor.around(
        _invocation(),
        next_call,
        config={
            "max_attempts": 3,
            "retry_on_error": False,
            "backoff_seconds": 0,
        },
    )

    assert isinstance(outcome.error, ToolTimeout)
    assert len(calls) == 3


async def test_retry_ignores_non_timeout_errors_by_default() -> None:
    processor = RetryProcessor()
    next_call, calls = _next_sequence([
        ToolOutcome(error=RuntimeError("boom")),
    ])

    outcome = await processor.around(
        _invocation(),
        next_call,
        config={
            "max_attempts": 3,
            "retry_on_error": False,
            "backoff_seconds": 0,
        },
    )

    assert isinstance(outcome.error, RuntimeError)
    assert len(calls) == 1


async def test_retry_on_error_true_retries_non_timeout_errors() -> None:
    processor = RetryProcessor()
    next_call, calls = _next_sequence([
        ToolOutcome(error=RuntimeError("boom")),
        ToolOutcome(error=RuntimeError("boom")),
        ToolOutcome(value="ok"),
    ])

    outcome = await processor.around(
        _invocation(),
        next_call,
        config={
            "max_attempts": 3,
            "retry_on_error": True,
            "backoff_seconds": 0,
        },
    )

    assert outcome.ok
    assert len(calls) == 3


async def test_retry_never_retries_denied() -> None:
    processor = RetryProcessor()
    next_call, calls = _next_sequence([
        ToolOutcome(error=ToolCallDenied("no")),
    ])

    outcome = await processor.around(
        _invocation(),
        next_call,
        config={
            "max_attempts": 3,
            # 即使打开「重试所有错误」，ToolCallDenied 也永远不重试。
            "retry_on_error": True,
            "backoff_seconds": 0,
        },
    )

    assert isinstance(outcome.error, ToolCallDenied)
    assert len(calls) == 1


async def test_retry_returns_first_success_without_extra_calls() -> None:
    processor = RetryProcessor()
    next_call, calls = _next_sequence([
        ToolOutcome(value="ok"),
    ])

    outcome = await processor.around(
        _invocation(),
        next_call,
        config={"max_attempts": 3, "backoff_seconds": 0},
    )

    assert outcome.ok
    assert len(calls) == 1


def test_retryable_modes() -> None:
    """
    默认只认 TimeoutError（含 ToolTimeout）；开关打开后认所有错误。
    """

    assert RetryProcessor._retryable(ToolTimeout(1.0), False)
    assert not RetryProcessor._retryable(
        RuntimeError("boom"), False
    )
    assert RetryProcessor._retryable(
        RuntimeError("boom"), True
    )


def test_old_retry_on_config_is_rejected() -> None:
    """旧版的 retry_on 白名单字段不再接受（extra=forbid）。"""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetryConfig(retry_on=["TimeoutError"])


async def test_invalid_config_falls_back_to_one_call() -> None:
    """
    库里残留旧配置（retry_on=[]）时，不重试但也不打挂工具。
    """

    processor = RetryProcessor()
    next_call, calls = _next_sequence([ToolOutcome(value="ok")])

    outcome = await processor.around(
        _invocation(),
        next_call,
        config={"retry_on": [], "max_attempts": 3},
    )

    assert outcome.ok
    assert len(calls) == 1


async def test_timeout_inside_retry_is_per_attempt() -> None:
    """
    retry(outer) -> timeout(inner)：每次尝试都有独立的超时预算。

    两次超时 + 第三次成功，共执行 3 次。
    """

    retry = RetryProcessor()
    timeout = TimeoutProcessor()
    attempts = {"n": 0}

    async def attempt() -> ToolOutcome:
        attempts["n"] += 1

        if attempts["n"] < 3:
            await asyncio.sleep(1.0)

        return ToolOutcome(value={"attempt": attempts["n"]})

    async def next_call() -> ToolOutcome:
        return await timeout.around(
            _invocation(),
            attempt,
            config={"timeout_seconds": 0.01},
        )

    outcome = await retry.around(
        _invocation(),
        next_call,
        config={
            "max_attempts": 3,
            "retry_on_error": False,
            "backoff_seconds": 0,
        },
    )

    assert outcome.ok
    assert outcome.value == {"attempt": 3}
    assert attempts["n"] == 3
