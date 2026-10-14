"""
Retry（规格 §22 第二条）。

    attempt 1 -> timeout
    attempt 2 -> success

最终：

    LLM Node executed once      <- 见 test_llm_node_isolation.py
    Runtime attempted twice     <- 这里

关键不是"能重试"，而是**只对可重试的错误重试**（规格 §8.2：
"不能对所有异常无脑 retry"）。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from fakes import FakeAdapter, Outcome, RecordingObserver, make_definition, make_registry

from app.agent.model import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelProfile,
    ModelProviderError,
    ModelRateLimitError,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
    ModelRuntimePolicy,
    ModelTimeoutError,
    ModelUnavailableError,
)


def _setup(adapter, observer=None, **policy_kwargs):
    """单候选的 runtime（重试是**同一个候选**内部的事）。"""

    policy_kwargs.setdefault("timeout", 1.0)
    policy_kwargs.setdefault("max_retries", 2)
    policy_kwargs.setdefault("retry_backoff", 0.0)

    registry = make_registry(
        models=[make_definition("primary")],
        profiles=[
            ModelProfile(
                id="p",
                primary="fake:primary",
                policy=ModelRuntimePolicy(**policy_kwargs),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react", model_role=ModelRole.REASONING, profile="p"
            )
        ],
        adapter=adapter,
        observer=observer,
    )

    return registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )


def _request() -> ModelRequest:
    return ModelRequest(messages=("你好",))


# ============================================================
# 可重试
# ============================================================


async def test_timeout_then_success_attempts_twice() -> None:
    """attempt 1 超时、attempt 2 成功 —— 对外是一次调用、两次尝试。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelTimeoutError("第一次超时")),
                Outcome(message="第二次成功"),
            ]
        }
    )

    observer = RecordingObserver()
    runtime = _setup(adapter, observer)

    response = await runtime.ainvoke(_request())

    assert response.message == "第二次成功"
    assert response.attempts == 2
    assert response.degraded is True

    # 两次都打在同一个模型上（没有换 provider）
    assert adapter.calls == [
        ("fake:primary", 1.0),
        ("fake:primary", 1.0),
    ]

    # 观测里看得到两次 attempt，第一次是失败、第二次是成功
    assert observer.codes() == ["timeout", None]
    assert observer.models() == ["fake:primary", "fake:primary"]

    assert len(observer.summaries) == 1
    summary = observer.summaries[0]
    assert summary.success is True
    assert summary.provider_attempts == 2
    assert summary.degraded is True


async def test_retry_reuses_the_same_request_object() -> None:
    """
    规格 §9：retry 重发的是**同一个 request**，不是重新构建一个。

    这条语义是"节点不会因为 retry/fallback 被重跑"的微观版本。
    """

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelUnavailableError("临时不可用")),
                Outcome(message="ok"),
            ]
        }
    )

    runtime = _setup(adapter)
    request = _request()

    await runtime.ainvoke(request)

    assert len(adapter.requests) == 2
    assert adapter.requests[0] is request
    assert adapter.requests[1] is request


async def test_provider_rate_limit_is_retried() -> None:
    """限流（provider 异常，经归一化）也是可重试的。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelRateLimitError("429")),
                Outcome(message="ok"),
            ]
        }
    )

    runtime = _setup(adapter)

    response = await runtime.ainvoke(_request())

    assert response.attempts == 2
    assert adapter.call_count() == 2


async def test_retry_stops_at_max_retries() -> None:
    """max_retries 是硬上限：不会无限重试。"""

    adapter = FakeAdapter(
        default=Outcome(error=ModelUnavailableError("一直不可用"))
    )

    runtime = _setup(adapter, max_retries=2)

    with pytest.raises(ModelUnavailableError):
        await runtime.ainvoke(_request())

    # 1 次初始 + 2 次重试 = 3 次，正好是 max_retries + 1
    assert adapter.call_count() == 3


async def test_max_retries_zero_means_single_attempt() -> None:
    """max_retries=0 就是不重试。"""

    adapter = FakeAdapter(
        default=Outcome(error=ModelTimeoutError("超时"))
    )

    runtime = _setup(adapter, max_retries=0)

    with pytest.raises(ModelTimeoutError):
        await runtime.ainvoke(_request())

    assert adapter.call_count() == 1


async def test_retry_backoff_is_actually_waited() -> None:
    """配了退避就真的等 —— 限流时"立刻再撞一次"没有意义。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelRateLimitError("429")),
                Outcome(message="ok"),
            ]
        }
    )

    runtime = _setup(adapter, retry_backoff=0.15)

    started = time.perf_counter()
    response = await runtime.ainvoke(_request())
    elapsed = time.perf_counter() - started

    assert response.attempts == 2
    assert elapsed >= 0.15


# ============================================================
# 不可重试
# ============================================================


async def test_invalid_request_is_not_retried() -> None:
    """
    请求不合法 -> 立刻抛，**不重试**。

    同一份请求再发一次还是会 400；重试只是把同一个错误重复几次并
    多花几次钱。
    """

    adapter = FakeAdapter(
        default=Outcome(error=ModelInvalidRequestError("400 参数不合法"))
    )

    observer = RecordingObserver()
    runtime = _setup(adapter, observer)

    with pytest.raises(ModelInvalidRequestError):
        await runtime.ainvoke(_request())

    assert adapter.call_count() == 1
    assert observer.codes() == ["invalid_request"]


async def test_authentication_error_is_not_retried() -> None:
    """凭据被拒 -> 不重试同一个模型（换 key 才有用）。"""

    adapter = FakeAdapter(
        default=Outcome(error=ModelAuthenticationError("401"))
    )

    runtime = _setup(adapter, max_retries=3)

    with pytest.raises(ModelAuthenticationError):
        await runtime.ainvoke(_request())

    assert adapter.call_count() == 1


async def test_policy_can_narrow_retryable_errors() -> None:
    """
    `retryable_errors` 能把某个可重试错误**收窄**成不可重试。

    这让"这次别重试"变成一个按 profile 配置的开关，而不必改代码。
    """

    adapter = FakeAdapter(
        default=Outcome(error=ModelTimeoutError("超时"))
    )

    runtime = _setup(adapter, retryable_errors=())

    with pytest.raises(ModelTimeoutError):
        await runtime.ainvoke(_request())

    assert adapter.call_count() == 1


async def test_unknown_provider_error_is_retried_as_provider_error() -> None:
    """归不了类的 provider 异常落到可重试的兜底，而不是直接失败。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                # 一个名字不在表里、也没有状态码的异常
                Outcome(error=RuntimeError("provider 抽风了")),
                Outcome(message="ok"),
            ]
        }
    )

    observer = RecordingObserver()
    runtime = _setup(adapter, observer)

    response = await runtime.ainvoke(_request())

    assert response.message == "ok"
    assert observer.codes() == ["provider_error", None]

    # 兜底**一定**是 Model 层异常 —— provider 异常不会穿透
    assert isinstance(
        observer.attempts[0][1].error_type,
        str,
    )
    assert ModelProviderError.code == "provider_error"


# ============================================================
# 超时
# ============================================================


async def test_attempt_timeout_is_enforced() -> None:
    """
    超时由 Runtime 统一执行（asyncio.wait_for），不依赖 provider 自觉。

    客户端构造时也会拿到 timeout，但那不覆盖 bind_tools 之类的
    事前工作，也可能被某些兼容端点忽略 —— 两处都要。
    """

    adapter = FakeAdapter(
        default=Outcome(message="太慢了", delay=5.0)
    )

    runtime = _setup(adapter, timeout=0.05, max_retries=0)

    started = time.perf_counter()

    with pytest.raises(ModelTimeoutError):
        await runtime.ainvoke(_request())

    elapsed = time.perf_counter() - started

    # 真的在 0.05s 左右就返回了，而不是等 provider 的 5 秒
    assert elapsed < 1.0


async def test_total_timeout_stops_the_whole_chain() -> None:
    """
    总预算耗尽 -> 停止整条链（不再尝试剩余候选）。

    没有它，最坏情况是 候选数 × (max_retries+1) × timeout —— 而 LLM
    Node 的上游（HTTP 请求 / 图运行）几乎一定等不了那么久。
    """

    adapter = FakeAdapter(
        default=Outcome(message="太慢了", delay=5.0)
    )

    runtime = _setup(
        adapter,
        timeout=1.0,
        max_retries=5,
        total_timeout=0.05,
    )

    with pytest.raises(ModelTimeoutError) as excinfo:
        await runtime.ainvoke(_request())

    assert "总预算" in str(excinfo.value)

    # 第一次 attempt 就把预算用完，第二次在发出去之前就被拦下
    assert adapter.call_count() == 1
