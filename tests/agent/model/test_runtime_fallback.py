"""
Fallback / 降级（规格 §22 第三条 + "All providers failed" 那条）。

    Primary -> unavailable
    Fallback -> success

最终：

    LLM Node executed once        <- 见 test_llm_node_isolation.py
    Runtime invoked once          <- 这里：一次 ainvoke
    Provider attempts = 2         <- 这里：两次 attempt

以及"所有候选都失败"时必须返回**统一的 ModelUnavailableError**，
而不是把最后那家的异常抛出去。
"""

from __future__ import annotations

import pytest

from fakes import (
    FakeAdapter,
    Outcome,
    RecordingObserver,
    fake_exception,
    make_definition,
    make_registry,
)

from app.agent.model import (
    ModelAuthenticationError,
    ModelCandidatesExhaustedError,
    ModelInvalidRequestError,
    ModelProfile,
    ModelRateLimitError,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
    ModelRuntimePolicy,
    ModelTimeoutError,
    ModelUnavailableError,
)


def _runtime(adapter, *, fallbacks=("fake:backup",), observer=None, **policy_kwargs):
    policy_kwargs.setdefault("timeout", 1.0)
    policy_kwargs.setdefault("max_retries", 2)
    policy_kwargs.setdefault("retry_backoff", 0.0)

    registry = make_registry(
        models=[
            make_definition("primary"),
            make_definition("backup"),
        ],
        profiles=[
            ModelProfile(
                id="p",
                primary="fake:primary",
                fallbacks=tuple(fallbacks),
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
# 降级成功
# ============================================================


async def test_primary_unavailable_falls_back() -> None:
    """primary 不可用 -> 换 backup；一次 invoke、两次 provider attempt。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelUnavailableError("503")),
                Outcome(error=ModelUnavailableError("503")),
                Outcome(error=ModelUnavailableError("503")),
            ],
            "fake:backup": [Outcome(message="备份答的")],
        }
    )

    observer = RecordingObserver()
    runtime = _runtime(adapter, observer=observer)

    response = await runtime.ainvoke(_request())

    assert response.message == "备份答的"
    assert response.provider == "fake"
    assert response.model == "backup"

    # primary 试满 max_retries+1 = 3 次，再换 backup 一次 -> 共 4 次 attempt
    assert response.attempts == 4
    assert response.degraded is True
    assert response.metadata["fallback_used"] is True
    assert response.metadata["retried"] is True

    assert adapter.called_models() == [
        "fake:primary",
        "fake:primary",
        "fake:primary",
        "fake:backup",
    ]

    # 观测：一次调用、四次 attempt、最终成功
    assert len(observer.summaries) == 1
    summary = observer.summaries[0]
    assert summary.success is True
    assert summary.provider_attempts == 4
    assert summary.model == "backup"


async def test_fallback_when_retry_is_not_allowed() -> None:
    """
    不可重试的错误**直接换候选**，不在原地重试。

    这是 disposition=FALLBACK 的实际效果：同一把 key 再试还是被拒，
    但换一家常常就好了。
    """

    adapter = FakeAdapter(
        script={
            "fake:primary": [Outcome(error=ModelAuthenticationError("401"))],
            "fake:backup": [Outcome(message="备份答的")],
        }
    )

    runtime = _runtime(adapter)

    response = await runtime.ainvoke(_request())

    assert response.message == "备份答的"
    assert response.attempts == 2

    # primary 只试了**一次**（auth 不重试），然后换 backup
    assert adapter.called_models() == ["fake:primary", "fake:backup"]


async def test_retry_then_fallback() -> None:
    """先重试、再降级：两种机制按顺序接力。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelRateLimitError("429")),
                Outcome(error=ModelRateLimitError("429")),
                Outcome(error=ModelRateLimitError("429")),
            ],
            "fake:backup": [Outcome(message="ok")],
        }
    )

    observer = RecordingObserver()
    runtime = _runtime(adapter, observer=observer)

    response = await runtime.ainvoke(_request())

    assert observer.codes() == [
        "rate_limit",
        "rate_limit",
        "rate_limit",
        None,
    ]
    assert response.attempts == 4
    assert response.degraded is True


async def test_provider_exception_drives_fallback() -> None:
    """
    **provider SDK 的异常**同样驱动降级 —— 它先被归一化，再被决策。

    这正是"LLM Node 不需要判断 provider 异常"的实现基础。
    """

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=fake_exception("APIConnectionError", message="连不上")),
            ],
            "fake:backup": [Outcome(message="ok")],
        }
    )

    runtime = _runtime(adapter, retryable_errors=())

    response = await runtime.ainvoke(_request())

    assert response.message == "ok"
    assert adapter.called_models() == ["fake:primary", "fake:backup"]


# ============================================================
# 全部失败
# ============================================================


async def test_all_candidates_failed_raises_unavailable() -> None:
    """
    所有 provider 都失败 -> 统一的 `ModelUnavailableError`。

    （具体类型是它的子类 `ModelCandidatesExhaustedError`，details 里
    带逐家明细；调用方 `except ModelUnavailableError` 一定能接住。）
    """

    adapter = FakeAdapter(
        default=Outcome(error=ModelUnavailableError("503"))
    )

    observer = RecordingObserver()
    runtime = _runtime(adapter, observer=observer, max_retries=0)

    with pytest.raises(ModelUnavailableError) as excinfo:
        await runtime.ainvoke(_request())

    error = excinfo.value

    assert isinstance(error, ModelCandidatesExhaustedError)
    assert error.code == "exhausted"

    # 两家都试过（各一次，因为 max_retries=0）
    assert adapter.called_models() == ["fake:primary", "fake:backup"]

    # details 里带完整明细：不能只留一句"失败了"
    assert error.details["candidates"] == ["fake:primary", "fake:backup"]
    assert len(error.details["failures"]) == 2
    assert error.details["provider_attempts"] == 2

    # 观测记录的是失败，而不是"没发生过"
    assert len(observer.summaries) == 1
    assert observer.summaries[0].success is False
    assert observer.summaries[0].error_code == "exhausted"


async def test_provider_exception_never_escapes() -> None:
    """provider 异常**不会**原样冒到调用方 —— 永远是 Model 层异常。"""

    adapter = FakeAdapter(
        default=Outcome(error=fake_exception("SomeWeirdSdkError", message="???"))
    )

    runtime = _runtime(adapter, max_retries=0)

    with pytest.raises(ModelUnavailableError) as excinfo:
        await runtime.ainvoke(_request())

    # 抛出的是 Model 层异常，不是那个假 SDK 的类
    assert type(excinfo.value).__module__.startswith("app.agent.model")

    assert not isinstance(excinfo.value, type(
        fake_exception("SomeWeirdSdkError")
    ))


async def test_single_candidate_keeps_original_error() -> None:
    """
    单候选（没有 fallback）时抛**原始错误**，不包成"全部候选失败"。

    没有链可怪；包一层只会把"请求不合法"糊成"服务不可用"，而这两种
    错的修法完全不同。
    """

    adapter = FakeAdapter(
        default=Outcome(error=ModelInvalidRequestError("400 上下文超长"))
    )

    runtime = _runtime(adapter, fallbacks=())

    with pytest.raises(ModelInvalidRequestError) as excinfo:
        await runtime.ainvoke(_request())

    assert not isinstance(excinfo.value, ModelCandidatesExhaustedError)


async def test_fallback_disabled_only_calls_primary() -> None:
    """`fallback_enabled=False` 时备选根本不参与。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [Outcome(error=ModelUnavailableError("503"))],
            "fake:backup": [Outcome(message="不该被调到")],
        }
    )

    runtime = _runtime(adapter, fallback_enabled=False, max_retries=0)

    with pytest.raises(ModelUnavailableError):
        await runtime.ainvoke(_request())

    assert adapter.called_models() == ["fake:primary"]


async def test_candidate_chain_order_is_preserved() -> None:
    """降级顺序就是 profile 里写的顺序（不是字典序、不是随机的）。"""

    adapter = FakeAdapter()

    registry = make_registry(
        models=[
            make_definition("a"),
            make_definition("b"),
            make_definition("c"),
        ],
        profiles=[
            ModelProfile(
                id="p",
                primary="fake:c",
                fallbacks=("fake:a", "fake:b"),
                policy=ModelRuntimePolicy(),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react", model_role=ModelRole.REASONING, profile="p"
            )
        ],
        adapter=adapter,
    )

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    assert [item.id for item in runtime.candidates()] == [
        "fake:c",
        "fake:a",
        "fake:b",
    ]


async def test_timeout_error_from_fallback_chain_is_reported() -> None:
    """整条链都是超时 -> 统一的不可用错误（而不是某个 provider 的异常）。"""

    adapter = FakeAdapter(
        default=Outcome(error=ModelTimeoutError("超时"))
    )

    runtime = _runtime(adapter, max_retries=0)

    with pytest.raises(ModelUnavailableError) as excinfo:
        await runtime.ainvoke(_request())

    assert isinstance(excinfo.value, ModelCandidatesExhaustedError)
    assert "timeout" in str(excinfo.value.details["last_error"])
