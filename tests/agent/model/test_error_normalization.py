"""
Provider 异常归一化（规格 §18、§22 第六条）。

    Provider SDK Exception
            ↓  Provider Adapter
    Model Exception
            ↓  Model Runtime
    LLM Node

两条断言贯穿全部用例：

    1. 分类**正确**（超时归超时、限流归限流…）
    2. 出口**一定是 Model 层异常** —— 兜底也必须是，认不出来不等于
       让它原样冒到上层

⚠️ 这些用例不 import openai / anthropic
------------------------------------
归一化是按**类名 + 状态码**做的，因此造一个"叫 RateLimitError 的异常"
就足以驱动它。这也正是那套设计的好处：不强依赖任何 provider SDK，
所以"没装 anthropic 的环境里归一化本身变成新的失败点"这件事不可能发生。
"""

from __future__ import annotations

import asyncio

import pytest

from fakes import FakeAdapter, Outcome, fake_exception, make_definition, make_registry

from app.agent.model import (
    ModelAuthenticationError,
    ModelCapabilityError,
    ModelError,
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
    ErrorDisposition,
    normalize_error,
)


# ============================================================
# 内建异常
# ============================================================


def test_builtin_timeout_maps_to_model_timeout() -> None:

    assert isinstance(normalize_error(TimeoutError("慢")), ModelTimeoutError)
    assert isinstance(
        normalize_error(asyncio.TimeoutError("慢")), ModelTimeoutError
    )


def test_builtin_connection_error_maps_to_unavailable() -> None:

    assert isinstance(
        normalize_error(ConnectionError("断了")), ModelUnavailableError
    )


# ============================================================
# 类名表（各家 SDK 的公开异常名）
# ============================================================


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("APITimeoutError", ModelTimeoutError),
        ("Timeout", ModelTimeoutError),
        ("ReadTimeout", ModelTimeoutError),
        ("RateLimitError", ModelRateLimitError),
        ("APIConnectionError", ModelUnavailableError),
        ("ConnectError", ModelUnavailableError),
        ("AuthenticationError", ModelAuthenticationError),
        ("PermissionDeniedError", ModelAuthenticationError),
        ("BadRequestError", ModelInvalidRequestError),
        ("UnprocessableEntityError", ModelInvalidRequestError),
        ("InternalServerError", ModelProviderError),
        ("APIError", ModelProviderError),
    ],
)
def test_sdk_exception_names_are_classified(name, expected) -> None:

    error = normalize_error(fake_exception(name, message="boom"))

    assert type(error) is expected
    assert error.provider is None  # 调用方没给上下文时就是 None


def test_subclass_name_is_used_via_mro() -> None:
    """真实 SDK 的异常常常是继承出来的 —— 父类名同样要被认出来。"""

    base = type("APIConnectionError", (Exception,), {})
    child = type("SpecializedConnectionError", (base,), {})

    assert isinstance(normalize_error(child("x")), ModelUnavailableError)


# ============================================================
# 状态码启发式
# ============================================================


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ModelAuthenticationError),
        (403, ModelAuthenticationError),
        (400, ModelInvalidRequestError),
        (404, ModelInvalidRequestError),
        (422, ModelInvalidRequestError),
        (429, ModelRateLimitError),
        (500, ModelUnavailableError),
        (502, ModelUnavailableError),
        (503, ModelUnavailableError),
        (529, ModelUnavailableError),
    ],
)
def test_status_code_heuristic(status, expected) -> None:
    """认不出类名时，状态码是第二判据。"""

    error = normalize_error(
        fake_exception("WeirdProviderError", status_code=status)
    )

    assert type(error) is expected


def test_status_code_from_httpx_style_response() -> None:
    """httpx 把状态码挂在 `response` 上。"""

    class _Response:
        status_code = 503

    error = normalize_error(
        fake_exception("TransportError", response=_Response())
    )

    assert isinstance(error, ModelUnavailableError)


# ============================================================
# 兜底
# ============================================================


def test_unknown_error_becomes_retryable_provider_error() -> None:
    """
    认不出来 -> 可重试的 ModelProviderError。

    给可重试的兜底而不是 FAIL：一个未知的暂时性故障不该让整轮对话
    直接失败。
    """

    error = normalize_error(ValueError("谁知道这是什么"))

    assert isinstance(error, ModelProviderError)
    assert error.disposition is ErrorDisposition.RETRY

    # 原始类型保留在 details 里（排查时要知道底层到底抛了什么）
    assert error.details["provider_error_type"] == "ValueError"


def test_normalize_never_raises() -> None:
    """
    `normalize_error` **永不抛新异常** —— 这是"provider 异常不会泄漏"
    的结构性保证，不依赖类名表有多全。
    """

    for exc in (
        ValueError("x"),
        KeyError("y"),
        BaseException("z"),
        fake_exception("TotallyUnknown"),
    ):

        assert isinstance(normalize_error(exc), ModelError)


# ============================================================
# 已是 Model 层异常
# ============================================================


def test_model_error_passes_through_and_gains_context() -> None:
    """已经是 Model 异常的原样返回，但补上"是哪一家失败的"。"""

    original = ModelRateLimitError("429")

    result = normalize_error(original, provider="deepseek", model="chat")

    assert result is original
    assert result.provider == "deepseek"
    assert result.model == "chat"
    assert result.qualified_model == "deepseek:chat"


@pytest.mark.parametrize(
    ("klass", "code", "retried"),
    [
        (ModelTimeoutError, "timeout", True),
        (ModelRateLimitError, "rate_limit", True),
        (ModelUnavailableError, "unavailable", True),
        (ModelProviderError, "provider_error", True),
        (ModelAuthenticationError, "authentication", False),
        (ModelInvalidRequestError, "invalid_request", False),
        (ModelCapabilityError, "capability", False),
    ],
)
def test_dispositions_are_declared_on_the_class(klass, code, retried) -> None:

    assert klass.code == code

    disposition = klass.disposition

    assert isinstance(disposition, ErrorDisposition)

    if retried:
        assert disposition is ErrorDisposition.RETRY
    else:
        assert disposition is not ErrorDisposition.RETRY


def test_retryable_error_table_matches_dispositions() -> None:
    """
    默认重试码表与错误类的 disposition 必须一致。

    （import 期已有一道自检；这条用例是它的回归护栏 —— 有人改了
    disposition 却忘了改表时，这里会红。）
    """

    from app.agent.model import DEFAULT_RETRYABLE_ERRORS

    for code in DEFAULT_RETRYABLE_ERRORS:

        assert code != "authentication"
        assert code != "invalid_request"


# ============================================================
# 端到端：provider 异常不穿透 Runtime
# ============================================================


async def test_runtime_never_leaks_provider_exception() -> None:
    """
    适配器抛 provider 异常 -> Runtime 抛 Model 异常。

    这是 LLM Node 里可以写 `except ModelError` 而不必写
    `except openai.RateLimitError` 的原因。
    """

    sdk_error = fake_exception("RateLimitError", message="429 too many")

    adapter = FakeAdapter(default=Outcome(error=sdk_error))

    registry = make_registry(
        models=[make_definition("primary")],
        profiles=[
            ModelProfile(
                id="p",
                primary="fake:primary",
                policy=ModelRuntimePolicy(max_retries=0),
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

    with pytest.raises(ModelError) as excinfo:
        await runtime.ainvoke(ModelRequest(messages=("你好",)))

    assert type(excinfo.value) is ModelRateLimitError
    assert not isinstance(excinfo.value, type(sdk_error))

    # __cause__ 保留原始异常 —— 排查时能一路追到 SDK 那一层
    assert excinfo.value.__cause__ is sdk_error


async def test_client_creation_failure_is_a_configuration_error() -> None:
    """
    建客户端失败（provider 没装 / key 缺失）-> 配置错，**不进重试**。

    归一化成可重试的 provider 错误会把同一个配置错重试三次，而真正
    的报错（"该设哪个环境变量"）会被埋掉。
    """

    class ExplodingAdapter(FakeAdapter):

        def create_client(self, definition, *, credential, timeout, max_retries):
            raise ValueError("provider 'nope' 不在 init_chat_model 的支持列表里")

    registry = make_registry(
        models=[make_definition("primary")],
        profiles=[ModelProfile(id="p", primary="fake:primary")],
        bindings=[
            ModelRoleBinding(
                agent_type="react", model_role=ModelRole.REASONING, profile="p"
            )
        ],
        adapter=ExplodingAdapter(),
    )

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    from app.agent.model import ModelConfigurationError

    with pytest.raises(ModelConfigurationError):
        await runtime.ainvoke(ModelRequest(messages=("你好",)))
