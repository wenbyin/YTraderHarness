"""
Model 模块的异常层级，以及 **provider 异常归一化**。

错误处理链（规格 §18）
---------------------
    Provider SDK Exception
            ↓  Provider Adapter（尽量在这里归一化）
    Model Exception
            ↓  Model Runtime（据它决定 retry / fallback / fail）
    LLM Node

因此有一条硬规则：

    LLM Node 里**不允许**出现 `except openai.RateLimitError`

它只可能看到本模块的异常。这条规则由两个机制保证：适配器自己捕获
provider 异常并归一化；Runtime 在调用点还有一层兜底网（`normalize_error`），
任何漏网的异常进来也会变成 Model 层异常，而不是原样冒到节点上。

为什么用**类名 + 状态码**识别，而不是 import openai / anthropic
--------------------------------------------------------------
三个理由，任一条都足够：

    1. 那会把 Model 模块与具体 SDK 绑死 —— 本模块刻意只依赖 langchain
       的**调用约定**（bind_tools / ainvoke / usage_metadata），不 import
       任何 provider 包。
    2. 只装了 openai 而没装 anthropic 的环境里，`import anthropic` 会炸
       —— 于是"归一化"这件事本身变成新的失败点。
    3. 各家 SDK 的异常类型在版本间改名（openai 1.x -> 3.x 就动过），
       状态码却稳定得多。

因此 `normalize_error` 依次尝试：已是 Model 异常 -> 内建异常 ->
类名表 -> 状态码启发式 -> 兜底 `ModelProviderError`。**兜底一定是
Model 层异常**，所以"provider 异常泄漏到上层"在结构上不可能发生。
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, ClassVar, Mapping

from .redaction import redact_mapping, redact_text

__all__ = [
    "DEFAULT_RETRYABLE_ERRORS",
    "ErrorDisposition",
    "ModelAuthenticationError",
    "ModelCandidateFailure",
    "ModelCandidatesExhaustedError",
    "ModelCapabilityError",
    "ModelConfigurationError",
    "ModelError",
    "ModelInvalidRequestError",
    "ModelInvocationError",
    "ModelProviderError",
    "ModelRateLimitError",
    "ModelResolutionError",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "normalize_error",
]


class ErrorDisposition(str, Enum):
    """
    一个错误该怎么处理 —— Runtime 只认这三个动作。

        RETRY      同一个候选模型再试（有次数上限）
        FALLBACK   不再试这个候选，换下一个（重试它没有意义）
        FAIL       整条链停在这里，直接向上抛

    它是**类别**的属性（在类上声明），不是运行时的 if-else 树：
    新增一种错误时，"该 retry 还是该 fallback"必须显式写在类上，
    而不是散在 Runtime 的分支里 —— 后者会让"新错误悄悄落进 else
    分支被重试三次"这种事发生。
    """

    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL = "fail"


class ModelError(Exception):
    """
    本模块所有异常的基类。

    三个字段贯穿始终，且**都进观测**（见 as_metadata）：

        provider / model   是谁失败的（跨 provider 的 fallback 链里，
                           不写清楚就查不出"到底哪一家挂了"）
        details            结构化补充（状态码、尝试次数、候选链…）

    ⚠️ 消息与 details 在构造时就**脱敏**：provider SDK 会把 key 写进
    报错文本（`Incorrect API key provided: sk-…`），而这个消息会进日志
    与观测行。见 redaction.py。
    """

    #: 稳定错误码。它是 `ModelRuntimePolicy.retryable_errors` 的取值，
    #: 也进观测行 —— 因此**不要**跟着类名改。
    code: ClassVar[str] = "model_error"

    #: 该错误默认该怎么处理。见 ErrorDisposition。
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.FAIL

    def __init__(
            self,
            message: str,
            *,
            provider: str | None = None,
            model: str | None = None,
            details: Mapping[str, Any] | None = None,
    ) -> None:

        self.message = redact_text(str(message))

        super().__init__(self.message)

        self.provider = provider
        self.model = model
        self.details: dict[str, Any] = dict(
            redact_mapping(dict(details or {}))
        )

    # --------------------------------------------------------

    @property
    def qualified_model(self) -> str | None:
        """`provider:model`；缺一个就返回有的那个，都没有则 None。"""

        if self.provider and self.model:
            return f"{self.provider}:{self.model}"

        return self.provider or self.model

    def as_metadata(self) -> dict[str, Any]:
        """给观测用的一份快照。**已脱敏**。"""

        return {
            "error_type": type(self).__name__,
            "error_code": self.code,
            "error_message": self.message,
            "provider": self.provider,
            "model": self.model,
            "details": dict(self.details),
        }

    def __str__(self) -> str:

        if self.qualified_model:
            return f"{self.message}（{self.qualified_model}）"

        return self.message


# ============================================================
# 配置类：调用之前就该发现的错
# ============================================================


class ModelConfigurationError(ModelError):
    """
    注册 / 解析 / 客户端构造阶段的错。

    **不进 retry / fallback**：这类错换一家 provider 也一样（模型没
    注册、provider 没注册、凭据缺失都是配置事实），重试只是把同一个
    失败重复几次并把真正的原因盖掉。
    """

    code: ClassVar[str] = "configuration"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.FAIL


class ModelResolutionError(ModelConfigurationError):
    """
    `(agent_type, model_role)` 解析不出 Model Profile。

    它是**接线错误**：Agent Builder / JEV 决定用哪张图、哪个角色，而
    Registry 里没有对应的绑定。静默回落到某个"默认模型"会让一次
    配置遗漏表现为"模型答得不太对"，几乎查不出来。
    """

    code: ClassVar[str] = "resolution"


class ModelCapabilityError(ModelConfigurationError):
    """
    候选模型不满足本次调用要求的能力（见 ModelCapabilities）。

    两种触发点：
        primary 不满足        -> 直接抛（这是配置错，不该降级）
        fallback 不满足       -> **不抛**，在候选链里被剔除并记一条
                                 警告（见 runtime.py），因为"某个
                                 fallback 不够格"是常见且可接受的
    """

    code: ClassVar[str] = "capability"


# ============================================================
# 调用类：一次 provider attempt 的失败
# ============================================================


class ModelInvocationError(ModelError):
    """一次 provider attempt 失败的基类。"""

    code: ClassVar[str] = "invocation"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.FAIL


class ModelTimeoutError(ModelInvocationError):
    """这次调用超时。**可重试** —— 超时是典型的暂时性失败。"""

    code: ClassVar[str] = "timeout"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.RETRY


class ModelRateLimitError(ModelInvocationError):
    """被限流。**可重试**（Runtime 会按退避等待）。"""

    code: ClassVar[str] = "rate_limit"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.RETRY


class ModelUnavailableError(ModelInvocationError):
    """
    provider 暂时不可用（连接失败 / 5xx / 服务过载）。

    ⚠️ 它同时是**整条候选链耗尽之后**对外抛出的那个错误（见
    ModelCandidatesExhaustedError）。规格要求"所有 provider 都失败时
    返回统一的 ModelUnavailableError"，而不是把最后那家的异常抛出去
    —— 最后一家的异常只说明"最后试的那家怎么了"，不说明整体结果。
    """

    code: ClassVar[str] = "unavailable"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.RETRY


class ModelAuthenticationError(ModelInvocationError):
    """
    凭据被拒（401 / 403 / invalid api key）。

    disposition 是 **FALLBACK 而不是 RETRY**：同一把 key 再试一次
    还是会被拒，但**换一家**常常就好了（key 是按 provider 配的，
    只有一家的 key 过期是常态）。
    """

    code: ClassVar[str] = "authentication"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.FALLBACK


class ModelInvalidRequestError(ModelInvocationError):
    """
    请求本身不合法（400 / 404 / 422、上下文超长、参数不被接受）。

    disposition 是 **FAIL**：换一家 provider 也会以同样的理由拒绝
    —— 请求是错的。继续 fallback 只会把同一个错误在别家再花一次钱，
    并且把"我们的请求有 bug"这件事伪装成"模型不稳定"。
    """

    code: ClassVar[str] = "invalid_request"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.FAIL


class ModelProviderError(ModelInvocationError):
    """
    认得是 provider 的问题、但归不到上面任何一类的兜底。

    可重试：确实存在"provider 偶发抽风"这种情况。它也是
    `normalize_error` 的最后一条出口 —— 因此**没有**任何 provider
    异常能原样冒到上层。
    """

    code: ClassVar[str] = "provider_error"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.RETRY


class ModelCandidateFailure(ModelError):
    """
    一个候选模型的最终失败记录（供耗尽错误与观测引用，不会单独抛出）。

    它不是"又一种错误"，而是**证据**：哪几家、各试了几次、分别以
    什么错误告终。ModelCandidatesExhaustedError 把它列进 details。
    """

    code: ClassVar[str] = "candidate_failure"
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.FAIL


class ModelCandidatesExhaustedError(ModelUnavailableError):
    """
    候选链里的每一个模型都失败了 —— 多候选场景对外的**唯一**终态。

    ⚠️ 继承 `ModelUnavailableError` 是刻意的：规格要求"所有 provider
    失败 -> 统一的 ModelUnavailableError"，调用方（或上层的通用兜底）
    `except ModelUnavailableError` 必须能接住它；而 details 里带着
    逐家的失败明细，排查时不缺信息。

    单候选（没有 fallback）时**不**用它 —— 那种情况下原始错误更精确，
    包一层只会把"请求不合法"糊成"服务不可用"。见 runtime.py。
    """

    code: ClassVar[str] = "exhausted"

    #: ⚠️ 显式声明 FAIL，**不继承**父类的 RETRY。
    #:
    #: 这个错误一抛出就说明候选链已经走完 —— 没有任何东西可重试。
    #: 继承 RETRY 还会让 `_assert_retryable_codes_consistent` 认为
    #: "exhausted 是个该进重试码表的错误"，而那恰恰是错的。
    disposition: ClassVar[ErrorDisposition] = ErrorDisposition.FAIL


# ============================================================
# 归一化
# ============================================================


#: 类名 -> Model 层异常。**不 import 任何 provider SDK**，按名字认。
#:
#: 覆盖 openai / anthropic / httpx / google-genai 四家的公开异常名。
#: 认不出来时还有状态码启发式与兜底，因此这张表"不全"不会导致泄漏，
#: 只会让分类粗一点（落到 ModelProviderError，可重试）。
#:
#: ⚠️ 只按**类名**查（含 MRO 里的父类名），不按模块路径 —— 各家的
#: 异常常常是继承出来的，父类的名字反而更能说明性质（比如
#: `APITimeoutError` 继承 `APIConnectionError`，这里要取前者）。
_NAME_TABLE: Mapping[str, type[ModelError]] = {
    # ---- 超时 ----
    "APITimeoutError": ModelTimeoutError,
    "Timeout": ModelTimeoutError,
    "TimeoutException": ModelTimeoutError,
    "ReadTimeout": ModelTimeoutError,
    "ConnectTimeout": ModelTimeoutError,
    "PoolTimeout": ModelTimeoutError,
    # ---- 限流 ----
    "RateLimitError": ModelRateLimitError,
    "RateLimitException": ModelRateLimitError,
    # ---- 连不上 / 服务端 ----
    "APIConnectionError": ModelUnavailableError,
    "ConnectError": ModelUnavailableError,
    "RemoteProtocolError": ModelUnavailableError,
    "ServiceUnavailableError": ModelUnavailableError,
    "InternalServerError": ModelProviderError,
    "APIError": ModelProviderError,
    "OverloadedError": ModelProviderError,
    # ---- 认证 ----
    "AuthenticationError": ModelAuthenticationError,
    "PermissionDeniedError": ModelAuthenticationError,
    "UnauthorizedError": ModelAuthenticationError,
    # ---- 请求不合法 ----
    "BadRequestError": ModelInvalidRequestError,
    "UnprocessableEntityError": ModelInvalidRequestError,
    "NotFoundError": ModelInvalidRequestError,
    "ContextLengthExceededError": ModelInvalidRequestError,
}

#: `ModelRuntimePolicy.retryable_errors` 的默认取值。
#:
#: 只列**确实值得重试**的那几类。刻意不含 authentication（换 key 才有用）
#: 与 invalid_request / capability（请求或配置是错的）。
#:
#: 下面有一段 import 期自检，保证它与各类的 disposition 不漂移。
DEFAULT_RETRYABLE_ERRORS: tuple[str, ...] = (
    "timeout",
    "rate_limit",
    "unavailable",
    "provider_error",
)


def _assert_retryable_codes_consistent() -> None:
    """
    import 期自检：默认重试码表必须与错误类的 disposition 一致。

    漂移的后果是**静默的行为变化**：有人把某个错误的 disposition 从
    RETRY 改成 FALLBACK，却忘了从这张表里删掉它 —— 于是 Runtime 里
    "要不要重试"的两个判据打架，而实际行为取决于哪个先被读到。
    那是最难查的一类 bug，所以在这里就炸掉。

    （与 context/builder/contracts.py 的 `_assert_template_complete`
    是同一种做法、同一个理由。）
    """

    retry_classes = {
        cls.code: name
        for name, cls in globals().items()
        if isinstance(cls, type)
        and issubclass(cls, ModelError)
        and cls.__module__ == __name__
        and cls.disposition is ErrorDisposition.RETRY
    }

    for code in DEFAULT_RETRYABLE_ERRORS:

        if code not in retry_classes:
            raise RuntimeError(
                f"DEFAULT_RETRYABLE_ERRORS 里的 {code!r} 不对应任何一个"
                " disposition=RETRY 的错误类。\n"
                "它要么是拼错了，要么是那个类的 disposition 被改过 ——"
                "两者都会让 Runtime 的重试判据自相矛盾。"
            )

    stray = set(retry_classes) - set(DEFAULT_RETRYABLE_ERRORS)

    if stray:
        raise RuntimeError(
            f"这些错误类声明了 disposition=RETRY，却不在"
            f" DEFAULT_RETRYABLE_ERRORS 里：{sorted(stray)}。\n"
            "策略表是 Runtime 判断“值不值得重试”的依据：漏了它们，"
            "那些错误就永远不会被重试（表现为“明明可重试却没重试”）。"
        )


_assert_retryable_codes_consistent()


# ------------------------------------------------------------
# 状态码启发式
# ------------------------------------------------------------

#: 这些状态码是**暂时性**的（RFC 里的 retry 类）。
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: 这些状态码说明**请求或凭据**有问题，换 provider 也一样（或需要换 key）。
_AUTH_STATUS = frozenset({401, 403})
_INVALID_STATUS = frozenset({400, 404, 405, 413, 415, 422})


def normalize_error(
        exc: BaseException,
        *,
        provider: str | None = None,
        model: str | None = None,
) -> ModelError:
    """
    把任意异常翻译成 Model 层异常。

    ⚠️ **永不抛新异常**：认不出来的一律变成 `ModelProviderError`。
    这条性质是"provider 异常不会泄漏到 LLM Node"的结构性保证 ——
    它不依赖那张类名表有多全。

    已经是 Model 异常的原样返回（但补上 provider/model 上下文：同一个
    错误在链里可能来自不同候选，谁失败的必须写清楚）。
    """

    if isinstance(exc, ModelError):

        if exc.provider is None:
            exc.provider = provider

        if exc.model is None:
            exc.model = model

        return exc

    classified = _classify(exc)

    error = classified(
        _describe(exc),
        provider=provider,
        model=model,
        details={
            "provider_error_type": type(exc).__name__,
            "provider_error_module": type(exc).__module__,
        },
    )

    # 保留原始异常：归一化之后，排查者仍然要能顺着异常链一路追到
    # provider SDK 那一层（否则"到底是谁抛的"就只剩一个类名）。
    #
    # 手工设 __cause__ / __suppress_context__ 与 `raise X from Y` 等价 ——
    # 这里不能写 raise，因为本函数只负责**构造**，抛出时机由 Runtime 定。
    error.__cause__ = exc
    error.__suppress_context__ = True

    return error


# ============================================================
# internals
# ============================================================


def _classify(exc: BaseException) -> type[ModelError]:
    """按内建类型 -> 类名 -> 状态码的顺序定出该用哪个 Model 异常。"""

    # ---- 1. 内建 ------------------------------------------
    #
    # asyncio.TimeoutError 在 3.11+ 就是内建 TimeoutError，这里一并
    # 写上是为了让 3.10 上的语义也清楚。
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return ModelTimeoutError

    if isinstance(exc, ConnectionError):
        return ModelUnavailableError

    # ---- 2. 类名（含基类名） -------------------------------

    for klass in type(exc).__mro__:

        mapped = _NAME_TABLE.get(klass.__name__)

        if mapped is not None:
            return mapped

    # ---- 3. 状态码 ----------------------------------------

    status = _status_code(exc)

    if status is not None:

        if status in _AUTH_STATUS:
            return ModelAuthenticationError

        if status in _INVALID_STATUS:
            return ModelInvalidRequestError

        if status == 429:
            return ModelRateLimitError

        if status in _RETRYABLE_STATUS:
            return ModelUnavailableError

        if 500 <= status < 600:
            return ModelProviderError

    # ---- 4. 兜底 ------------------------------------------
    #
    # 到这里的都是"认得出是 provider 侧的错、但归不了类"。给可重试的
    # 兜底而不是 FAIL：一个未知的暂时性故障不该让整轮对话直接失败。
    return ModelProviderError


def _status_code(exc: BaseException) -> int | None:
    """
    从异常上取 HTTP 状态码。

    各家叫法不同（openai: `status_code`，httpx: `response.status_code`，
    anthropic: `status_code`），因此按几种常见形态依次摸一遍。
    """

    for attribute in ("status_code", "http_status", "status"):

        value = getattr(exc, attribute, None)

        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)

    if response is not None:

        value = getattr(response, "status_code", None)

        if isinstance(value, int):
            return value

    return None


def _describe(exc: BaseException) -> str:
    """给归一化后的异常写一句人能读懂的话。"""

    return f"{type(exc).__name__}: {exc}"
