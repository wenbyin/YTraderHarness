"""
密钥不泄漏（规格 §22 最后一条）。

日志里不能出现：api_key / authorization / secret / credential 的**值**。

⚠️ 为什么这条要用"真实调用 + 捕获日志"来测
------------------------------------------
泄漏的路径不是某个人手写的 log 语句，而是：

    异常消息里的 key（provider SDK 会把它写进报错文本）
    metadata 里"顺手带上"的 key
    definition.config 里写死的 key

三处都只有真的跑一次、真的记一次日志才覆盖得到。因此这里的断言是
"**整段日志文本里找不到那些值**"，而不是"某个函数返回了 ***"。
"""

from __future__ import annotations

import json
import logging

import pytest

from fakes import FakeAdapter, Outcome, fake_exception, make_definition, make_registry

from app.agent.model import (
    LiteralCredentialResolver,
    LoggingModelObserver,
    ModelProfile,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
    ModelRuntimePolicy,
    ModelUsage,
    redact_mapping,
    redact_text,
)


SECRET = "sk-super-secret-0123456789abcdef"
BEARER = "Bearer abcdef1234567890"
PLAIN_SECRET = "s3cr3t-value"

#: 这些值**必须**在日志里消失。
SECRETS = (SECRET, BEARER, PLAIN_SECRET)


def _registry(
        adapter,
        *,
        config=None,
        credential_ref=None,
        credentials=None,
        observer=None,
):
    registry = make_registry(
        models=[
            make_definition(
                "primary",
                config=config or {},
                credential_ref=credential_ref,
            )
        ],
        profiles=[
            ModelProfile(
                id="p",
                primary="fake:primary",
                policy=ModelRuntimePolicy(max_retries=0, retry_backoff=0.0),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react", model_role=ModelRole.REASONING, profile="p"
            )
        ],
        adapter=adapter,
        credentials=credentials,
        observer=observer,
    )

    return registry


def _runtime(registry):
    return registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )


def _log_text(caplog) -> str:
    """把捕获到的日志拼成一段文本（消息 + 参数 + 异常说明）。"""

    parts: list[str] = []

    for record in caplog.records:

        parts.append(record.getMessage())
        parts.append(repr(record.args))
        parts.append(record.exc_text or "")

    return "\n".join(parts)


# ============================================================
# 脱敏规则本身
# ============================================================


def test_redaction_keeps_observability_fields() -> None:
    """
    敏感键被抹掉，但**观测字段不受影响**。

    "见到 token 就抹"会顺手把 input_tokens / total_tokens 也毁掉 ——
    而它们是成本数据，那是最不该丢的东西。
    """

    redacted = redact_mapping({
        "api_key": SECRET,
        "openai_api_key": SECRET,
        "authorization": BEARER,
        "secret": PLAIN_SECRET,
        "client_secret": PLAIN_SECRET,
        "credential": PLAIN_SECRET,
        "password": PLAIN_SECRET,
        "access_token": PLAIN_SECRET,
        # 下面这些是**观测字段 / 名字**，不该被抹
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "max_tokens": 4096,
        "credential_ref": "deepseek-production",
        "token_count": 7,
    })

    for key in (
        "api_key",
        "openai_api_key",
        "authorization",
        "secret",
        "client_secret",
        "credential",
        "password",
        "access_token",
    ):
        assert redacted[key] == "***", key

    assert redacted["input_tokens"] == 10
    assert redacted["output_tokens"] == 20
    assert redacted["total_tokens"] == 30
    assert redacted["max_tokens"] == 4096
    assert redacted["token_count"] == 7

    # ⚠️ credential_ref 是**名字**，不是秘密 —— 抹掉它只会让人查不出
    # "这次用的是哪份凭据"。
    assert redacted["credential_ref"] == "deepseek-production"


def test_redaction_of_text_patterns() -> None:

    assert SECRET not in redact_text(f"Invalid API key provided: {SECRET}")
    assert "abcdef1234567890" not in redact_text(f"Authorization: {BEARER}")
    assert PLAIN_SECRET not in redact_text(f"api_key={PLAIN_SECRET}")

    # 正常文本原样保留
    assert redact_text("total_tokens: 120") == "total_tokens: 120"


def test_redaction_is_not_in_place() -> None:
    """脱敏不能就地改调用方还要用的那份数据。"""

    source = {"api_key": SECRET, "nested": {"secret": PLAIN_SECRET}}

    redact_mapping(source)

    assert source["api_key"] == SECRET
    assert source["nested"]["secret"] == PLAIN_SECRET


# ============================================================
# 真实调用链
# ============================================================


async def test_no_secret_in_logs_on_success(caplog) -> None:
    """成功路径：配置里有 key、metadata 里有 key —— 日志里都不许出现。"""

    caplog.set_level(logging.DEBUG)

    adapter = FakeAdapter(
        default=Outcome(
            message="ok",
            usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        )
    )

    registry = _registry(
        adapter,
        config={"api_key": SECRET},
        credential_ref="test-ref",
        credentials=LiteralCredentialResolver({"test-ref": SECRET}),
        observer=LoggingModelObserver(),
    )

    response = await _runtime(registry).ainvoke(
        ModelRequest(
            messages=("你好",),
            metadata={
                "api_key": SECRET,
                "authorization": BEARER,
                "secret": PLAIN_SECRET,
                "credential": PLAIN_SECRET,
                "input_tokens": 10,
                "trace_id": "t-1",
            },
        )
    )

    log_text = _log_text(caplog)

    for secret in SECRETS:
        assert secret not in log_text, f"日志里出现了密钥：{secret!r}"

    # 但观测字段仍在 —— 脱敏不能把可观测性一起抹掉
    assert "total_tokens" in log_text or "tokens=3" in log_text


async def test_secret_reaching_the_adapter_is_expected(caplog) -> None:
    """
    凭据**应当**到达适配器（它就是去建客户端的），但**不该**进日志。

    这两件事必须分开断言：前者是功能，后者是安全。
    """

    caplog.set_level(logging.DEBUG)

    adapter = FakeAdapter(script={"fake:primary": [Outcome(message="ok")]})

    registry = _registry(
        adapter,
        credential_ref="test-ref",
        credentials=LiteralCredentialResolver({"test-ref": SECRET}),
        observer=LoggingModelObserver(),
    )

    await _runtime(registry).ainvoke(ModelRequest(messages=("你好",)))

    # 功能：适配器拿到了秘密（否则它没法建客户端）
    assert adapter.created[0]["credential"] == SECRET

    # 安全：日志里没有
    assert SECRET not in _log_text(caplog)

    # 安全：缓存的键里也没有（缓存键会进调试器 / repr）
    assert SECRET not in repr(list(registry.factory._clients.keys()))


async def test_response_metadata_is_redacted() -> None:
    """响应 metadata 会进节点观测并落库 —— 那里同样不能有秘密。"""

    adapter = FakeAdapter(script={"fake:primary": [Outcome(message="ok")]})

    registry = _registry(adapter)

    response = await _runtime(registry).ainvoke(
        ModelRequest(
            messages=("你好",),
            metadata={"api_key": SECRET, "input_tokens": 10},
        )
    )

    request_metadata = response.metadata["request_metadata"]

    assert request_metadata["api_key"] == "***"
    assert request_metadata["input_tokens"] == 10

    # 整个 metadata 必须可 JSON 序列化（它要写进 JSONB 观测行）
    assert SECRET not in json.dumps(response.metadata, ensure_ascii=False)


async def test_provider_error_message_is_redacted(caplog) -> None:
    """
    provider 把 key 写进报错文本 —— 那条路径也要被堵住。

    `Incorrect API key provided: sk-...` 是真实发生过的事，而异常消息
    会进日志、进观测行的 error_message。
    """

    caplog.set_level(logging.DEBUG)

    adapter = FakeAdapter(
        default=Outcome(
            error=fake_exception(
                "AuthenticationError",
                message=f"Incorrect API key provided: {SECRET}",
            )
        )
    )

    registry = _registry(adapter, observer=LoggingModelObserver())

    from app.agent.model import ModelAuthenticationError

    with pytest.raises(ModelAuthenticationError) as excinfo:
        await _runtime(registry).ainvoke(ModelRequest(messages=("你好",)))

    # 异常消息
    assert SECRET not in str(excinfo.value)

    # 观测 metadata（落库的那份）
    assert SECRET not in json.dumps(
        excinfo.value.as_metadata(), ensure_ascii=False
    )

    # 日志
    assert SECRET not in _log_text(caplog)


def test_literal_resolver_repr_hides_values() -> None:
    """pytest 失败时打印 fixture / 调试器里 repr 一个对象是很常见的。"""

    resolver = LiteralCredentialResolver({"a": SECRET})

    assert SECRET not in repr(resolver)
    assert "a" in repr(resolver)


def test_definition_describe_redacts_config() -> None:
    """目录接口会输出 config —— 里面若有 key，输出必须是 ***。"""

    definition = make_definition("primary", config={"api_key": SECRET, "temperature": 0.2})

    described = definition.describe()

    assert described["config"]["api_key"] == "***"
    assert described["config"]["temperature"] == 0.2
    assert SECRET not in json.dumps(described, ensure_ascii=False)


def test_registry_describe_has_no_secret() -> None:
    """整份注册表摘要里也不许出现秘密。"""

    adapter = FakeAdapter()

    registry = _registry(
        adapter,
        config={"api_key": SECRET},
        credential_ref="deepseek-production",
    )

    blob = json.dumps(registry.describe(), ensure_ascii=False)

    assert SECRET not in blob
    assert "deepseek-production" in blob  # 名字要留着，否则查不出用了哪份凭据
