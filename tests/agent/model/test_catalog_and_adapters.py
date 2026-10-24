"""
目录装配与 provider 适配器（规格 §7、§17、§20）。

三块：

    目录      model.yaml 能装配出注册表；写错的配置在**启动期**报错
    适配器    真的转交 app/common/llm.py 建客户端（**不联网**），
              并把工具 / 结构化输出绑上去、把用量读出来、把异常归一化
    凭据      credential_ref -> 环境变量的两条解析路径

⚠️ 这里会构造**真实**的 langchain 客户端（ChatDeepSeek / ChatMoonshot）
--------------------------------------------------------------
构造不联网（SDK 只在第一次调用时才建连接），因此它是安全的集成检查：
它验证的是"model.yaml 的声明 + Factory + 适配器"这条链真的能产出可用
的客户端。真发请求的检查不在这里（那需要密钥与网络）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.model import (
    DEFAULT_CATALOG_PATH,
    EnvVarCredentialResolver,
    LangChainProviderAdapter,
    ModelConfigurationError,
    ModelDefinition,
    ModelPricing,
    ModelProfile,
    ProviderResult,
    build_model_registry,
    load_model_catalog,
)
from app.agent.model.credentials import CredentialResolver
from app.agent.model.provider import ModelClient


# ============================================================
# 目录
# ============================================================


def test_shipped_catalog_loads() -> None:
    """随代码提供的那份目录必须是自洽的（引用都解析得到）。"""

    catalog = load_model_catalog()

    assert catalog.source.endswith("model.yaml")
    assert {item.name for item in catalog.providers} == {
        "deepseek",
        "kimi",
        "qwen",
        "mock",
    }
    assert "deepseek:deepseek-v4-flash" in catalog.model_ids()
    assert {item.id for item in catalog.profiles} == {
        "reasoning_default",
        "fast_default",
        "evaluation_default",
        "classification_default",
        "mock_default",
    }

    # 核心绑定必须在
    bindings = {(item.agent_type, item.model_role) for item in catalog.bindings}

    assert ("react", "reasoning") in bindings


def test_shipped_catalog_has_no_wildcard_binding() -> None:
    """
    随代码提供的目录**刻意没有**兜底绑定。

    兜底会让"新加一个角色忘了配绑定 -> 静默用了 fast_default"，而症状
    是"模型答得不太对"。没有兜底时它会明确报错并列出已有绑定。
    """

    catalog = load_model_catalog()

    for binding in catalog.bindings:
        assert binding.agent_type != "*"
        assert binding.model_role != "*"


def test_shipped_catalog_credentials_hold_names_only() -> None:
    """credentials 段里只有**环境变量名**，没有秘密。"""

    catalog = load_model_catalog()

    for ref, env_name in catalog.credentials.items():

        assert env_name.isupper()
        assert env_name.isidentifier()

        # 值看起来像密钥就说明有人写错了地方
        assert not env_name.startswith("sk-")


def test_shipped_registry_assembles() -> None:
    """装配一份可用的注册表（这一步不建任何客户端）。"""

    registry = build_model_registry()

    assert [item.name for item in registry.providers()] == [
        "deepseek",
        "kimi",
        "mock",
        "qwen",
    ]

    runtime = registry.get_runtime(agent_type="react", model_role="reasoning")

    assert runtime.profile_id == "reasoning_default"
    assert [item.id for item in runtime.candidates()][0] == (
        "deepseek:deepseek-v4-flash"
    )


def test_default_catalog_path_points_at_app_config() -> None:
    """
    目录住在 `app/config/`，**不**在读它的那个模块旁边。

    这条断言在搬家前是反过来的（那时要求它跟代码同目录）。改判据而不是
    删用例：这条路径写错的后果很安静 —— 走到 `load_model_catalog()` 才报
    "文件不存在"，而那时离"路径少写了一层"已经很远。

    ⚠️ 同时钉住**不在 datasource/ 子目录里**：`app/config/datasource/` 是
    唯一被数据源加载器 glob 的目录，model.yaml 落进去会被当成数据源定义
    去解析（顶层没有 `type`，于是被 skip 掉，另加一条看不懂的 warning）。
    """

    assert DEFAULT_CATALOG_PATH.exists()

    assert DEFAULT_CATALOG_PATH.parent.name == "config"
    assert DEFAULT_CATALOG_PATH.parent.parent.name == "app"


# ------------------------------------------------------------
# 配置写错时在启动期报错
# ------------------------------------------------------------


def _write(tmp_path, text: str):

    path = tmp_path / "model.yaml"
    path.write_text(text, encoding="utf-8")

    return path


def test_unknown_top_level_key_is_rejected(tmp_path) -> None:
    """拼错一个顶层键的后果是整段静默失效 —— 所以报错。"""

    path = _write(tmp_path, "providers: {}\nmodles: []\n")

    with pytest.raises(ModelConfigurationError) as excinfo:
        load_model_catalog(path)

    assert "modles" in str(excinfo.value)


def test_model_referencing_unknown_provider_is_rejected(tmp_path) -> None:

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - provider: openai
    model: gpt-5
""",
    )

    with pytest.raises(ModelConfigurationError) as excinfo:
        load_model_catalog(path)

    assert "openai" in str(excinfo.value)


def test_unknown_capability_name_is_rejected(tmp_path) -> None:
    """能力名写错一个字母 -> 报错，不忽略。"""

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - provider: deepseek
    model: chat
    capabilities:
      tool_callingg: true
""",
    )

    with pytest.raises(ModelConfigurationError) as excinfo:
        load_model_catalog(path)

    assert "tool_callingg" in str(excinfo.value)


def test_profile_referencing_unknown_model_is_rejected(tmp_path) -> None:

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - provider: deepseek
    model: chat
profiles:
  p:
    primary: deepseek:ghost
""",
    )

    with pytest.raises(ModelConfigurationError) as excinfo:
        load_model_catalog(path)

    assert "deepseek:ghost" in str(excinfo.value)


def test_binding_referencing_unknown_profile_is_rejected(tmp_path) -> None:

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - provider: deepseek
    model: chat
profiles:
  p:
    primary: deepseek:chat
bindings:
  - agent_type: react
    model_role: reasoning
    profile: nope
""",
    )

    with pytest.raises(ModelConfigurationError) as excinfo:
        load_model_catalog(path)

    assert "nope" in str(excinfo.value)


def test_alias_to_unknown_target_is_rejected(tmp_path) -> None:

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - provider: deepseek
    model: chat
aliases:
  cheap: ghost
""",
    )

    with pytest.raises(ModelConfigurationError):
        load_model_catalog(path)


def test_policy_null_total_timeout_means_unlimited(tmp_path) -> None:
    """
    `total_timeout: null`（显式不限）与"没写这个键"（用默认）是两件事。
    """

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - provider: deepseek
    model: chat
profiles:
  unlimited:
    primary: deepseek:chat
    policy:
      total_timeout: null
  defaulted:
    primary: deepseek:chat
""",
    )

    catalog = load_model_catalog(path)

    by_id = {item.id: item for item in catalog.profiles}

    assert by_id["unlimited"].policy.total_timeout is None
    assert by_id["defaulted"].policy.total_timeout == 180.0


def test_model_id_can_be_overridden(tmp_path) -> None:

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - id: my-flash
    provider: deepseek
    model: deepseek-v4-flash
    aliases: [flash]
profiles:
  p:
    primary: flash
bindings:
  - agent_type: react
    model_role: reasoning
    profile: p
""",
    )

    catalog = load_model_catalog(path)

    assert catalog.models[0].id == "my-flash"
    assert catalog.models[0].model == "deepseek-v4-flash"
    assert catalog.profiles[0].primary == "flash"


# ============================================================
# 适配器
# ============================================================


class _FakeChatModel:
    """鸭子类型的 langchain 客户端（只实现适配器会用到的那些）。"""

    def __init__(self, **kwargs) -> None:

        self.kwargs = kwargs
        self.bound_tools = None
        self.output_format = None
        #: 传给 with_structured_output 的其余关键字（如 extra_body）。
        #: 真实 langchain 的那个方法收 **kwargs，鸭子类型要跟上，
        #: 否则适配器一传就 TypeError。
        self.structured_kwargs: dict = {}
        self.messages = None
        self.response = SimpleNamespace(content="ok", usage_metadata=None)
        self.error: BaseException | None = None

    def bind_tools(self, tools):

        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, fmt, **kwargs):

        self.output_format = fmt
        self.structured_kwargs = dict(kwargs)
        return self

    async def ainvoke(self, messages, config=None):

        self.messages = list(messages)

        if self.error is not None:
            raise self.error

        return self.response


def _client(model: _FakeChatModel, definition: ModelDefinition) -> ModelClient:

    return ModelClient(
        definition=definition,
        client=model,
        adapter=LangChainProviderAdapter(
            chat_model_factory=lambda **kwargs: model
        ),
        timeout=5.0,
    )


async def test_adapter_binds_tools_and_response_format() -> None:
    """有工具才 bind_tools；有 response_format 才绑结构化输出。"""

    adapter = LangChainProviderAdapter()

    model = _FakeChatModel()

    client = _client(model, ModelDefinition.make("deepseek", "chat"))

    from app.agent.model import ModelRequest

    tool = object()
    fmt = object()

    await adapter.ainvoke(
        client,
        ModelRequest(messages=("你好",), tools=(tool,), response_format=fmt),
        timeout=5.0,
    )

    assert model.bound_tools == [tool]
    assert model.output_format is fmt
    assert model.messages == ["你好"]


async def test_adapter_does_not_bind_tools_when_absent() -> None:
    """"这次没有工具"是完全正常的请求 —— 不该调 bind_tools。"""

    adapter = LangChainProviderAdapter()

    model = _FakeChatModel()

    client = ModelClient(
        definition=ModelDefinition.make("deepseek", "chat"),
        client=model,
        adapter=adapter,
    )

    from app.agent.model import ModelRequest

    await adapter.ainvoke(client, ModelRequest(messages=("你好",)), timeout=5.0)

    assert model.bound_tools is None
    assert model.output_format is None


def test_thinking_off_extra_body_precedence() -> None:
    """
    ReAct 主循环默认保留思考；JSON schema / 显式关闭仍然优先。
    """

    from dataclasses import replace

    from app.agent.model import ModelRequest
    from app.agent.model.adapters import thinking_off_extra_body

    definition = replace(
        ModelDefinition.make("deepseek", "chat"),
        structured_output_extra_body={"thinking": {"type": "disabled"}},
    )

    tool = object()

    # 主循环：带 tools，但显式要求思考 -> 不关。
    assert thinking_off_extra_body(
        definition,
        ModelRequest(
            messages=(),
            tools=(tool,),
            enable_thinking=True,
        ),
    ) == {}

    # JSON schema 优先：即使 enable_thinking=True 也关。
    assert thinking_off_extra_body(
        definition,
        ModelRequest(
            messages=(),
            tools=(tool,),
            response_format=object(),
            enable_thinking=True,
        ),
    ) == {"extra_body": {"thinking": {"type": "disabled"}}}

    # 普通工具调用（没有 enable_thinking）-> 关。
    assert thinking_off_extra_body(
        definition,
        ModelRequest(messages=(), tools=(tool,)),
    ) == {"extra_body": {"thinking": {"type": "disabled"}}}

    # 纯文本 -> 不关。
    assert thinking_off_extra_body(
        definition,
        ModelRequest(messages=()),
    ) == {}

    # 模型级硬开关优先于 enable_thinking。
    forced_off = replace(definition, disable_thinking=True)
    assert thinking_off_extra_body(
        forced_off,
        ModelRequest(messages=(), enable_thinking=True),
    ) == {"extra_body": {"thinking": {"type": "disabled"}}}


async def test_adapter_reads_usage_from_both_places() -> None:
    """用量可能在 usage_metadata，也可能在 response_metadata.token_usage。"""

    adapter = LangChainProviderAdapter()

    model = _FakeChatModel()

    model.response = SimpleNamespace(
        content="ok",
        usage_metadata={
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
        },
        response_metadata={"finish_reason": "stop", "model_name": "chat"},
    )

    client = ModelClient(
        definition=ModelDefinition.make("deepseek", "chat"),
        client=model,
        adapter=adapter,
    )

    from app.agent.model import ModelRequest

    result = await adapter.ainvoke(
        client, ModelRequest(messages=("你好",)), timeout=5.0
    )

    assert isinstance(result, ProviderResult)
    assert result.usage.total_tokens == 10
    assert result.metadata["finish_reason"] == "stop"

    # OpenAI 兼容端点把用量放在 response_metadata.token_usage
    model.response = SimpleNamespace(
        content="ok",
        response_metadata={
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 2}
        },
    )

    result = await adapter.ainvoke(
        client, ModelRequest(messages=("你好",)), timeout=5.0
    )

    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 2
    assert result.usage.total_tokens == 7  # 两个分项都有时自己加


async def test_adapter_normalizes_provider_exception() -> None:
    """适配器是最靠近 provider 的归一化点。"""

    from fakes import fake_exception

    from app.agent.model import ModelRateLimitError

    adapter = LangChainProviderAdapter()

    model = _FakeChatModel()
    model.error = fake_exception("RateLimitError", message="429")

    client = ModelClient(
        definition=ModelDefinition.make("deepseek", "chat"),
        client=model,
        adapter=adapter,
    )

    from app.agent.model import ModelRequest

    with pytest.raises(ModelRateLimitError) as excinfo:
        await adapter.ainvoke(
            client, ModelRequest(messages=("你好",)), timeout=5.0
        )

    assert excinfo.value.provider == "deepseek"
    assert excinfo.value.model == "chat"


async def test_adapter_passes_definition_config_to_the_client() -> None:
    """
    模型定义里的 config 全部作为 provider 参数透传，retry 除外。

    `max_retries` 从 config 里被剔除：它属于可靠性策略，不该在
    "客户端也要重试"这件事上再生效一次（见 factory.py）。
    """

    recorded: dict = {}

    def factory(**kwargs):

        recorded.update(kwargs)
        return _FakeChatModel()

    adapter = LangChainProviderAdapter(chat_model_factory=factory)

    adapter.create_client(
        ModelDefinition.make(
            "deepseek",
            "chat",
            config={"max_tokens": 512, "top_p": 0.9, "max_retries": 5},
        ),
        credential="secret-key",
        timeout=30.0,
        max_retries=0,
    )

    assert recorded["provider"] == "deepseek"
    assert recorded["model"] == "chat"
    assert recorded["timeout"] == 30.0
    assert recorded["max_retries"] == 0  # 不是 config 里的 5
    assert recorded["api_key"] == "secret-key"
    assert recorded["max_tokens"] == 512
    assert recorded["top_p"] == 0.9


def test_adapter_client_creation_error_is_a_configuration_error() -> None:
    """建不出来 = 配置错（不是可重试的 provider 故障）。"""

    def factory(**kwargs):

        raise ValueError("provider 'nope' 不在 init_chat_model 的支持列表里")

    adapter = LangChainProviderAdapter(chat_model_factory=factory)

    with pytest.raises(ModelConfigurationError) as excinfo:
        adapter.create_client(
            ModelDefinition.make("nope", "x"),
            credential=None,
            timeout=10.0,
            max_retries=0,
        )

    message = str(excinfo.value)

    assert "nope:x" in message
    assert "init_chat_model" in message  # 底层原因被保留


# ============================================================
# 真实 langchain 客户端（不联网）
# ============================================================


def test_real_clients_are_constructed_from_the_shipped_catalog() -> None:
    """
    model.yaml 的声明真的能产出可用的 provider 客户端。

    这一步覆盖三件事：
        provider 名 -> 客户端类         （llm.py 的分派表）
        模型名覆盖                       （同一 provider 下多个模型）
        客户端内部重试被关掉              （retry 归 Runtime）
    """

    pytest.importorskip("langchain_deepseek")
    pytest.importorskip("langchain_moonshot")

    from app.common.deepseek_chat import ReasoningChatDeepSeek

    registry = build_model_registry()

    flash = registry.factory.get(
        registry.definition("deepseek:deepseek-v4-flash"), timeout=15
    )

    # DeepSeek 用子类补回 reasoning_content（思考模式多轮要求），
    # 见 app/common/deepseek_chat.py。
    assert isinstance(flash.client, ReasoningChatDeepSeek)
    assert flash.client.model_name == "deepseek-v4-flash"
    # 16384 而不是 4096 或 8192：`max_tokens` 是**思考与正文共用**的额度，
    # 而这条端点的推理占比约九成 —— 4096 被思考吃光时正文一个字都写不出来
    # （2026-10-13，content 是空数组），8192 在出题那条路上又被吃光
    # （2026-10-16，输出 456 字符断在字符串中间）。两次的实测与推理占比
    # 都记在 model.yaml 里 flash 的 `max_tokens` 那段说明里。
    #
    # ⚠️ 这个数**故意钉住**：它是"catalog 里的值真的透传到客户端"的证据，
    # 改配置的人应该在这里被拦一下、去看那段说明（那是第二次为同一件事
    # 翻倍了，说明更对症的做法是让这类调用别思考）。
    assert getattr(flash.client, "max_tokens", None) == 16384

    # 同一个 provider 的**另一个模型**：定义里的 model 覆盖了 llm.yaml 的
    reasoner = registry.factory.get(
        registry.definition("deepseek:deepseek-reasoner"), timeout=15
    )

    assert reasoner.client.model_name == "deepseek-reasoner"

    # 客户端内部重试默认 0（retry 的所有者是 Runtime）
    assert getattr(flash.client, "max_retries", None) == 0

    # 走 _DIRECT_PROVIDERS 那一家的（langchain-moonshot 不在
    # init_chat_model 的分派表里，见 app/common/llm.py）
    kimi = registry.factory.get(registry.definition("kimi:kimi-k2.7-code"), timeout=15)

    assert type(kimi.client).__name__ == "ChatMoonshot"

    # 同一个定义取第二次 -> 命中缓存，不再构造
    assert registry.factory.get(
        registry.definition("deepseek:deepseek-v4-flash"), timeout=15
    ) is flash

    assert registry.factory.stats.created == 3


# ============================================================
# 凭据解析
# ============================================================


def test_env_resolver_uses_declared_env_name(monkeypatch) -> None:

    monkeypatch.setenv("MY_DEEPSEEK_KEY", "value-from-env")

    resolver = EnvVarCredentialResolver(
        {"deepseek-production": "MY_DEEPSEEK_KEY"}
    )

    assert resolver.resolve("deepseek-production") == "value-from-env"


def test_env_resolver_falls_back_to_name_convention(monkeypatch) -> None:
    """没登记的 ref 按约定归一成环境变量名。"""

    monkeypatch.setenv("DEEPSEEK_SANDBOX", "sandbox-key")

    resolver = EnvVarCredentialResolver()

    assert resolver.resolve("deepseek-sandbox") == "sandbox-key"


def test_env_resolver_returns_none_when_absent(monkeypatch) -> None:
    """
    解析不到 -> None（= 不覆盖，用 llm.yaml 里 provider 自己那份）。

    这不是错误：绝大多数模型本来就不需要 credential_ref。
    """

    monkeypatch.delenv("DEEPSEEK_MISSING", raising=False)

    resolver = EnvVarCredentialResolver()

    assert resolver.resolve("deepseek-missing") is None


def test_env_resolver_can_be_made_strict(monkeypatch) -> None:
    """要"声明的凭据必须存在"时显式要求（fail fast）。"""

    monkeypatch.delenv("DEEPSEEK_MISSING", raising=False)

    resolver = EnvVarCredentialResolver(required=True)

    with pytest.raises(ModelConfigurationError) as excinfo:
        resolver.resolve("deepseek-missing")

    assert "DEEPSEEK_MISSING" in str(excinfo.value)


def test_resolvers_satisfy_the_protocol() -> None:
    """两个实现都满足 CredentialResolver 这个端口（runtime_checkable）。"""

    from app.agent.model import LiteralCredentialResolver, NullCredentialResolver

    assert isinstance(EnvVarCredentialResolver(), CredentialResolver)
    assert isinstance(NullCredentialResolver(), CredentialResolver)
    assert isinstance(LiteralCredentialResolver({}), CredentialResolver)


def test_pricing_from_catalog_is_optional(tmp_path) -> None:
    """没写 pricing -> None（不估成本），写了才有。"""

    path = _write(
        tmp_path,
        """
providers:
  deepseek:
    adapter: langchain
models:
  - provider: deepseek
    model: chat
    pricing:
      input_per_1k: 0.5
      output_per_1k: 1.5
  - provider: deepseek
    model: other
""",
    )

    catalog = load_model_catalog(path)

    assert catalog.models[0].pricing == ModelPricing(
        input_per_1k=0.5, output_per_1k=1.5
    )
    assert catalog.models[1].pricing is None


def test_model_profile_requires_primary() -> None:

    with pytest.raises(ValueError):
        ModelProfile(id="p", primary="")
