"""
客户端的创建与复用（规格 §17、§22 "Node 与 Runtime 隔离"的一半）。

    不要在每一次 invoke() 中重新初始化 SDK client
    Factory 应当支持 client reuse / cache

这条约束在代码上唯一可验证的形态就是"同一个定义建了几次"，因此这里
直接断言 `adapter.created` 与 `factory.stats`。

⚠️ 重试的所有者是 Runtime，不是客户端
-----------------------------------
客户端内部重试默认被关掉（`max_retries=0`）。旧实现是"客户端内部重试
2 次"，与节点级 retry_policy 叠乘会变成 9 次**可计费**调用。现在这条
风险在结构上不存在 —— 但它必须被测试盯住，否则某次改动会把默认值
调回 2，而症状只是"账单变高了"。
"""

from __future__ import annotations

import pytest

from fakes import FakeAdapter, FakeCredentialResolver, Outcome, make_definition

from app.agent.model import (
    DefaultModelRegistry,
    ModelConfigurationError,
    ModelProfile,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
    ModelRuntimePolicy,
)


def _registry(adapter, *, models=None, profiles=None, bindings=None, credentials=None):
    registry = DefaultModelRegistry(credentials=credentials)

    for definition in models or [make_definition("primary")]:
        registry.register_provider(definition.provider, adapter)
        registry.register_model(definition)

    for profile in profiles or [ModelProfile(id="p", primary="fake:primary")]:
        registry.register_profile(profile)

    for binding in bindings or [
        ModelRoleBinding(
            agent_type="react", model_role=ModelRole.REASONING, profile="p"
        )
    ]:
        registry.register_binding(binding)

    return registry


def _request() -> ModelRequest:
    return ModelRequest(messages=("你好",))


# ============================================================
# 复用
# ============================================================


async def test_client_is_created_once_and_reused() -> None:
    """两次调用 -> 一个客户端。"""

    adapter = FakeAdapter()
    registry = _registry(adapter)

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    await runtime.ainvoke(_request())
    await runtime.ainvoke(_request())

    assert len(adapter.created) == 1
    assert registry.factory.stats.created == 1
    assert registry.factory.stats.hits >= 1

    # 但请求是**两次**（复用客户端 ≠ 复用结果）
    assert adapter.call_count() == 2


async def test_same_model_different_timeout_is_a_different_client() -> None:
    """
    超时是**建客户端时**的参数，因此进缓存键。

    两个 profile 用同一个模型但超时不同 -> 两个客户端。这不是浪费，
    而是必要条件（超时值在构造时就固定了）。
    """

    adapter = FakeAdapter()

    registry = _registry(
        adapter,
        models=[make_definition("primary")],
        profiles=[
            ModelProfile(
                id="slow",
                primary="fake:primary",
                policy=ModelRuntimePolicy(timeout=120),
            ),
            ModelProfile(
                id="fast",
                primary="fake:primary",
                policy=ModelRuntimePolicy(timeout=10),
            ),
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react", model_role="reasoning", profile="slow"
            ),
            ModelRoleBinding(
                agent_type="react", model_role="summarization", profile="fast"
            ),
        ],
    )

    await registry.get_runtime(
        agent_type="react", model_role="reasoning"
    ).ainvoke(_request())

    await registry.get_runtime(
        agent_type="react", model_role="summarization"
    ).ainvoke(_request())

    assert len(adapter.created) == 2
    assert sorted(item["timeout"] for item in adapter.created) == [10, 120]


async def test_credential_rotation_invalidates_cache() -> None:
    """
    同一把 ref 换了新 key -> 缓存自动失效。

    否则轮换密钥之后，旧的客户端（拿着旧 key）会继续被复用，而
    症状是"我已经换了 key 但还是 401"。
    """

    adapter = FakeAdapter()

    rotating = FakeCredentialResolver({"ref": "key-one"})

    registry = _registry(
        adapter,
        models=[make_definition("primary", credential_ref="ref")],
        credentials=rotating,
    )

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    await runtime.ainvoke(_request())

    rotating.secrets["ref"] = "key-two"

    await runtime.ainvoke(_request())

    assert len(adapter.created) == 2
    assert [item["credential"] for item in adapter.created] == [
        "key-one",
        "key-two",
    ]


async def test_no_credential_ref_means_provider_default() -> None:
    """
    没有 credential_ref 时**什么都不做**：让 provider 用它自己配置里的
    凭据（app/common/llm.yaml）。默认路径上本模块不认识任何密钥。
    """

    adapter = FakeAdapter()

    registry = _registry(adapter, credentials=FakeCredentialResolver())

    await registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    ).ainvoke(_request())

    assert adapter.created[0]["credential"] is None


# ============================================================
# 重试归属
# ============================================================


async def test_client_internal_retry_is_off_by_default() -> None:
    """
    客户端内部重试默认 0 —— retry 归 Runtime 管，不叠加。

    （叠加的症状是"账单变高了"，几乎不会有人从代码里看出来。）
    """

    adapter = FakeAdapter()
    registry = _registry(adapter)

    await registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    ).ainvoke(_request())

    assert adapter.created[0]["max_retries"] == 0


async def test_client_retry_can_be_requested_explicitly() -> None:
    """要恢复客户端重试必须**显式**写在模型定义的 config 里。"""

    adapter = FakeAdapter()
    registry = _registry(
        adapter,
        models=[make_definition("primary", config={"max_retries": 3})],
    )

    await registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    ).ainvoke(_request())

    assert adapter.created[0]["max_retries"] == 3


async def test_invalid_client_retry_config_is_rejected() -> None:
    """config.max_retries 写错 -> 配置错，而不是静默忽略。"""

    adapter = FakeAdapter()
    registry = _registry(
        adapter,
        models=[make_definition("primary", config={"max_retries": "两次"})],
    )

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    with pytest.raises(ModelConfigurationError):
        await runtime.ainvoke(_request())

    assert adapter.created == []


# ============================================================
# 生命周期与登记
# ============================================================


async def test_aclose_closes_every_client() -> None:
    """进程关闭时每个缓存客户端都被释放一次。"""

    adapter = FakeAdapter()

    registry = _registry(
        adapter,
        models=[make_definition("primary"), make_definition("backup")],
        profiles=[
            ModelProfile(
                id="p", primary="fake:primary", fallbacks=("fake:backup",)
            )
        ],
    )

    registry.factory.get(
        registry.definition("fake:primary"), timeout=10
    )
    registry.factory.get(
        registry.definition("fake:backup"), timeout=10
    )

    await registry.aclose()

    assert adapter.closed == 2
    assert registry.factory.stats.closed == 2


async def test_unknown_provider_is_a_configuration_error() -> None:
    """模型声明了一个没有适配器的 provider -> 建客户端时报配置错。"""

    adapter = FakeAdapter()
    registry = _registry(adapter)

    # 直接往注册表里塞一个"provider 没登记"的模型定义
    registry.register_model(make_definition("ghost", provider="ghost"))

    registry.register_profile(
        ModelProfile(id="ghost_profile", primary="ghost:ghost")
    )
    registry.register_binding(
        ModelRoleBinding(
            agent_type="ghost",
            model_role="reasoning",
            profile="ghost_profile",
        )
    )

    runtime = registry.get_runtime(
        agent_type="ghost", model_role="reasoning"
    )

    with pytest.raises(ModelConfigurationError) as excinfo:
        await runtime.ainvoke(_request())

    assert "ghost" in str(excinfo.value)
    assert adapter.created == []


def test_provider_conflict_is_rejected() -> None:
    """同一 provider 接两个不同的适配器 -> 报错，不静默取其一。"""

    registry = DefaultModelRegistry()

    registry.register_provider("fake", FakeAdapter())

    with pytest.raises(ModelConfigurationError):
        registry.register_provider("fake", FakeAdapter())


def test_duplicate_model_registration_is_rejected() -> None:
    registry = _registry(FakeAdapter())

    with pytest.raises(ModelConfigurationError):
        registry.register_model(make_definition("primary"))
