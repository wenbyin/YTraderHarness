"""
基础调用链（规格 §22 第一条）。

    Registry -> Resolver -> Factory -> Runtime -> Provider -> Response

一次调用里每一环都该留痕：解析出的 profile、建过的客户端、发出的
attempt、回来的响应。这里逐环断言，任何一环被短路都会失败。
"""

from __future__ import annotations

import pytest

from fakes import FakeAdapter, Outcome, make_definition, make_registry

from app.agent.model import (
    ModelCapabilities,
    ModelProfile,
    ModelRequest,
    ModelResolutionError,
    ModelResponse,
    ModelRole,
    ModelRoleBinding,
    ModelRuntimePolicy,
)


def _registry(adapter: FakeAdapter, **policy_kwargs):
    return make_registry(
        models=[make_definition("primary", capabilities=ModelCapabilities(
            tool_calling=True, structured_output=True,
        ))],
        profiles=[
            ModelProfile(
                id="reasoning_default",
                primary="fake:primary",
                policy=ModelRuntimePolicy(**policy_kwargs),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.REASONING,
                profile="reasoning_default",
            )
        ],
        adapter=adapter,
    )


def _request(text: str = "你好") -> ModelRequest:
    return ModelRequest(messages=(text,))


async def test_full_chain_returns_response() -> None:
    """一次完整的调用：解析 -> 建客户端 -> 发请求 -> 统一响应。"""

    adapter = FakeAdapter()
    registry = _registry(adapter)

    runtime = registry.get_runtime(
        agent_type="react",
        model_role=ModelRole.REASONING,
    )

    response = await runtime.ainvoke(_request())

    assert isinstance(response, ModelResponse)
    assert response.message == "ok"
    assert response.provider == "fake"
    assert response.model == "primary"
    assert response.model_id == "fake:primary"
    assert response.attempts == 1
    assert response.degraded is False
    assert response.metadata["profile_id"] == "reasoning_default"
    assert response.metadata["agent_type"] == "react"
    assert response.metadata["model_role"] == "reasoning"
    assert response.metadata["fallback_used"] is False


async def test_chain_touches_every_stage_once() -> None:
    """每一环各做各的事，且只做一次。"""

    adapter = FakeAdapter()
    registry = _registry(adapter)

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    await runtime.ainvoke(_request())

    # Resolver 给出的 profile
    assert runtime.profile_id == "reasoning_default"

    # Factory 建了一个客户端，且用的是 policy 的 timeout
    assert len(adapter.created) == 1
    assert adapter.created[0]["model_id"] == "fake:primary"
    assert adapter.created[0]["timeout"] == runtime.policy.timeout

    # Provider 发了一次请求
    assert adapter.calls == [("fake:primary", runtime.policy.timeout)]


async def test_runtime_is_cached_per_resolution() -> None:
    """同一个 (profile, agent_type, role, context) 共用同一个 Runtime。"""

    adapter = FakeAdapter()
    registry = _registry(adapter)

    first = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )
    second = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    assert first is second

    # 不同的角色解析出不同的 profile -> 不同的 runtime。
    registry.register_profile(
        ModelProfile(id="fast_default", primary="fake:primary")
    )
    registry.register_binding(
        ModelRoleBinding(
            agent_type="react",
            model_role=ModelRole.SUMMARIZATION,
            profile="fast_default",
        )
    )

    other = registry.get_runtime(
        agent_type="react", model_role=ModelRole.SUMMARIZATION
    )

    assert other is not first
    assert other.profile_id == "fast_default"


async def test_alias_resolves_to_profile() -> None:
    """逻辑别名能解析到 profile（Agent 侧只该用逻辑名）。"""

    adapter = FakeAdapter()
    registry = _registry(adapter)

    registry.register_alias("cheap_default", "reasoning_default")

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    assert runtime.profile_id == "reasoning_default"
    assert registry.profile("cheap_default").id == "reasoning_default"


async def test_unknown_binding_fails_loudly() -> None:
    """
    解析不出来就报错，**不回落到某个默认模型**。

    静默回落的症状是"模型答得不太对"，几乎查不出来。
    """

    adapter = FakeAdapter()
    registry = _registry(adapter)

    with pytest.raises(ModelResolutionError) as excinfo:
        registry.get_runtime(agent_type="react", model_role="vision")

    message = str(excinfo.value)

    assert "vision" in message
    assert "reasoning_default" in message  # 报错里列出已有的绑定


async def test_empty_messages_is_rejected_before_provider_call() -> None:
    """空 messages 是调用方的 bug，不该发到 provider 那里去。"""

    from app.agent.model import ModelInvalidRequestError

    adapter = FakeAdapter()
    registry = _registry(adapter)

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    with pytest.raises(ModelInvalidRequestError):
        await runtime.ainvoke(ModelRequest(messages=()))

    assert adapter.calls == []


async def test_usage_and_cost_are_reported() -> None:
    """用量与成本走标准化字段，缺报价时是 None 而不是 0。"""

    from app.agent.model import ModelUsage

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(
                    message="ok",
                    usage=ModelUsage(
                        input_tokens=100,
                        output_tokens=50,
                        total_tokens=150,
                    ),
                )
            ]
        }
    )

    registry = make_registry(
        models=[make_definition("primary")],
        profiles=[
            ModelProfile(
                id="reasoning_default",
                primary="fake:primary",
                policy=ModelRuntimePolicy(retry_backoff=0.0),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.REASONING,
                profile="reasoning_default",
            )
        ],
        adapter=adapter,
    )

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    response = await runtime.ainvoke(_request())

    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 50
    assert response.usage.total_tokens == 150

    # 没配 pricing -> 不估成本（编个 0 出来会让"没报价"与"真的免费"分不清）
    assert response.metadata["estimated_cost"] is None


async def test_pricing_estimates_cost_only_when_configured() -> None:
    """配了单价才算成本；缺哪一半就不算那一半。"""

    from app.agent.model import ModelPricing, ModelUsage

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(
                    message="ok",
                    usage=ModelUsage(input_tokens=1000, output_tokens=1000),
                )
            ]
        }
    )

    registry = make_registry(
        models=[
            make_definition(
                "primary",
                pricing=ModelPricing(input_per_1k=1.0, output_per_1k=2.0),
            )
        ],
        profiles=[
            ModelProfile(
                id="reasoning_default",
                primary="fake:primary",
                policy=ModelRuntimePolicy(),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.REASONING,
                profile="reasoning_default",
            )
        ],
        adapter=adapter,
    )

    runtime = registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )

    response = await runtime.ainvoke(_request())

    # 1000 token * 1.0/1000 + 1000 token * 2.0/1000
    assert response.metadata["estimated_cost"] == pytest.approx(3.0)

    # 只报了一半时只算一半（另一半当"没报"，不当 0）。
    assert ModelPricing(input_per_1k=1.0).estimate(
        ModelUsage(input_tokens=1000, total_tokens=2000)
    ) == pytest.approx(1.0)

    # 什么都没报 -> None
    assert ModelPricing(input_per_1k=1.0).estimate(None) is None
