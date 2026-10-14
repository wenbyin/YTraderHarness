"""
能力相容性（规格 §22 第四条）。

    Primary requires tool_calling
    Fallback does not support tool_calling
    -> 必须禁止使用该 fallback

这条约束是硬的：它不依赖"选模型的人记得检查"，而是 Runtime 在每次
调用前按**这次请求要求的能力**过滤候选链。
"""

from __future__ import annotations

import pytest

from fakes import FakeAdapter, Outcome, make_definition, make_registry

from app.agent.model import (
    ModelCapabilities,
    ModelCapabilityError,
    ModelProfile,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
    ModelRuntimePolicy,
    ModelUnavailableError,
    required_capabilities_for,
)


TOOLS = (object(),)  # 一个"工具"占位：只需要非空

TOOL_CAPABLE = ModelCapabilities(tool_calling=True, structured_output=True)
NO_TOOLS = ModelCapabilities(structured_output=True)


def _runtime(adapter, primary_caps, fallback_caps=NO_TOOLS, **policy_kwargs):
    policy_kwargs.setdefault("max_retries", 0)
    policy_kwargs.setdefault("retry_backoff", 0.0)

    registry = make_registry(
        models=[
            make_definition("primary", capabilities=primary_caps),
            make_definition("backup", capabilities=fallback_caps),
        ],
        profiles=[
            ModelProfile(
                id="p",
                primary="fake:primary",
                fallbacks=("fake:backup",),
                policy=ModelRuntimePolicy(**policy_kwargs),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react", model_role=ModelRole.REASONING, profile="p"
            )
        ],
        adapter=adapter,
    )

    return registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )


# ============================================================
# 需求来源
# ============================================================


def test_request_derives_requirements_from_content() -> None:
    """请求内容能确定地推导出两件事：工具调用与结构化输出。"""

    plain = ModelRequest(messages=("你好",))
    with_tools = ModelRequest(messages=("你好",), tools=TOOLS)
    with_format = ModelRequest(messages=("你好",), response_format=dict)

    assert required_capabilities_for(plain).as_dict() == {
        "tool_calling": False,
        "structured_output": False,
        "streaming": False,
        "vision": False,
        "reasoning": False,
    }

    assert required_capabilities_for(with_tools).tool_calling is True
    assert required_capabilities_for(with_format).structured_output is True


def test_vision_and_reasoning_are_not_guessed() -> None:
    """
    vision / reasoning **不**从消息内容猜。

    "有图片所以需要 vision"看起来聪明，实际会错（图片可能是 URL 字符串、
    markdown 或多模态 block），而猜错的代价是"本来能跑的调用被判成能力
    不符"。这两项必须显式声明。
    """

    request = ModelRequest(
        messages=("这是一张图：https://example.com/a.png",),
    )

    assert required_capabilities_for(request).vision is False
    assert required_capabilities_for(request).reasoning is False


# ============================================================
# fallback 相容性
# ============================================================


async def test_incompatible_fallback_is_excluded() -> None:
    """
    需要 tool_calling 时，不支持工具的 fallback **不进候选链**。

    注意这里 primary 是能跑的 —— 断言的重点是 fallback 被剔除了，
    而且这件事在响应里看得见（incompatible_candidates）。
    """

    adapter = FakeAdapter()
    runtime = _runtime(adapter, TOOL_CAPABLE)

    response = await runtime.ainvoke(
        ModelRequest(messages=("你好",), tools=TOOLS)
    )

    assert response.message == "ok"

    # 候选链里只有 primary
    assert [item.id for item in runtime.candidates(
        ModelCapabilities(tool_calling=True)
    )] == ["fake:primary"]

    # 被剔除的候选出现在观测里 —— 否则"为什么没降级"永远查不出来
    assert response.metadata["incompatible_candidates"] == ["fake:backup"]
    assert response.degraded is True


async def test_incompatible_fallback_is_never_called() -> None:
    """
    primary 失败时，不相容的 fallback **不会被调用** —— 调用链到此为止。

    这是规格 §8.3 那个例子的实际检验：一个需要 Tool Calling 的节点
    不能降级到不支持 Tool Calling 的模型上，宁可失败。
    """

    adapter = FakeAdapter(
        default=Outcome(error=ModelUnavailableError("503"))
    )

    runtime = _runtime(adapter, TOOL_CAPABLE, max_retries=0)

    with pytest.raises(ModelUnavailableError) as excinfo:
        await runtime.ainvoke(
            ModelRequest(messages=("你好",), tools=TOOLS)
        )

    # 只打了 primary —— backup 一次都没被调用
    assert adapter.called_models() == ["fake:primary"]

    # 而且报错里说明了"有一个候选被剔除"
    assert excinfo.value.details["incompatible_candidates"] == ["fake:backup"]


async def test_compatible_fallback_is_used() -> None:
    """能力相容时 fallback 正常参与（避免把上面的断言写成"永远不降级"）。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [Outcome(error=ModelUnavailableError("503"))],
            "fake:backup": [Outcome(message="备份答的")],
        }
    )

    runtime = _runtime(
        adapter,
        primary_caps=TOOL_CAPABLE,
        fallback_caps=TOOL_CAPABLE,
        max_retries=0,
    )

    response = await runtime.ainvoke(
        ModelRequest(messages=("你好",), tools=TOOLS)
    )

    assert response.message == "备份答的"
    assert adapter.called_models() == ["fake:primary", "fake:backup"]


async def test_structured_output_requirement_gates_fallback() -> None:
    """response_format 需要 structured_output，同样参与过滤。"""

    adapter = FakeAdapter()
    runtime = _runtime(
        adapter,
        primary_caps=ModelCapabilities(structured_output=True),
        fallback_caps=ModelCapabilities(),  # 不支持结构化输出
    )

    response = await runtime.ainvoke(
        ModelRequest(messages=("你好",), response_format=dict)
    )

    assert response.metadata["incompatible_candidates"] == ["fake:backup"]


# ============================================================
# primary 不相容
# ============================================================


async def test_primary_missing_capability_fails_before_any_call() -> None:
    """
    primary 不满足要求 -> **调用前**就报错，且一次 provider 都不调。

    静默降级会让"我们一直在用一个做不到这件事的模型"长期无人发现。
    """

    adapter = FakeAdapter()
    runtime = _runtime(adapter, primary_caps=NO_TOOLS)

    with pytest.raises(ModelCapabilityError) as excinfo:
        await runtime.ainvoke(
            ModelRequest(messages=("你好",), tools=TOOLS)
        )

    assert adapter.calls == []

    error = excinfo.value

    assert error.details["missing"] == ["tool_calling"]
    assert error.details["profile_id"] == "p"
    assert "tool_calling" in str(error)


async def test_extra_requirement_at_call_site() -> None:
    """调用点可以临时追加要求（不必改 profile，也不必重建 Runtime）。"""

    adapter = FakeAdapter()
    runtime = _runtime(adapter, primary_caps=TOOL_CAPABLE)

    # 不追加时一切正常
    await runtime.ainvoke(ModelRequest(messages=("你好",)))

    # 追加 vision 要求 -> primary 不支持 -> 报错
    with pytest.raises(ModelCapabilityError) as excinfo:
        await runtime.ainvoke(
            ModelRequest(messages=("你好",)),
            required_capabilities=ModelCapabilities(vision=True),
        )

    assert excinfo.value.details["missing"] == ["vision"]


async def test_node_declared_requirement_applies_to_every_call() -> None:
    """
    节点在 `ModelResolveContext` 里声明的要求对**每一次**调用都生效。

    它是"这个节点知道自己要什么"的表达（比如一个必然要工具的节点），
    与"这一次请求恰好带没带 tools"是两件事。
    """

    adapter = FakeAdapter()

    registry = make_registry(
        models=[make_definition("primary", capabilities=NO_TOOLS)],
        profiles=[ModelProfile(id="p", primary="fake:primary")],
        bindings=[
            ModelRoleBinding(
                agent_type="react", model_role=ModelRole.REASONING, profile="p"
            )
        ],
        adapter=adapter,
    )

    from app.agent.model import ModelResolveContext

    runtime = registry.get_runtime(
        agent_type="react",
        model_role=ModelRole.REASONING,
        context=ModelResolveContext(
            required_capabilities=ModelCapabilities(tool_calling=True),
        ),
    )

    # 请求里没有 tools，但节点声明了需要工具调用 -> 仍然不够用
    with pytest.raises(ModelCapabilityError):
        await runtime.ainvoke(ModelRequest(messages=("你好",)))

    assert adapter.calls == []
