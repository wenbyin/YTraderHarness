"""
State Path 与 gate 语义（规格 §九、§十一、§十二）。

这是整份规格里**最强制**的一段：MISSING / FALSE / TRUE / INVALID 四态
必须严格分开，且**不做隐式 bool 转换**。

为什么逐条测
----------
`bool("yes") == True`、`bool(1) == True`、`bool(None) == False` —— 三个
都会把"上游写错了"伪装成"照常执行"或"恰好跳过"。它们是同一个 bug 的
三种长相，而症状（模型看到一份缺东西或多余的上下文）几乎不可能从日志
里看出来。因此四种情况各有一条用例。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.context.processor.errors import ContextStateError
from app.agent.context.processor.spec import (
    ContextProcessorSpec,
    StateBinding,
)
from app.agent.context.processor.state import (
    MISSING,
    resolve_inputs,
    resolve_state_path,
)
from app.common.context import StageId


# ============================================================
# resolve_state_path
# ============================================================


def test_resolve_nested_mapping_path() -> None:
    """dot path 穿过嵌套映射。"""

    state = {"context_policy": {"include_memory": True}}

    assert resolve_state_path(
        state, "context_policy.include_memory"
    ) is True


def test_resolve_object_attribute_path() -> None:
    """也认对象属性（dataclass / pydantic / SimpleNamespace）。"""

    @dataclass
    class Run:
        phase: str

    assert resolve_state_path({"run": Run(phase="initial")}, "run.phase") == (
        "initial"
    )


def test_resolve_single_segment() -> None:
    assert resolve_state_path({"user_id": "u1"}, "user_id") == "u1"


def test_missing_key_returns_sentinel() -> None:
    """键不存在 -> MISSING（不是 None）。"""

    result = resolve_state_path({"context_policy": {}}, "context_policy.include_memory")

    assert result is MISSING
    assert repr(result) == "<MISSING>"


def test_missing_intermediate_returns_sentinel() -> None:
    """中间段不存在 / 不是容器 -> MISSING。"""

    assert resolve_state_path({}, "context_policy.include_memory") is MISSING

    assert resolve_state_path(
        {"a": 1}, "a.b"
    ) is MISSING


def test_none_value_is_found_not_missing() -> None:
    """
    存在且为 None 与"不存在"必须分得开。

    这是 `None` gate 归到 INVALID（而不是干脆 MISSING）的基础：两种情况的
    报错信息不同，指向的原因也不同（漏配 vs 赋值成了 None）。
    """

    state = {"context_policy": {"include_memory": None}}

    result = resolve_state_path(state, "context_policy.include_memory")

    assert result is None
    assert result is not MISSING


def test_missing_sentinel_is_falsy() -> None:
    """
    哨兵是假值：万一有人漏了 `is MISSING`，`if value` 会走"跳过"这条
    **保守**分支，而不是凭空执行一个前置条件未知的处理器。
    """

    assert bool(MISSING) is False


def test_empty_path_is_rejected() -> None:

    with pytest.raises(ValueError):
        resolve_state_path({}, "")


def test_default_is_returned_when_missing() -> None:

    assert resolve_state_path({}, "a.b", default="x") == "x"


# ============================================================
# StateBinding / ContextProcessorSpec
# ============================================================


def test_binding_rejects_malformed_paths() -> None:

    for bad in (".a", "a.", "a..b", "  "):

        with pytest.raises(ValueError):
            StateBinding(bad)


def test_spec_accepts_string_shorthand_for_bindings() -> None:
    """规格 §十三：`inputs={"user_id": "user_id"}` 这种简写必须成立。"""

    spec = ContextProcessorSpec(
        name="x",
        order=10,
        stage=StageId.PROFILE_LOADER,
        when="context_policy.include_user_profile",
        inputs={"user_id": "user_id", "profile": StateBinding("p", required=False)},
    )

    assert isinstance(spec.when, StateBinding)
    assert spec.when.path == "context_policy.include_user_profile"
    assert spec.inputs["user_id"].required is True
    assert spec.inputs["profile"].required is False


def test_spec_rejects_unknown_stage() -> None:

    with pytest.raises(ValueError):
        ContextProcessorSpec(name="x", order=1, stage="profile_loader")  # type: ignore[arg-type]


def test_spec_rejects_bad_order() -> None:

    with pytest.raises(ValueError):
        ContextProcessorSpec(name="x", order=True, stage=StageId.PROFILE_LOADER)

    with pytest.raises(ValueError):
        ContextProcessorSpec(name="x", order=-1, stage=StageId.PROFILE_LOADER)


def test_spec_describe_matches_catalog_shape() -> None:
    """
    `describe()` 的形状贴 `context_builder.yaml` 的 StagePolicy ——
    将来 `/context-builders` 从代码派生目录时不必再猜形状。
    """

    spec = ContextProcessorSpec(
        name="memory",
        order=50,
        stage=StageId.MEMORY_RETRIEVER,
        when="context_policy.include_memory",
        inputs={"user_id": "user_id"},
        optional=True,
    )

    described = spec.describe()

    assert described["execute_when"] == ["context_policy.include_memory"]
    assert described["stage"] == "memory_retriever"
    assert described["required"] is False
    assert described["optional"] is True
    assert described["input_contract"]["fields"] == [
        {
            "name": "user_id",
            "type": "state_path",
            "required": True,
            "description": "user_id",
        }
    ]


# ============================================================
# resolve_inputs
# ============================================================


def test_resolve_inputs_binds_multiple_paths() -> None:
    """一个 Processor 可以声明多个 inputs（规格 §十一）。"""

    state = {
        "user_id": "u1",
        "context_policy": {"max_recent_turns": 4},
        "summary": {"text": "旧摘要"},
    }

    bound = resolve_inputs(state, {
        "user_id": StateBinding("user_id"),
        "turns": StateBinding("context_policy.max_recent_turns"),
        "summary": StateBinding("summary", required=False),
    })

    assert bound.values == {
        "user_id": "u1",
        "turns": 4,
        "summary": {"text": "旧摘要"},
    }
    assert dict(bound.missing) == {}
    assert bound.missing_required == ()


def test_required_input_missing_is_reported() -> None:
    """必需输入缺失 -> 由流水线升级为 FAILED（规格 §十二）。"""

    bound = resolve_inputs({"other": 1}, {"user_id": StateBinding("user_id")})

    assert bound.missing_required == ("user_id",)
    assert "user_id" not in bound.values


def test_required_input_with_none_counts_as_missing() -> None:
    """必需输入拿到 None 与没有它一样没用。"""

    bound = resolve_inputs({"user_id": None}, {"user_id": StateBinding("user_id")})

    assert bound.missing_required == ("user_id",)


def test_optional_input_missing_is_delivered_as_none() -> None:
    """
    可选缺失以 None 交付（规格 §十二 明确允许的形态）。

    于是 Processor 可以写 `inputs["summary"]`，而不必到处 `.get()`。
    """

    bound = resolve_inputs(
        {}, {"summary": StateBinding("summary", required=False)}
    )

    assert bound.values["summary"] is None
    assert dict(bound.missing) == {"summary": False}
    assert bound.missing_required == ()


def test_none_value_for_optional_input_is_preserved() -> None:
    """可选输入真的给了 None -> 交付 None（不是"缺失"）。"""

    bound = resolve_inputs(
        {"summary": None}, {"summary": StateBinding("summary", required=False)}
    )

    assert bound.values["summary"] is None
    assert dict(bound.missing) == {}


# ============================================================
# gate 的四种语义（在流水线里测，这里只测解析那一半）
# ============================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (None, None),
        ("yes", "yes"),
        (1, 1),
        (0, 0),
    ],
)
def test_gate_value_is_returned_verbatim(value, expected) -> None:
    """
    解析这一层**不做任何类型转换** —— 类型判定是流水线的事
    （只有真正的 bool 才放行）。
    """

    state = {"context_policy": {"include_memory": value}}

    assert resolve_state_path(
        state, "context_policy.include_memory"
    ) == expected


def test_state_error_carries_processor_and_stage() -> None:
    """报错要指得出"哪一步、哪个阶段"。"""

    error = ContextStateError(
        "missing", processor="memory", stage="memory_retriever"
    )

    assert error.processor == "memory"
    assert error.stage == "memory_retriever"
    assert error.as_metadata()["error_type"] == "ContextStateError"
    assert "memory_retriever" in str(error)
