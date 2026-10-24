"""默认能力策略：模式判定与 AGENT_RUNTIME 子树展开。"""

from __future__ import annotations

from app.agent.graph.react import capability_defaults
from app.agent.graph.react.capability_defaults import (
    allowed_capability_codes,
    default_capability_codes,
    resolve_capability_exposure,
    resolve_mode,
)

__all__: list[str] = []


CODES = [
    "AGENT_RUNTIME",
    "AGENT_INTERRUPT",
    "AGENT_RUNTIME_ORCHESTRATION",
    "AGENT_SKILL",
    "AGENT_SUB_AGENT",
    "AGENT_TASK",
    "CAPABILITY",
    "TOOL",
    "DATETIME_TOOL",
    "TEST_TOOL",
    "WEB_TOOL",
    "STOCK",
]

PARENT_OF = {
    "AGENT_RUNTIME": None,
    "AGENT_INTERRUPT": "AGENT_RUNTIME",
    "AGENT_RUNTIME_ORCHESTRATION": "AGENT_RUNTIME",
    "AGENT_SKILL": "AGENT_RUNTIME",
    "AGENT_SUB_AGENT": "AGENT_RUNTIME",
    "AGENT_TASK": "AGENT_RUNTIME",
    "CAPABILITY": None,
    "TOOL": None,
    "DATETIME_TOOL": "TOOL",
    "TEST_TOOL": "TOOL",
    "WEB_TOOL": "TOOL",
    "STOCK": None,
}

RUNTIME_SUBTREE = {
    "AGENT_RUNTIME",
    "AGENT_INTERRUPT",
    "AGENT_RUNTIME_ORCHESTRATION",
    "AGENT_SKILL",
    "AGENT_SUB_AGENT",
    "AGENT_TASK",
}


# ============================================================
# 模式
# ============================================================


def test_plan_enter_request_wins() -> None:
    assert resolve_mode(
        execution_plan="react",
        plan_status="off",
        plan_enter_requested=True,
    ) == "plan"


def test_active_plan_wins_over_execution_plan() -> None:
    assert resolve_mode(
        execution_plan="react",
        plan_status="plan",
        plan_enter_requested=False,
    ) == "plan"


def test_react_mode() -> None:
    assert resolve_mode(
        execution_plan="react",
        plan_status="off",
        plan_enter_requested=False,
    ) == "react"


def test_other_falls_back_to_direct() -> None:
    assert resolve_mode(
        execution_plan="other",
        plan_status="off",
        plan_enter_requested=False,
    ) == "direct"

    assert resolve_mode(
        execution_plan=None,
        plan_status=None,
        plan_enter_requested=False,
    ) == "direct"


# ============================================================
# 默认能力
# ============================================================


def test_plan_expands_the_whole_runtime_subtree() -> None:
    codes = default_capability_codes(
        "plan",
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(codes) == RUNTIME_SUBTREE
    assert "STOCK" not in codes
    assert len(codes) == len(set(codes))


def test_planner_uses_the_same_runtime_subtree() -> None:
    codes = default_capability_codes(
        "planner",
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(codes) == RUNTIME_SUBTREE


def test_planner_global_capability_is_the_catalog_query() -> None:
    assert capability_defaults.MODEL_CAPABILITIES["planner"] == (
        "CAPABILITY",
    )


def test_direct_and_react_have_no_mode_level_defaults() -> None:
    for mode in ("direct", "react"):
        assert default_capability_codes(
            mode,
            codes=CODES,
            parent_of=PARENT_OF,
        ) == []


def test_unknown_mode_is_empty_not_an_error() -> None:
    assert default_capability_codes(
        "no-such-mode",
        codes=CODES,
        parent_of=PARENT_OF,
    ) == []


def test_missing_root_is_empty() -> None:
    assert default_capability_codes(
        "plan",
        codes=["STOCK"],
        parent_of={"STOCK": None},
    ) == []


def test_children_of_inactive_nodes_are_not_pulled_in() -> None:
    # AGENT_RUNTIME 不在可用集合里 -> 整棵子树不展开，
    # 即使 parent_of 里还有它的子节点。
    assert default_capability_codes(
        "plan",
        codes=["AGENT_SUB_AGENT"],
        parent_of={"AGENT_SUB_AGENT": "AGENT_RUNTIME"},
    ) == []


# ============================================================
# 全局暴露策略（agent -> allowed capabilities）
# ============================================================


def test_general_agent_exposes_configured_capabilities() -> None:
    allowed = allowed_capability_codes(
        "general",
        codes=CODES,
        parent_of=PARENT_OF,
    )

    # 普通 code 是精确匹配：TOOL 不会把 TEST_TOOL / WEB_TOOL 带进来。
    # CAPABILITY 是 `capability.list_capabilities` 的默认入口。
    assert set(allowed) == {"TOOL", "DATETIME_TOOL", "CAPABILITY"}
    assert "AGENT_RUNTIME" not in allowed
    assert "TEST_TOOL" not in allowed
    assert "WEB_TOOL" not in allowed


def test_unknown_agent_is_fail_closed() -> None:
    assert allowed_capability_codes(
        "no-such-agent",
        codes=CODES,
        parent_of=PARENT_OF,
    ) == []


def test_policy_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_defaults,
        "MODEL_CAPABILITIES",
        {"lower": ("tool", "agent_runtime/*")},
    )

    allowed = allowed_capability_codes(
        "lower",
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(allowed) == RUNTIME_SUBTREE | {"TOOL"}


def test_subtree_syntax_expands_descendants(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_defaults,
        "MODEL_CAPABILITIES",
        {"tree": ("tool/*",)},
    )

    allowed = allowed_capability_codes(
        "tree",
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(allowed) == {"TOOL", "DATETIME_TOOL", "TEST_TOOL", "WEB_TOOL"}


def test_exposure_unions_classifier_capabilities_by_default() -> None:
    """
    defaults 是地板，分类器本次识别的能力是增量，默认并入。
    """

    exposure = resolve_capability_exposure(
        "general",
        mode_defaults=[],
        requested=["AGENT_SUB_AGENT", "STOCK"],
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(exposure.effective) == {
        "TOOL",
        "DATETIME_TOOL",
        "CAPABILITY",
        "AGENT_SUB_AGENT",
        "STOCK",
    }
    assert exposure.clipped == ()


def test_plan_mode_adds_the_runtime_subtree() -> None:
    mode_defaults = default_capability_codes(
        "plan",
        codes=CODES,
        parent_of=PARENT_OF,
    )

    exposure = resolve_capability_exposure(
        "general",
        mode_defaults=mode_defaults,
        requested=[],
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(exposure.effective) == (
        RUNTIME_SUBTREE | {"TOOL", "DATETIME_TOOL", "CAPABILITY"}
    )
    assert exposure.clipped == ()


def test_classifier_capabilities_can_be_disabled(monkeypatch) -> None:
    """严格模式：只信全局配置，requested 只进 clipped。"""

    monkeypatch.setattr(
        capability_defaults,
        "ALLOW_CLASSIFIER_CAPABILITIES",
        False,
    )

    exposure = resolve_capability_exposure(
        "general",
        mode_defaults=[],
        requested=["STOCK"],
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(exposure.effective) == {
        "TOOL", "DATETIME_TOOL", "CAPABILITY",
    }
    assert set(exposure.clipped) == {"STOCK"}


def test_exposure_reports_clipped_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_defaults,
        "ALLOW_CLASSIFIER_CAPABILITIES",
        False,
    )
    monkeypatch.setattr(
        capability_defaults,
        "MODEL_CAPABILITIES",
        {"limited": ("TOOL",)},
    )

    exposure = resolve_capability_exposure(
        "limited",
        mode_defaults=[],
        requested=["AGENT_SUB_AGENT", "AGENT_RUNTIME"],
        codes=CODES,
        parent_of=PARENT_OF,
    )

    assert set(exposure.effective) == {"TOOL"}
    assert set(exposure.clipped) == {"AGENT_SUB_AGENT", "AGENT_RUNTIME"}
