"""
Mock 分类结果的 execution_plan 固定策略。

stress_eval 用 mock 模型时只该走 react：压测流量必须落在同一条主链路上，
随机分叉到 other 会让同一批指标混入两种执行路径。

这里钉住三层：
    1. 配置解析（react / random / 非法值）
    2. _classification_args 在强制 react 时的输出
    3. 强制值不在本次 schema 枚举里时回退到合法值
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.common.mock_model_config import MockModelConfig
from app.mock.model_server import MockModelServer
from app.mock.settings import (
    DEFAULT_MOCK_CONFIG_PATH,
    load_mock_config,
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "behavior": {"enum": ["qa"]},
        "execution_plan": {"enum": ["react", "other"]},
    },
    "required": ["behavior", "execution_plan"],
}


def _server(**overrides) -> MockModelServer:
    return MockModelServer(
        MockModelConfig(seed=7, **overrides)
    )


def test_execution_plan_config_parsing() -> None:
    assert (
        MockModelConfig(execution_plan="react").execution_plan
        == "react"
    )
    assert (
        MockModelConfig(execution_plan="REACT").execution_plan
        == "react"
    )

    # random / 空串 = 恢复随机。
    assert MockModelConfig(execution_plan="random").execution_plan is None
    assert MockModelConfig(execution_plan="").execution_plan is None

    with pytest.raises(ValidationError):
        MockModelConfig(execution_plan="plan")

    # direct 已从 ExecutionPlan 移除，不再是合法值。
    with pytest.raises(ValidationError):
        MockModelConfig(execution_plan="direct")


def test_repo_config_pins_react() -> None:
    config = load_mock_config(str(DEFAULT_MOCK_CONFIG_PATH))

    assert config.execution_plan == "react"


def test_classification_args_force_react() -> None:
    server = _server(execution_plan="react")

    for _ in range(20):
        args = server._classification_args("", _SCHEMA)

        assert args["execution_plan"] == "react"


def test_classification_args_random_when_unset() -> None:
    server = _server(execution_plan=None)

    plans = {
        server._classification_args("", _SCHEMA)["execution_plan"]
        for _ in range(20)
    }

    assert plans
    assert plans <= {"react", "other"}


def test_forced_plan_falls_back_when_schema_disallows_it() -> None:
    server = _server(execution_plan="react")

    schema = {
        "type": "object",
        "properties": {
            "behavior": {"enum": ["qa"]},
            "execution_plan": {"enum": ["other"]},
        },
        "required": ["behavior", "execution_plan"],
    }

    args = server._classification_args("", schema)

    assert args["execution_plan"] == "other"
