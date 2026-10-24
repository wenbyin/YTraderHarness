"""
Mock 模型服务的启用开关与配置加载。

配置放在 **独立文件** ``app/config/mock_model.yaml``，不写进 app.yaml：
``app/common/app_config.py`` 的 ``AppConfig`` 是 ``extra="forbid"`` 的，
新增一个顶层键会让现有加载器拒绝整份 app.yaml。mock 是开发功能，独立一份
文件既不用动生产配置的校验，也方便整体删除。

配置模型住在 ``app/common/mock_model_config.py``：它只依赖 pydantic，
common 层不会反过来 import 这个功能包。

启动参数 ``--mock`` / ``--no-mock`` 由 ``app/serve.py`` 写进环境变量
``MOCK_MODEL_ENABLED``，优先级高于 yaml 里的 ``enabled``。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from app.common.mock_model_config import (
    MockCapabilityConfig,
    MockClassificationConfig,
    MockModelConfig,
    MockModeConfig,
    MockToolCallConfig,
    NormalSpec,
)
from app.common.utils import expand_env

__all__ = [
    "DEFAULT_MOCK_CONFIG_PATH",
    "MockCapabilityConfig",
    "MockClassificationConfig",
    "MockModelConfig",
    "MockModeConfig",
    "MockToolCallConfig",
    "NormalSpec",
    "is_mock_enabled",
    "load_mock_config",
]

#: 默认配置文件位置 —— app/config/mock_model.yaml。
DEFAULT_MOCK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "mock_model.yaml"
)

_TRUE = frozenset({"1", "true", "yes", "on", "y"})


@lru_cache(maxsize=4)
def _load_mock_config(path: str | None = None) -> MockModelConfig:
    """
    读并校验 mock 配置。

    文件不存在时返回全默认值而不是报错：mock 是可选功能，关闭它的时候
    不应该因为少一份开发配置让服务起不来。
    """

    file_path = Path(path) if path else DEFAULT_MOCK_CONFIG_PATH

    if not file_path.exists():
        return MockModelConfig()

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}

    # 允许两种写法：
    #   mock_model: {...}     与 app.yaml 风格一致
    #   {...}                 直接把文件当 mock_model 段
    section = raw.get("mock_model", raw) if isinstance(raw, dict) else raw

    return MockModelConfig.model_validate(expand_env(section))


def load_mock_config(path: str | None = None) -> MockModelConfig:
    """读当前进程的 mock 配置（按路径缓存）。"""

    return _load_mock_config(path)


def is_mock_enabled() -> bool:
    """
    当前进程是否启用 mock。

    优先级：环境变量 ``MOCK_MODEL_ENABLED`` > ``mock_model.yaml`` 的
    ``enabled``。
    """

    raw = os.environ.get("MOCK_MODEL_ENABLED")

    if raw is not None and raw.strip():
        return raw.strip().lower() in _TRUE

    return bool(load_mock_config().enabled)
