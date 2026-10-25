"""
`providers.yaml` 的共享读取。

为什么要合并
------------
`llm` / `web` / `rerank` 三类外部 HTTP provider 的连接信息本来就同源：

    web     缺项时从 llm 的 deepseek provider 兜底
    rerank  缺项时从 llm 的 qwen provider 兜底

分成三份文件时，同一把 key 要抄三处，抄漏一处的症状是「某一类调用突然
401」。合并后只有一份密钥文件，三个加载器读同一份文档、各取自己那段。

这里只做两件事
--------------
1. 读文件；
2. 取段并展开 `${ENV}`。

**不做 DTO 校验**：每段的形状仍由各自的 Pydantic 模型负责，所以三份
配置的校验语义、缓存语义、缺项回落行为都不变。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from app.common.utils.env import expand_env

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "get_providers_section",
    "load_providers_document",
]


#: 默认配置文件位置 —— `app/config/providers.yaml`。
#:
#: 与旧版 `llm.yaml` / `web.yaml` / `rerank.yaml` 同一层。文件含密钥，
#: 已被 `.gitignore` 按路径排除；仓库里只留 `.template`。
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "providers.yaml"
)


def load_providers_document(
        path: str | Path | None = None,
) -> dict[str, Any]:
    """
    读整份 `providers.yaml`。**不做环境变量展开**。

    文件不存在时抛 ValueError（与旧 `load_llm_config` 一致）。需要
    「缺文件就回落」的调用方（web / rerank）自己先判 `exists()`，不要
    把两种语义混在一起。

    Raises:
        ValueError: 文件不存在，或顶层不是映射。
    """

    file_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not file_path.exists():
        raise ValueError(
            f"provider 配置文件不存在：{file_path}。"
            "请从 app/config/providers.yaml.template 复制一份，"
            "或检查部署时的配置挂载。"
        )

    raw = yaml.safe_load(
        file_path.read_text(encoding="utf-8")
    ) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"provider 配置的顶层必须是映射：{file_path}。"
            f"实际是 {type(raw).__name__}。"
        )

    return raw


def get_providers_section(
        document: Mapping[str, Any],
        name: str,
) -> dict[str, Any]:
    """
    取一段并展开其中的 `${ENV}`。

    段不存在返回空 dict —— 这与旧实现「文件缺失 / section 为空就回落」
    的语义一致，由调用方决定是否再套默认值。

    Raises:
        ValueError: 段存在但不是映射（例如写成了列表或字符串）。
    """

    value = document.get(name)

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(
            f"providers.yaml 的 {name!r} 段必须是映射，"
            f"实际是 {type(value).__name__}。"
        )

    return expand_env(value)
