"""
`app/common/providers_config.py` 的读取语义。

只验三件容易回归的事：

    1. 文件缺失时抛 ValueError（llm 依赖这条 fail fast）
    2. 段不存在时返回空 dict（web / rerank 依赖这条回落）
    3. 段内 `${ENV}` 被展开，且只展开取到的那一段
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.common.providers_config import (
    get_providers_section,
    load_providers_document,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不存在"):
        load_providers_document(tmp_path / "providers.yaml")


def test_missing_section_returns_empty(tmp_path: Path) -> None:
    path = _write(tmp_path / "providers.yaml", "llm:\n  default_provider: kimi\n")

    document = load_providers_document(path)

    assert get_providers_section(document, "web") == {}


def test_section_expands_env(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROVIDERS_TEST_KEY", "from-env")

    path = _write(
        tmp_path / "providers.yaml",
        "llm:\n"
        "  default_provider: kimi\n"
        "web:\n"
        "  api_key: ${PROVIDERS_TEST_KEY}\n"
        "  max_count: 7\n",
    )

    document = load_providers_document(path)

    assert get_providers_section(document, "web") == {
        "api_key": "from-env",
        "max_count": 7,
    }


def test_non_mapping_section_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "providers.yaml", "web:\n  - not-a-mapping\n")

    document = load_providers_document(path)

    with pytest.raises(ValueError, match="必须是映射"):
        get_providers_section(document, "web")
