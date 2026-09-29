"""
节点共享的小工具。

state 里的 ID 一律是 str（TypedDict 用 str 是为了 checkpoint
序列化友好），而 artifact_store / 仓储要 UUID。每次转换都手写
`UUID(state[...])` 容易漏、也容易在 None 上炸，因此统一走这里。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

__all__ = ["file_id_of", "tenant_of"]


def tenant_of(state: dict[str, Any]) -> UUID | None:
    """
    取当前版本的租户。可能为空（类型定义未设租户时）。

    产物按 (tenant_id, checksum) 去重，因此这里必须传准 ——
    传错会让不同租户的产物互相复用。
    """
    raw = state.get("tenant_id")
    return None if raw is None else UUID(str(raw))


def file_id_of(ref: dict[str, Any]) -> UUID:
    """
    从 ArtifactRef 取出 file_id。

    产物就是 File 模块里的一条 File，file_id 是它唯一的身份。
    """
    value = ref.get("file_id")

    if not value:
        raise ValueError(
            f"ArtifactRef is missing file_id: {sorted(ref)}"
        )

    return UUID(str(value))
