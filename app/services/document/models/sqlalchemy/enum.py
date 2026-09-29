"""
兼容层。

枚举已统一到 domain/schemas/enum.py，避免两处定义漂移。

本模块只做转发，保留旧导入路径可用：

    from ...models.sqlalchemy.enum import ProcessingStatus

新代码请直接从 domain.schemas.enum 导入。
"""

from __future__ import annotations

from ...domain.schemas.enum import (
    ChunkMode,
    DefinitionStatus,
    DocumentStatus,
    ProcessingStatus,
    RetrievalMethod,
    VersionMode,
    VersionStatus,
)

__all__ = [
    "ChunkMode",
    "DefinitionStatus",
    "DocumentStatus",
    "ProcessingStatus",
    "RetrievalMethod",
    "VersionMode",
    "VersionStatus",
]
