"""
Capability 模块的对外契约。

domain/schemas/
└── capability.py
    └── Capability 及其 Create / Update / Replace

全部结构逐字段对应 capability.yaml 的 components/schemas；
字段名与枚举取值都以那份 spec 为准。

⚠️ 工具 ↔ 能力的**关联**契约不在这里 —— 它留在 tool 模块
（ToolClassificationUpdate，对应 /tools/{tool_id}/classifications），
因为那个路径的主语是工具。
"""

from .capability import (
    Capability,
    CapabilityCreate,
    CapabilityReplace,
    CapabilityUpdate,
)

__all__ = [
    "Capability",
    "CapabilityCreate",
    "CapabilityReplace",
    "CapabilityUpdate",
]
