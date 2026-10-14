"""
能力的对外契约。

对应 capability.yaml 的 components/schemas：
    Capability / CapabilityCreate / CapabilityUpdate /
    CapabilityReplace
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "Capability",
    "CapabilityCreate",
    "CapabilityReplace",
    "CapabilityUpdate",
]


class Capability(APIModel):
    """一个能力节点。"""

    id: UUID = Field(description="能力 ID。")

    code: str = Field(
        description="稳定业务标识，全局唯一。关联与筛选都按它。",
    )

    name: str = Field(description="展示名。")

    description: str | None = Field(
        default=None,
        description="能力说明。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父能力 ID。为 None 表示一级能力。",
    )

    sort_order: int = Field(
        default=0,
        description="同级排序，小的在前。",
    )

    is_active: bool = Field(
        default=True,
        description="是否启用。",
    )


class CapabilityCreate(APIModel):
    """新建能力。"""

    code: str = Field(
        min_length=1,
        max_length=255,
        description="稳定业务标识，必须全局唯一。",
    )

    name: str = Field(
        min_length=1,
        max_length=255,
        description="展示名。",
    )

    description: str | None = Field(
        default=None,
        description="能力说明。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父能力 ID。",
    )

    sort_order: int = Field(
        default=0,
        description="同级排序。",
    )


class CapabilityUpdate(APIModel):
    """
    局部更新能力。

    刻意**不含 code** —— 它是稳定标识，改了会让已有的
    tool 关联与外部引用对不上。要换标识请新建一个能力。
    """

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="展示名。",
    )

    description: str | None = Field(
        default=None,
        description="能力说明。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父能力 ID。",
    )

    sort_order: int | None = Field(
        default=None,
        description="同级排序。",
    )

    is_active: bool | None = Field(
        default=None,
        description="是否启用。",
    )


class CapabilityReplace(APIModel):
    """
    整体替换能力节点（PUT）。

    字段与 CapabilityUpdate 相同，区别在**语义**：这里是替换，
    没传的 description / parent_id 按 null 处理（清空说明、提升为
    一级能力），而不是"保持原值"。局部更新请用 PATCH。

    sort_order 与 is_active 是**必填**，刻意不给默认值：
    PUT 的常见触发是"我只想改个名字"，若二者可省，一次改名就会
    把排序重置成 0、把停用的节点重新启用 —— 两种都是静默的、
    事后很难发现的破坏。可空字段能安全地"缺省即清空"，这两个不行。

    code 同样不在这里 —— 理由见 CapabilityUpdate。
    """

    name: str = Field(
        min_length=1,
        max_length=255,
        description="展示名。",
    )

    description: str | None = Field(
        default=None,
        description="能力说明。不传即为 null（清空）。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父能力 ID。不传即为 null（提升为一级能力）。",
    )

    sort_order: int = Field(
        description="同级排序，小的在前。必填，不给默认值。",
    )

    is_active: bool = Field(
        description="是否启用。必填，不给默认值。",
    )
