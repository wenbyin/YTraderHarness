"""
工具分类的对外契约。

对应 tool.yaml 的 components/schemas：
    Classification / ClassificationCreate / ClassificationUpdate /
    ToolClassificationUpdate
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "Classification",
    "ClassificationCreate",
    "ClassificationUpdate",
    "ToolClassificationUpdate",
]


class Classification(APIModel):
    """一个分类节点。"""

    id: UUID = Field(description="分类 ID。")

    code: str = Field(
        description="稳定业务标识，全局唯一。关联与筛选都按它。",
    )

    name: str = Field(description="展示名。")

    description: str | None = Field(
        default=None,
        description="分类说明。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父分类 ID。为 None 表示一级分类。",
    )

    sort_order: int = Field(
        default=0,
        description="同级排序，小的在前。",
    )

    is_active: bool = Field(
        default=True,
        description="是否启用。",
    )


class ClassificationCreate(APIModel):
    """新建分类。"""

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
        description="分类说明。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父分类 ID。",
    )

    sort_order: int = Field(
        default=0,
        description="同级排序。",
    )


class ClassificationUpdate(APIModel):
    """
    局部更新分类。

    刻意**不含 code** —— 它是稳定标识，改了会让已有的
    tool 关联与外部引用对不上。要换标识请新建一个分类。
    """

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="展示名。",
    )

    description: str | None = Field(
        default=None,
        description="分类说明。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父分类 ID。",
    )

    sort_order: int | None = Field(
        default=None,
        description="同级排序。",
    )

    is_active: bool | None = Field(
        default=None,
        description="是否启用。",
    )


class ToolClassificationUpdate(APIModel):
    """
    整体替换一个 Tool 的分类关联。

    语义是"替换"不是"追加"：没列出来的分类会被解除关联。
    这样前端可以拿一个多选控件直接提交，不必自己算增删。
    """

    classification_ids: list[UUID] = Field(
        description="替换后的完整分类集合。传空数组即解除全部关联。",
    )
