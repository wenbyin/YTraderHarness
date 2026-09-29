"""
Domain entity for ToolClassification.

分类是一棵**树**，不是标签表：
    tool_classification 自己带 parent_id，靠它组织层级
    （Non-responsibilities 里写明的 Tag management 是另一回事）。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ToolClassification"]


class ToolClassification(BaseModel):
    """
    一个业务分类节点。

    用途是给 LLM 与 JEV 筛选工具：一个 Agent 只该看到
    "检索类"工具，就不必把写操作也塞进它的上下文。

    层级与启用状态是两把独立的开关：
        parent_id   组织关系，决定树的形状
        is_active   启用状态，决定这个节点（及其子树）要不要被用

    删一个分类会连带删掉它的关联行（见 ORM 的 CASCADE），
    因此"停用"优先用 is_active=False 而不是删除。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    classification_id: UUID = Field(
        description="分类的全局唯一 ID。",
    )

    code: str = Field(
        description=(
            "稳定业务标识，全局唯一。"
            "与 ToolDefinition.tool_key 同理 —— 关联与筛选都按它，"
            "不按 ID。"
        ),
    )

    name: str = Field(
        description="展示名。",
    )

    description: str | None = Field(
        default=None,
        description="分类说明。",
    )

    parent_id: UUID | None = Field(
        default=None,
        description="父分类 ID。为 None 表示它是一级分类。",
    )

    sort_order: int = Field(
        default=0,
        description="同级排序，小的在前。",
    )

    is_active: bool = Field(
        default=True,
        description="是否启用。False 时不应被用于筛选。",
    )

    created_at: datetime = Field(
        description="创建时间。",
    )

    updated_at: datetime = Field(
        description="最后更新时间。",
    )
