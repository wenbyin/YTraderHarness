"""
Domain entity for ToolPresentation.

一个工具的展示描述：**存在即显示**。没有行 = 默认不显示。

⚠️ 与 ToolDefinitionState 是两张表、两条规则
------------------------------------------
    tool_definition_state   有行 = **禁用**（默认启用）
    tool_presentation       有行 = **显示**（默认隐藏）

默认方向相反，且写入生命周期不同（下线是 DELETE 行；显示是
upsert 整份描述符），所以不合并。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import ToolRenderer

__all__ = ["ToolPresentation"]


class ToolPresentation(BaseModel):
    """一条 presenter 记录。"""

    model_config = ConfigDict(
        from_attributes=True,
    )

    tool_key: str = Field(
        description="稳定业务标识，与 tool_definition.tool_key 对应。",
    )

    renderer: ToolRenderer = Field(
        description="渲染形态。",
    )

    display_name: str | None = Field(
        default=None,
        description="给人看的标题。",
    )

    summary: str | None = Field(
        default=None,
        description="点路径模板。",
    )

    show_input: bool = Field(
        default=False,
        description="是否把入参放进事件载荷。",
    )

    show_output: bool = Field(
        default=False,
        description="是否把结果放进事件载荷。",
    )

    props: dict[str, Any] = Field(
        default_factory=dict,
        description="renderer 专属内容。",
    )

    updated_by: str | None = Field(
        default=None,
        description="最后修改人。",
    )

    updated_at: datetime = Field(
        description="最后修改时间。",
    )
