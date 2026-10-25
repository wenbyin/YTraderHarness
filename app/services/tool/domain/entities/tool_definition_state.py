"""
Domain entity for ToolDefinitionState.

一条"显式禁用"记录：**存在即禁用**，没有 enabled 字段。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ToolDefinitionState"]


class ToolDefinitionState(BaseModel):
    """
    Tool 的运行时开关。

    与 `ToolDefinition` 刻意分开：那个是代码侧事实，同步整行覆盖；
    这个是人在运行期写的状态，同步不会碰（见
    models/sqlalchemy/tool_definition_state.py 的说明）。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    tool_key: str = Field(
        description="被显式禁用的 Tool 的稳定业务标识。",
    )

    disabled_by: str | None = Field(
        default=None,
        description="最后执行禁用的人。",
    )

    disabled_at: datetime = Field(
        description="禁用时间。",
    )
