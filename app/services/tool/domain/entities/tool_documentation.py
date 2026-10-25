"""
Domain entities for tool documentation.

两层的说明文字：
    ToolDocumentation          整个工具的描述 / 返回值描述
    ParameterDocumentation     单个参数的描述，按 JSON Pointer 定位
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ParameterDocumentation",
    "ToolDocumentation",
]


class ParameterDocumentation(BaseModel):
    """
    单个参数的说明**覆盖**。

    为什么要 path 而不是参数名
    -------------------------
    input_schema 是一棵 JSON Schema 树，参数可能嵌在
    properties / items / $defs 里。用 JSON Pointer 能唯一指到
    任意深度的一个节点：

        /query                       顶层参数
        /top_k
        /filters/date_from           嵌套对象的字段
        /filters/ranges/0/field      数组元素里的字段

    只存**覆盖值**：代码里的 docstring 是 source，
    这里为空表示"沿用代码里的说明"。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    parameter_documentation_id: UUID = Field(
        description="参数说明的全局唯一 ID。",
    )

    tool_id: UUID = Field(
        description="所属 Tool 定义 ID。",
    )

    parameter_path: str = Field(
        description="参数在 input_schema 里的 JSON Pointer，例如 /query。",
    )

    description_override: str | None = Field(
        default=None,
        description="人工覆盖的参数说明。为空表示沿用代码里的说明。",
    )

    updated_by: str | None = Field(
        default=None,
        description="最后修改人。",
    )

    updated_at: datetime = Field(
        description="最后修改时间。",
    )


class ToolDocumentation(BaseModel):
    """
    一个 Tool 的人工说明覆盖。

    说明文字这一侧**唯一**允许人写的地方 —— definition 归代码
    （见 tool_definition.py 的说明），同步不会碰这里。

    运行期开关（禁用 / 启用）是另一件事，住在
    tool_definition_state；那边同样是人工写、同步不碰，但语义不同：
    它不属于"说明"。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    tool_id: UUID = Field(
        description="所属 Tool 定义 ID。一个 Tool 至多一行。",
    )

    description_override: str | None = Field(
        default=None,
        description="整个人工覆盖的工具描述。为空表示用代码里的 docstring。",
    )

    return_description_override: str | None = Field(
        default=None,
        description="人工覆盖的返回值描述。",
    )

    updated_by: str | None = Field(
        default=None,
        description="最后修改人。",
    )

    updated_at: datetime = Field(
        description="最后修改时间。",
    )

    # ========================================================
    # Composition
    # ========================================================

    parameter_documentations: list[ParameterDocumentation] = Field(
        default_factory=list,
        description="各参数的说明覆盖。",
    )
