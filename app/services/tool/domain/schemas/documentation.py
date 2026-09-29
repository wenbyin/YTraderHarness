"""
Tool 说明文字的对外契约。

对应 tool.yaml 的 components/schemas：
    ToolDocumentation / ParameterDocumentation /
    ToolDocumentationUpdate / ParameterDocumentationUpdate

本模块刻意**不 import tool.py** —— 那边要引用这里的
ToolDocumentation，单向依赖才不会有循环。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "ParameterDocumentation",
    "ParameterDocumentationUpdate",
    "ToolDocumentation",
    "ToolDocumentationUpdate",
]


class ParameterDocumentation(APIModel):
    """
    单个参数的说明：代码里的原始值 + 人工覆盖 + 生效值。

    三个都给出来，前端才能显示"这个参数被人工改过"，
    而不是只看到一个分不清来源的字符串。
    """

    parameter_path: str = Field(
        description="参数在 input_schema 里的 JSON Pointer，例如 /query。",
    )

    source_description: str | None = Field(
        default=None,
        description="代码 docstring / schema 里的原始说明。只读。",
    )

    description_override: str | None = Field(
        default=None,
        description="人工覆盖值。为空表示沿用原始说明。",
    )

    effective_description: str | None = Field(
        default=None,
        description="生效值：覆盖优先，否则回退到原始说明。只读。",
    )


class ParameterDocumentationUpdate(APIModel):
    """单个参数的覆盖写入。"""

    parameter_path: str = Field(
        min_length=1,
        max_length=500,
        description="参数路径（JSON Pointer）。",
    )

    description_override: str | None = Field(
        default=None,
        description="覆盖值。传 null 表示撤回到代码里的原始说明。",
    )


class ToolDocumentation(APIModel):
    """一个 Tool 的说明总览。"""

    tool_id: UUID = Field(description="所属 Tool 定义 ID。")

    description_override: str | None = Field(
        default=None,
        description="人工覆盖的工具描述。",
    )

    return_description_override: str | None = Field(
        default=None,
        description="人工覆盖的返回值描述。",
    )

    parameter_descriptions: list[ParameterDocumentation] = Field(
        default_factory=list,
        description="各参数的说明。",
    )

    effective_description: str | None = Field(
        default=None,
        description="生效的工具描述。只读。",
    )

    effective_return_description: str | None = Field(
        default=None,
        description="生效的返回值描述。只读。",
    )

    updated_by: str | None = Field(
        default=None,
        description="最后修改人。只读。",
    )

    updated_at: datetime | None = Field(
        default=None,
        description="最后修改时间。从未人工修改过时为 None。只读。",
    )


class ToolDocumentationUpdate(APIModel):
    """
    整体替换一个 Tool 的说明覆盖。

    与 ToolClassificationUpdate 同样是**替换**语义：
    parameter_descriptions 里没列出的参数，其覆盖会被清掉
    （回到代码里的原始说明）。
    """

    description_override: str | None = Field(
        default=None,
        description="工具描述覆盖。传 null 表示撤回到代码里的 docstring。",
    )

    return_description_override: str | None = Field(
        default=None,
        description="返回值描述覆盖。",
    )

    parameter_descriptions: list[ParameterDocumentationUpdate] = Field(
        default_factory=list,
        description="参数说明覆盖的完整集合。",
    )
