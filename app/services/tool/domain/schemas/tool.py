"""
Tool 定义的对外契约。

对应 tool.yaml 的 components/schemas：
    ToolManifest / ToolSummary / ToolDefinitionDetail /
    PagedToolSummaryResponse
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

from .classification import Classification
from .documentation import ToolDocumentation
from .enum import SourceState, SourceType

__all__ = [
    "PagedToolSummaryResponse",
    "ToolDefinitionDetail",
    "ToolManifest",
    "ToolSummary",
]


class ToolManifest(APIModel):
    """
    代码侧事实的完整快照。

    ⚠️ 这里的字段**全部**是只读的 —— 一个都没有对应的写接口。
    要改它们只能改代码，然后跑一次同步（POST /tools/discovery/sync）。

    想改说明文字请用 ToolDocumentation 的 override 字段，
    那条路不会被同步覆盖。
    """

    id: UUID = Field(
        description="Tool 定义 ID。",
    )

    tool_key: str = Field(
        description="稳定业务标识，例如 document.list。",
    )

    name: str = Field(
        description="展示名。",
    )

    source_type: SourceType = Field(
        description="来源种类。",
    )

    source_ref: str | None = Field(
        default=None,
        description="来源对象，例如所属业务模块名。",
    )

    entrypoint: str | None = Field(
        default=None,
        description=(
            "方法入口，形如 python.module:Class.method。"
            "仅供人看与审计，不是执行路径。"
        ),
    )

    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="输入 JSON Schema。",
    )

    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="输出 JSON Schema。",
    )

    source_docstring: dict[str, Any] | None = Field(
        default=None,
        description=(
            "从代码 docstring 解析出的结构："
            "raw 原文、description 正文、"
            "returns 返回值说明（DTO 优先）、"
            "parameters 参数说明（JSON Pointer -> 说明）。"
            "取不到的键不出现。"
        ),
    )

    manifest_hash: str = Field(
        description="内容摘要，同步靠它判断有没有变。",
    )

    source_state: SourceState = Field(
        description="ACTIVE=代码里还在；MISSING=代码里已找不到。",
    )

    discovered_at: datetime = Field(
        description="首次发现时间。",
    )

    last_synced_at: datetime = Field(
        description="最近一次同步时间。",
    )


class ToolSummary(APIModel):
    """列表页用的精简视图。"""

    id: UUID = Field(description="Tool 定义 ID。")

    tool_key: str = Field(description="稳定业务标识。")

    name: str = Field(description="展示名。")

    source_type: SourceType = Field(description="来源种类。")

    source_state: SourceState = Field(description="代码侧状态。")

    description: str | None = Field(
        default=None,
        description=(
            "生效描述：人工覆盖优先，没有则回退到代码里的 docstring。"
        ),
    )

    classifications: list[Classification] = Field(
        default_factory=list,
        description="已关联的分类。",
    )


class ToolDefinitionDetail(APIModel):
    """详情页：事实 + 人工说明 + 分类，一次给全。"""

    manifest: ToolManifest = Field(
        description="代码侧事实。",
    )

    documentation: ToolDocumentation = Field(
        description="人工说明与生效值。",
    )

    classifications: list[Classification] = Field(
        default_factory=list,
        description="已关联的分类。",
    )


class PagedToolSummaryResponse(APIModel):
    """分页的工具摘要列表。"""

    items: list[ToolSummary] = Field(
        default_factory=list,
        description="本页数据。",
    )

    page: int = Field(description="当前页，从 1 开始。")

    page_size: int = Field(description="每页条数。")

    total: int = Field(description="符合条件的总条数。")
