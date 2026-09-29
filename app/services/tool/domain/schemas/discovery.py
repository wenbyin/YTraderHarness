"""
发现与同步的对外契约。

对应 tool.yaml 的 components/schemas：
    ToolSyncRequest / ToolSyncResult
"""

from __future__ import annotations

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "ToolSyncRequest",
    "ToolSyncResult",
]


class ToolSyncRequest(APIModel):
    """同步请求。整体可选 —— 不带 body 就是"全量同步"。"""

    source_types: list[str] = Field(
        default_factory=list,
        description=(
            "可选的来源种类过滤。为空表示全部可发现的来源。"
        ),
    )

    source_refs: list[str] = Field(
        default_factory=list,
        description="可选的来源对象过滤，例如只要 document 模块的。",
    )

    remove_missing: bool = Field(
        default=False,
        description=(
            "是否**物理删除**本次没发现的定义。"
            "默认 false —— 建议保持：Tool 可能已被编排进某个 "
            "Agent / 工作流，行删了引用就断了，报错也指不到原因。"
            "默认只转成 MISSING，让调用方自己决定怎么处理。"
        ),
    )


class ToolSyncResult(APIModel):
    """同步结果计数。"""

    scanned: int = Field(
        default=0,
        description="本次扫描到的工具定义数。",
    )

    created: int = Field(
        default=0,
        description="新建的行数。",
    )

    updated: int = Field(
        default=0,
        description="内容有变化、被覆盖的行数。",
    )

    unchanged: int = Field(
        default=0,
        description="摘要一致、原样跳过的行数。",
    )

    missing: int = Field(
        default=0,
        description="库里 ACTIVE 但本次没扫到、已标记为 MISSING 的行数。",
    )

    orphaned_documentations: int = Field(
        default=0,
        description=(
            "随 remove_missing 一起被级联删除的说明行数。"
            "remove_missing=false 时恒为 0。"
        ),
    )
