"""
工具 ↔ 能力的关联契约。

对应 tool.yaml 的 /tools/{tool_id}/classifications（tag：
Tool Classifications）。

⚠️ 能力本体的契约不在这里 —— 它是 app/services/capability 的
Capability / CapabilityCreate / CapabilityUpdate / CapabilityReplace，
对应 capability.yaml。

为什么这一半留在 tool 模块
--------------------------
路径的主语是**工具**：GET/PUT /tools/{tool_id}/classifications 说的是
"这个工具关联了哪些能力"。关联行的生命周期跟着工具走（删工具就
解除关联），所以读写关联的契约、仓储、服务都留在 tool 模块。

术语也刻意保留 classification：能力本体改名 capability，但这条路径
描述的是工具这一侧的关联，没有理由跟着改 —— 改名的连带面越小越好。
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "ToolClassificationUpdate",
]


class ToolClassificationUpdate(APIModel):
    """
    整体替换一个 Tool 的能力关联。

    语义是"替换"不是"追加"：没列出来的能力会被解除关联。
    这样前端可以拿一个多选控件直接提交，不必自己算增删。
    """

    classification_ids: list[UUID] = Field(
        description="替换后的完整能力集合。传空数组即解除全部关联。",
    )
