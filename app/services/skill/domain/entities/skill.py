"""
Domain entity for Skill.

一行 skill = 一份可被主 agent 选中、由子 agent 执行的行为定义。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import SkillStatus

__all__ = ["Skill"]


class Skill(BaseModel):
    """
    一个 Skill 定义。

    description 与 instructions 是**两个读者、两种用途**：
        description   主 agent 选 skill 时读
        instructions  子 agent 节点启动时注入

    不要合成一个字段 —— 主 agent 不需要看到完整 SOP，子 agent 也不该
    靠一句摘要来行动。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    skill_id: UUID = Field(
        description="Skill 的全局唯一 ID。",
    )

    skill_key: str = Field(
        description=(
            "稳定业务标识，创建后不可改。"
            "后续 task board / run 记录按它或 skill_id 引用本行。"
        ),
    )

    name: str = Field(
        description="展示名。",
    )

    description: str = Field(
        description="给主 agent 选择 skill 时读的文本。",
    )

    # ========================================================
    # A2A AgentSkill 元数据
    # ========================================================

    tags: list[str] = Field(
        default_factory=list,
        description=(
            "A2A AgentSkill.tags。标准必填字段，允许空数组。"
        ),
    )

    examples: list[str] = Field(
        default_factory=list,
        description="A2A AgentSkill.examples（可选）。",
    )

    instructions: str = Field(
        description="注入子 agent 节点的行为提示词 / SOP。",
    )

    status: SkillStatus = Field(
        description="生命周期状态。只有 ACTIVE 会被主 agent 选中。",
    )

    created_at: datetime = Field(
        description="创建时间。",
    )

    updated_at: datetime = Field(
        description="最后更新时间。",
    )
