"""
Skill 的对外契约。

对应 skill.yaml 的 components/schemas：
    SkillCreate / SkillUpdate / SkillSummary / SkillDetail /
    PagedSkillResponse / SkillCapabilityUpdate / CapabilityRef
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

from .enum import SkillStatus

__all__ = [
    "CapabilityRef",
    "PagedSkillResponse",
    "SkillCapabilityUpdate",
    "SkillCreate",
    "SkillDetail",
    "SkillSummary",
    "SkillUpdate",
]


class CapabilityRef(APIModel):
    """
    Skill 侧看到的能力。

    只带识别一个绑定所需的字段；能力本体（层级、启用状态、说明）
    以 capability.yaml 的 Capability 为准，不在这里复制一份。
    """

    id: UUID = Field(description="能力 ID。")

    code: str = Field(description="能力的稳定业务标识。")

    name: str = Field(description="能力展示名。")


class SkillCreate(APIModel):
    """新建 Skill。"""

    skill_key: str = Field(
        min_length=2,
        max_length=120,
        pattern=r"^[a-z][a-z0-9_]{1,119}$",
        description=(
            "稳定业务标识。创建后不可改，"
            "后续表按它或 id 引用本行。"
        ),
    )

    name: str = Field(
        min_length=1,
        max_length=200,
        description="展示名。",
    )

    description: str = Field(
        min_length=1,
        description="给主 agent 选择 skill 时读的文本。",
    )

    tags: list[str] = Field(
        default_factory=list,
        description=(
            "A2A AgentSkill.tags。标准必填字段，允许空数组；"
            "只在发布 AgentCard 时使用。"
        ),
    )

    examples: list[str] = Field(
        default_factory=list,
        description="A2A AgentSkill.examples（可选）。",
    )

    instructions: str = Field(
        min_length=1,
        description="注入子 agent 节点的行为提示词 / SOP。",
    )

    status: SkillStatus = Field(
        default=SkillStatus.DRAFT,
        description="初始状态。默认 DRAFT。",
    )


class SkillUpdate(APIModel):
    """
    局部更新 Skill。

    刻意**不含 skill_key** —— 它是稳定标识，后续 task board / run
    记录会引用它。要换标识请新建一个 Skill。

    字段都可选，但**不接受显式 null**：这几列在库里都是 NOT NULL，
    传 null 会被服务层以 400 拒绝，而不是让数据库抛完整性错误。
    """

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="展示名。",
    )

    description: str | None = Field(
        default=None,
        min_length=1,
        description="给主 agent 选择 skill 时读的文本。",
    )

    tags: list[str] | None = Field(
        default=None,
        description=(
            "A2A AgentSkill.tags。显式 null 会被拒绝；"
            "清空请传空数组。"
        ),
    )

    examples: list[str] | None = Field(
        default=None,
        description=(
            "A2A AgentSkill.examples。显式 null 会被拒绝；"
            "清空请传空数组。"
        ),
    )

    instructions: str | None = Field(
        default=None,
        min_length=1,
        description="注入子 agent 节点的行为提示词 / SOP。",
    )

    status: SkillStatus | None = Field(
        default=None,
        description="生命周期状态。",
    )


class SkillSummary(APIModel):
    """列表页用的精简视图。"""

    id: UUID = Field(description="Skill ID。")

    skill_key: str = Field(description="稳定业务标识。")

    name: str = Field(description="展示名。")

    description: str = Field(
        description="给主 agent 选择 skill 时读的文本。",
    )

    tags: list[str] = Field(
        description="A2A AgentSkill.tags。",
    )

    status: SkillStatus = Field(description="生命周期状态。")

    created_at: datetime = Field(description="创建时间。")

    updated_at: datetime = Field(description="最后更新时间。")


class SkillDetail(SkillSummary):
    """详情页：摘要 + A2A 元数据 + 提示词 + 能力绑定。"""

    examples: list[str] = Field(
        description="A2A AgentSkill.examples。",
    )

    instructions: str = Field(
        description="注入子 agent 节点的行为提示词 / SOP。",
    )

    capabilities: list[CapabilityRef] = Field(
        description=(
            "已绑定的能力。空列表表示该 skill 看不到任何工具。"
        ),
    )


class PagedSkillResponse(APIModel):
    """分页的 Skill 摘要列表。"""

    items: list[SkillSummary] = Field(
        description="本页数据。",
    )

    page: int = Field(description="当前页，从 1 开始。")

    page_size: int = Field(description="每页条数。")

    total: int = Field(description="符合条件的总条数。")


class SkillCapabilityUpdate(APIModel):
    """
    全量替换 Skill 的能力绑定。

    语义是替换不是追加：没列出来的会被解除绑定。
    传空数组表示清空 —— 该 skill 将看不到任何工具。
    """

    capability_ids: list[UUID] = Field(
        description=(
            "替换后的能力 ID 集合。必须显式提供 —— 缺省成空数组"
            "会让一次手误的 PUT 静默清空全部绑定。"
        ),
    )
