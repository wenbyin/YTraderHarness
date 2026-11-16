from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ...domain.schemas.enum import SkillStatus


class Skill(Base):
    """
    一份 Skill 定义。

    没有版本表：更新就地生效。运行时在 spawn 子 agent 时把定义
    快照进节点状态，运行中的分支不受后续 PATCH 影响。

    skill_key 是稳定标识（创建后不可改）—— 后续 task board / run
    记录引用它或 id；改名只改 name。
    """

    __tablename__ = "skill"

    skill_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Skill 全局唯一 ID。",
    )

    skill_key: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        unique=True,
        comment="稳定业务标识，创建后不可改。",
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="展示名。",
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="给主 agent 选择 skill 时读的文本。",
    )

    # ========================================================
    # A2A AgentSkill 元数据
    # ========================================================
    #
    # 这些字段只在**对外发布 AgentCard 时**使用；内部 subagent
    # 运行不读它们。instructions 永远不进卡片。

    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
        server_default=text("'{}'::text[]"),
        comment=(
            "A2A AgentSkill.tags。标准要求必填，但允许空数组。"
        ),
    )

    examples: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
        comment="A2A AgentSkill.examples（可选，字符串数组）。",
    )

    instructions: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "注入子 agent 节点的行为提示词 / SOP。"
            "内部字段，never exported to an A2A AgentCard."
        ),
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=SkillStatus.DRAFT.value,
        server_default=SkillStatus.DRAFT.value,
        comment="DRAFT / ACTIVE / DISABLED。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后更新时间。",
    )

    # ========================================================
    # 容量度量（派生列，只读）
    # ========================================================
    #
    # 这些列**不接受写请求**：SkillCreate / SkillUpdate 是
    # additionalProperties: false，多带一个字段就 422。它们只由服务层
    # 在 create / update / 换绑定时测量并落库。
    #
    # 字符数刻意不存：`instructions.length` 前端实时算即可，落库就会
    # 和原文产生第二份可漂移的事实。token 必须后端算 —— 前端没有编码器，
    # 而且要把口径（编码名）连同数字一起存下来，否则数字无法解释。
    #
    # 存量行的默认态是「未测量」：0 / 0 / 'unmeasured' / NULL。
    # 前端据此显示「未测量」，不会把 0 当成真实结果。任何一次保存或
    # 改绑定都会把它们刷成实测值。

    instructions_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        comment="仅 instructions 原文的 token 数；0 + unmeasured 表示未测量。",
    )

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        comment=(
            "SkillSnapshot.render_prompt() 渲染出的完整 prompt 的 token 数。"
        ),
    )

    token_encoding: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="unmeasured",
        server_default="unmeasured",
        comment=(
            "实测口径：o200k_base / heuristic:3.0 / unmeasured。"
        ),
    )

    token_computed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="最近一次测量时间（UTC）；NULL 表示从未测量。",
    )

    __table_args__ = (
        CheckConstraint(
            "skill_key ~ '^[a-z][a-z0-9_]{1,119}$'",
            name="ck_skill_key_format",
        ),
        CheckConstraint(
            "status IN ('DRAFT', 'ACTIVE', 'DISABLED')",
            name="ck_skill_status",
        ),
        CheckConstraint(
            "jsonb_typeof(examples) = 'array'",
            name="ck_skill_examples_array",
        ),
        Index("idx_skill_status", "status"),
    )
