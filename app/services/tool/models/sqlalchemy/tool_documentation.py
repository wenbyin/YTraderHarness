from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base


class ToolDocumentation(Base):
    """
    一个 Tool 的人工说明覆盖。

    这是**唯一**允许人写的地方 —— definition 归代码。
    同步永远不会写这张表，因此人工内容不会被代码升级冲掉。

    主键就是 tool_id：一个 Tool 至多一行说明，
    因此不需要自己的 id 列（与 models.sql 一致）。
    """

    __tablename__ = "tool_documentation"

    tool_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "tool_definition.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
        comment="所属 Tool 定义 ID。同时也是主键 —— 一对一。",
    )

    description_override: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="人工覆盖的工具描述。为空表示用代码里的 docstring。",
    )

    return_description_override: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="人工覆盖的返回值描述。",
    )

    updated_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="最后修改人。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后修改时间。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    tool: Mapped["ToolDefinition"] = relationship(
        back_populates="documentation",
    )
