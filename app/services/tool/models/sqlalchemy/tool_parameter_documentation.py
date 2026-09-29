from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base


class ToolParameterDocumentation(Base):
    """
    单个参数的说明覆盖。

    用 JSON Pointer 定位参数（例如 /filters/date_from），
    而不是参数名 —— input_schema 是一棵 JSON Schema 树，
    同名参数可能出现在不同深度。

    只存覆盖值：代码 docstring 里的是 source，
    这里为空表示"沿用代码里的说明"。
    """

    __tablename__ = "tool_parameter_documentation"

    parameter_documentation_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="参数说明全局唯一 ID。",
    )

    tool_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "tool_definition.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属 Tool 定义 ID。",
    )

    parameter_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="参数在 input_schema 里的 JSON Pointer，例如 /query。",
    )

    description_override: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="人工覆盖的参数说明。",
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
        back_populates="parameter_documentations",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tool_id",
            "parameter_path",
            name="uq_tool_parameter_doc_path",
        ),
        Index("idx_tool_parameter_doc_tool", "tool_id"),
    )
