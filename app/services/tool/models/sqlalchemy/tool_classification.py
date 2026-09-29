from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base


class ToolClassification(Base):
    """
    业务分类树的一个节点。

    用途是给 LLM 与 JEV 筛选工具（"这个 Agent 只该看到检索类工具"）。

    自引用外键构成树。刻意**不加** CASCADE：
    删父节点时如果级联删掉整棵子树，会连带解除一批 Tool 的关联，
    影响面太大且无声。删之前请先把子节点挪走或停用 ——
    停用（is_active=False）才是常规操作。

    它与 Tool 的关联是多对多，见 ToolDefinitionClassification。
    """

    __tablename__ = "tool_classification"

    classification_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="分类全局唯一 ID。",
    )

    code: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="稳定业务标识，全局唯一。关联与筛选都按它。",
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="展示名。",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="分类说明。",
    )

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tool_classification.id"),
        nullable=True,
        comment="父分类 ID。为 None 表示一级分类。",
    )

    sort_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="同级排序，小的在前。",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="是否启用。",
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
    # Relationships
    # ========================================================

    definition_links: Mapped[
        list["ToolDefinitionClassification"]
    ] = relationship(
        back_populates="classification",
        cascade="all, delete-orphan",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        Index("idx_tool_classification_parent", "parent_id"),
        Index("idx_tool_classification_active", "is_active"),
    )


class ToolDefinitionClassification(Base):
    """
    Tool 与分类的多对多关联。

    ⚠️ **这张表不在 models.sql 里** —— 是我补的。

    models.sql 给了 tool_classification（带 UNIQUE code 与 parent_id，
    显然是一棵分类树），但 tool.yaml 有
    `PUT /tools/{tool_id}/classifications`（body 是 classification_ids），
    那需要的是 tool 与分类之间的关联，规格里缺这张表。

    双列主键而不是自增 id：关联本身没有身份，唯一的约束就是
    "同一个 (tool, classification) 只能出现一次"，而主键正好表达它。

    两个索引方向都要：按 tool 查它的分类（详情页），
    以及按 classification 反查有哪些 tool（按分类筛工具）。
    tool_id 那个方向由主键的前缀覆盖，因此只需再补 classification 方向。
    """

    __tablename__ = "tool_definition_classification"

    tool_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "tool_definition.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
        comment="Tool 定义 ID。",
    )

    classification_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "tool_classification.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
        comment="分类 ID。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    tool: Mapped["ToolDefinition"] = relationship(
        back_populates="classification_links",
    )

    classification: Mapped["ToolClassification"] = relationship(
        back_populates="definition_links",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        Index(
            "idx_tool_def_classification_classification",
            "classification_id",
        ),
    )
