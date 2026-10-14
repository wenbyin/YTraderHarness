from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:  # pragma: no cover - 仅为 relationship 目标名
    from app.services.tool.models.sqlalchemy.tool_classification import (
        ToolDefinitionClassification,
    )


class Capability(Base):
    """
    业务能力树的一个节点。

    用途是给 LLM 与 JEV 筛选工具（"这个 Agent 只该看到检索类能力"）。
    表名是 capability —— 历史上叫 tool_classification，改名时它已经
    与工具解耦：它是一棵独立的树，只是被工具引用。

    自引用外键构成树。刻意**不加** CASCADE：
    删父节点时如果级联删掉整棵子树，会连带解除一批 Tool 的关联，
    影响面太大且无声。删之前请先把子节点挪走或停用 ——
    停用（is_active=False）才是常规操作。

    它与 Tool 的关联是多对多，关联表在 tool 模块
    （tool_definition_classification），实体是 ToolDefinitionClassification。
    """

    __tablename__ = "capability"

    capability_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="能力全局唯一 ID。",
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
        comment="能力说明。",
    )

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("capability.id"),
        nullable=True,
        comment="父能力 ID。为 None 表示一级能力。",
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

    #: 指向 Tool 的关联行。**跨模块**（关联模型在 tool 模块），
    #: 因此这里只能是字符串目标 + TYPE_CHECKING 导入。
    #: 代码里从不遍历它 —— 关联的读写都走显式 JOIN/SELECT
    #: （见 repository/capability_impl.py 的 count_links）；
    #: 保留它只为表达"删掉这个能力，关联行会跟着走"这一层
    #: 数据库事实（ON DELETE CASCADE）。
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
        Index("idx_capability_parent", "parent_id"),
        Index("idx_capability_active", "is_active"),
    )
