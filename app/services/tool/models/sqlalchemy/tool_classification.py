from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

if TYPE_CHECKING:  # pragma: no cover - 仅为 relationship 目标名
    from app.services.capability.models.sqlalchemy.capability import (
        Capability,
    )


class ToolDefinitionClassification(Base):
    """
    Tool 与能力（capability）的多对多关联。

    ⚠️ **这张表不在 models.sql 里** —— 是我补的。

    models.sql 给了 capability（带 UNIQUE code 与 parent_id，
    显然是一棵分类树），但 tool.yaml 有
    `PUT /tools/{tool_id}/classifications`（body 是 classification_ids），
    那需要的是 tool 与能力之间的关联，规格里缺这张表。

    双列主键而不是自增 id：关联本身没有身份，唯一的约束就是
    "同一个 (tool, classification) 只能出现一次"，而主键正好表达它。

    两个索引方向都要：按 tool 查它的能力（详情页），
    以及按 classification 反查有哪些 tool（按能力筛工具）。
    tool_id 那个方向由主键的前缀覆盖，因此只需再补 classification 方向。

    名字保持 classification 不变
    ----------------------------
    被引用的那张表从 tool_classification 改名成了 capability
    （搬到了 app/services/capability/），但**关联这一侧不动**：
    表名、列名 classification_id、以及它支撑的
    /tools/{tool_id}/classifications 路径都保持原样 —— 那些路径的
    主语是工具，工具这一侧的术语没有理由跟着能力改名。

    外键指向改成了 capability.id，那是改名的必然连带：
    PostgreSQL 侧 RENAME 会自动更新约束，这里跟着改字符串即可。
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
            "capability.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
        comment="能力 ID。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    tool: Mapped["ToolDefinition"] = relationship(
        back_populates="classification_links",
    )

    #: 指向能力。**跨模块**（Capability 在 app/services/capability/），
    #: 因此只能是字符串目标 + TYPE_CHECKING 导入，SQLAlchemy 到
    #: mapper 配置时才解析。代码里从不遍历它 —— 关联的读写都走
    #: 显式 JOIN/SELECT（见 repository/tool_classification_impl.py）。
    classification: Mapped["Capability"] = relationship(
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
