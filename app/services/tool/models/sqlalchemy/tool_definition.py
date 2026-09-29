from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

from ...domain.schemas.enum import SourceState


class ToolDefinition(Base):
    """
    一个 Tool 的代码侧事实快照。

    这一行的内容全部由代码产生、由同步写入 ——
    没有任何接口能改它（见 domain/entities/tool_definition.py 的说明）。
    人工只能改 tool_documentation 的 override 字段。

    列名与 models.sql 一致（主键列叫 id），
    Python 属性叫 tool_id —— 与其余实体保持一致的读法，
    同时不必让 SQL 迁就 ORM 的命名习惯。
    """

    __tablename__ = "tool_definition"

    # ========================================================
    # Primary Key
    # ========================================================

    tool_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Tool 定义全局唯一 ID。",
    )

    # ========================================================
    # Business Identity
    # ========================================================

    tool_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment=(
            "Tool 的稳定业务标识，例如 document.list。"
            "同步按它比对 —— 不用主键是因为重新发现时主键会变。"
        ),
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="展示名。默认取方法名。",
    )

    # ========================================================
    # Source
    # ========================================================

    source_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="local_method / mcp / openapi。",
    )

    source_ref: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="来源对象，例如 service/module/mcp-server/openapi spec。",
    )

    entrypoint: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment=(
            "实际方法入口，例如 python.module:Class.method。"
            "仅供展示与审计，不是执行路径。"
        ),
    )

    # ========================================================
    # Schema
    # ========================================================

    input_schema: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="代码产生的输入 schema。",
    )

    output_schema: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="代码产生的输出 schema。",
    )

    source_docstring: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="代码 docstring 的原始/解析结果。",
    )

    # ========================================================
    # Change Detection
    # ========================================================

    manifest_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "上面这些字段的稳定摘要（sha256 十六进制，恰好 64 字符）。"
            "同步靠它判断 created / updated / unchanged。"
        ),
    )

    source_state: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=SourceState.ACTIVE.value,
        comment=(
            "ACTIVE=代码里还在；MISSING=代码里已找不到。"
            "⚠️ 取值大写，与 models.sql 的 DEFAULT 'ACTIVE' 一致。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="首次发现时间。",
    )

    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="最近一次同步时间。",
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
    #
    # 目标类写成**字符串**，不在本模块 import 它们 ——
    # SQLAlchemy 在 mapper 配置时（首次使用前）从注册表里解析，
    # 那时 models/sqlalchemy/__init__.py 已经把 5 个模块都加载了。
    # 这也避免了三个模型文件互相 import 形成环。
    # 与 document/models/sqlalchemy/document.py 同一套写法。

    documentation: Mapped["ToolDocumentation"] = relationship(
        back_populates="tool",
        cascade="all, delete-orphan",
        uselist=False,
    )

    parameter_documentations: Mapped[
        list["ToolParameterDocumentation"]
    ] = relationship(
        back_populates="tool",
        cascade="all, delete-orphan",
        order_by="ToolParameterDocumentation.parameter_path",
    )

    classification_links: Mapped[
        list["ToolDefinitionClassification"]
    ] = relationship(
        back_populates="tool",
        cascade="all, delete-orphan",
    )

    #: 处理器链。order_by 写死 sequence —— 链的语义就是有序的，
    #: 依赖调用方自己 sort 早晚会有人漏掉（那会静默改变执行顺序）。
    processors: Mapped[list["ToolProcessor"]] = relationship(
        back_populates="tool",
        cascade="all, delete-orphan",
        order_by="ToolProcessor.sequence",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        Index("idx_tool_definition_source_type", "source_type"),
        Index("idx_tool_definition_source_state", "source_state"),
    )
