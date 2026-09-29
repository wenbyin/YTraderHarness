from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

from ...domain.schemas.enum import DocumentStatus


# ============================================================
# 2. Document
# ============================================================

class Document(Base):
    """
    一个逻辑 Document。

    Document 表示“业务上的同一个文档”。

    例如：

        user_profile
            version 1 -> profile.md
            version 2 -> profile_v2.md
            version 3 -> profile_v3.md

    Document 本身不保存 source file 信息。

    特别注意：
        file_id 不应该放在 Document，
        而应该放在 DocumentVersion。

    原因：
        每一次 update 都可能产生一个新的 File。
    """

    __tablename__ = "document"

    # ========================================================
    # Primary Key
    # ========================================================

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="逻辑 Document 全局唯一 ID。",
    )

    # ========================================================
    # Foreign Key
    # ========================================================

    document_type_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_type_definition.document_type_definition_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        comment=(
            "该 Document 使用的 DocumentTypeDefinition。"
            "Document 创建后不允许改变。"
        ),
    )

    # ========================================================
    # Tenant
    # ========================================================

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="租户 ID。",
    )

    # ========================================================
    # Business Identity
    # ========================================================

    document_key: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment=(
            "业务侧可选稳定标识。"
            "用于业务系统关联 Document，"
            "不是 Document 数据库主键。"
        ),
    )

    title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Document 的逻辑标题。",
    )

    # ========================================================
    # Current Version Pointer
    # ========================================================
    #
    # 保留 current_document_version_id。
    #
    # 它不是 source file 信息，
    # 而是 Document 的当前逻辑版本指针。
    #
    # 不直接建立 FK 到 document_version，
    # 避免 Document <-> DocumentVersion 循环 FK。
    # ========================================================

    current_document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "当前生效的 DocumentVersion ID。"
            "创建/处理成功后由 Service 更新。"
            "它属于逻辑版本指针，不属于文件存储信息。"
        ),
    )

    # ========================================================
    # Lifecycle
    # ========================================================

    status: Mapped[DocumentStatus] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.ACTIVE,
        index=True,
        comment=(
            "Document 生命周期。"
            "active=正常；deleted=逻辑删除。"
        ),
    )

    # ========================================================
    # Document Metadata
    # ========================================================

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "Document 级业务 metadata。"
            "这里保存的是业务语义 metadata，"
            "不是文件存储 metadata。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="创建 Document 的用户 ID。",
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

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="逻辑删除时间。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    document_type_definition: Mapped["DocumentTypeDefinition"] = relationship(
        back_populates="documents",
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        Index(
            "ix_document_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_document_document_type_definition",
            "document_type_definition_id",
        ),
    )
