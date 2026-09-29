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

from ...domain.schemas.enum import ProcessingStatus


# ============================================================
# 9. DocumentVersionRetrievalBinding
# ============================================================

class DocumentVersionRetrievalBinding(Base):
    """
    DocumentVersion 与 RetrievalDefinition 的运行实例关系。

    与 DocumentProcessorRetrievalBinding 的区别：

        DocumentProcessorRetrievalBinding
            = 配置层
            = “这个前处理是否应该进入这个 Retrieval”

        DocumentVersionRetrievalBinding
            = 执行层
            = “这个具体 DocumentVersion 在这个 Retrieval
               上当前处理到了什么状态”

    例如：

        Version 3
           │
           ├── keyword
           │      └── completed
           │
           ├── dense_vector
           │      └── completed
           │
           └── graph
                  └── processing
    """

    __tablename__ = "document_version_retrieval_binding"

    # ========================================================
    # Primary Key
    # ========================================================

    document_version_retrieval_binding_id: Mapped[uuid.UUID] = (
        mapped_column(
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            comment="DocumentVersionRetrievalBinding 主键。",
        )
    )

    # ========================================================
    # Foreign Keys
    # ========================================================

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_version.document_version_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="实际处理的 DocumentVersion ID。",
    )

    document_retrieval_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_retrieval_definition.document_retrieval_definition_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
        comment="实际执行的 RetrievalDefinition ID。",
    )

    document_index_binding_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_index_binding.document_index_binding_id",
            ondelete="RESTRICT",
        ),
        nullable=True,
        index=True,
        comment=(
            "实际写入的 DocumentIndexBinding。"
            "PostgreSQL document retrieval 等不需要独立 Index 实例的"
            "Retrieval 可以为空。"
        ),
    )

    # ========================================================
    # Processing State
    # ========================================================

    processing_status: Mapped[ProcessingStatus] = mapped_column(
        String(32),
        nullable=False,
        default=ProcessingStatus.PENDING,
        index=True,
        comment=(
            "该 Retrieval 的执行状态。"
            "pending=等待处理；"
            "processing=处理中；"
            "completed=完成；"
            "failed=失败。"
        ),
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="处理失败时的错误信息。",
    )

    # ========================================================
    # Audit
    # ========================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Binding 创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Binding 最后更新时间。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    document_version: Mapped["DocumentVersion"] = relationship(
        back_populates="retrieval_bindings",
    )

    retrieval_definition: Mapped[
        "DocumentRetrievalDefinition"
    ] = relationship(
        back_populates="version_bindings",
    )

    index_binding: Mapped[
        "DocumentIndexBinding | None"
    ] = relationship(
        back_populates="version_bindings",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "document_retrieval_definition_id",
            name="uq_document_version_retrieval_binding",
        ),
        Index(
            "ix_document_version_retrieval_status",
            "document_version_id",
            "processing_status",
        ),
    )
