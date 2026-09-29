from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base


# ============================================================
# 6. DocumentProcessorRetrievalBinding
# ============================================================

class DocumentProcessorRetrievalBinding(Base):
    """
    DocumentProcessorDefinition 与
    DocumentRetrievalDefinition 之间的 N:N 关系。

    它表达：

        “这种文件前处理结果，
         是否需要进入这种 Retrieval？”

    例如：

        .pdf Processing
             │
             ├──────── keyword
             ├──────── dense_vector
             └──────── graph

        .md Processing
             │
             ├──────── keyword
             └──────── dense_vector

    一个 ProcessingDefinition 可以绑定多个 RetrievalDefinition。

    一个 RetrievalDefinition 也可以被多个 ProcessingDefinition 使用。

    这是配置关系，不是运行状态。
    """

    __tablename__ = "document_processor_retrieval_binding"

    # ========================================================
    # Primary Key
    # ========================================================

    document_processor_retrieval_binding_id: Mapped[uuid.UUID] = (
        mapped_column(
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            comment="DocumentProcessorRetrievalBinding 主键。",
        )
    )

    # ========================================================
    # Foreign Keys
    # ========================================================

    document_processor_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_processor_definition.document_processor_definition_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="文件前处理定义 ID。",
    )

    document_retrieval_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_retrieval_definition.document_retrieval_definition_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="检索定义 ID。",
    )

    # ========================================================
    # Audit
    # ========================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="关系创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="关系最后更新时间。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    processor_definition: Mapped[
        "DocumentProcessorDefinition"
    ] = relationship(
        back_populates="retrieval_bindings",
    )

    retrieval_definition: Mapped[
        "DocumentRetrievalDefinition"
    ] = relationship(
        back_populates="processor_bindings",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "document_processor_definition_id",
            "document_retrieval_definition_id",
            name="uq_document_processor_retrieval_binding",
        ),
        Index(
            "ix_document_processor_retrieval_binding_processor",
            "document_processor_definition_id",
        ),
        Index(
            "ix_document_processor_retrieval_binding_retrieval",
            "document_retrieval_definition_id",
        ),
    )
