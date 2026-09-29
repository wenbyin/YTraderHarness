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
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

from ...domain.schemas.enum import DefinitionStatus


# ============================================================
# 4. DocumentProcessorDefinition
# ============================================================

class DocumentProcessorDefinition(Base):
    """
    文件前处理定义。

    它只负责：

        file_extension
            ↓
        Reader / Loader
            ↓
        Processor
            ↓
        DocumentTypeDefinition.modality_type

    它不负责：

        - Chunker
        - Retrieval
        - Index
        - pipeline_key
        - Canonical Pipeline

    一个 DocumentTypeDefinition 下：

        UNIQUE(file_extension)

    因此同一种文件后缀，在同一个业务类型下只能有一个
    前处理定义。

    新文件进入后，先根据 file extension 选择唯一的
    DocumentProcessorDefinition，然后再通过
    DocumentProcessorRetrievalBinding 进入 N 个 RetrievalDefinition。
    """

    __tablename__ = "document_processor_definition"

    # ========================================================
    # Primary Key
    # ========================================================

    document_processor_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="DocumentProcessorDefinition 全局唯一 ID。",
    )

    # ========================================================
    # Document Type
    # ========================================================

    document_type_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_type_definition.document_type_definition_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment=(
            "所属 DocumentTypeDefinition ID。"
        ),
    )

    # ========================================================
    # File Format
    # ========================================================

    file_extension: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment=(
            "该前处理定义适用的文件扩展名。"
            "例如 .pdf、.md、.html、.txt。"
        ),
    )

    mime_types: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment=(
            "该文件扩展名允许对应的 MIME Type 集合。"
            "用于对源文件实际 MIME Type 进行约束。"
        ),
    )

    # ========================================================
    # Loader
    # ========================================================

    loader_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment=(
            "Loader / Reader 能力名称。"
            "负责从 File 中读取并解析原始内容。"
            "例如 pdf_loader、markdown_loader。"
        ),
    )

    loader_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "Loader 的具体运行参数。"
        ),
    )

    # ========================================================
    # Processor
    # ========================================================

    processor_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "Processor 能力名称。"
            "负责对 Loader 输出进行规范化、清洗、结构化处理。"
            "如果不需要额外处理，则为空。"
        ),
    )

    processor_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "Processor 的具体运行参数。"
        ),
    )

    # ========================================================
    # Lifecycle
    # ========================================================

    status: Mapped[DefinitionStatus] = mapped_column(
        String(32),
        nullable=False,
        default=DefinitionStatus.ACTIVE,
        index=True,
        comment=(
            "生命周期状态。"
            "active=正常；deleted=逻辑删除。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="创建该前处理定义的用户 ID。",
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

    document_type_definition: Mapped[
        "DocumentTypeDefinition"
    ] = relationship(
        back_populates="processor_definitions",
    )

    document_versions: Mapped[
        list["DocumentVersion"]
    ] = relationship(
        back_populates="processor_definition",
    )

    retrieval_bindings: Mapped[
        list["DocumentProcessorRetrievalBinding"]
    ] = relationship(
        back_populates="processor_definition",
        cascade="all, delete-orphan",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "document_type_definition_id",
            "file_extension",
            name="uq_document_processor_definition_extension",
        ),
        Index(
            "ix_document_processor_definition_type_extension",
            "document_type_definition_id",
            "file_extension",
        ),
    )
