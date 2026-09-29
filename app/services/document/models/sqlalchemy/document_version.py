from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

from ...domain.schemas.enum import (
    ProcessingStatus,
    VersionStatus,
)


# ============================================================
# 3. DocumentVersion
# ============================================================

class DocumentVersion(Base):
    """
    Document 的一个具体版本。

    一个 Version 对应一个 File。

        Document A
        ├── Version 1 -> file_id=A
        ├── Version 2 -> file_id=B
        └── Version 3 -> file_id=C

    原始文件的存储信息全部由 File 模块管理，
    这里只保存 file_id。

    一个真实文件只有一个文件格式，因此一个
    DocumentVersion 只会选择一个
    DocumentProcessorDefinition。

        file.extension
              ↓
        DocumentProcessorDefinition
              ↓
        Reader → Processor
              ↓
        N 个 RetrievalDefinition
    """

    __tablename__ = "document_version"

    # ========================================================
    # Primary Key
    # ========================================================

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="DocumentVersion 全局唯一 ID。",
    )

    # ========================================================
    # Foreign Keys
    # ========================================================

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document.document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属逻辑 Document ID。",
    )

    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "file.file_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        comment=(
            "该版本对应的原始 File ID。"
            "File 的具体存储信息由 File 模块管理。"
        ),
    )

    document_processor_definition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_processor_definition.document_processor_definition_id",
            ondelete="RESTRICT",
        ),
        nullable=True,
        comment=(
            "该版本实际使用的 DocumentProcessorDefinition。"
            "由 file extension 选择。"
            "analysis_mode=default 时可以为空，表示使用系统默认 Reader。"
        ),
    )

    # ========================================================
    # Version Number
    # ========================================================

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment=(
            "Document 内部版本号，从 1 开始递增。"
        ),
    )

    # ========================================================
    # Version State
    # ========================================================

    version_status: Mapped[VersionStatus] = mapped_column(
        String(32),
        nullable=False,
        default=VersionStatus.PENDING,
        index=True,
        comment=(
            "版本业务状态。"
            "pending=尚未成为可用版本；"
            "active=当前可用；"
            "superseded=被后续版本替代；"
            "failed=不可用；"
            "deleted=逻辑删除。"
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
            "该版本所有 Retrieval 的聚合处理状态。"
            "pending=尚未开始；"
            "processing=至少一个 Retrieval 在处理中；"
            "completed=所有 Retrieval 均已完成；"
            "failed=任一 Retrieval 失败。"
        ),
    )

    # ========================================================
    # Metadata
    # ========================================================

    meta_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "版本级业务 metadata。"
            "由调用方在创建 / 追加版本时传入，"
            "用于承载该版本特有的业务语义，"
            "例如来源批次、解析参数快照、"
            "关联出来的图片或资源引用等。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="创建该版本的用户 ID。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="版本创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="版本最后更新时间。",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "处理流水线全部完成的时间。"
            "processing_status 变为 completed 时写入。"
        ),
    )

    # ========================================================
    # Relationships
    # ========================================================

    document: Mapped["Document"] = relationship(
        back_populates="versions",
    )

    processor_definition: Mapped[
        "DocumentProcessorDefinition | None"
    ] = relationship(
        back_populates="document_versions",
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document_version",
        cascade="all, delete-orphan",
        # 一个版本下可能有多份切分产出（不同 chunker），只按 chunk_index
        # 排会让它们按序号交错。先按 chunker_key 归拢。
        order_by=(
            "DocumentChunk.chunker_key, DocumentChunk.chunk_index"
        ),
    )

    retrieval_bindings: Mapped[
        list["DocumentVersionRetrievalBinding"]
    ] = relationship(
        back_populates="document_version",
        cascade="all, delete-orphan",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_version_document_number",
        ),
        Index(
            "ix_document_version_document_status",
            "document_id",
            "version_status",
        ),
        Index(
            "ix_document_version_file_id",
            "file_id",
        ),
        Index(
            "ix_document_version_processor_definition",
            "document_processor_definition_id",
        ),
    )
