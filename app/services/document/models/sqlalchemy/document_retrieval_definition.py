from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

from ...domain.schemas.enum import (
    DefinitionStatus,
    RetrievalType,
    StorageBackend,
)


# ============================================================
# 5. DocumentRetrievalDefinition
# ============================================================

class DocumentRetrievalDefinition(Base):
    """
    Retrieval 定义。

    它描述：

        “这个业务 Document 需要以什么检索方式被存储和召回？”

    一条 RetrievalDefinition：

        Chunker
          ↓
        Retriever
          ↓
        Storage

    例如：

        keyword
          ↓
        text chunker
          ↓
        bm25_retriever
          ↓
        Elasticsearch

        dense_vector
          ↓
        semantic chunker
          ↓
        vector_retriever
          ↓
        Milvus

        graph
          ↓
        graph processor
          ↓
        graph_retriever
          ↓
        Neo4j

    一个 DocumentTypeDefinition 下：

        UNIQUE(retrieval_type)

    因此每种 retrieval_type 只有一个定义。
    """

    __tablename__ = "document_retrieval_definition"

    # ========================================================
    # Primary Key
    # ========================================================

    document_retrieval_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="DocumentRetrievalDefinition 主键。",
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
        comment="所属 DocumentTypeDefinition ID。",
    )

    # ========================================================
    # Retrieval Identity
    # ========================================================

    retrieval_type: Mapped[RetrievalType] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "业务检索类型。"
            "例如 document、keyword、sparse_vector、"
            "dense_vector、graph。"
        ),
    )

    # ========================================================
    # Chunker
    # ========================================================

    chunker_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "Chunker 能力名称。"
            "某些 Retrieval 类型可能不需要传统 Chunk。"
        ),
    )

    chunker_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Chunker 的运行参数。",
    )

    # ========================================================
    # Retriever
    # ========================================================

    retriever_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "Retriever 能力名称。"
            "例如 bm25_retriever、vector_retriever、graph_retriever。"
        ),
    )

    retriever_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "Retriever 的召回参数。"
            "例如 top_k、score_threshold 等。"
        ),
    )

    # ========================================================
    # Storage
    # ========================================================

    storage_backend: Mapped[StorageBackend] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "实际存储后端。"
            "通常由 retrieval_type 默认映射得到。"
            "例如 sparse_vector -> Elasticsearch；"
            "dense_vector -> Milvus；"
            "graph -> Neo4j；"
            "document -> PostgreSQL。"
        ),
    )

    storage_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="实际存储后端的配置。",
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
        comment="创建该 RetrievalDefinition 的用户 ID。",
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
        back_populates="retrieval_definitions",
    )

    processor_bindings: Mapped[
        list["DocumentProcessorRetrievalBinding"]
    ] = relationship(
        back_populates="retrieval_definition",
        cascade="all, delete-orphan",
    )

    index_bindings: Mapped[
        list["DocumentIndexBinding"]
    ] = relationship(
        back_populates="retrieval_definition",
        cascade="all, delete-orphan",
    )

    version_bindings: Mapped[
        list["DocumentVersionRetrievalBinding"]
    ] = relationship(
        back_populates="retrieval_definition",
        cascade="all, delete-orphan",
    )

    # 曾有一条 chunks 关系，与 DocumentChunk 上的反向关系配对。
    #
    # chunk 现在归属 **chunker 身份**而不是 retrieval，那条外键已删除，
    # 所以这条关系也一并去掉 —— 留着会让 SQLAlchemy 在 mapper 配置阶段
    # 抛 NoForeignKeysError。
    #
    # 要取"这一路用哪些 chunk"：把本定义的 (chunker_name, chunker_config)
    # 过一遍 chunker_identity() 拿到身份，再按
    # (document_version_id, chunker_key, chunker_config) 查 document_chunk。

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "document_type_definition_id",
            "retrieval_type",
            name="uq_document_retrieval_definition_type",
        ),
        Index(
            "ix_document_retrieval_definition_type_retrieval",
            "document_type_definition_id",
            "retrieval_type",
        ),
    )
