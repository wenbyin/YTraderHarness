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
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base


# ============================================================
# 8. DocumentIndexBinding
# ============================================================

class DocumentIndexBinding(Base):
    """
    DocumentRetrievalDefinition 与实际 Index / Storage 实例之间的关联。

    这是“Index 实例关系”，不是 Retrieval 配置定义。

    DocumentRetrievalDefinition 决定：
        使用什么 Retrieval Type、
        Chunker、
        Retriever、
        Storage Backend。

    DocumentIndexBinding 决定：
        实际使用哪个 Index / Collection / Graph 实例。

    例如：

        dense_vector
            ↓
        Milvus
            ↓
        collection = company_information_dense

        graph
            ↓
        Neo4j
            ↓
        graph = company_information
    """

    __tablename__ = "document_index_binding"

    # ========================================================
    # Primary Key
    # ========================================================

    document_index_binding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="DocumentIndexBinding 全局唯一 ID。",
    )

    # ========================================================
    # Foreign Keys
    # ========================================================

    document_retrieval_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_retrieval_definition.document_retrieval_definition_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="对应的 DocumentRetrievalDefinition。",
    )

    # ========================================================
    # Index Instance
    # ========================================================

    backend_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "实际存储后端类型。"
            "例如 postgresql、elasticsearch、milvus、neo4j。"
        ),
    )

    index_identifier: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment=(
            "实际 Index / Collection / Graph 标识。"
            "例如 Elasticsearch index name、"
            "Milvus collection name、Neo4j graph name。"
        ),
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

    retrieval_definition: Mapped[
        "DocumentRetrievalDefinition"
    ] = relationship(
        back_populates="index_bindings",
    )

    version_bindings: Mapped[
        list["DocumentVersionRetrievalBinding"]
    ] = relationship(
        back_populates="index_binding",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "document_retrieval_definition_id",
            "backend_type",
            "index_identifier",
            name="uq_document_index_binding_instance",
        ),
    )
