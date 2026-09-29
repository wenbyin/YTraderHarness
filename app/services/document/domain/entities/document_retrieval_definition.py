"""
Domain entity for DocumentRetrievalDefinition.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import (
    DefinitionStatus,
    RetrievalType,
    StorageBackend,
)


class DocumentRetrievalDefinition(BaseModel):
    """
    检索定义。

    它是 [chunker + retriever + storage] 的配置集合，
    只描述“以什么方式切分、召回和存储”，
    不持有任何 chunk 内容。

        Chunker
          ↓
        Retriever
          ↓
        Storage

    例如：

        keyword       → text chunker    → bm25_retriever   → Elasticsearch
        dense_vector  → semantic chunker → vector_retriever → Milvus
        document      → 不切分           → 直读             → PostgreSQL

    一个 DocumentTypeDefinition 下 UNIQUE(retrieval_type)，
    因此每种检索类型只有一个定义。

    被哪些文件后缀名使用，由
    DocumentProcessorRetrievalBinding 决定。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    retrieval_definition_id: UUID = Field(
        description="DocumentRetrievalDefinition 主键。",
    )

    document_type_definition_id: UUID = Field(
        description="所属 DocumentTypeDefinition ID。",
    )

    retrieval_type: RetrievalType = Field(
        description=(
            "业务检索类型。"
            "同一 DocumentTypeDefinition 下唯一。"
        ),
    )

    # ========================================================
    # Chunker
    # ========================================================

    chunker_name: str | None = Field(
        default=None,
        description=(
            "Chunker 能力名称。"
            "某些 retrieval 类型不需要传统 Chunk。"
        ),
    )

    chunker_config: dict = Field(
        default_factory=dict,
        description="Chunker 的运行参数。",
    )

    # ========================================================
    # Retriever
    # ========================================================

    retriever_name: str | None = Field(
        default=None,
        description=(
            "Retriever 能力名称。"
            "例如 bm25_retriever、vector_retriever、graph_retriever。"
        ),
    )

    retriever_config: dict = Field(
        default_factory=dict,
        description="Retriever 的召回参数，例如 top_k、score_threshold。",
    )

    # ========================================================
    # Storage
    # ========================================================

    storage_backend: StorageBackend = Field(
        description=(
            "实际存储后端。"
            "通常由 retrieval_type 默认映射得到。"
        ),
    )

    storage_config: dict = Field(
        default_factory=dict,
        description="实际存储后端的配置。",
    )

    # ========================================================
    # Lifecycle
    # ========================================================

    status: DefinitionStatus = Field(
        default=DefinitionStatus.ACTIVE,
        description="active=正常；deleted=逻辑删除。",
    )

    # ========================================================
    # Audit
    # ========================================================

    created_by: UUID | None = Field(
        default=None,
        description="创建该 RetrievalDefinition 的用户 ID。",
    )

    created_at: datetime = Field(
        description="创建时间。",
    )

    updated_at: datetime = Field(
        description="最后更新时间。",
    )

    deleted_at: datetime | None = Field(
        default=None,
        description="逻辑删除时间，未删除时为 None。",
    )
