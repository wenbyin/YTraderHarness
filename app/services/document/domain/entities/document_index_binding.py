"""
Domain entity for DocumentIndexBinding.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import StorageBackend


class DocumentIndexBinding(BaseModel):
    """
    DocumentRetrievalDefinition 与实际 Index / Storage 实例的关联。

    这是“Index 实例关系”，不是 Retrieval 配置定义：

        DocumentRetrievalDefinition 决定
            用什么 retrieval_type、chunker、retriever、storage_backend

        DocumentIndexBinding 决定
            实际落到哪个 Index / Collection / Graph 实例

    例如：

        dense_vector → Milvus  → collection = company_information_dense
        graph        → Neo4j   → graph      = company_information

    注意：
        版本级的执行状态不在这里，而在
        DocumentVersionRetrievalBinding 上。
        一个 Index 实例可以被多个 DocumentVersion 复用，
        因此状态不能挂在本实体上。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    document_index_binding_id: UUID = Field(
        description="DocumentIndexBinding 全局唯一 ID。",
    )

    document_retrieval_definition_id: UUID = Field(
        description="对应的 DocumentRetrievalDefinition。",
    )

    # ========================================================
    # Index Instance
    # ========================================================

    backend_type: StorageBackend = Field(
        description=(
            "实际存储后端类型。"
            "例如 postgresql、elasticsearch、milvus、neo4j。"
        ),
    )

    index_identifier: str = Field(
        description=(
            "实际 Index / Collection / Graph 标识。"
            "例如 Elasticsearch index name、"
            "Milvus collection name、Neo4j graph name。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    created_at: datetime = Field(
        description="Binding 创建时间。",
    )

    updated_at: datetime = Field(
        description="Binding 最后更新时间。",
    )
