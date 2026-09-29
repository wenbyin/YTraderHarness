"""
Domain entity for DocumentProcessorRetrievalBinding.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import RetrievalType


class DocumentProcessorRetrievalBinding(BaseModel):
    """
    DocumentProcessorDefinition 与
    DocumentRetrievalDefinition 之间的 N:N 关系。

    它表达“这种文件前处理结果，是否需要进入这种 Retrieval”：

        .pdf Processing
             │
             ├──────── keyword
             ├──────── dense_vector
             └──────── graph

    这是配置关系，不是运行状态。

    运行状态由 DocumentVersionRetrievalBinding 记录。

    注意：
        该关系在 API 层用双自然键
        (file_extension, retrieval_type) 表达，
        而不是用两端的 id。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    document_processor_retrieval_binding_id: UUID = Field(
        description="Binding 主键。",
    )

    document_processor_definition_id: UUID = Field(
        description="文件前处理定义 ID。",
    )

    document_retrieval_definition_id: UUID = Field(
        description="检索定义 ID。",
    )

    # ========================================================
    # Audit
    # ========================================================

    created_at: datetime = Field(
        description="关系创建时间。",
    )

    updated_at: datetime = Field(
        description="关系最后更新时间。",
    )
