"""
Domain entity for DocumentVersionRetrievalBinding.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import ProcessingStatus


class DocumentVersionRetrievalBinding(BaseModel):
    """
    DocumentVersion 与 DocumentRetrievalDefinition 的运行实例关系。

    这是**执行层**的关系，与配置层的
    DocumentProcessorRetrievalBinding 相对：

        DocumentProcessorRetrievalBinding
            配置层 —— “这个前处理是否应该进入这个 Retrieval”

        DocumentVersionRetrievalBinding
            执行层 —— “这个具体 DocumentVersion 在这个 Retrieval
                       上当前处理到了什么状态”

    例如：

        Version 3
           │
           ├── keyword       → completed
           ├── dense_vector  → completed
           └── graph         → processing

    与 document_pipeline_run / document_pipeline_node_run 的分工：

        本实体
            当前业务状态。每 (version, retrieval) 一行，upsert。
            回答“这个版本现在在 dense 索引里可搜吗”，
            被 DocumentVersion.processing_status 聚合消费。

        pipeline_run / pipeline_node_run
            追加型执行历史。无上界，每次尝试一行。
            回答“发生了什么、多久、为什么挂的”。

    单向派生：pipeline 的 finalize 节点写本表，
    再从本表聚合出 DocumentVersion.processing_status。
    执行历史的行永远不喂业务状态，
    本表的行永远不携带耗时与历史。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    document_version_retrieval_binding_id: UUID = Field(
        description="本绑定全局唯一 ID。",
    )

    document_version_id: UUID = Field(
        description="实际处理的 DocumentVersion ID。",
    )

    document_retrieval_definition_id: UUID = Field(
        description="实际执行的 DocumentRetrievalDefinition ID。",
    )

    document_index_binding_id: UUID | None = Field(
        default=None,
        description=(
            "实际写入的 DocumentIndexBinding。"
            "PostgreSQL document retrieval 等不需要独立 Index 实例的"
            "Retrieval 可以为空。"
        ),
    )

    # ========================================================
    # Processing State
    # ========================================================

    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description=(
            "该 Retrieval 的执行状态。"
            "pending=等待处理；"
            "processing=处理中；"
            "completed=完成；"
            "failed=失败。"
        ),
    )

    error_message: str | None = Field(
        default=None,
        description="处理失败时的错误信息。",
    )

    # ========================================================
    # Audit
    # ========================================================

    created_at: datetime = Field(
        description="绑定创建时间。",
    )

    updated_at: datetime = Field(
        description="绑定最后更新时间。",
    )
