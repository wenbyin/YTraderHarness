"""
Domain entity for DocumentTypeDefinition.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import (
    DefinitionStatus,
    AnalysisMode,
    ModalityType,
    VersionMode,
)


class DocumentTypeDefinition(BaseModel):
    """
    Document 类型定义。

    它描述的是：

        “这是什么业务类型的 Document？”

    不直接定义文件格式、Loader、Processor、Chunker、Retrieval，
    这些分别由：

        DocumentProcessorDefinition   后缀名 → loader → processor
        DocumentRetrievalDefinition   chunker → retriever → storage

    负责，两者通过 DocumentProcessorRetrievalBinding 做 N:N 绑定。

    一个 DocumentTypeDefinition 只有一种 modality_type。

    modality_type 表示该业务类型统一的语义载体类型，
    既向前约束文件前处理，也向后约束检索配置。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    document_type_definition_id: UUID = Field(
        description="DocumentTypeDefinition 主键。",
    )

    tenant_id: UUID | None = Field(
        default=None,
        description="租户 ID，用于多租户数据隔离。",
    )

    type_key: str = Field(
        description=(
            "文档类型唯一编码。"
            "例如 company_announcement、company_material。"
            "创建后不可修改。"
        ),
    )

    # ========================================================
    # Basic Information
    # ========================================================

    name: str = Field(
        description="文档类型名称。",
    )

    description: str | None = Field(
        default=None,
        description="该文档类型的业务语义描述。",
    )

    # ========================================================
    # Reader Policy
    # ========================================================

    modality_type: ModalityType = Field(
        description=(
            "该类型唯一的业务内容类型。"
            "创建后不可修改。"
        ),
    )

    analysis_mode: AnalysisMode = Field(
        default=AnalysisMode.DEFAULT,
        description=(
            "default=使用系统预制后缀名 Reader，"
            "不建 ProcessorDefinition，该类型下全部 retrieval 都参与；"
            "config=使用自定义 ProcessorDefinition，"
            "缺少匹配后缀名时不允许上传；"
            "none=不解析文件。"
        ),
    )

    # ========================================================
    # Optimistic Lock
    # ========================================================

    revision: int = Field(
        default=1,
        ge=1,
        description=(
            "乐观锁版本号。每次成功更新 +1。"
            "更新时必须回传，不匹配则拒绝。"
        ),
    )

    # ========================================================
    # Policy
    # ========================================================

    allow_update: bool = Field(
        default=True,
        description="是否允许在该文档类型下创建新的内容版本。",
    )

    version_mode: VersionMode = Field(
        default=VersionMode.VERSIONED,
        description=(
            "版本策略。"
            "versioned=允许更新；"
            "immutable=Document 创建后不可更新。"
        ),
    )

    # ========================================================
    # Retrieval Result Fusion
    # ========================================================

    fusion_strategy: str | None = Field(
        default=None,
        description=(
            "多个 Retrieval 召回结果的 Fusion 策略。"
            "例如 rrf、weighted_score。"
        ),
    )

    fusion_config: dict = Field(
        default_factory=dict,
        description="Fusion 策略参数。",
    )

    # ========================================================
    # Rerank
    # ========================================================

    rerank_strategy: str | None = Field(
        default=None,
        description=(
            "Fusion 后的 Rerank 策略。"
            "None 表示不进行 Rerank。"
        ),
    )

    rerank_config: dict = Field(
        default_factory=dict,
        description="Rerank 参数。",
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata: dict = Field(
        default_factory=dict,
        description="该文档类型的业务扩展 Metadata。",
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

    created_at: datetime = Field(
        description="创建时间。",
    )

    updated_at: datetime = Field(
        description=(
            "最后更新时间。"
            "逻辑删除时该字段即为删除时间。"
        ),
    )
