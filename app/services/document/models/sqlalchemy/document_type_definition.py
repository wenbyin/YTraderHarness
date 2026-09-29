from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

from ...domain.schemas.enum import (
    DefinitionStatus,
    AnalysisMode,
    ModalityType,
    VersionMode,
)


# ============================================================
# 1. DocumentTypeDefinition
# ============================================================

class DocumentTypeDefinition(Base):
    """
    Document 类型定义。

    它描述的是：

        “这是什么业务类型的 Document？”

    不直接定义：

        - 文件格式
        - Loader
        - Processor
        - Chunker
        - Retrieval

    这些分别由：

        DocumentProcessorDefinition
        DocumentRetrievalDefinition

    负责。

    一个 DocumentTypeDefinition 可以对应多个
    DocumentProcessorDefinition。

    一个 DocumentTypeDefinition 可以对应多个
    DocumentRetrievalDefinition。

    一个 DocumentTypeDefinition 只有一种 modality_type。

    modality_type 表示：
        该业务类型统一的语义载体类型。
    """

    __tablename__ = "document_type_definition"

    # ========================================================
    # Primary Key
    # ========================================================

    document_type_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="DocumentTypeDefinition 主键。",
    )

    # ========================================================
    # Tenant
    # ========================================================

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="租户 ID，用于多租户数据隔离。",
    )

    # ========================================================
    # Basic Information
    # ========================================================

    type_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        comment=(
            "文档类型唯一编码。"
            "例如 company_announcement、company_material。"
            "创建后不可修改。"
        ),
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="文档类型名称。",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="该文档类型的业务语义描述。",
    )

    # ========================================================
    # Reader Policy
    # ========================================================

    modality_type: Mapped[ModalityType] = mapped_column(
        String(32),
        nullable=False,
        comment=(
            "该 DocumentTypeDefinition 唯一的业务内容类型。"
            "例如 text、structured、image、audio、video、multimodal。"
            "同一个 DocumentTypeDefinition 只能存在一种 modality_type。"
        ),
    )

    analysis_mode: Mapped[AnalysisMode] = mapped_column(
        String(32),
        nullable=False,
        default=AnalysisMode.DEFAULT,
        comment=(
            "文件解析策略。"
            "default=使用系统默认的文件后缀名 Reader；"
            "config=使用该 DocumentTypeDefinition 自定义的"
            "DocumentProcessorDefinition；"
            "none=不进行文件 Reader，要求输入已经是标准化内容。"
        ),
    )

    # ========================================================
    # Optimistic Lock
    # ========================================================

    revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment=(
            "乐观锁版本号，每次成功更新 +1。"
            "级联保存时必须回传，不匹配则拒绝，"
            "避免两个管理员同时编辑时静默覆盖。"
        ),
    )

    # ========================================================
    # Policy
    # ========================================================

    allow_update: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment=(
            "是否允许在该文档类型下创建新的内容版本。"
        ),
    )

    version_mode: Mapped[VersionMode] = mapped_column(
        String(32),
        nullable=False,
        default=VersionMode.VERSIONED,
        comment=(
            "版本策略。"
            "versioned=允许更新；"
            "immutable=Document 创建后不可更新。"
        ),
    )

    # ========================================================
    # Retrieval Result Fusion
    # ========================================================

    fusion_strategy: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "多个 Retrieval 召回结果的 Fusion 策略。"
            "例如 rrf、weighted_score。"
        ),
    )

    fusion_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Fusion 策略参数。",
    )

    # ========================================================
    # Rerank
    # ========================================================

    rerank_strategy: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "Fusion 后的 Rerank 策略。"
            "NULL 表示不进行 Rerank。"
        ),
    )

    rerank_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Rerank 参数。",
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "该文档类型的业务扩展 Metadata。"
            "注意 Python 属性名使用 metadata_，"
            "数据库列名仍为 metadata。"
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
            "不单独记录删除时间，"
            "以 updated_at 作为最后一次状态变更时间。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

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

    # ========================================================
    # Relationships
    # ========================================================

    processor_definitions: Mapped[list["DocumentProcessorDefinition"]] = relationship(
        back_populates="document_type_definition",
        cascade="all, delete-orphan",
    )

    retrieval_definitions: Mapped[list["DocumentRetrievalDefinition"]] = relationship(
        back_populates="document_type_definition",
        cascade="all, delete-orphan",
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="document_type_definition",
    )
