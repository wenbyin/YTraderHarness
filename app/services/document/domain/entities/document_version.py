"""
Domain entity for DocumentVersion.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import (
    ProcessingStatus,
    VersionStatus,
)


class DocumentVersion(BaseModel):
    """
    Document 的一个具体版本。

    一个 Version 对应一个 File。

        Document A
        ├── Version 1 -> file_id=A
        ├── Version 2 -> file_id=B
        └── Version 3 -> file_id=C

    原始文件的存储信息全部由 File 模块管理，
    这里只保存 file_id。

    两个状态字段的分工：

        version_status
            版本在业务上是否可用（生命周期）

        processing_status
            该版本所有 Pipeline 的聚合处理状态（执行状态）
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    document_version_id: UUID = Field(
        description="DocumentVersion 全局唯一 ID。",
    )

    document_id: UUID = Field(
        description="所属逻辑 Document ID。",
    )

    file_id: UUID = Field(
        description=(
            "该版本对应的原始 File ID。"
            "File 的具体存储信息由 File 模块管理。"
        ),
    )

    version_number: int = Field(
        ge=1,
        description="Document 内部版本号，从 1 开始递增。",
    )

    # ========================================================
    # Lifecycle
    # ========================================================

    version_status: VersionStatus = Field(
        default=VersionStatus.PENDING,
        description=(
            "版本业务状态。"
            "pending=尚未成为可用版本；"
            "active=当前可用；"
            "superseded=被后续版本替代；"
            "failed=不可用；"
            "deleted=逻辑删除。"
        ),
    )

    # Actual processing definition chosen for this version
    document_processor_definition_id: UUID | None = Field(
        default=None,
        description=(
            "Actual custom DocumentProcessorDefinition selected by file extension. "
            "May be null when the system default extension reader is used."
        ),
    )

    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description=(
            "该版本处理流水线的聚合状态。"
            "由该版本下所有 DocumentIndexBinding 聚合得到。"
        ),
    )

    # ========================================================
    # Metadata
    # ========================================================

    meta_data: dict = Field(
        default_factory=dict,
        description=(
            "版本级业务 metadata。"
            "与 Document.metadata 的区别："
            "document 级描述“这是什么文档”，"
            "version 级描述“这一版带来了什么”。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    created_by: UUID | None = Field(
        default=None,
        description="创建该版本的用户 ID。",
    )

    created_at: datetime = Field(
        description="版本创建时间。",
    )

    updated_at: datetime = Field(
        description="版本最后更新时间。",
    )

    completed_at: datetime | None = Field(
        default=None,
        description=(
            "处理流水线全部完成的时间。"
            "processing_status 变为 completed 时写入。"
        ),
    )
