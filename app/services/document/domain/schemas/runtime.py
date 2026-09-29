"""
Pydantic schemas for the Document runtime API.

对应 app/services/document/document.yaml 的 Documents 段：

    POST   /documents
    GET    /documents
    GET    /documents/{document_id}
    PUT    /documents/{document_id}
    DELETE /documents/{document_id}
    GET    /documents/{document_id}/versions
    GET    /documents/{document_id}/versions/{document_version_id}

注意：
    processing_status（处理进度）暂不对外暴露，
    后续通过 progress 相关接口提供。
    当前版本状态一律取 document_version.version_status。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from app.common.schemas import APIModel

from .enum import (
    DocumentStatus,
    VersionStatus,
)


# ============================================================
# Create
# ============================================================


class DocumentCreateRequest(APIModel):
    """
    创建逻辑 Document 及其首个 DocumentVersion。

    file_id 必须已经通过 File Service 上传完成。
    """

    document_type_definition_id: UUID = Field(
        description="所选 DocumentTypeDefinition ID。",
    )

    tenant_id: UUID | None = Field(
        default=None,
        description=(
            "租户 ID。为空时从 DocumentTypeDefinition 继承。"
        ),
    )

    document_key: str | None = Field(
        default=None,
        max_length=500,
        description="业务侧可选稳定标识。",
    )

    title: str | None = Field(
        default=None,
        max_length=500,
        description="Document 逻辑标题。",
    )

    file_id: UUID = Field(
        description="已上传完成的 File ID，用于首个版本。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document 级业务 metadata。",
    )


# ============================================================
# Update
# ============================================================


class DocumentUpdateRequest(APIModel):
    """
    更新 Document。

    三种操作，可单独或组合使用，但至少提供一个字段：

        title / metadata
            修改 Document 本身，不创建版本。

        file_id
            用新文件追加一个 DocumentVersion，
            版本号 = 当前最大 + 1。

        current_document_version_id
            切换当前生效版本。

    约束：
        file_id 与 current_document_version_id 互斥。
    """

    title: str | None = Field(
        default=None,
        max_length=500,
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
    )

    file_id: UUID | None = Field(
        default=None,
        description="新文件 ID。提供后创建新的 DocumentVersion。",
    )

    current_document_version_id: UUID | None = Field(
        default=None,
        description="切换当前生效版本。",
    )

    @model_validator(mode="after")
    def validate_update(self) -> "DocumentUpdateRequest":
        if (
                self.file_id is not None
                and self.current_document_version_id is not None
        ):
            raise ValueError(
                "file_id 与 current_document_version_id 互斥，"
                "不能同时提供。"
            )
        return self


# ============================================================
# Response
# ============================================================


class DocumentResponse(APIModel):
    """Document 本体。"""

    document_id: UUID

    document_type_definition_id: UUID

    tenant_id: UUID | None

    document_key: str | None

    title: str | None

    current_document_version_id: UUID | None

    status: DocumentStatus

    metadata: dict[str, Any]

    created_by: UUID | None

    created_at: datetime

    updated_at: datetime

    deleted_at: datetime | None


class DocumentVersionSummary(APIModel):
    """
    版本列表中的单个版本。

    is_current:
        由 document.current_document_version_id 与
        document_version.document_version_id 比较得到，
        不是数据库列。
    """

    document_version_id: UUID

    document_id: UUID

    version_number: int

    file_id: UUID

    document_processor_definition_id: UUID | None

    version_status: VersionStatus

    is_current: bool

    created_at: datetime

    meta_data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "版本级 metadata。"
            "当前存放 pipeline_config（当次执行的参数快照），"
            "未来可承载版本级业务语义，"
            "例如来源批次、关联出来的图片或资源引用。"
        ),
    )


class DocumentVersionDetailResponse(DocumentVersionSummary):
    """
    单个版本的详情。

    pipeline_config:
        该版本创建 / 处理时实际使用的完整 pipeline 配置快照，
        取自 document_version.meta_data 中的同名键。

        该配置属于版本执行 provenance，不属于 Document 业务 metadata，
        用于支持按历史配置重现处理结果。
        结构见 document.yaml 的 PipelineConfigSnapshot。
    """

    pipeline_config: dict[str, Any] = Field(
        default_factory=dict,
    )

    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    """Document 详情：本体 + 当前版本 + 全部版本。"""

    current_document_version: DocumentVersionSummary | None = None

    versions: list[DocumentVersionSummary] = Field(
        default_factory=list,
    )


class DocumentListResponse(APIModel):
    """分页的 Document 列表。"""

    items: list[DocumentResponse] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int


class DocumentVersionListResponse(APIModel):
    """分页的版本列表。"""

    items: list[DocumentVersionSummary] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int
