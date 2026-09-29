"""
Pydantic schemas related to the logical Document.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from .common import APIModel, FileSource


class CreateDocumentRequest(APIModel):
    """
    创建新的逻辑 Document。

    语义：
        一次 create/upload 始终创建一个新的 Document。

    document_type_code:
        Document 创建后固定，不允许变更。
    """

    tenant_id: UUID | None = Field(
        default=None,
        description="租户 ID。",
    )

    document_type_code: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "DocumentTypeDefinition.type_code。"
            "创建后不可修改。"
        ),
    )

    file: FileSource

    title: str | None = Field(
        default=None,
        max_length=500,
        description="逻辑文档标题。",
    )

    document_key: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "业务侧可选的稳定唯一标识。"
            "用于业务系统关联逻辑 Document。"
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document 级业务 Metadata。",
    )

    created_by: UUID | None = Field(
        default=None,
        description="创建者用户 ID。",
    )


class DocumentVersionReference(APIModel):
    """
    DocumentVersion 的轻量引用。
    """

    document_version_id: UUID

    version_number: int = Field(
        ge=1
    )

    version_status: str

    processing_status: str


class CreateDocumentResponse(APIModel):
    """
    创建 Document 的返回值。

    注意：
        Document 创建成功不一定代表 Index 已经完成。
        processing_status 可能仍为 pending / processing。
    """

    document_id: UUID

    document_type_definition_id: UUID

    current_document_version_id: UUID | None

    status: str

    version: DocumentVersionReference

    created_at: datetime


class UpdateDocumentRequest(APIModel):
    """
    更新已有 Document。

    语义：
        update 不创建新的 Document，
        而是在现有 Document 下创建新的 DocumentVersion。

    不允许传 document_type_code：
        DocumentTypeDefinition 一旦绑定，不允许修改。
    """

    document_id: UUID

    file: FileSource

    title: str | None = Field(
        default=None,
        max_length=500,
        description="更新后的逻辑文档标题。",
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Document 级 Metadata。"
            "None 表示不修改已有 Document metadata。"
        ),
    )

    updated_by: UUID | None = Field(
        default=None,
        description="更新者用户 ID。",
    )


class UpdateDocumentResponse(APIModel):
    """
    Document 更新结果。
    """

    document_id: UUID

    document_version_id: UUID

    version_number: int

    previous_document_version_id: UUID | None

    processing_status: str

    current_document_version_id: UUID | None

    created_at: datetime


class GetDocumentRequest(APIModel):
    """
    获取逻辑 Document。
    """

    document_id: UUID

    include_metadata: bool = True

    include_current_version: bool = True


class CurrentVersionInfo(APIModel):
    """
    当前版本信息。

    current version 通过：

        1. current_document_version_id
        2. 若为空，则 fallback 到 active version 中最大的 version_number

    获取。
    """

    document_version_id: UUID

    version_number: int

    version_status: str

    processing_status: str

    file_name: str

    content_type: str | None


class DocumentResponse(APIModel):
    """
    Document 查询结果。
    """

    document_id: UUID

    document_type_definition_id: UUID

    document_type_code: str

    tenant_id: UUID | None

    document_key: str | None

    title: str | None

    status: str

    current_document_version_id: UUID | None

    current_version: CurrentVersionInfo | None

    metadata: dict[str, Any]

    index_status: dict[str, Any]

    created_at: datetime

    updated_at: datetime

    deleted_at: datetime | None


class DeleteDocumentRequest(APIModel):
    """
    逻辑删除 Document。

    第一版不会物理删除：
        - 原始文件
        - ES
        - Milvus
        - 其他 Index

    后续由 Cleanup Job 决定是否清理。
    """

    document_id: UUID

    deleted_by: UUID | None = Field(
        default=None,
        description="执行删除操作的用户 ID。",
    )


class DeleteDocumentResponse(APIModel):
    """
    逻辑删除结果。
    """

    document_id: UUID

    status: str

    deleted_at: datetime


class GetDocumentContentRequest(APIModel):
    """
    获取 Document 的完整解析文本。

    document_version_id:
        None → 当前版本
        指定 UUID → 获取指定历史版本
    """

    document_id: UUID

    document_version_id: UUID | None = None

    include_metadata: bool = False


class DocumentContentResponse(APIModel):
    """
    完整解析文本内容。
    """

    document_id: UUID

    document_version_id: UUID

    version_number: int

    file_name: str

    content_type: str | None

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
