"""
Domain entity for Document.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import DocumentStatus


class Document(BaseModel):
    """
    一个逻辑 Document。

    表示“业务上的同一个文档”，例如：

        user_profile
            version 1 -> profile.md
            version 2 -> profile_v2.md
            version 3 -> profile_v3.md

    Document 本身不保存 source file 信息。

    file_id 属于 DocumentVersion：
        每一次更新都可能产生一个新的 File。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    document_id: UUID = Field(
        description="逻辑 Document 全局唯一 ID。",
    )

    document_type_definition_id: UUID = Field(
        description=(
            "该 Document 使用的 DocumentTypeDefinition。"
            "创建后不允许改变。"
        ),
    )

    tenant_id: UUID | None = Field(
        default=None,
        description="租户 ID，创建时从 DocumentTypeDefinition 继承。",
    )

    document_key: str | None = Field(
        default=None,
        description="业务侧可选稳定标识。",
    )

    title: str | None = Field(
        default=None,
        description="Document 的逻辑标题。",
    )

    # ========================================================
    # Current Version Pointer
    # ========================================================

    current_document_version_id: UUID | None = Field(
        default=None,
        description=(
            "当前生效的 DocumentVersion ID。"
            "版本处理成功后由 Service 更新。"
        ),
    )

    # ========================================================
    # Lifecycle
    # ========================================================

    status: DocumentStatus = Field(
        default=DocumentStatus.ACTIVE,
        description="active=正常；deleted=逻辑删除。",
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata: dict = Field(
        default_factory=dict,
        description="Document 级业务 metadata。",
    )

    # ========================================================
    # Audit
    # ========================================================

    created_by: UUID | None = Field(
        default=None,
        description="创建 Document 的用户 ID。",
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
