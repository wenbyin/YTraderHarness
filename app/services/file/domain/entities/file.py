"""
Domain entities for the File Service.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import FileOrigin, FileStatus


class File(BaseModel):
    """
    File domain entity.

    File 表示一个逻辑文件资产。

    职责：
        - 描述文件身份
        - 描述文件基本属性
        - 描述物理存储位置
        - 描述文件生命周期
        - 保存文件级业务 metadata

    不负责：
        - 文件 IO
        - PostgreSQL 操作
        - Document
        - Chunk
        - Index

    关系：

        File
          |
          +-- file_id
          |
          +-- storage_key
          |
          +-- metadata
    """

    model_config = ConfigDict(
        # extra="forbid",
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    file_id: UUID = Field(
        description="文件唯一 ID。",
    )

    # ========================================================
    # Tenant
    # ========================================================

    tenant_id: UUID | None = Field(
        default=None,
        description="租户 ID，用于多租户/用户数据隔离。",
    )

    # ========================================================
    # Original File Metadata
    # ========================================================

    file_name: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "用户上传时的原始文件名。"
            "仅用于展示和下载时恢复文件名，"
            "不作为物理存储唯一标识。"
        ),
    )

    content_type: str | None = Field(
        default=None,
        max_length=200,
        description="文件 MIME Type，例如 application/pdf。",
    )

    file_size: int = Field(
        ge=0,
        description="文件大小，单位 Byte。",
    )

    checksum: str = Field(
        min_length=1,
        max_length=128,
        description="文件内容 checksum，默认 SHA-256。",
    )

    # ========================================================
    # Physical Storage
    # ========================================================

    storage_backend: str = Field(
        min_length=1,
        max_length=50,
        description=(
            "文件实际存储后端，例如 local、s3、minio。"
        ),
    )

    storage_key: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "文件在 Storage 中的唯一 Key。"
            "通常基于 file_id 生成。"
        ),
    )

    # ========================================================
    # Lifecycle
    # ========================================================

    origin: FileOrigin = Field(
        default=FileOrigin.USER_UPLOAD,
        description=(
            "user_upload = 用户上传的资产；"
            "pipeline_artifact = 流水线中间产物。"
        ),
    )

    status: FileStatus = Field(
        default=FileStatus.ACTIVE,
        description=(
            "文件生命周期状态。"
            "active=正常；"
            "deleted=逻辑删除。"
        ),
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata: dict = Field(
        default_factory=dict,
        description=(
            "文件级业务 Metadata。"
            "例如来源、标签、业务扩展字段等。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    uploaded_by: str | None = Field(
        default=None,
        description=(
            "上传用户 ID —— Casdoor 的 sub（app/common/auth），字符串。"
            "原为 UUID，见 models/sqlalchemy/file.py 的换类型说明。"
        ),
    )

    created_at: datetime = Field(
        description="文件记录创建时间。",
    )

    updated_at: datetime = Field(
        description="文件记录最后更新时间。",
    )

    deleted_at: datetime | None = Field(
        default=None,
        description="逻辑删除时间，未删除时为 None。",
    )
