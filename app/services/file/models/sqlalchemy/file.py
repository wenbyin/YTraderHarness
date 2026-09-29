from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    text,
    BigInteger,
    DateTime,
    Index,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.common.models import Base
from ...domain.schemas.enum import FileOrigin, FileStatus


class File(Base):
    """
    文件资产。

    File 与 Document 解耦：

        File
          ↓
        物理文件

        DocumentVersion
          ↓
        file_id
          ↓
        File

    FileService 负责：
        - 文件上传
        - 文件元数据
        - 文件读取
        - 文件逻辑删除
        - 后续物理清理

    DocumentService 不直接管理物理文件。
    """

    __tablename__ = "file"

    # ========================================================
    # Primary Key
    # ========================================================

    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="文件唯一 ID。也是 Object Storage 唯一标识的基础。",
    )

    # ========================================================
    # Tenant
    # ========================================================

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="租户 ID，用于用户/租户数据隔离。",
    )

    # ========================================================
    # Original File Metadata
    # ========================================================

    file_name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment=(
            "用户上传时的原始文件名。"
            "只用于展示和下载时恢复文件名，"
            "不作为 Object Storage 的唯一 Key。"
        ),
    )

    content_type: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="文件 MIME Type，例如 application/pdf。",
    )

    file_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="文件大小，单位 Byte。",
    )

    checksum: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="原始文件内容 checksum，默认使用 SHA-256。",
    )

    # ========================================================
    # Physical Storage
    # ========================================================

    storage_backend: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="local",
        comment=(
            "物理文件存储后端。"
            "第一版默认 local。"
            "未来可支持 s3、minio、oss、cos 等。"
        ),
    )

    storage_key: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        unique=True,
        comment=(
            "Object Storage 中的唯一 Key。"
            "推荐基于 file_id 生成，而不是直接使用 file_name。"
        ),
    )

    # ========================================================
    # Lifecycle
    # ========================================================

    origin: Mapped[FileOrigin] = mapped_column(
        String(32),
        nullable=False,
        default=FileOrigin.USER_UPLOAD,
        server_default=FileOrigin.USER_UPLOAD.value,
        index=True,
        comment=(
            "user_upload = 用户上传的资产；"
            "pipeline_artifact = 流水线中间产物。"
            "产物默认不出现在文件列表里。"
        ),
    )

    status: Mapped[FileStatus] = mapped_column(
        String(32),
        nullable=False,
        default=FileStatus.ACTIVE,
        index=True,
        comment=(
            "File 生命周期状态。"
            "active=正常；deleted=逻辑删除。"
        ),
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
            "文件级扩展 Metadata。"
            "注意 Python 属性名使用 metadata_，"
            "数据库列名仍为 metadata。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="上传用户 ID。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="文件记录创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="文件记录最后更新时间。",
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="逻辑删除时间。",
    )

    __table_args__ = (
        Index(
            "ix_file_tenant_checksum",
            "tenant_id",
            "checksum",
        ),

        # 产物的内容唯一约束 —— **只作用于 pipeline_artifact**。
        #
        # 为什么需要它：去重是"先查后插"，而 Send 扇出的多个分支
        # 是**并发**执行的。两个分支同时查到"不存在"、然后各插一行，
        # 实测出现过同 checksum 产物相差 2ms 各写一份。
        # 顺序场景（reprocess）不会暴露这个问题，只有并发才触发。
        #
        # 用户上传**故意不约束** —— 同一份文件被两个用户上传，
        # 应当各有各的记录，那是独立资产而不是重复。
        #
        # NULLS NOT DISTINCT：tenant_id 为空时也算同一个租户。
        # 默认语义下多个 NULL 互不相等，约束会形同虚设
        # （有 46 行产物的 tenant_id 就是 NULL）。
        Index(
            "uq_file_artifact_content",
            "tenant_id",
            "checksum",
            "file_name",
            unique=True,
            postgresql_where=text("origin = 'pipeline_artifact'"),
            postgresql_nulls_not_distinct=True,
        ),
    )
