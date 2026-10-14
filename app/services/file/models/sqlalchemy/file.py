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

    ⚠️ uploaded_by 换过类型，已有库要手动改
    ------------------------------------
    它原本是 `UUID`，而且**从来没被写过** —— 上传接口不填它
    （见 api/v1/files.py）。改成字符串是因为值换成了 Casdoor 的 sub
    （app/common/auth 的 AuthenticatedUser.subject）：那是**字符串**，
    形如 UUID 但没有任何保证。

    仓库没有迁移工具，initialize_database 用的是
    `Base.metadata.create_all`，它只建缺失的表、**不改已存在的表**。
    本模块没有 models.sql，DDL 就写在这里 —— 已有库要执行：

        ALTER TABLE file
            ALTER COLUMN uploaded_by TYPE VARCHAR(255)
            USING uploaded_by::text;

    现存值全是 NULL，USING 是空操作。**先跑它、再上新代码**：列还是
    uuid 时写入 dev-bypass 这类非 UUID 值会直接报 invalid input
    syntax，症状是"上传 500、其余接口正常"。
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

    uploaded_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "上传用户 ID —— Casdoor 的 sub（见 app/common/auth），字符串。"
            "原为 UUID，换类型的原因见类 docstring。\n\n"
            "只有用户上传（origin=user_upload）会写它；流水线产物"
            "（pipeline_artifact）恒为 NULL —— 产物不属于任何用户。"
        ),
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
