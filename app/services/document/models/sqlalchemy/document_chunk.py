from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base


# ============================================================
# 7. DocumentChunk
# ============================================================

class DocumentChunk(Base):
    """
    一个 DocumentVersion 的 chunk —— **版本级的事实，不属于任何一路 retrieval**。

    DocumentChunk 不是 CanonicalChunk（后者是投影给索引后端的形态）。

    它表示：

        DocumentVersion
              ↓
        (清洗后正文 + Chunker + Chunker 配置)
              ↓
        DocumentChunk
              ↓
        PostgreSQL

    归属是 **chunker 身份**，不是 RetrievalDefinition
    ----------------------------------------------
    切分的产出只取决于 (正文, chunker, chunker 配置)，与 retrieval_type /
    retriever / storage_backend 都无关。所以这里记的是产出它的 chunker
    身份，而"哪些 retrieval 用这份 chunk"是**推导**出来的：retrieval 定义
    里的 (chunker_name, chunker_config) 归一化后与本表两项相等即可。

    不存绑定表 —— 存一份必然与 chunk 行漂移，且"谁为准"没有答案。

    于是 keyword 与 dense_vector 两路共用同一 chunker 时，命中同一块拿到
    的是**同一个 document_chunk_id**，融合阶段才认得出"这是同一块"。

    DocumentChunk 保留：
        - 原始内容
        - Chunk 顺序
        - 原文定位
        - 业务 Metadata

    不负责：
        - 文件解析
        - Chunk 执行
        - Index 写入
        - 检索
    """

    __tablename__ = "document_chunk"

    # ========================================================
    # Primary Key
    # ========================================================

    document_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="DocumentChunk 全局唯一 ID。",
    )

    # ========================================================
    # Foreign Keys
    # ========================================================

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_version.document_version_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="所属 DocumentVersion ID。",
    )

    chunker_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment=(
            "产出这批 chunk 的 chunker 契约键。"
            "与 chunker_config 一起决定这份 chunk 归谁用 —— "
            "retrieval 定义里的同两项归一化后相等，就用这份。"
        ),
    )

    chunker_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "归一化之后的 chunker 配置（见 chunker/identity.py）。"
            "用 jsonb 而不是 text：Postgres 的 jsonb 相等是**语义相等**"
            "（键序无关），因此可以直接进唯一约束。"
        ),
    )

    document_processor_definition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_processor_definition.document_processor_definition_id",
            ondelete="RESTRICT",
        ),
        nullable=True,
        index=True,
        comment=(
            "产出该 Chunk 源内容的前处理定义。"
            "使用系统默认 Reader 时为空，"
            "该字段提供 Chunk 到 Pipeline 的 provenance。"
        ),
    )

    # ========================================================
    # Position
    # ========================================================

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Chunk 在该版本内的顺序编号，从 0 开始。",
    )

    start_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Chunk 在原文中的起始字符位置。",
    )

    end_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Chunk 在原文中的结束字符位置。",
    )

    # ========================================================
    # Content
    # ========================================================

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Document Retrieval 使用的 Chunk 文本内容。",
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Chunk 内容 hash，用于内容一致性校验与去重。",
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Token 数量。",
    )

    char_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="字符数量。",
    )

    # ========================================================
    # Locator / Metadata
    # ========================================================

    locator: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "Chunk 在原始文件中的位置。"
            "例如 page、paragraph、section、line 等。"
        ),
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "Chunk 级 Metadata。"
            "可保存资源引用、图片地址、表格信息、"
            "原始文件定位等业务信息。"
        ),
    )

    # ========================================================
    # Audit
    # ========================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Chunk 创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Chunk 最后更新时间。",
    )

    # ========================================================
    # Relationships
    # ========================================================

    document_version: Mapped["DocumentVersion"] = relationship(
        back_populates="chunks",
    )

    # 曾有一条指向 DocumentRetrievalDefinition 的关系，靠
    # document_retrieval_definition_id 外键连起来。
    #
    # 那条外键已随列一起删除 —— chunk 是版本级的事实，归属 chunker 身份
    # 而不是某一路 retrieval（见实体说明）。关系留着会让 SQLAlchemy 在
    # mapper 配置阶段直接抛 NoForeignKeysError（找不到连接条件）。
    #
    # "哪些 retrieval 用这份 chunk" 现在是**推导**的，不是关联：
    # 比较 retrieval 定义的 (chunker_name, chunker_config) 归一化后与
    # chunk 行上的 (chunker_key, chunker_config) 是否相等。

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        # 一个版本下一份切分产出就是一行 —— 身份是 (版本, chunker, 配置)。
        #
        # 原先是 (document_version_id, retrieval_definition_id, chunk_index)，
        # 那让"一个 chunk 属于一路 retrieval"，于是共用同一个 chunker 的
        # 两路会各存一份内容完全相同、ID 不同的 chunk，查询期融合认不出
        # 是同一块。改成按 chunker 身份之后，共用 chunker 的 retrieval 自动
        # 拿到同一行。
        #
        # chunker_config 是 jsonb：Postgres 的 jsonb 相等是语义相等
        # （键序无关），所以能直接进唯一约束。
        UniqueConstraint(
            "document_version_id",
            "chunker_key",
            "chunker_config",
            "chunk_index",
            name="uq_document_chunk_version_chunker_index",
        ),
        Index(
            "ix_document_chunk_version_chunker",
            "document_version_id",
            "chunker_key",
        ),
    )
