from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentChunk(BaseModel):
    """
    一个 DocumentVersion 的 chunk —— **版本级的事实，不属于任何一路 retrieval**。

    归属：chunker 身份，不是 RetrievalDefinition
    -------------------------------------------
    切分的产出只取决于 (正文, chunker, chunker 配置)，与
    retrieval_type / retriever / storage_backend 都无关。因此本实体记的是
    **产出它的 chunker 身份**（chunker_key + chunker_config），而不是某一路
    retrieval。

    哪些 retrieval 用这份 chunk 是**推导出来的**：
    该 retrieval 定义里的 (chunker_name, chunker_config) 归一化后与这里的
    两项相等，就用它。不存绑定表 —— 存一份必然与 chunk 行漂移，而且
    "谁为准"没有答案。

    这样同一块被 keyword 与 dense_vector 两路同时命中时，两侧拿到的是
    **同一个 document_chunk_id**，融合阶段才认得出"这是同一块"。

    canonical 程度
    --------------
    DocumentChunk 是 PostgreSQL 侧的物理行；CanonicalChunk 是投影给索引
    后端的形态（带向量、不带 chunker 身份）。两者不是同一个东西。
    """

    model_config = ConfigDict(from_attributes=True)

    # Identity
    document_chunk_id: UUID = Field(
        description="DocumentChunk global ID."
    )

    document_version_id: UUID = Field(
        description="Owning DocumentVersion ID."
    )

    chunker_key: str = Field(
        description=(
            "产出这批 chunk 的 chunker 契约键，例如 "
            "TEXT_TXT_RECURSIVE_CHUNKER。与 chunker_config 一起构成"
            "切分产出的身份。"
        )
    )

    chunker_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "归一化之后的 chunker 配置。"
            "必须与 DocumentRetrievalDefinition.chunker_config 用同一个"
            "归一化函数（见 chunker/identity.py）算出来，否则 retrieval "
            "会找不到自己的 chunk。"
        )
    )
    document_processor_definition_id: UUID | None = Field(
        default=None,
        description=(
            "Processing definition that produced the source content. "
            "Nullable when the system default reader is used."
        )
    )

    # Position
    chunk_index: int = Field(
        ge=0,
        description="Chunk sequence number within this retrieval representation.",
    )
    start_index: int | None = Field(
        default=None,
        ge=0,
        description="Start character offset in processed text when available.",
    )
    end_index: int | None = Field(
        default=None,
        ge=0,
        description="End character offset in processed text when available.",
    )

    # Content
    content: str = Field(
        description="Text content stored for PostgreSQL document retrieval."
    )
    content_hash: str | None = Field(
        default=None,
        description="Content hash for change detection/idempotency.",
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Token count when available.",
    )
    char_count: int | None = Field(
        default=None,
        ge=0,
        description="Character count when available.",
    )

    # Locator / resource metadata
    locator: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Location in the source file, e.g. page, paragraph, section, line."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Chunk metadata, including references to related images/files/resources."
        ),
    )

    # 检索前缀（瞬态）
    #
    # 阶段 8 的增强产出：`前缀 + 空格 + content`，只喂阶段 9 的 dense 与
    # sparse 向量化。前缀是"这条 chunk 属于谁"的几个元素（文件名 / section /
    # DTD 名 / 业务主键 / 上传时间），正文里往往不重复它们。
    #
    # ⚠️ 它是**瞬态字段**，三条边界都要守住：
    #   1. 不落 PG —— repository 的 add_many() 是逐列显式映射，天然带不过去。
    #   2. 不落索引 —— 阶段 10 的 _to_canonical_chunks() 逐字段构造，读的是
    #      content，embed_text 到不了 ES / Milvus。
    #   3. 不参与 content_hash 与计数 —— 那些是 chunker 算的，这里不碰。
    #
    # 之所以不拼进 content：`content == 正文[start_index:end_index]` 是硬
    # 不变式，qa/overlap.py 的区间求交与 eval 的 offset 切片都靠它，改了会
    # **静默错位**。
    embed_text: str | None = Field(
        default=None,
        description=(
            "Text fed to stage 9 embedding (prefix + space + content)."
            " Transient: not persisted to PostgreSQL, not projected to any"
            " index, not part of content_hash or the counters."
        ),
    )

    # Audit
    created_at: datetime | None = Field(
        default=None,
        description="Creation timestamp.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp.",
    )

    @model_validator(mode="after")
    def validate_position(self) -> "DocumentChunk":
        if (
                self.start_index is not None
                and self.end_index is not None
                and self.end_index < self.start_index
        ):
            raise ValueError("end_index must be greater than or equal to start_index")
        return self
