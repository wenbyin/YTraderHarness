"""
处理结果页的对外契约：按 版本 × retrieval 查 Chunk。

对应 app/api/v1/document_versions.py 暴露的接口，
规格见 app/services/document/document_version.yaml。

页面怎么用这两个接口
--------------------
    一个 Document -> N 个 DocumentVersion -> 每个版本下 N 个 retrieval（= tab）
    每个 tab 下：该路的 chunk 列表 + 该路运行时用的 chunker 配置

所以两个接口是**成对**的：

    GET .../retrievals                     列出 tab（含 chunker 与 chunk 数）
    GET .../retrievals/{rid}/chunks        某个 tab 的内容

为什么不做成一个扁平接口
------------------------
document_type_definition.yaml 里的 GET /document-chunks 是通用过滤
（按 version + processor_definition），**不支持按 retrieval 筛** ——
而"哪一路 retrieval"正是 tab 的唯一标识，也是 document_chunk 表上唯一约束
(document_version_id, document_retrieval_definition_id, chunk_index) 的一部分。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

from .enum import (
    EmbeddingOutput,
    ProcessingStatus,
    RetrievalType,
    StorageBackend,
)

__all__ = [
    "ChunkProjection",
    "ChunkProjectionEntry",
    "ChunkProjectionListResponse",
    "ChunkProjectionRequest",
    "ChunkerBinding",
    "DocumentChunkListResponse",
    "DocumentChunkSummary",
    "DocumentVersionRetrievalListResponse",
    "DocumentVersionRetrievalSummary",
    "SparseSummary",
    "VectorSummary",
]


class ChunkerBinding(APIModel):
    """
    该路运行时实际使用的 chunker 及其配置 —— **运行时冻结值**。

    它回答的是"这一路当时是怎么切的"，而不是"现在配置里怎么写"。
    改了 retrieval 定义之后，历史版本这里仍然是旧值。

    ⚠️ 当前**恒为 None**
    -------------------
    流水线目前没有把这个信息记下来（chunking 节点只把 chunker_name
    写进了节点行的 input_summary，配置没记），所以这里返回 None。

    按规格约定，前端遇到 None 应当显示「未知」，**不要**回退去读当前
    定义 —— 那会撒谎（历史版本显示的是今天的配置）。

    要让它有值：在 chunking 节点把 chunker 名与配置一起写进 input_summary
    （那里是逐分支的，天然对齐 tab），然后这里从节点行读。
    """

    processor_key: str = Field(
        description=(
            "能力键，例如 TEXT_TXT_GRAPH_CHUNKER。"
            "取值见 GET /document-processor-capabilities?role=chunk。"
        ),
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "该次运行传给 chunker 的配置原文。字段含义随 processor_key 而变 —— "
            "用能力目录的 config_schema 才能解释它。"
        ),
    )


class DocumentVersionRetrievalSummary(APIModel):
    """某个版本下的一路 retrieval —— 也就是前端的一个 tab。"""

    document_version_retrieval_binding_id: UUID

    document_retrieval_definition_id: UUID = Field(
        description="tab 标识。用它去取该 tab 的 chunk。",
    )

    retrieval_type: RetrievalType

    storage_backend: StorageBackend

    chunker: ChunkerBinding | None = Field(
        default=None,
        description=(
            "该路运行时使用的 chunker；当前恒为 None，"
            "原因见 ChunkerBinding 的说明。前端应显示「未知」。"
        ),
    )

    chunk_count: int = Field(
        ge=0,
        description=(
            "该路写入了多少 chunk。0 表示这一路没有产出 chunk"
            "（例如不需要切分的 retrieval 类型，或该路还没跑完）。"
        ),
    )

    processing_status: ProcessingStatus

    error_message: str | None = Field(
        default=None,
        description=(
            "该路失败时的原因。只在 processing_status=failed 时有值。"
        ),
    )


class DocumentVersionRetrievalListResponse(APIModel):
    """tab 列表。不分页 —— 一个类型下的 retrieval 数量级很小。"""

    items: list[DocumentVersionRetrievalSummary] = Field(
        default_factory=list,
    )

    total: int = Field(
        ge=0,
        description="tab 数量。",
    )


class DocumentChunkSummary(APIModel):
    """
    一条 chunk。

    内容、定位信息、元数据三者分开给：

        content    正文
        locator    在源文件里的位置（页码 / 章节 / 行）
        metadata   文档级元数据 + 阶段 8 增强补上的字段

    后两者是自由 dict，前端做展示时应当容错 —— 它们的内容随 retrieval
    与 enricher 配置而变。
    """

    document_chunk_id: UUID

    chunk_index: int = Field(
        ge=0,
        description=(
            "本 retrieval 内从 0 连续。列表按它升序，"
            "因此翻页不会重复或漏块。"
        ),
    )

    content: str = Field(
        description="chunk 正文。传了 preview_chars 时是截断后的。",
    )

    content_truncated: bool = Field(
        default=False,
        description=(
            "该条 content 被 preview_chars 截断过。"
            "前端应给出「查看完整内容」的入口 —— 用 document_chunk_id "
            "去取单条（GET /document-chunks/{document_chunk_id}）。"
        ),
    )

    char_count: int = Field(
        ge=0,
        description=(
            "**截断前**的字符数 —— 因此它可能大于 content 的长度。"
        ),
    )

    token_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "token 数。只有按 token 度量长短的 chunker 填得出来，"
            "其余为 None（不估算）。"
        ),
    )

    start_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "在归一化后正文中的起始字符偏移，可用它做高亮定位。"
            "满足 content == 正文[start_index:end_index]（未截断时）。"
        ),
    )

    end_index: int | None = Field(default=None, ge=0)

    content_hash: str | None = Field(
        default=None,
        description="content 的 sha256，用于变更检测与去重。",
    )

    locator: dict[str, Any] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime | None = None


class DocumentChunkListResponse(APIModel):
    """某个 (版本, retrieval) 下的 chunk 分页。"""

    items: list[DocumentChunkSummary] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int = Field(
        ge=0,
        description="该 (版本, retrieval) 下的 chunk 总数。",
    )


# ============================================================
# 投影：这一条 chunk 当时写给索引的是什么
# ============================================================


class VectorSummary(APIModel):
    """
    稠密向量的摘要。

    给的是**判断"这条投影正不正常"够用的那几个数** —— 维度对不对、
    是不是零向量、数值量级是否离谱。看全部值要传 include_vector。
    """

    dimensions: int = Field(description="向量维度。应与索引 schema 一致。")

    nonzero_count: int = Field(
        ge=0,
        description=(
            "非零分量个数。稠密向量通常接近满值；"
            "异常地少说明 embedding 有问题。"
        ),
    )

    l2_norm: float = Field(
        description=(
            "L2 模长。embedder 开了归一化时应当接近 1.0；"
            "明显偏离说明没归一化或向量异常。"
        ),
    )

    head: list[float] = Field(
        default_factory=list,
        description="前 8 个分量，用来看数值量级。",
    )


class SparseSummary(APIModel):
    """稀疏向量的摘要。"""

    nonzero_count: int = Field(
        ge=0,
        description="非零项个数 —— 也就是这条文本命中了多少个词项。",
    )

    top: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "权重最大的前 8 项，每项 {dim, weight}。"
            "dim 是 token id（见 embedder/sparse.py 的说明）。"
        ),
    )


class ChunkProjectionEntry(APIModel):
    """
    一条 chunk 在**某一路 retrieval** 下的投影。

    为什么要按 retrieval 分开
    ------------------------
    chunk 现在是**版本级的事实**，不归任何一路 retrieval（见
    DocumentChunk 的说明）。同一个 chunk 会被多路 retrieval 各自写进
    不同的索引，形态还不同（稠密向量 / 稀疏向量 / 透传），所以"这条
    chunk 的投影"本来就不是一个东西，而是一组。

    ⚠️ 只有需要向量化的 retrieval 才有投影产物
    ------------------------------------------
    阶段 9 对 keyword / document 那两路是**透传**的（它们不需要向量），
    因此没有单独的投影产物。这种情况下 kind / fields 都是 None，
    unavailable_reason 会说明原因。
    """

    document_retrieval_definition_id: UUID

    retrieval_type: RetrievalType = Field(
        description=(
            "回显在这里，前端不必再去查一次定义就知道这一项是哪一路。"
        ),
    )

    kind: EmbeddingOutput | None = Field(
        default=None,
        description="投影的形态：dense / sparse / graph。",
    )

    embedder: str | None = Field(
        default=None,
        description="产出它的 embedder 能力键，例如 TEXT_BGE_SMALL_ZH_EMBEDDER。",
    )

    dimensions: int | None = Field(
        default=None,
        description=(
            "该批向量声明的维度。与 vector_summary.dimensions 可能不同 —— "
            "那个是这条向量实际的长度，这个是产物里声明的值，"
            "两者不一致本身就是个问题。"
        ),
    )

    fields: dict[str, Any] | None = Field(
        default=None,
        description=(
            "投影里**非向量**的那些字段 —— 也就是当时发给索引后端的"
            " content / content_hash / 计数 / metadata 等。"
        ),
    )

    vector_summary: VectorSummary | None = None

    sparse_summary: SparseSummary | None = None

    vector: list[float] | None = Field(
        default=None,
        description="完整稠密向量。只有 include_vector=true 时返回。",
    )

    sparse: dict[int, float] | None = Field(
        default=None,
        description=(
            "完整稀疏向量（维度 -> 权重）。只有 include_vector=true 时返回。"
            "键是整数维度索引。"
        ),
    )

    unavailable_reason: str | None = Field(
        default=None,
        description=(
            "这一路取不到投影时的原因。有值时其余字段都是 None —— "
            "前端应当把它显示出来，而不是当作「这一路没有投影」。"
        ),
    )


class ChunkProjection(APIModel):
    """
    一条 chunk 当时**写给索引的东西**（投影）—— 按 retrieval 分组。

    与 DocumentChunkSummary 的区别
    ------------------------------
        DocumentChunkSummary   原件：这一条是什么（从 PG 读）
        ChunkProjection        投影：它被变成了什么样子写进索引（从产物读）

    两者不是同一份数据：投影里带向量、而且只带索引需要的字段。

    外层仍是"一条 chunk 一项"，与请求一一对应且同序 —— 变的是里面
    多了一个 projections 列表，因为同一条 chunk 会被多路 retrieval
    以不同形态写进不同索引。
    """

    document_chunk_id: UUID

    document_version_id: UUID | None = None

    chunker_key: str | None = Field(
        default=None,
        description=(
            "产出这条 chunk 的 chunker 能力键。列出来是因为"
            "「哪几路 retrieval 用到了它」正是由它推导出来的。"
        ),
    )

    projections: list[ChunkProjectionEntry] = Field(
        default_factory=list,
        description=(
            "这条 chunk 在各路 retrieval 下的投影。空列表表示"
            "**没有任何一路用到它**（正常不该发生）。"
            "某一项取不到时列表里仍有它，看该项的 unavailable_reason。"
        ),
    )

    unavailable_reason: str | None = Field(
        default=None,
        description=(
            "chunk 本身就取不到时的原因（document_chunk 里没有这条 id）。"
            "有值时 projections 必为空。"
        ),
    )


class ChunkProjectionRequest(APIModel):
    """
    批量查投影的入参。

    用 POST 而不是 GET：chunk_id 可能一次传几十上百个，
    塞进 query string 会撞 URL 长度上限。语义上它是只读的。
    """

    document_chunk_ids: list[UUID] = Field(
        min_length=1,
        max_length=200,
        description=(
            "要查的 chunk id。返回结果与它**一一对应且同序**，"
            "包含取不到的那些（看 unavailable_reason）。"
        ),
    )

    document_retrieval_definition_id: UUID | None = Field(
        default=None,
        description=(
            "只看某一路 retrieval 的投影。不传则返回这条 chunk 在"
            "**所有**用到它的 retrieval 下的投影。"
        ),
    )


class ChunkProjectionListResponse(APIModel):
    """按 chunk_id 批量查询投影的结果。"""

    items: list[ChunkProjection] = Field(
        description=(
            "与请求的 chunk_ids **一一对应且同序**，包括取不到的那些"
            "（它们的 unavailable_reason 有值）。"
        ),
    )

    total: int = Field(ge=0)

    include_vector: bool = Field(
        description="回显本次是否返回了完整向量，方便前端判断要不要再请求一次。",
    )
