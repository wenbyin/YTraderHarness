"""
Pydantic schemas for internal document processing pipeline.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from .common import APIModel


# ============================================================
# Processor
# ============================================================


class ProcessorRequest(APIModel):
    """
    Processor 链中单步的输入。

    Processor 覆盖流水线的所有处理：
        chunk_enrichment
        chunking
        cleaning
        embedding
        finalize
        format_detection
        indexing
        ingestion
        metadata_extraction
        normalization
        parsing

    所有处理共用同一个契约，靠 config 区分行为 ——
    这样"公共 Processor 链"就只是一串名字，而不是三种类型。
    """

    document_id: UUID | None = None

    document_version_id: UUID | None = None

    text: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    config: dict[str, Any] = Field(
        default_factory=dict
    )


class ProcessorResult(APIModel):
    """
    Processor 单步的输出。

    metadata 是**累积**的：每一步都可以往里加东西，
    后续步骤看到的是前面所有步骤的并集。

    chunks 为什么在这里
    -------------------
    分块类角色（role=chunk）的产物是**列表**，text 表达不了 ——
    硬塞成 JSON 字符串会把类型和可读性都丢掉。

    它装的是 domain entity `DocumentChunk`（带
    document_retrieval_definition_id / chunk_index / content_hash
    的完整实体），而不是 processing.py 里的 CanonicalChunk ——
    后者是写给索引后端的投影，由阶段 10 现算。

    用 list[Any] 而不是 list[DocumentChunk] 是**刻意的**：
    domain/schemas 是叶子层，反过来 import domain/entities 会把
    两层耦合成环。类型靠这段说明与实现侧的注解保证。
    """

    document_id: UUID | None = None

    document_version_id: UUID | None = None

    text: str | None = None

    text_checksum: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    chunks: list[Any] | None = Field(
        default=None,
        description=(
            "仅 role=chunk 填：DocumentChunk 实体列表。"
            "其它 role 留 None。"
        ),
    )

    vectors: list[Any] | None = Field(
        default=None,
        description=(
            "仅 role=embed 填：与输入 texts **等长且顺序严格对应**"
            " 的向量列表。其它 role 留 None。"
            "形态由 embedder 的 output_kind 决定 —— "
            "dense 是 list[list[float]]，sparse 是 list[dict[int, float]]"
            "（维度索引 -> 权重，**不展开**成定长稠密数组）。"
        ),
    )


# ============================================================
# Parser
# ============================================================


class ParseRequest(ProcessorRequest):
    """
    Parser 输入。

    第一版通过 storage_key 读取文件，
    避免大型文件在模块之间反复传输 bytes。
    """
    storage_key: str

    file_name: str

    content_type: str | None = None

    origin_text: str | None = None

    raw: bytes | None = None


class ParseResult(ProcessorResult):
    """
    Parser 标准化输出。
    """
    pass


class CleanRequest(ProcessorRequest):
    """
    Parser 输入。

    第一版通过 storage_key 读取文件，
    避免大型文件在模块之间反复传输 bytes。
    """
    storage_key: str

    file_name: str

    content_type: str | None = None

    origin_text: str | None = None

    raw: bytes | None = None


class CleanResult(ProcessorResult):
    result: list | None = None


# ============================================================
# Chunk
# ============================================================


class ChunkRequest(APIModel):
    """
    Chunker 输入。

    为什么**没有** document_retrieval_definition_id
    ---------------------------------------------
    切分的产出只取决于 (正文, chunker, chunker 配置)，与哪一路 retrieval
    无关。所以调用方不需要（也不应该）告诉 chunker"你在为哪路 retrieval
    切"—— 产物记的是 **chunker 身份**（chunker_key + chunker_config），
    由 chunker 自己盖上去。

    这带来一个直接好处：共用同一个 chunker 的 keyword 与 dense_vector
    两路，切分只跑一次、只落一份 chunk 行。查询期两路命中同一块时拿到
    的是同一个 document_chunk_id，融合才认得出"这是同一块"。

    config 必须是**归一化后**的配置
    ------------------------------
    它会原样落到 document_chunk.chunker_config，而查询侧要靠它反查
    "这份 chunk 归哪一路用"。所以调用方给之前要过一遍
    `chunker/identity.py` 的归一化 —— 否则库里存的与查时算的对不上，
    表现是"retrieval 找不到自己的 chunk"，且不报错。

    document_type_definition_id 为什么也要传
    ---------------------------------------
    chunk 落库时要把「所属文档类型」一起记下来（document_chunk 表没有
    这一列，按约定放 metadata）—— 前端拿 chunk 时经常需要它去查类型
    定义，而只靠 document_version_id 还得多跳一次。
    """

    document_id: UUID

    document_version_id: UUID

    document_type_definition_id: UUID

    text: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    config: dict[str, Any] = Field(
        default_factory=dict
    )


class EnrichRequest(ProcessorRequest):
    """
    阶段 8（chunk_enrichment）的输入。

    与 ChunkRequest 的区别：切分是"正文 → chunk"，增强是
    "chunk → 更完整的 chunk"，所以入参是**已经切好的 chunk**，
    而不是待切的正文。

    document_metadata 是增强的主要输入
    ---------------------------------
    增强要做的事（继承文档级元数据、补定位）全都依赖它 —— 没有它这个
    阶段就只能原样透传。它由 metadata 产物（阶段 6）提供，至少应当有：

        title            文档标题
        source           来源标识（文件名 / 外部系统 id）
        file_name        原始文件名 —— **后缀名**是格式识别的第一依据
        business keys    业务标识（公司代码之类），继承给每个 chunk
                         供查询侧做过滤

    缺键不报错（增强按"有什么用什么"处理），但缺得太多就等于没增强。
    """

    chunks: list[Any] = Field(
        default_factory=list,
        description=(
            "待增强的 DocumentChunk 实体。"
            "用 list[Any] 而不是 list[DocumentChunk] 与 ProcessorResult "
            "同样的理由：schemas 是叶子层，不能反过来 import entities。"
        ),
    )

    document_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "**文档级**元数据 —— 增强要往下继承的那份（标题、业务标识、"
            "来源）。由阶段 6 的产物提供。"
            "\n\n"
            "为什么单独一个字段而不是复用继承来的 `metadata`："
            "`ProcessorRequest.metadata` 的既定语义是**累积的元数据袋**"
            "（每个阶段往里面加东西、原样往下带，见 ProcessorResult 的"
            "说明）。两者来源与生命周期都不同："
            "\n"
            "    metadata           阶段间累积，逐阶段增长\n"
            "    document_metadata  文档级事实，阶段 6 一次给定\n"
            "\n"
            "混用会出现「增强拿到的是累积袋而不是文档事实」这种错，"
            "而且不报错 —— 增强只会默默地什么都不加。"
        ),
    )


class CanonicalChunk(APIModel):
    """
    Canonical Chunk。

    这是整个 Document 系统中唯一的逻辑 Chunk 表达。

    ES / Milvus / FAISS 等 Index 必须以这里的数据作为来源，
    不允许各自重新 Chunk。
    """

    document_chunk_id: UUID

    document_id: UUID

    document_version_id: UUID

    chunk_index: int = Field(
        ge=0
    )

    content: str

    content_hash: str

    token_count: int | None = Field(
        default=None,
        ge=0,
    )

    char_count: int = Field(
        ge=0
    )

    locator: dict[str, Any] = Field(
        default_factory=dict
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    # ========================================================
    # 向量（只对需要向量的后端有意义）
    # ========================================================
    #
    # CanonicalChunk 是"写给索引后端的投影"，因此它必须携带后端需要的
    # 东西 —— **包括向量**。之前没有这两个字段，后端根本拿不到向量。
    #
    # 两者互斥，取决于后端：
    #     Milvus / dense_vector     填 vector
    #     ES / sparse_vector        填 sparse_vector
    #     关键字 / document retrieval  两个都留 None

    vector: list[float] | None = Field(
        default=None,
        description=(
            "稠密向量。dense 后端（Milvus FLOAT_VECTOR）用。"
            "维度必须与该索引实例一致。"
        ),
    )

    sparse_vector: dict[int, float] | None = Field(
        default=None,
        description=(
            "稀疏向量：维度索引 -> 权重。"
            "sparse 后端（ES sparse_vector）用。"
            "键是**整数**维度索引 —— 从 JSON 读回来时是字符串，"
            "构造前要转回 int（pydantic 会自动转，但别依赖它）。"
        ),
    )

    context_text: str | None = Field(
        default=None,
        description=(
            "从文档级继承下来的上下文（标题/公司/来源/section 拼成的一段），"
            "由阶段 8 的增强产出、阶段 10 组装。"
            "\n\n"
            "它**不是**正文的一部分，也绝不并进 content —— content 是"
            "切分的产物，content_hash 是它的指纹，唯一约束靠它去重，"
            "改了两者的语义就崩（且不报错）。"
            "\n\n"
            "存在意义：一份按公司组织的长文档，中间的纯数据 chunk 正文里"
            "不重复公司名，查询「东方财富 股东户数」时公司名在 chunk 层面"
            "根本不可见 —— 结果里会混进别家公司的同类数据。"
            "\n\n"
            "只有文本类后端用得上（ES 拿它做 multi_match 与高亮）。"
            "向量那一路要同样的效果应当拼给 embedder 的输入，"
            "与本字段无关。"
        ),
    )


class ChunkResult(APIModel):
    """
    Chunk Strategy 批量输出。
    """

    document_version_id: UUID

    chunks: list[CanonicalChunk]


# ============================================================
# Embed
# ============================================================


class EmbedRequest(ProcessorRequest):
    """
    Embedder 输入。

    注意输入是**一批**文本而不是单条 —— 这与 ChunkRequest 刚好相反：
    chunk 的产物是列表，embed 的输入**和**产物都是列表。

    为什么接口层面就要批量：本地模型每次调用都有固定开销（上下文
    切换、显存搬运），逐条调用会把开销放大 N 倍。把批量做进契约，
    调用方就不必自己拼批次 —— 而自己拼批次正是最容易把顺序搞乱的
    地方（见 embedder/base.py 的"顺序即正确性"）。
    """

    texts: list[str] = Field(
        default_factory=list,
        description="待向量化的文本。返回的向量与之等长且同序。",
    )


# ============================================================
# Graph
# ============================================================


class EntityMention(APIModel):
    """
    从一段文本里抽出的一处实体提及。

    注意是**提及**（mention）而不是实体本身：同一家公司在一篇文档里
    出现十次就是十条提及，将来归并成图上的一个节点是后续阶段的事
    （实体消解 / 别名归并）。抽取阶段不做归并，只如实记录。
    """

    name: str = Field(
        min_length=1,
        description="实体名，例如「贵州茅台」。",
    )

    type: str = Field(
        default="",
        description=(
            "实体类型，例如 公司 / 人 / 产品 / 指标。"
            "取值由抽取实现定义（通常来自它接受的 entity_types 配置）。"
        ),
    )

    description: str = Field(
        default="",
        description="对该实体的简短说明，来自原文上下文。",
    )

    start_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "在**本段文本**里的起始字符偏移，用于前端高亮定位。"
            "抽不出来就留 None —— 不要瞎估，错了的高亮比没有更糟。"
        ),
    )

    end_index: int | None = Field(
        default=None,
        ge=0,
    )


class RelationMention(APIModel):
    """
    两处实体提及之间的一条关系。

    source / target 指向 EntityMention.name（同一段文本内），
    不是全局实体 ID —— 归并是后续阶段的事。
    """

    source: str = Field(min_length=1, description="头实体名。")

    target: str = Field(min_length=1, description="尾实体名。")

    type: str = Field(
        default="",
        description="关系类型，例如 持股 / 任职 / 供应。",
    )

    description: str = Field(
        default="",
        description="支撑该关系的原文依据。",
    )

    weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "该关系的置信度或强度。语义由抽取实现定义 —— "
            "如果实现不打分，一律填 1.0，不要留下没意义的 0。"
        ),
    )


class GraphFragment(APIModel):
    """
    一段文本抽出的局部图 —— 实体提及 + 它们之间的关系。

    「局部」是关键词：它只描述**这一段**里看到了什么，
    不保证与别段的同一实体连上。全局图由后续的归并阶段拼出来。
    """

    entities: list[EntityMention] = Field(
        default_factory=list,
    )

    relations: list[RelationMention] = Field(
        default_factory=list,
    )


# ============================================================
# Index
# ============================================================


class IndexUpsertRequest(APIModel):
    """
    IndexBackend 写入请求。

    所有 Index 都使用同一批 CanonicalChunk。
    """

    document_id: UUID

    document_version_id: UUID

    document_type_definition_id: UUID

    index_identifier: str

    index_config: dict[str, Any] = Field(
        default_factory=dict
    )

    chunks: list[CanonicalChunk]


class IndexDeleteRequest(APIModel):
    """
    删除一个 DocumentVersion 在某个 Index 中的数据。
    """

    document_id: UUID

    document_version_id: UUID

    index_identifier: str

    index_config: dict[str, Any] = Field(
        default_factory=dict
    )


class IndexSearchRequest(APIModel):
    """
    单个 Index 的独立召回请求。

    DocumentService / RetrievalService 会生成这个对象，
    再交给具体 IndexBackend。

    query_vector 由**调用方**算好传进来，索引后端不自己算
    --------------------------------------------------------
    IndexBackend 不该持有 embedder：召回的后端是 ES / Milvus，
    而"用哪个模型向量化"是**业务配置**（DTD 的 retrieval 定义决定）。
    把 embedder 塞进索引后端会有两个后果：

        - 同一个模型被两个后端各持一份客户端（连接池翻倍）
        - 换模型要改索引层，而它本该对模型一无所知

    所以由查询侧用写入期**同一个** embedder 算好查询向量传进来。
    写入期与查询期模型不一致时**不会报错**：维度对得上就照常返回
    结果，只是排序没有意义 —— 这正是必须由调用方显式给向量的原因。

    与 query 的关系
    ---------------
    query 是**原文**，始终要有（BM25 那一路直接用它，结果 metadata
    与观测也靠它）。query_vector 是它的稠密表示，只有走向量召回时
    才需要。两者都给，让后端自己决定用哪个：

        keyword        用 query
        dense_vector   用 query_vector
    """

    query: str

    query_vector: list[float] | None = Field(
        default=None,
        description=(
            "查询文本的稠密向量，由查询侧的 embedder 算出。"
            "走稠密向量召回（dense_vector / Milvus）的必须给；"
            "其余留 None。"
        ),
    )

    sparse_query: dict[int, float] | None = Field(
        default=None,
        description=(
            "查询文本的稀疏表示：维度索引 -> 权重，由查询侧的"
            "稀疏 embedder 算出（见 embedder/sparse.py）。"
            "走稀疏召回（sparse_vector / ES）的必须给。"
            "\n\n"
            "键是**整数**维度索引，与 CanonicalChunk.sparse_vector "
            "一致；ES 要求字符串键，由 index/es.py 在构造查询时转 —— "
            "与写入侧 _to_document 的转法是同一处约定。"
            "\n\n"
            "与 query_vector 一样，由调用方算好传进来：索引后端不持有"
            "embedder（理由见 IndexSearchRequest 的类说明）。"
        ),
    )

    filters: dict[str, Any] = Field(
        default_factory=dict
    )

    top_k: int = Field(
        default=50,
        ge=1,
        le=500,
    )

    min_score: float | None = Field(
        default=None,
        description=(
            "**原始分**下限，由各后端下推到索引层执行：\n"
            "    ES      -> search body 的 min_score\n"
            "    Milvus  -> search_params 的 radius\n"
            "\n"
            "None = 不过滤。\n\n"
            "必须收**原始分**：调用方在拿到结果后会立刻做按最大值"
            "归一化（见 document_retrieve_service_impl 的 _recall_one），"
            "归一化之后头名恒为 1.0，那时再比阈值就永远滤不掉东西。"
            "\n\n"
            "语义随后端而变（BM25 无上界、余弦 [-1,1]），所以由调用方"
            "按 retrieval_type 分别给（见 RetrievalRecallConfig）。"
        ),
    )

    search_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "该后端特有的检索参数，原样透传。"
            "例如 Milvus HNSW 的 ef / nprobe。"
            "\n\n"
            "**不包含 metric_type** —— 那是建表时定死的物理事实，"
            "由后端自己在合并时覆盖掉，调用方给了也无效"
            "（见 index/milvus.py）。"
            "\n\n"
            "ES 没有对应概念，传了会被忽略。"
        ),
    )

    document_version_ids: list[UUID] | None = Field(
        default=None,
        description=(
            "只在这些版本里召回。None 表示不按版本过滤。"
            "\n\n"
            "为什么是**复数**：查询期要过滤的是'当前生效的那些版本'，"
            "而一次检索覆盖整个 DTD（可能有很多文档，每个文档各有一个"
            "生效版本），所以过滤条件天然是一个集合。"
            "\n\n"
            "这一条**必须**下推到索引层，不能在召回后回表时再剔："
            "索引里同时存着旧版本与新版本的 chunk（reprocess 只增不删），"
            "回表过滤会让 top_k 被陈旧版本吃掉，返回的条数少得莫名其妙。"
        ),
    )

    index_identifier: str

    index_config: dict[str, Any] = Field(
        default_factory=dict
    )


class IndexSearchResult(APIModel):
    """
    单个 Index 的召回结果。

    只需要返回统一 Chunk ID + score + 必要 metadata，
    不要求 Index 自己返回完整 Document。

    highlights 是**索引侧**给出的标记片段（不含正文本身）。
    """

    document_chunk_id: UUID

    score: float

    retrieval_source: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    highlights: list[str] | None = Field(
        default=None,
        description=(
            "索引侧高亮片段：带标记的正文片段，由**产生这次召回的"
            "那一种查询**给出。\n\n"
            "None 与空列表都表示'这一路没有高亮'，但原因不同：\n"
            "    None  该后端/该查询类型不产生高亮\n"
            "    []    会产生，但这次没有匹配到可标的片段\n\n"
            "**只有做了词项匹配的查询才标得出来**：ES 的 BM25 走"
            "fvh、Milvus 的稠密召回走 LexicalHighlighter（按查询词），"
            "而 ES 的 sparse_vector 查询是向量相似度、不经过高亮器，"
            "标不出来。"
        ),
    )


class IndexSearchResponse(APIModel):
    """
    单个 Index 的召回结果集合。
    """

    results: list[IndexSearchResult]


# ============================================================
# Fusion
# ============================================================


class FusionRequest(APIModel):
    """
    多 Index 召回结果 Fusion 请求。
    """

    query: str

    result_sets: list[list[IndexSearchResult]]

    top_k: int = Field(
        default=20,
        ge=1,
        le=500,
    )

    config: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# Rerank
# ============================================================


class RerankRequest(APIModel):
    """
    Fusion 后的 Rerank 请求。
    """

    query: str

    results: list[IndexSearchResult]

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    config: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# Processing
# ============================================================


class ProcessDocumentVersionRequest(APIModel):
    """
    Worker / ProcessingService 使用。

    不暴露给 Agent。
    """

    document_id: UUID

    document_version_id: UUID


class ProcessDocumentVersionResponse(APIModel):
    """
    Version Processing 结果。
    """

    document_id: UUID

    document_version_id: UUID

    processing_status: str

    index_status: dict[str, Any] = Field(
        default_factory=dict
    )
