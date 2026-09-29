from enum import Enum


class VersionMode(str, Enum):
    """
    Document 版本策略。

    versioned:
        允许在该 DocumentTypeDefinition 下创建新的
        DocumentVersion。

    immutable:
        Document 创建后不允许更新。
    """

    VERSIONED = "versioned"
    IMMUTABLE = "immutable"


class RetrievalMethod(str, Enum):
    """
    DocumentProcessorDefinition 的检索方法。

    text:
        基于原始文本进行检索。
        通常用于简单的全文匹配或文本过滤，
        不依赖向量化表示。

    bm25:
        基于 BM25 的关键词相关性检索。
        适合精确词匹配、专业术语、实体名称、编号等场景。

    dense_vector:
        基于稠密向量的语义检索。
        将文档内容转换为 Dense Embedding，
        通过向量相似度进行语义匹配。

    sparse_vector:
        基于稀疏向量的语义检索。
        保留较强的词项匹配特征，同时引入语义扩展能力，
        适合兼顾关键词和语义相关性的场景。

    graph:
        基于图结构的关系检索。
        文档内容或业务实体以节点、关系形式组织，
        通过图遍历、关系匹配等方式进行检索。
    """
    TEXT = "text"
    BM25 = "bm25"
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    GRAPH = "graph"


class ChunkMode(str, Enum):
    """
    DocumentTypeDefinition 的 Chunk 模式。

    none:
        不进行 Chunk。
        整个文档内容作为一个逻辑内容单元。

    fixed:
        按固定 token / character 数量切分。

    recursive:
        按分隔符层级递归切分，
        在结构较规整的文本上比 fixed 更贴合语义边界。

    semantic:
        根据语义边界进行 Chunk，通常依赖 Embedding 模型。

    semantic:
        根据Token进行Chunk
    """

    NONE = "none"
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC_DENSE = "semantic"
    TOKEN = "token"


class DefinitionStatus(str, Enum):
    """
    DocumentTypeDefinition / DocumentProcessorDefinition
    的生命周期状态。

    active:
        正常可用。

    deleted:
        逻辑删除。
        不物理删除记录，由后续 Cleanup Job 决定是否清理。
    """

    ACTIVE = "active"
    DELETED = "deleted"


class DocumentStatus(str, Enum):
    """
    逻辑 Document 的生命周期状态。

    与 DefinitionStatus 取值相同，但描述的是
    业务文档而不是定义配置，因此单独定义。
    """

    ACTIVE = "active"
    DELETED = "deleted"


class VersionStatus(str, Enum):
    """
    DocumentVersion 的版本生命周期状态。

    pending:
        已创建，尚未成为可用版本。

    active:
        当前可用版本，可以作为 current version。

    superseded:
        已被更新的版本替代。

    failed:
        处理失败，不可用。

    deleted:
        逻辑删除。
    """

    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    DELETED = "deleted"


class ProcessingStatus(str, Enum):
    """
    处理流水线的执行状态。

    同时用于两个层级：

        DocumentVersion.processing_status
            该版本所有 Pipeline 的聚合状态。
            全部 completed 才是 completed；
            任一 failed 即 failed；
            任一 processing 即 processing；
            否则 pending。

        DocumentIndexBinding.processing_status
            单个 Pipeline 在单个 Index 上的执行状态。

    取值与 document_type_definition.yaml 的 ProcessingStatus 一致。
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisMode(str, Enum):
    """
    DocumentTypeDefinition 的文件解析策略。

    回答的是“这个类型的文件要不要解析、按谁的配置解析”，
    与 ModalityType（内容是什么模态）是两个不同维度。

    DEFAULT:
        使用系统预制的中缀名 Reader。
        不需要配置 DocumentProcessorDefinition，
        该类型下的全部 RetrievalDefinition 都会参与。

    CONFIG:
        使用该 DocumentTypeDefinition 自定义的
        DocumentProcessorDefinition 解析。
        缺少匹配文件后缀名的配置时，不允许上传。

    NONE:
        不解析文件，要求输入已经是标准化内容。
    """

    NONE = "none"
    DEFAULT = "default"
    CONFIG = "config"


class ProcessorMode(str, Enum):
    """
    Processor 执行模式。

    NONE：
        不执行 Processor 链。
        Parser 输出的数据直接进入后续 Retrieval 流程。

    COMMON：
        使用系统默认的公共 Processor 执行链。
        当前默认链为：
            Normalize → Clean → Enrich Metadata

    CONFIG：
        使用当前 DocumentProcessorDefinition.processor_config
        中配置的 Processor 执行链。
        用于需要自定义处理步骤或参数的业务类型。
    """

    NONE = "none"
    COMMON = "common"
    CONFIG = "config"


class ProcessorRole(str, Enum):
    """
    处理能力在写入流水线里承担的阶段。

    与 ProcessorMode 是两个维度，不要混：

        ProcessorMode  怎么执行（这一步要不要跑、按谁的配置跑）
        ProcessorRole  是流水线里的哪一步（这个能力是干什么的）

    取值与 pipeline/catalog.py 的节点划分、以及各 role 的 ABC 对应：

        loader           DocumentParser     阶段 3  文件 -> 文本
        normalize        DocumentProcessor  阶段 4  编码 / 换行 / 空白
        clean            DocumentProcessor  阶段 5  去页眉页脚等噪声
        enrich_metadata  DocumentProcessor  阶段 6  抽取文档级元数据
        chunk            ChunkStrategy      阶段 7  按 retrieval 切分
        chunk_enrich     ChunkEnricher      阶段 8  补定位与继承元数据
        embed            Embedder           阶段 9  文本 -> 向量

    刻意不含查询期能力（retrieve / fuse / rerank）与 IndexBackend：
    前者不在写入流水线上，后者按 StorageBackend 注册而非按名字，
    注册模型不同。需要时另行开接口，不要塞进能力目录。
    """

    LOADER = "loader"
    NORMALIZE = "normalize"
    CLEAN = "clean"
    ENRICH_METADATA = "enrich_metadata"
    CHUNK = "chunk"
    CHUNK_ENRICH = "chunk_enrich"
    EMBED = "embed"


class EmbeddingOutput(str, Enum):
    """
    向量化能力产出的**向量形态**。

    这个区分不是修饰性的 —— 两种向量在数学上不可互换，索引侧也
    存在完全不同的字段类型里：

        dense    定长浮点数组，例如 512 维
                 存 Milvus 的 FLOAT_VECTOR / ES 的 dense_vector

        sparse   维度索引 -> 权重 的稀疏映射，例如 {1024: 0.31, 88: 0.7}
                 存 Milvus 的 SPARSE_FLOAT_VECTOR / ES 的 sparse_vector

        graph    **根本不是向量**：实体提及与关系的局部图（GraphFragment）
                 存 Neo4j 之类的图库

    它同时决定了"哪种 retrieval_type 需要哪种能力"：

        dense_vector  -> DENSE
        sparse_vector -> SPARSE
        graph         -> GRAPH

    拿稠密模型去服务 sparse_vector 不会报错 —— 维度对得上、检索照常
    返回结果，只是排序没有意义。所以选取时必须按这个枚举匹配，
    而不是"有 embedder 就用"。

    ⚠️ GRAPH 的存在说明这个枚举名已经不太贴切了
    --------------------------------------------
    它叫 EmbeddingOutput，值里却有一个不产出向量的成员。

    机制上是对的：阶段 9 本来就是"按 retrieval 需要什么，挑一个能力
    来加工这一批 chunk"，dense/sparse 产出向量、graph 产出图，只是
    "产出物"的三种形态。但**名字**（以及 DocumentEmbedder / embed()）
    是从"向量化"那个角度起的，现在名不副实了。
    要不要把这一族改名（例如 ChunkRepresentation / produce()），
    是个独立决定，见 embedder/graph.py 的说明。
    """

    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"


class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class RetrievalType(str, Enum):
    """
    Retrieval type is a business-level retrieval/storage representation.

    Examples:
        document     -> PostgreSQL document chunks
        keyword      -> lexical / BM25 retrieval
        sparse_vector
        dense_vector
        graph
    """

    DOCUMENT = "document"
    KEYWORD = "keyword"
    SPARSE_VECTOR = "sparse_vector"
    DENSE_VECTOR = "dense_vector"
    GRAPH = "graph"


#: retrieval_type -> 它需要哪一种**向量形态**。
#:
#: 不在表里的（document / keyword）不需要向量：前者直接查 chunk 表，
#: 后者走 BM25 全文，都用原文即可。
#:
#: 为什么放在这里：写入期（决定"要产出什么向量"）与查询期（决定
#: "要拿什么向量去查"）用的是**同一个对应关系**，两边不一致时维度
#: 可能照样对得上、检索照样返回结果，只是排序没有意义。原来两边各
#: 手写一份，正是那种"改了一处忘了另一处"的温床，所以收敛到枚举
#: 旁边这一份 —— 与 StorageBackend 注释里"删除并列枚举以免漂移"
#: 是同一条纪律。
#:
#: ⚠️ 查询期目前只用得到前两项（graph 的召回还没实现），它那边读的
#: 是同一个表，因此**拿不到**就是"这一路暂时没有查询向量"，不必
#: 单独维护一份子集。
RETRIEVAL_EMBEDDING_KIND: dict[str, EmbeddingOutput] = {
    RetrievalType.DENSE_VECTOR.value: EmbeddingOutput.DENSE,
    RetrievalType.SPARSE_VECTOR.value: EmbeddingOutput.SPARSE,
    RetrievalType.GRAPH.value: EmbeddingOutput.GRAPH,
}


class StorageBackend(str, Enum):
    """
    实际存储 / Index 后端类型。

    这是**唯一**的后端枚举，被三处引用：

        DocumentRetrievalDefinition.storage_backend
        DocumentIndexBinding.backend_type
        DocumentIndexBinding 的 model 列

    曾有一个并列的 BackendType（ELASTICSEARCH/MILVUS/FAISS/
    QDRANT/OPENSEARCH）描述同一件事，但从未被任何地方引用，
    已删除以免两个枚举漂移。新增后端一律加在这里。

    注意它只描述"用什么 Index 系统"，
    不规定 Retrieval 类型（见 RetrievalType）。
    """

    POSTGRESQL = "postgresql"
    ELASTICSEARCH = "elasticsearch"
    MILVUS = "milvus"
    NEO4J = "neo4j"


# ============================================================
# 查询期（RetrieveProcessor）
# ============================================================
#
# 对应 app/services/document/document_retrieve.yaml。
#
# 这几个枚举描述的是**一次检索轮次**的形态，与写入链上的
# ProcessorRole 不共用 —— 查询期能力（fuse / rerank）刻意不在
# 能力目录里，见 processor_capability.yaml 的 "role 的覆盖范围"。


class RetrieveScope(str, Enum):
    """
    本次查询覆盖了几个 DocumentTypeDefinition。

    single:
        一个 DTD，走它自己的 RetrieveProcessor。
    multi:
        多个 DTD，各自跑完再统一过 COMMON_RETRIEVE_PROCESSOR_CHAIN。
    """

    SINGLE = "single"
    MULTI = "multi"


class RetrieveProcessorMode(str, Enum):
    """
    本次查询实际使用了哪条 RetrieveProcessor 装配路径。

    dtd:
        单个 DTD 的 RetrieveProcessor。
    common_chain:
        每个 DTD 各自处理，之后再过 COMMON_RETRIEVE_PROCESSOR_CHAIN。
    """

    DTD = "dtd"
    COMMON_CHAIN = "common_chain"


class RetrieveProcessorSource(str, Enum):
    """
    生效的 RetrieveProcessor 从哪来 —— 响应里回显给调用方。

    document_type_definition:
        完全来自 DTD 配置。
    common_retrieve_processor_chain:
        来自系统级的公共链配置。
    runtime_override:
        本次请求带了至少一项真正改变了生效配置的 override。
    """

    DOCUMENT_TYPE_DEFINITION = "document_type_definition"
    COMMON_RETRIEVE_PROCESSOR_CHAIN = (
        "common_retrieve_processor_chain"
    )
    RUNTIME_OVERRIDE = "runtime_override"


class FusionerType(str, Enum):
    """
    多路召回结果的融合算法。

    与 DTD 的 fusion_strategy 字符串取值一致
    （例如 rrf、weighted_sum）。
    """

    RRF = "rrf"
    WEIGHTED_SUM = "weighted_sum"


class RerankerType(str, Enum):
    """
    融合之后的精排算法。

    与 DTD 的 rerank_strategy 字符串取值一致。
    """

    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"
    CUSTOM = "custom"
