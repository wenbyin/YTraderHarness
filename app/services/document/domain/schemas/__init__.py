"""
Public Pydantic schemas for the Document domain.

domain/schemas/
│
├── common.py
│   └── 公共类型 / 文件引用
│
├── enum.py
│   └── VersionMode / ChunkMode / RetrievalMethod /
│       DefinitionStatus / DocumentStatus / VersionStatus /
│       ProcessingStatus / StorageBackend / ProcessorRole
│
├── capability.py
│   └── 处理能力目录（processor_key 契约表 + config_schema）的对外契约
│
├── definition.py
│   └── DocumentTypeDefinition / DocumentProcessorDefinition 管理契约
│
├── runtime.py
│   └── Document / DocumentVersion / DocumentChunk 运行时契约
│
├── observability.py
│   └── PipelineRun / PipelineNodeRun / 节点目录 / reprocess 的对外契约
│
├── processing.py
│   └── Parser / Chunk / Index / Fusion / Rerank 内部契约
│
└── retrieve.py
    └── POST /documents/retrieve 的对外契约（RetrieveProcessor 四段配置）

注意：
    document.py / version.py 两个模块描述的是
    document_type_definition.yaml 之前的旧契约（DocumentService ABC 那一套），
    其中的 CreateDocumentRequest / CreateDocumentResponse
    已与 runtime.py 同名冲突，因此不再从本包导出。

    同属那套旧契约的 search.py 与 application/service.py
    已随 document_retrieve.yaml 的落地删除 —— 检索的对外契约
    现在唯一地由 retrieve.py 给出，避免两套 API 契约长期漂移。
"""

from .common import (
    APIModel,
    ErrorInfo,
    FileSource,
    PageInfo,
    StoredFile,
)

from .enum import (
    ChunkMode,
    DefinitionStatus,
    DocumentStatus,
    ProcessingStatus,
    RetrievalMethod,
    VersionMode,
    VersionStatus,
)

from .capability import (
    ProcessorCapabilityCatalogResponse,
    ProcessorCapabilityEntry,
    ProcessorCapabilitySelector,
)

from .definition import (
    DocumentProcessorDefinitionListResponse,
    DocumentProcessorDefinitionRequest,
    DocumentProcessorDefinitionResponse,
    DocumentRetrievalDefinitionListResponse,
    DocumentRetrievalDefinitionRequest,
    DocumentRetrievalDefinitionResponse,
    DocumentTypeDefinitionCreateRequest,
    DocumentTypeDefinitionDetailResponse,
    DocumentTypeDefinitionListResponse,
    DocumentTypeDefinitionResponse,
    DocumentTypeDefinitionUpdateRequest,
    ProcessorRetrievalBindingRequest,
    ProcessorRetrievalBindingResponse,
)

from .version_chunks import (
    ChunkProjection,
    ChunkProjectionListResponse,
    ChunkProjectionRequest,
    ChunkerBinding,
    DocumentChunkListResponse,
    DocumentChunkSummary,
    DocumentVersionRetrievalListResponse,
    DocumentVersionRetrievalSummary,
    SparseSummary,
    VectorSummary,
)

from .runtime import (
    DocumentCreateRequest,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentVersionDetailResponse,
    DocumentVersionListResponse,
    DocumentVersionSummary,
)

from .observability import (
    PipelineNodeCatalogEntry,
    PipelineNodeCatalogResponse,
    PipelineNodeRunListResponse,
    PipelineNodeRunResponse,
    PipelineRunDetailResponse,
    PipelineRunListResponse,
    PipelineRunResponse,
    ReprocessDocumentVersionRequest,
    ReprocessDocumentVersionResponse,
)

from .retrieve import (
    CutterConfig,
    DegradedRoute,
    DocumentRetrieveRequest,
    DocumentRetrieveResponse,
    DocumentRetrieveResult,
    EffectiveRetrieveProcessor,
    FilterConfig,
    FusionerConfig,
    RetrievalRecallConfig,
    RetrieveProcessorOverride,
    RerankerConfig,
)

from .processing import (
    CleanRequest,
    CleanResult,
    EnrichRequest,
    CanonicalChunk,
    ChunkRequest,
    ChunkResult,
    EmbedRequest,
    EntityMention,
    FusionRequest,
    GraphFragment,
    IndexDeleteRequest,
    IndexSearchRequest,
    IndexSearchResponse,
    IndexSearchResult,
    IndexUpsertRequest,
    ParseRequest,
    ParseResult,
    ProcessDocumentVersionRequest,
    ProcessorRequest,
    ProcessorResult,
    ProcessDocumentVersionResponse,
    RelationMention,
    RerankRequest,
)
