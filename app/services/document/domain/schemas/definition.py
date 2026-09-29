"""
Pydantic schemas for Document definition management.

对应 app/services/document/document_type_definition.yaml 的定义层契约：

    DocumentTypeDefinition        写入的唯一入口
    DocumentProcessorDefinition   后缀名 → loader → processor
    DocumentRetrievalDefinition   chunker → retriever → storage
    ProcessorRetrievalBinding     两者之间的 N:N

关键语义：

    写只走父级
        子表没有独立写接口，级联保存全部通过
        DocumentTypeDefinitionCreateRequest /
        DocumentTypeDefinitionUpdateRequest 完成。

    子项按自然键识别
        processor 用 file_extension，retrieval 用 retrieval_type。
        不传子项 id，因此创建与更新共用同一套子项结构。

    全量替换
        payload 中出现的子项即其最终状态；
        未出现的既有子项会被软删除。
        父级顶层字段仍为部分更新。

    乐观锁
        更新必须回传 revision，不匹配则拒绝。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

from .enum import (
    DefinitionStatus,
    AnalysisMode,
    ModalityType,
    RetrievalType,
    StorageBackend,
    VersionMode,
)


# ============================================================
# 子项请求（创建与更新共用）
# ============================================================


class DocumentProcessorDefinitionRequest(APIModel):
    """
    文件前处理定义。

    按 file_extension 自然键识别，不传 id。
    """

    file_extension: str = Field(
        min_length=1,
        max_length=32,
        description=(
            "该前处理定义适用的文件扩展名。"
            "例如 .pdf、.md、.html。"
            "同一 DocumentTypeDefinition 下唯一。"
        ),
    )

    mime_types: list[str] = Field(
        default_factory=list,
        description=(
            "该扩展名允许对应的 MIME Type 集合。"
            "例如 ['application/pdf']。"
        ),
    )

    loader_name: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "Loader / Reader 能力名称。"
            "该字段决定能否解析该后缀名。"
            "例如 pdf_loader、markdown_loader。"
        ),
    )

    loader_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Loader 的具体运行参数。",
    )

    processor_name: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Processor 能力名称。"
            "为空表示不需要额外处理。"
        ),
    )

    processor_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Processor 的具体运行参数。",
    )


class DocumentRetrievalDefinitionRequest(APIModel):
    """
    检索定义。

    按 retrieval_type 自然键识别，不传 id。
    只是 [chunker + retriever + storage] 的配置集合，
    不持有任何 chunk 内容。
    """

    retrieval_type: RetrievalType = Field(
        description=(
            "业务检索类型。"
            "同一 DocumentTypeDefinition 下唯一。"
        ),
    )

    chunker_name: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Chunker 能力名称。"
            "某些 retrieval 类型不需要传统 Chunk。"
        ),
    )

    chunker_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunker 的运行参数。",
    )

    retriever_name: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Retriever 能力名称。"
            "例如 bm25_retriever、vector_retriever。"
        ),
    )

    retriever_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Retriever 的召回参数。",
    )

    storage_backend: StorageBackend | None = Field(
        default=None,
        description=(
            "实际存储后端。"
            "不传时由 retrieval_type 默认映射得到。"
        ),
    )

    storage_config: dict[str, Any] = Field(
        default_factory=dict,
        description="实际存储后端的配置。",
    )


class ProcessorRetrievalBindingRequest(APIModel):
    """
    一条绑定关系。

    用同一次 payload 中两个数组的**下标**描述：

        processor_index  -> processors[processor_index]
        retrieval_index  -> retrievals[retrieval_index]

    选择下标而不是自然键的原因：
        绑定关系只在这份 payload 内有意义，
        下标不受大小写、空格、别名等匹配问题影响。

    下标指的是 **payload 数组中的位置**，
    不是数据库里的顺序，因此服务端必须按
    payload 顺序保留子项，不能重新排序。
    """

    processor_index: int = Field(
        ge=0,
        description="processors 数组中的下标。",
    )

    retrieval_index: int = Field(
        ge=0,
        description="retrievals 数组中的下标。",
    )


# ============================================================
# DocumentTypeDefinition 写入
# ============================================================


class DocumentTypeDefinitionCreateRequest(APIModel):
    """
    创建类型定义，并级联创建子项。

    子项按自然键识别，因此创建时不需要子项 id。
    """

    tenant_id: UUID | None = Field(
        default=None,
        description="租户 ID。",
    )

    type_key: str = Field(
        min_length=1,
        max_length=100,
        description="文档类型唯一编码，创建后不可修改。",
    )

    name: str = Field(
        min_length=1,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
    )

    modality_type: ModalityType = Field(
        description="该类型唯一的业务内容类型，创建后不可修改。",
    )

    analysis_mode: AnalysisMode = Field(
        default=AnalysisMode.DEFAULT,
        description=(
            "default=使用系统预制后缀名 Reader，"
            "不建 ProcessorDefinition，该类型下全部 retrieval 都参与；"
            "config=使用自定义 ProcessorDefinition；"
            "none=不解析文件。"
        ),
    )

    allow_update: bool = Field(
        default=True,
    )

    version_mode: VersionMode = Field(
        default=VersionMode.VERSIONED,
    )

    fusion_strategy: str | None = Field(
        default=None,
        max_length=100,
    )

    fusion_config: dict[str, Any] = Field(
        default_factory=dict,
    )

    rerank_strategy: str | None = Field(
        default=None,
        max_length=100,
    )

    rerank_config: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    processors: list[DocumentProcessorDefinitionRequest] = Field(
        default_factory=list,
        description="随该类型一并创建的文件前处理定义。",
    )

    retrievals: list[DocumentRetrievalDefinitionRequest] = Field(
        default_factory=list,
        description="随该类型一并创建的检索定义。",
    )

    bindings: list[ProcessorRetrievalBindingRequest] = Field(
        default_factory=list,
        description=(
            "N:N 绑定。两个自然键都必须命中上面两个数组。"
        ),
    )


class DocumentTypeDefinitionUpdateRequest(APIModel):
    """
    更新类型定义（级联全量替换）。

    父级顶层字段部分更新：未提供则保持不变。
    三个子项数组为全量替换：未出现的既有子项会被软删除，
    其名下的 binding 一并级联删除。
    """

    revision: int = Field(
        ge=1,
        description=(
            "乐观锁。必须回传 GET 得到的 revision，"
            "不匹配返回 409。"
        ),
    )

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
    )

    analysis_mode: AnalysisMode | None = Field(
        default=None,
    )

    allow_update: bool | None = Field(
        default=None,
    )

    version_mode: VersionMode | None = Field(
        default=None,
    )

    fusion_strategy: str | None = Field(
        default=None,
        max_length=100,
    )

    fusion_config: dict[str, Any] | None = Field(
        default=None,
    )

    rerank_strategy: str | None = Field(
        default=None,
        max_length=100,
    )

    rerank_config: dict[str, Any] | None = Field(
        default=None,
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
    )

    processors: list[
        DocumentProcessorDefinitionRequest
    ] | None = Field(
        default=None,
        description=(
            "None=不触碰子表；数组=全量替换，"
            "未出现的既有 processor 会被软删除。"
        ),
    )

    retrievals: list[
        DocumentRetrievalDefinitionRequest
    ] | None = Field(
        default=None,
        description=(
            "None=不触碰子表；数组=全量替换，"
            "未出现的既有 retrieval 会被软删除。"
        ),
    )

    bindings: list[
        ProcessorRetrievalBindingRequest
    ] | None = Field(
        default=None,
        description=(
            "None=不触碰绑定；数组=全量替换。"
        ),
    )


# ============================================================
# 子项响应
# ============================================================


class DocumentProcessorDefinitionResponse(APIModel):

    processor_definition_id: UUID

    document_type_definition_id: UUID

    file_extension: str

    mime_types: list[str]

    loader_name: str

    loader_config: dict[str, Any]

    processor_name: str | None

    processor_config: dict[str, Any]

    retrieval_types: list[RetrievalType] = Field(
        default_factory=list,
        description=(
            "只读。该前处理定义当前绑定到的 retrieval，"
            "由 bindings 推导。"
        ),
    )

    status: DefinitionStatus

    created_at: datetime

    updated_at: datetime


class DocumentRetrievalDefinitionResponse(APIModel):

    retrieval_definition_id: UUID

    document_type_definition_id: UUID

    retrieval_type: RetrievalType

    chunker_name: str | None

    chunker_config: dict[str, Any]

    retriever_name: str | None

    retriever_config: dict[str, Any]

    storage_backend: StorageBackend

    storage_config: dict[str, Any]

    file_extensions: list[str] = Field(
        default_factory=list,
        description=(
            "只读。绑定到该 retrieval 的文件后缀名，"
            "由 bindings 推导。"
        ),
    )

    status: DefinitionStatus

    created_at: datetime

    updated_at: datetime


class ProcessorRetrievalBindingResponse(APIModel):
    """
    一条绑定关系的读模型。

    读侧仍然给出**自然键**而不是下标：
    下标脱离请求上下文就没有意义，
    而前端详情页需要知道"哪条 Pipeline 绑了哪个检索"，
    用文件名后缀与 retrieval_type 表达才自描述。

    前端做 GET -> 编辑 -> PUT 回环时，
    按下标在 processors / retrievals 数组中定位即可。
    """

    file_extension: str

    retrieval_type: RetrievalType


# ============================================================
# DocumentTypeDefinition 响应
# ============================================================


class DocumentTypeDefinitionResponse(APIModel):

    document_type_definition_id: UUID

    revision: int = Field(
        description="乐观锁版本号，每次成功更新 +1。",
    )

    tenant_id: UUID | None

    type_key: str

    name: str

    description: str | None

    modality_type: ModalityType

    analysis_mode: AnalysisMode

    allow_update: bool

    version_mode: VersionMode

    fusion_strategy: str | None

    fusion_config: dict[str, Any]

    rerank_strategy: str | None

    rerank_config: dict[str, Any]

    metadata: dict[str, Any]

    status: DefinitionStatus

    created_at: datetime

    updated_at: datetime


class DocumentTypeDefinitionDetailResponse(
    DocumentTypeDefinitionResponse
):
    """详情：附带三个子项集合。"""

    processors: list[
        DocumentProcessorDefinitionResponse
    ] = Field(
        default_factory=list,
    )

    retrievals: list[
        DocumentRetrievalDefinitionResponse
    ] = Field(
        default_factory=list,
    )

    bindings: list[
        ProcessorRetrievalBindingResponse
    ] = Field(
        default_factory=list,
    )


class DocumentTypeDefinitionListResponse(APIModel):

    items: list[DocumentTypeDefinitionResponse] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int

    total_pages: int


class DocumentProcessorDefinitionListResponse(APIModel):

    items: list[DocumentProcessorDefinitionResponse] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int

    total_pages: int


class DocumentRetrievalDefinitionListResponse(APIModel):

    items: list[DocumentRetrievalDefinitionResponse] = Field(
        default_factory=list,
    )

    page: int

    page_size: int

    total: int

    total_pages: int


DocumentTypeDefinitionDetailResponse.model_rebuild()
