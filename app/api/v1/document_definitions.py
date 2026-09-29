from __future__ import annotations

from uuid import UUID

from fastapi import (
    APIRouter,
    Query,
    status,
)

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.document.application.definition_service import (
    DocumentDefinitionService,
)
from app.services.document.domain.schemas import (
    DocumentProcessorDefinitionListResponse,
    DocumentProcessorDefinitionResponse,
    DocumentRetrievalDefinitionListResponse,
    DocumentRetrievalDefinitionResponse,
    DocumentTypeDefinitionCreateRequest,
    DocumentTypeDefinitionDetailResponse,
    DocumentTypeDefinitionListResponse,
    DocumentTypeDefinitionUpdateRequest,
)
from app.services.document.domain.schemas.enum import (
    DefinitionStatus,
    ModalityType,
    RetrievalType,
)

router = APIRouter(
    tags=["document-definitions"],
)


# ============================================================
# DocumentTypeDefinition —— 唯一写入入口
# ============================================================


@router.get(
    "/document-type-definitions",
    response_model=DocumentTypeDefinitionListResponse,
)
async def list_type_definitions(
        tenant_id: UUID | None = Query(default=None),
        modality_type: ModalityType | None = Query(default=None),
        keyword: str | None = Query(
            default=None,
            max_length=255,
            description="同时匹配 type_key 与 name，不区分大小写。",
        ),
        status_: DefinitionStatus | None = Query(
            default=None, alias="status",
        ),
        include_deleted: bool = Query(default=False),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """分页查询类型定义。排序固定为 created_at 倒序。"""

    try:
        return await service.list_type_definitions(
            tenant_id=tenant_id,
            modality_type=modality_type,
            keyword=keyword,
            status=status_,
            include_deleted=include_deleted,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/document-type-definitions",
    response_model=DocumentTypeDefinitionDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_type_definition(
        request: DocumentTypeDefinitionCreateRequest,
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """
    创建类型定义，并级联创建 processors / retrievals / bindings。

    子项按自然键识别（processor 用 file_extension，
    retrieval 用 retrieval_type），创建时不需要子项 id。
    """

    try:
        return await service.create_type_definition(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/document-type-definitions/{document_type_definition_id}",
    response_model=DocumentTypeDefinitionDetailResponse,
)
async def get_type_definition(
        document_type_definition_id: UUID,
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """获取类型定义详情（含 processors / retrievals / bindings）。"""

    try:
        return await service.get_type_definition(
            document_type_definition_id
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/document-type-definitions/{document_type_definition_id}",
    response_model=DocumentTypeDefinitionDetailResponse,
)
async def update_type_definition(
        document_type_definition_id: UUID,
        request: DocumentTypeDefinitionUpdateRequest,
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """
    更新类型定义（级联全量替换）。

    父级顶层字段部分更新，未提供则保持不变。

    processors / retrievals / bindings 传数组时全量替换，
    未出现的既有子项会被软删除，其名下 binding 一并级联删除；
    传 null 表示不触碰该子项集合。

    必须回传 revision，不匹配返回 409。
    """

    try:
        return await service.update_type_definition(
            document_type_definition_id,
            request,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.delete(
    "/document-type-definitions/{document_type_definition_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_type_definition(
        document_type_definition_id: UUID,
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """逻辑删除类型定义及其全部子项。不物理删除。"""

    try:
        await service.delete_type_definition(
            document_type_definition_id
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 扁平读 —— DocumentProcessorDefinition
# ============================================================


@router.get(
    "/document-processors",
    response_model=DocumentProcessorDefinitionListResponse,
)
async def list_processors(
        document_type_definition_id: UUID | None = Query(
            default=None,
            description="按所属 DocumentTypeDefinition 过滤。",
        ),
        file_extension: str | None = Query(default=None, max_length=32),
        status_: DefinitionStatus | None = Query(
            default=None, alias="status",
        ),
        include_deleted: bool = Query(default=False),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """
    扁平查询文件前处理定义。

    按所属类型过滤请用 query 参数，而不是路径嵌套。
    """

    try:
        return await service.list_processors(
            document_type_definition_id=document_type_definition_id,
            file_extension=file_extension,
            status=status_,
            include_deleted=include_deleted,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/document-processors/{processor_definition_id}",
    response_model=DocumentProcessorDefinitionResponse,
)
async def get_processor(
        processor_definition_id: UUID,
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """
    扁平读取单个文件前处理定义。

    资源归属由响应中的 document_type_definition_id 给出。
    """

    try:
        return await service.get_processor(processor_definition_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 扁平读 —— DocumentRetrievalDefinition
# ============================================================


@router.get(
    "/document-retrievals",
    response_model=DocumentRetrievalDefinitionListResponse,
)
async def list_retrievals(
        document_type_definition_id: UUID | None = Query(
            default=None,
            description="按所属 DocumentTypeDefinition 过滤。",
        ),
        retrieval_type: RetrievalType | None = Query(default=None),
        status_: DefinitionStatus | None = Query(
            default=None, alias="status",
        ),
        include_deleted: bool = Query(default=False),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """扁平查询检索定义。"""

    try:
        return await service.list_retrievals(
            document_type_definition_id=document_type_definition_id,
            retrieval_type=retrieval_type,
            status=status_,
            include_deleted=include_deleted,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/document-retrievals/{retrieval_definition_id}",
    response_model=DocumentRetrievalDefinitionResponse,
)
async def get_retrieval(
        retrieval_definition_id: UUID,
        service: DocumentDefinitionService = Service[DocumentDefinitionService],
):
    """扁平读取单个检索定义。"""

    try:
        return await service.get_retrieval(retrieval_definition_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc
