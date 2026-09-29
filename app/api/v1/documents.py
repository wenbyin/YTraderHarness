from __future__ import annotations

from uuid import UUID

from fastapi import (
    APIRouter,
    Query,
    status,
)

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.document.application.document_service import (
    DocumentRuntimeService,
)
from app.services.document.domain.schemas import (
    DocumentCreateRequest,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentVersionDetailResponse,
    DocumentVersionListResponse,
)
from app.services.document.domain.schemas.enum import (
    DocumentStatus,
    VersionStatus,
)

router = APIRouter(
    tags=["Documents"],
)


# ============================================================
# Document
# ============================================================


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
        request: DocumentCreateRequest,
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """
    用已上传的 File 创建逻辑 Document 及其首个版本。

    注意：
        解析 / 切分 / 建索引尚未接入，
        创建成功后版本不做后续状态流转。
    """

    try:
        return await service.create_document(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/documents",
    response_model=DocumentListResponse,
)
async def list_documents(
        tenant_id: UUID | None = Query(default=None),
        document_type_definition_id: UUID | None = Query(default=None),
        status_: DocumentStatus | None = Query(
            default=None, alias="status",
        ),
        keyword: str | None = Query(
            default=None,
            max_length=500,
            description="匹配 title 或 document_key。",
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """分页查询 Document。排序固定为 created_at 倒序。"""

    try:
        return await service.list_documents(
            tenant_id=tenant_id,
            document_type_definition_id=(
                document_type_definition_id
            ),
            status=status_,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
)
async def get_document(
        document_id: UUID,
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """Document 详情：本体 + 当前版本 + 全部版本。"""

    try:
        return await service.get_document(document_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/documents/{document_id}",
    response_model=DocumentResponse,
)
async def update_document(
        document_id: UUID,
        request: DocumentUpdateRequest,
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """
    更新 Document。

    三种操作，可单独或组合使用，但至少提供一个字段：

        title / metadata
            修改 Document 本身，不创建版本。

        file_id
            用新文件追加一个 DocumentVersion。

        current_document_version_id
            切换当前生效版本。

    file_id 与 current_document_version_id 互斥。
    """

    try:
        return await service.update_document(
            document_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
        document_id: UUID,
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """
    逻辑删除 Document。

    只改状态，不物理删除版本、Chunk 或原始文件。
    """

    try:
        await service.delete_document(document_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# DocumentVersion
# ============================================================


@router.get(
    "/documents/{document_id}/versions",
    response_model=DocumentVersionListResponse,
)
async def list_document_versions(
        document_id: UUID,
        status_: VersionStatus | None = Query(
            default=None, alias="status",
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """分页查询某个 Document 的版本，按版本号升序。"""

    try:
        return await service.list_versions(
            document_id,
            status=status_,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/documents/{document_id}/versions/{document_version_id}",
    response_model=DocumentVersionDetailResponse,
)
async def get_document_version(
        document_id: UUID,
        document_version_id: UUID,
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """
    单个版本详情。

    pipeline_config 是该版本创建时使用的配置快照，
    取自 document_version.meta_data。
    """

    try:
        return await service.get_version(
            document_id, document_version_id
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
