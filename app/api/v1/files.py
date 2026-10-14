from __future__ import annotations

from datetime import datetime
from urllib.parse import quote
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File as FastAPIFile,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response

from app.common.auth import AuthenticatedUser, current_user
from app.services.dependencies import Service
from app.services.file.application.service import FileService

from app.services.file.domain.schemas.enum import (
    FileOrigin,
    FileStatus,
)
from app.services.file.domain.schemas.file import (
    DeleteFileRequest,
    DeleteFileResponse,
    FileResponse,
    GetFileContentRequest,
    GetFileRequest,
    GetFileResponse,
    ListFilesRequest,
    ListFilesResponse,
    UploadFileRequest,
)

router = APIRouter(
    prefix="/files",
    tags=["files"],
)


def _content_disposition(
        file_name: str,
) -> str:
    """
    Build an RFC 6266 / RFC 5987 compliant
    Content-Disposition header.

    HTTP header 只能承载 latin-1 字符。
    中文等非 ASCII 文件名直接拼进 filename=
    会抛 UnicodeEncodeError，因此必须：

        filename=
            ASCII 回退名，给不支持 RFC 5987 的老客户端

        filename*=UTF-8''<percent-encoded>
            真实的 UTF-8 文件名，现代浏览器优先使用它

    同时过滤引号与控制字符（含 CR/LF），
    避免破坏 header 结构或造成 header 注入。
    """

    ascii_name = "".join(
        ch
        if (
                ch.isascii()
                and ch.isprintable()
                and ch not in '"\\'
        )
        else "_"
        for ch in file_name
    ).strip()

    if not ascii_name:
        ascii_name = "download"

    encoded_name = quote(
        file_name,
        safe="",
    )

    return (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{encoded_name}"
    )


# ⚠️ 上传会记下 uploaded_by，但**查询不做归属过滤**（刻意的取舍）
# ------------------------------------------------------------
# 只落库、先不设权限边界：GET /files、GET /files/{id}、/{id}/content、
# DELETE /{id} 目前对**任何**登录用户开放 —— 猜中 file_id 就能拿到别人的
# 文件。要收口时在 FileService 的 list_files / get_file / delete_file 上
# 加 `uploaded_by = 当前用户 OR uploaded_by IS NULL` 即可（与 session
# 那一套同形，NULL 同样是"加这列之前的存量"）。
#
# 值一律来自 access token、**不从请求体取**：multipart 端点本来就不接收
# 任何用户字段，保持这样 —— 让调用方声明上传人等于允许替别人署名。
@router.post(
    "",
    response_model=FileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
        upload: UploadFile = FastAPIFile(...),
        service: FileService = Service[FileService],
        user: AuthenticatedUser = Depends(current_user),
):
    """
    上传一个新的文件资产。

    注意：
        这里只创建 File。
        不创建 Document。
    """

    content = await upload.read()

    request = UploadFileRequest(
        file_name=upload.filename or "unknown",
        content_type=upload.content_type,
        content=content,
        # 空 subject 归一成 None（列里 NULL）—— 与会话那套同一条规矩，
        # 见 api/v1/sessions.py 的 _owner_id。
        uploaded_by=user.subject or None,
    )

    try:
        return await service.upload_file(
            request
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.get(
    "",
    response_model=ListFilesResponse,
)
async def list_files(
        origin: FileOrigin = Query(
            default=FileOrigin.USER_UPLOAD,
            description=(
                "文件来源。默认只返回用户上传的资产；"
                "流水线中间产物需显式传 pipeline_artifact。"
            ),
        ),

        tenant_id: UUID | None = Query(
            default=None,
            description="租户 ID。",
        ),

        status: FileStatus | None = Query(
            default=None,
            description=(
                "文件生命周期状态。"
                "显式指定时忽略 include_deleted。"
            ),
        ),

        file_name: str | None = Query(
            default=None,
            max_length=500,
            description="文件名模糊匹配，不区分大小写。",
        ),

        content_type: str | None = Query(
            default=None,
            max_length=200,
            description=(
                "MIME Type 前缀匹配。"
                "例如 image/ 匹配所有图片。"
            ),
        ),

        uploaded_by: str | None = Query(
            default=None,
            max_length=255,
            description=(
                "按上传用户过滤（Casdoor 的 sub，字符串）。"
                "⚠️ 它是过滤条件、**不是**授权边界 —— 见本文件上方那段说明。"
            ),
        ),

        created_from: datetime | None = Query(
            default=None,
            description="创建时间下界，闭区间。",
        ),

        created_to: datetime | None = Query(
            default=None,
            description="创建时间上界，闭区间。",
        ),

        include_deleted: bool = Query(
            default=False,
            description="是否包含已逻辑删除的文件。",
        ),

        page: int = Query(
            default=1,
            ge=1,
            description="页码，从 1 开始。",
        ),

        page_size: int = Query(
            default=20,
            ge=1,
            le=100,
            description="每页数量。",
        ),

        service: FileService = Service[FileService],
):
    """
    分页查询 File 列表。

    只查询 PostgreSQL，不访问物理存储。

    排序固定为 created_at 倒序。

    status 与 include_deleted 的优先级：

        status 显式指定       -> 只按该 status 过滤
        include_deleted=true  -> active 与 deleted 都返回
        否则                  -> 只返回 active
    """

    request = ListFilesRequest(
        origin=origin,
        tenant_id=tenant_id,
        status=status,
        file_name=file_name,
        content_type=content_type,
        uploaded_by=uploaded_by,
        created_from=created_from,
        created_to=created_to,
        include_deleted=include_deleted,
        page=page,
        page_size=page_size,
    )

    try:
        return await service.list_files(
            request
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.get(
    "/{file_id}",
    response_model=GetFileResponse,
)
async def get_file(
        file_id: UUID,

        service: FileService = Service[FileService],
):
    """
    获取文件 Metadata。
    """

    try:
        return await service.get_file(
            GetFileRequest(
                file_id=file_id
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.get(
    "/{file_id}/content",
)
async def get_file_content(
        file_id: UUID,

        service: FileService = Service[FileService],
):
    """
    获取文件内容。

    当前先直接返回 bytes。
    后续可以改为 StreamingResponse。
    """

    try:
        result = await service.get_file_content(
            GetFileContentRequest(
                file_id=file_id
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return Response(
        content=result.content,
        media_type=(
                result.content_type
                or "application/octet-stream"
        ),
        headers={
            "Content-Disposition": (
                _content_disposition(
                    result.file_name
                )
            )
        },
    )


@router.delete(
    "/{file_id}",
    response_model=DeleteFileResponse,
)
async def delete_file(
        file_id: UUID,

        service: FileService = Service[FileService],
):
    """
    逻辑删除 File。

    不物理删除文件。
    """

    try:
        return await service.delete_file(
            DeleteFileRequest(
                file_id=file_id
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
