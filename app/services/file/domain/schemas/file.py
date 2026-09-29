from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel
from .enum import FileOrigin, FileStatus


class StoredFile(APIModel):
    """
    FileStorage.put() 成功后的标准返回结果。

    表示：
        文件已经成功写入具体的物理存储后，
        Storage 层返回给 FileService 的结果。

    注意：
        StoredFile 只描述“存储结果”，
        不描述 FileService 的业务生命周期。

    因此这里不包含：
        - file_id
        - tenant_id
        - status
        - uploaded_by
        - created_at
        - updated_at

    这些字段属于 File Entity。
    """

    storage_backend: str = Field(
        min_length=1,
        max_length=50,
        description=(
            "实际使用的文件存储后端。"
            "第一版固定为 local，未来可支持 s3、minio、oss、cos 等。"
        ),
    )

    storage_bucket: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Object Storage Bucket。"
            "Local filesystem 不使用 Bucket，因此通常为 None。"
        ),
    )

    storage_key: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "文件在具体 Storage 中的唯一 Key。"
            "通常基于 file_id 生成，不直接使用用户原始文件名。"
        ),
    )

    file_name: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "用户上传时的原始文件名。"
            "用于展示以及下载文件时恢复原始文件名。"
            "该字段不作为物理存储的唯一标识。"
        ),
    )

    file_size: int = Field(
        ge=0,
        description=(
            "实际写入 Storage 的文件大小，单位 Byte。"
        ),
    )

    content_type: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "文件 MIME Type，例如 application/pdf。"
        ),
    )

    checksum: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "实际写入 Storage 的文件内容 checksum。"
            "默认使用 SHA-256。"
        ),
    )


class UploadFileRequest(APIModel):
    tenant_id: UUID | None = None

    file_name: str = Field(
        min_length=1,
        max_length=500,
    )

    content_type: str | None = Field(
        default=None,
        max_length=200,
    )

    content: bytes

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    uploaded_by: UUID | None = None


class FileResponse(APIModel):
    file_id: UUID

    tenant_id: UUID | None

    file_name: str

    content_type: str | None

    file_size: int

    checksum: str

    storage_backend: str

    storage_key: str

    status: str

    metadata: dict[str, Any]

    created_at: datetime

    updated_at: datetime

    deleted_at: datetime | None


class GetFileRequest(APIModel):
    file_id: UUID


class GetFileResponse(FileResponse):
    pass


class DeleteFileRequest(APIModel):
    file_id: UUID

    deleted_by: UUID | None = None


class DeleteFileResponse(APIModel):
    file_id: UUID

    status: str

    deleted_at: datetime


class GetFileContentRequest(APIModel):
    file_id: UUID


class FileContentResponse(APIModel):
    file_id: UUID

    file_name: str

    content_type: str | None

    content: bytes


class ListFilesRequest(APIModel):
    """
    分页查询 File 列表。

    过滤条件之间是 AND 关系。

    status 与 include_deleted 的优先级：

        status 显式指定       -> 只按该 status 过滤
        include_deleted=True  -> 不限制 status，active 与 deleted 都返回
        否则                  -> 只返回 active

    排序固定为 created_at 倒序，
    调用方不能指定排序字段。

    origin 默认 user_upload —— 流水线中间产物不会出现在列表里。
    想连产物一起看时显式传 pipeline_artifact。
    """

    origin: FileOrigin = Field(
        default=FileOrigin.USER_UPLOAD,
        description=(
            "文件来源。默认只看用户上传的资产；"
            "流水线产物需显式指定 pipeline_artifact。"
        ),
    )

    tenant_id: UUID | None = Field(
        default=None,
        description=(
            "租户 ID。为空时不按租户过滤。"
        ),
    )

    status: FileStatus | None = Field(
        default=None,
        description=(
            "文件生命周期状态。"
            "显式指定时优先级最高，忽略 include_deleted。"
        ),
    )

    file_name: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "文件名模糊匹配，不区分大小写的包含匹配。"
            "为空时不按文件名过滤。"
        ),
    )

    content_type: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "MIME Type 前缀匹配。"
            "例如 image/ 可以同时匹配 image/png 与 image/jpeg，"
            "application/pdf 则精确匹配 PDF。"
        ),
    )

    uploaded_by: UUID | None = Field(
        default=None,
        description="上传用户 ID。",
    )

    created_from: datetime | None = Field(
        default=None,
        description="创建时间下界，闭区间。",
    )

    created_to: datetime | None = Field(
        default=None,
        description="创建时间上界，闭区间。",
    )

    include_deleted: bool = Field(
        default=False,
        description=(
            "是否包含已逻辑删除的文件。"
            "status 显式指定时该项被忽略。"
        ),
    )

    page: int = Field(
        default=1,
        ge=1,
        description="页码，从 1 开始。",
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="每页数量。",
    )


class FileListItem(APIModel):
    """
    文件列表中的单个条目。

    注意：
        这里刻意不包含 storage_key / storage_backend。

        storage_key 是存储层实现细节。
        列表调用方不需要它，也不应该绕过 FileService
        直接访问物理文件。

        item 也不包含 metadata：
        列表页只需要展示信息，
        需要完整 metadata 时走 get_file。
    """

    file_id: UUID

    tenant_id: UUID | None

    file_name: str

    content_type: str | None

    file_size: int

    checksum: str

    status: str

    uploaded_by: UUID | None

    created_at: datetime

    updated_at: datetime

    deleted_at: datetime | None


class ListFilesResponse(APIModel):
    """
    分页的文件列表结果。

    total_pages:
        由 total 与 page_size 计算得到。
        total=0 时为 0。
    """

    items: list[FileListItem]

    page: int

    page_size: int

    total: int

    total_pages: int
