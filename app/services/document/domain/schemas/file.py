from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )


class FileUpload(APIModel):
    """
    DocumentService 内部的文件输入。

    source_uri：
        已经存在于临时文件 / 文件服务中的文件引用。

    DocumentService 不强制要求直接传 bytes。
    """

    source_uri: str
    file_name: str
    content_type: str | None = None
    file_size: int | None = Field(default=None, ge=0)


class StoredFile(APIModel):
    storage_key: str

    storage_bucket: str | None = None

    file_name: str

    file_size: int

    content_type: str | None = None

    checksum: str
