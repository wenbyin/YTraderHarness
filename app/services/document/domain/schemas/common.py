"""
Common Pydantic schemas used by the Document Service.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.common.schemas import APIModel


class FileSource(APIModel):
    """
    输入文件引用。

    DocumentService 不要求上层直接传 bytes。
    文件本身由上传层 / File Service 先保存到可访问位置，
    DocumentService 通过 source_uri 读取。

    source_uri:
        文件输入位置，例如：
        - 本地临时文件
        - file service URL
        - 内部文件引用

    注意：
        source_uri 是“输入引用”，
        不是最终 Object Storage 的 storage_key。
    """

    source_uri: str = Field(
        min_length=1,
        description="输入文件的来源 URI 或内部文件引用。",
    )

    file_name: str = Field(
        min_length=1,
        max_length=500,
        description="用户上传时的原始文件名。",
    )

    content_type: str | None = Field(
        default=None,
        max_length=200,
        description="文件 MIME Type，例如 application/pdf。",
    )

    file_size: int | None = Field(
        default=None,
        ge=0,
        description="文件大小，单位 Byte。",
    )


class StoredFile(APIModel):
    """
    ObjectStorage 保存文件后的结果。
    """

    storage_bucket: str | None = Field(
        default=None,
        description="Object Storage bucket。local storage 时通常为空。",
    )

    storage_key: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "Object Storage 中的唯一 Key。"
            "不直接使用用户原始文件名作为唯一标识。"
        ),
    )

    file_name: str = Field(
        min_length=1,
        max_length=500,
        description="原始文件名。",
    )

    file_size: int = Field(
        ge=0,
        description="实际保存的文件大小，单位 Byte。",
    )

    content_type: str | None = Field(
        default=None,
        max_length=200,
        description="文件 MIME Type。",
    )

    checksum: str = Field(
        min_length=1,
        max_length=128,
        description="原始文件内容 checksum，例如 SHA-256。",
    )


class PageInfo(APIModel):
    """
    可选的文档分页信息。
    """

    page_number: int | None = Field(
        default=None,
        ge=1,
        description="页码，从 1 开始。",
    )

    page_count: int | None = Field(
        default=None,
        ge=1,
        description="文档总页数。",
    )


class ErrorInfo(APIModel):
    """
    标准错误信息。
    """

    code: str
    message: str
    details: dict[str, Any] = Field(
        default_factory=dict
    )
