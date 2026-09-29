"""
Pydantic schemas related to DocumentVersion.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from .common import APIModel


class ListDocumentVersionsRequest(APIModel):
    """
    获取一个 Document 的版本历史。
    """

    document_id: UUID

    include_deleted: bool = False


class DocumentVersionListItem(APIModel):
    """
    Version 历史列表中的单个版本。
    """

    document_version_id: UUID

    document_id: UUID

    version_number: int = Field(
        ge=1
    )

    version_status: str

    processing_status: str

    file_name: str

    content_type: str | None

    file_size: int | None

    file_checksum: str | None

    text_checksum: str | None

    created_at: datetime

    updated_at: datetime


class ListDocumentVersionsResponse(APIModel):
    """
    Document Version 历史。
    """

    document_id: UUID

    versions: list[DocumentVersionListItem]
