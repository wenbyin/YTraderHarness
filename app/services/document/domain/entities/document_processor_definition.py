from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import DefinitionStatus


class DocumentProcessorDefinition(BaseModel):
    """
    File pre-processing definition.

    Responsibility:

        file_extension -> loader -> processor -> DocumentTypeDefinition.modality_type

    It does NOT contain chunker or retrieval configuration.

    Within one DocumentTypeDefinition, file_extension is unique.

    Example:
        .pdf  -> pdf_loader -> pdf_processor
        .md   -> markdown_loader -> markdown_processor

    Both may produce the same modality_type, such as text.
    """

    model_config = ConfigDict(from_attributes=True)

    document_processor_definition_id: UUID = Field(
        description="DocumentProcessorDefinition primary key."
    )
    document_type_definition_id: UUID = Field(
        description="Owning DocumentTypeDefinition ID."
    )

    # File format
    file_extension: str = Field(
        description="Supported file extension, e.g. .pdf, .md, .docx."
    )

    mime_types: list[str] = Field(
        default_factory=list,
        description="Allowed MIME types for the extension.",
    )

    # Reader / processor
    loader_name: str = Field(
        description="Reader/Loader capability name."
    )

    loader_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Loader runtime configuration.",
    )

    processor_name: str | None = Field(
        default=None,
        description="Processor capability name; null when no extra processing is needed.",
    )

    processor_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Processor runtime configuration.",
    )

    # Lifecycle
    status: DefinitionStatus = Field(
        default=DefinitionStatus.ACTIVE,
        description="active or logically deleted.",
    )

    # Audit
    created_by: UUID | None = Field(
        default=None,
        description="User who created this definition.",
    )
    created_at: datetime = Field(description="Creation timestamp.")
    updated_at: datetime = Field(description="Last update timestamp.")
    deleted_at: datetime | None = Field(
        default=None,
        description="Logical deletion timestamp.",
    )
