"""
单条 Chunk 的查询接口。

规格见 app/services/document/document_type_definition.yaml：

    GET /document-chunks/{document_chunk_id}    Get a DocumentChunk by ID

与 /document-versions/{vid}/retrievals/{rid}/chunks 的分工
---------------------------------------------------------
    那个   版本 × retrieval -> 这一路切出了什么（列表，分页）
    这个   只要一个 chunk id -> 这一块写了什么（单条，全文）

后者是给**手里已经有 id**的调用方的：检索评测的阶段 trace 逐层追踪
金标时只有 `gold_chunk_ids`（一串 UUID），列表接口那套 (版本, retrieval)
在评测侧**根本不存在**（`EvalRunCase` 里没有这两个 id）；带
`preview_chars` 拉过列表的调用方也需要一个按 id 回查全文的入口
（`DocumentChunkSummary.content_truncated` 的说明就是这么写的）。

为什么**不**顺带实现同一个 yaml 里的 GET /document-chunks（列表）
----------------------------------------------------------------
它需要 document_version_id / document_processor_definition_id /
chunk_index / keyword 四个过滤器的组合语义，而当前没有任何调用方
需要它 —— 版本页走的是上面那个按 retrieval 的列表接口。少实现一个
没人验过的过滤器组合，比多一个"看起来能用但没人跑过"的端点多。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import (
    APIRouter,
    Query,
)

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.document.application.document_chunk_service import (
    DocumentChunkService,
)
from app.services.document.domain.schemas.version_chunks import (
    DocumentChunkSummary,
)

router = APIRouter(
    tags=["Document Chunks"],
)


@router.get(
    "/document-chunks/{document_chunk_id}",
    response_model=DocumentChunkSummary,
)
async def get_document_chunk(
        document_chunk_id: UUID,
        preview_chars: int | None = Query(
            default=None,
            ge=1,
            le=100000,
            description=(
                "只返回 content 的前 N 个字符，并用 content_truncated 标记。"
                "留空返回**全文** —— 本端点的意义就是「看不到正文时按 id "
                "回查」，默认截断会让它失去意义。给「正文可能是整篇几十万"
                "字的不分块配置」留的防护由调用方显式开启。"
            ),
        ),
        service: DocumentChunkService = Service[DocumentChunkService],
):
    """
    按 `document_chunk_id` 取单条 chunk。

    不需要（也不接受）document_id / document_version_id /
    document_processor_definition_id —— 见契约里那句 "does not require
    ... to be supplied by the caller"。

    取不到返回 404，而不是空对象：id 不存在与"正文本来为空"必须分得开。
    """

    try:
        return await service.get_chunk(
            document_chunk_id,
            preview_chars=preview_chars,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
