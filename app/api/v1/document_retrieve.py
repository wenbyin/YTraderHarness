"""
业务级文档检索接口。

规格见 app/services/document/document_retrieve.yaml。

一次调用 = 一轮检索
-------------------
本接口不做查询改写、扩召回、重试或 agent loop。调用方（通常是
外部 LLM）拿结果后自己决定要不要再调一次 —— 这正是它作为
agentic RAG 工具组件的用法。

路由顺序（重要）
----------------
本路由注册在 documents_router **之前**，见 router.py。
/documents/retrieve 与 documents 里的宽匹配 /documents/{document_id}
前缀相同，且后者的路径参数是 UUID —— 先注册字面量路径，
避免 "retrieve" 被当成一个非法 UUID 去解析。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.common.utils import to_http_error

from app.services.dependencies import Service
from app.services.document.application.document_retrieve_service import (
    DocumentRetrieveService,
)
from app.services.document.domain.schemas import (
    DocumentRetrieveRequest,
    DocumentRetrieveResponse,
)

router = APIRouter(
    tags=["Document Retrieve"],
)


@router.post(
    "/documents/retrieve",
    response_model=DocumentRetrieveResponse,
)
async def retrieve_documents(
        request: DocumentRetrieveRequest,
        service: DocumentRetrieveService = Service[DocumentRetrieveService],
) -> DocumentRetrieveResponse:

    try:
        return await service.retrieve_documents(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc
