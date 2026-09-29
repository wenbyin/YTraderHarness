"""
工具分类的taxonomy 管理接口。

对应 tool.yaml 的 tag：Classifications。

工具与分类的**关联**不在本文件 —— 它在 tools.py 的
/tools/{tool_id}/classifications 下，因为那个路径的主语是工具。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, status

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.tool.application.classification_service import (
    ClassificationService,
)
from app.services.tool.domain.schemas import (
    Classification,
    ClassificationCreate,
    ClassificationUpdate,
)

router = APIRouter()


@router.get(
    "/classifications",
    response_model=list[Classification],
    tags=["Classifications"],
)
async def list_classifications(
        parent_id: UUID | None = Query(
            default=None,
            description=(
                "按父分类过滤。不传表示不过滤（列出全部层级的节点），"
                "而不是只列一级分类。"
            ),
        ),
        keyword: str | None = Query(
            default=None,
            description="按 code 或名称模糊匹配。",
        ),
        is_active: bool | None = Query(
            default=None,
            description="按启用状态过滤。",
        ),
        service: ClassificationService = Service[
            ClassificationService
        ],
):
    """列出分类。结果是扁平的，层级关系看每项的 parent_id。"""

    try:
        return await service.list_classifications(
            parent_id=parent_id,
            keyword=keyword,
            is_active=is_active,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/classifications",
    response_model=Classification,
    status_code=status.HTTP_201_CREATED,
    tags=["Classifications"],
)
async def create_classification(
        request: ClassificationCreate,
        service: ClassificationService = Service[
            ClassificationService
        ],
):
    """新建一个分类节点。"""

    try:
        return await service.create_classification(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/classifications/{classification_id}",
    response_model=Classification,
    tags=["Classifications"],
)
async def get_classification(
        classification_id: UUID,
        service: ClassificationService = Service[
            ClassificationService
        ],
):
    """取单个分类。"""

    try:
        return await service.get_classification(
            classification_id
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.patch(
    "/classifications/{classification_id}",
    response_model=Classification,
    tags=["Classifications"],
)
async def update_classification(
        classification_id: UUID,
        request: ClassificationUpdate,
        service: ClassificationService = Service[
            ClassificationService
        ],
):
    """
    局部更新分类。

    code 不在可改字段里 —— 它是稳定标识，关联与外部引用都按它。
    只传要改的字段即可，没传的保持原值。
    """

    try:
        return await service.update_classification(
            classification_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
