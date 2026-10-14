"""
业务能力树（taxonomy）的管理接口。

对应 capability.yaml 的 tag：Capabilities。规格与实现都在
app/services/capability/ 下，本文件只做 HTTP 层的转发 —— 与
tools.py / documents.py 等其余路由同一套分层。

工具与能力的**关联**不在本文件 —— 它在 tools.py 的
/tools/{tool_id}/classifications 下，因为那个路径的主语是工具。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, status

from app.common.utils import to_http_error
from app.services.capability.application.capability_service import (
    CapabilityService,
)
from app.services.capability.domain.schemas import (
    Capability,
    CapabilityCreate,
    CapabilityReplace,
    CapabilityUpdate,
)
from app.services.dependencies import Service

router = APIRouter()


@router.get(
    "/capabilities",
    response_model=list[Capability],
    tags=["Capabilities"],
)
async def list_capabilities(
        parent_id: UUID | None = Query(
            default=None,
            description=(
                "按父能力过滤。不传表示不过滤（列出全部层级的节点），"
                "而不是只列一级能力。"
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
        service: CapabilityService = Service[CapabilityService],
):
    """列出能力。结果是扁平的，层级关系看每项的 parent_id。"""

    try:
        return await service.list_capabilities(
            parent_id=parent_id,
            keyword=keyword,
            is_active=is_active,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/capabilities",
    response_model=Capability,
    status_code=status.HTTP_201_CREATED,
    tags=["Capabilities"],
)
async def create_capability(
        request: CapabilityCreate,
        service: CapabilityService = Service[CapabilityService],
):
    """新建一个能力节点。"""

    try:
        return await service.create_capability(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/capabilities/{capability_id}",
    response_model=Capability,
    tags=["Capabilities"],
)
async def get_capability(
        capability_id: UUID,
        service: CapabilityService = Service[CapabilityService],
):
    """取单个能力。"""

    try:
        return await service.get_capability(capability_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.patch(
    "/capabilities/{capability_id}",
    response_model=Capability,
    tags=["Capabilities"],
)
async def update_capability(
        capability_id: UUID,
        request: CapabilityUpdate,
        service: CapabilityService = Service[CapabilityService],
):
    """
    局部更新能力。

    code 不在可改字段里 —— 它是稳定标识，关联与外部引用都按它。
    只传要改的字段即可，没传的保持原值。
    """

    try:
        return await service.update_capability(
            capability_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/capabilities/{capability_id}",
    response_model=Capability,
    tags=["Capabilities"],
)
async def replace_capability(
        capability_id: UUID,
        request: CapabilityReplace,
        service: CapabilityService = Service[CapabilityService],
):
    """
    整体替换能力（与 PATCH 的局部更新相对）。

    区别在语义：没传的 description / parent_id 按 null 处理
    （清空说明、提升为一级能力），而不是保持原值。

    sort_order 与 is_active 必填 —— 一次"只想改个名字"的 PUT
    不该顺手重置排序、或把停用的节点重新启用。

    code 同样不在可改字段里，理由与 PATCH 相同。
    """

    try:
        return await service.replace_capability(
            capability_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.delete(
    "/capabilities/{capability_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Capabilities"],
)
async def delete_capability(
        capability_id: UUID,
        force: bool = Query(
            default=False,
            description=(
                "能力仍被工具引用时，带 true 表示确认删除，"
                "关联会一并解除。不能用来跳过子能力检查 —— "
                "有子能力时无条件 409。"
            ),
        ),
        service: CapabilityService = Service[CapabilityService],
):
    """
    删除能力。关联到此能力的 Tool 会变成未关联（库里 CASCADE）。

    两关，只放行第二关：
        1. 还有子能力        -> 409，force 也解不开。先删或挪走它们。
        2. 还被工具引用      -> 409，错误里给出引用数；
                              确认后带 force=true 重试。

    常规做法其实是停用（PATCH is_active=false）：数据留着，
    历史关联不断，只是不再被用于筛选。
    """

    try:
        await service.delete_capability(
            capability_id, force=force
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
