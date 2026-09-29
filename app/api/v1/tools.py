"""
工具定义管理接口。

对应 tool.yaml 的 tags：
    Tools                工具定义的查询
    Tool Documentation   说明文字的读写
    Tool Classifications 工具与分类的关联
    Discovery            发现与同步

⚠️ 路由顺序：/tools/discovery/sync 必须注册在 /tools/{tool_id} **之前**。
虽然一个是 POST、一个是 GET，看似不冲突，但保留这个顺序是因为
同类问题的成本极不对称 —— 万一以后有人给 /tools/{tool_id} 加上
POST，或者加一条 GET /tools/discovery，宽匹配就会把 "discovery"
当成一个非法 UUID 拿去解析，报错离原因很远。
document 模块的 /documents/retrieve vs /documents/{id} 踩过同一个坑。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.tool.application.classification_service import (
    ClassificationService,
)
from app.services.tool.application.tool_discovery_service import (
    ToolDiscoveryService,
)
from app.services.tool.application.tool_documentation_service import (
    ToolDocumentationService,
)
from app.services.tool.application.tool_query_service import (
    ToolQueryService,
)
from app.services.tool.domain.schemas import (
    Classification,
    PagedToolSummaryResponse,
    SourceState,
    SourceType,
    ToolClassificationUpdate,
    ToolDefinitionDetail,
    ToolDocumentation,
    ToolDocumentationUpdate,
    ToolSyncRequest,
    ToolSyncResult,
)

router = APIRouter()


# ============================================================
# Discovery
# ============================================================
#
# 必须排在最前面，理由见模块 docstring。


@router.post(
    "/tools/discovery/sync",
    response_model=ToolSyncResult,
    tags=["Discovery"],
)
async def sync_tools(
        request: ToolSyncRequest | None = None,
        service: ToolDiscoveryService = Service[
            ToolDiscoveryService
        ],
):
    """
    扫描代码里的工具声明，与库里的记录对齐。

    幂等：同一份代码连跑两次，第二次一定全是 unchanged。
    如果第二次仍报 updated，说明摘要不稳定，正在无谓地重写整张表。
    """

    try:
        return await service.sync(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# Tools
# ============================================================


@router.get(
    "/tools",
    response_model=PagedToolSummaryResponse,
    tags=["Tools"],
)
async def list_tools(
        keyword: str | None = Query(
            default=None,
            description="按 tool_key 或展示名模糊匹配。",
        ),
        source_type: SourceType | None = Query(
            default=None,
            description="来源种类过滤。",
        ),
        classification_code: str | None = Query(
            default=None,
            description="按分类的稳定标识过滤。",
        ),
        source_state: SourceState | None = Query(
            default=None,
            description="ACTIVE=代码里还在；MISSING=代码里已找不到。",
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=200),
        service: ToolQueryService = Service[ToolQueryService],
):
    """
    分页列出工具定义。

    返回的 description 是**生效值**（人工覆盖优先），
    与详情页一致 —— 否则列表和详情会显示两套说法。
    """

    try:
        return await service.list_tools(
            keyword=keyword,
            source_type=source_type,
            classification_code=classification_code,
            source_state=source_state,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/tools/{tool_id}",
    response_model=ToolDefinitionDetail,
    tags=["Tools"],
)
async def get_tool(
        tool_id: UUID,
        service: ToolQueryService = Service[ToolQueryService],
):
    """取一个工具的全部信息：代码侧事实 + 人工说明 + 分类。"""

    try:
        return await service.get_tool(tool_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# Tool Documentation
# ============================================================


@router.get(
    "/tools/{tool_id}/documentation",
    response_model=ToolDocumentation,
    tags=["Tool Documentation"],
)
async def get_tool_documentation(
        tool_id: UUID,
        service: ToolDocumentationService = Service[
            ToolDocumentationService
        ],
):
    """
    取说明：代码来源 + 人工覆盖 + 生效值。

    三者都给出来，前端才能标出"这条被人工改过"。
    """

    try:
        return await service.get_documentation(tool_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/tools/{tool_id}/documentation",
    response_model=ToolDocumentation,
    tags=["Tool Documentation"],
)
async def replace_tool_documentation(
        tool_id: UUID,
        request: ToolDocumentationUpdate,
        service: ToolDocumentationService = Service[
            ToolDocumentationService
        ],
):
    """
    整体替换说明覆盖。

    **替换**语义：body 里没列出的参数，其覆盖会被清掉
    （回到代码里的原始说明）。前端可以把整个表单直接提交上来。

    写在这里的内容不会被同步覆盖 —— 同步只碰代码侧的 definition。
    """

    try:
        return await service.replace_documentation(
            tool_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# Tool Classifications
# ============================================================


@router.get(
    "/tools/{tool_id}/classifications",
    response_model=list[Classification],
    tags=["Tool Classifications"],
)
async def get_tool_classifications(
        tool_id: UUID,
        service: ClassificationService = Service[
            ClassificationService
        ],
):
    """取某个工具已关联的分类。"""

    try:
        return await service.get_tool_classifications(tool_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/tools/{tool_id}/classifications",
    response_model=list[Classification],
    tags=["Tool Classifications"],
)
async def replace_tool_classifications(
        tool_id: UUID,
        request: ToolClassificationUpdate,
        service: ClassificationService = Service[
            ClassificationService
        ],
):
    """
    整体替换某个工具的分类关联。

    **替换**语义：没列出的分类会被解除关联。传空数组即全部解除。
    """

    try:
        return await service.replace_tool_classifications(
            tool_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
