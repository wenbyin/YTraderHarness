"""
工具定义管理接口。

对应 tool.yaml 的 tags：
    Tools                工具定义的查询
    Tool Documentation   说明文字的读写
    Tool Classifications 工具与能力的关联
    Discovery            发现与同步

⚠️ 能力（capability）本体的接口**不在这里** —— 它是能力树自己的
模块 app/services/capability/（路由也在那个模块里，对应
capability.yaml）。本文件只有"工具 ↔ 能力"的关联这两个端点：
路径的主语是工具，所以它们留在这里。

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
from app.services.capability.domain.schemas import Capability
from app.services.dependencies import Service
from app.services.tool.application.tool_classification_service import (
    ToolClassificationService,
)
from app.services.tool.application.tool_definition_state_service import (
    ToolDefinitionStateService,
)
from app.services.tool.application.tool_discovery_service import (
    ToolDiscoveryService,
)
from app.services.tool.application.tool_documentation_service import (
    ToolDocumentationService,
)
from app.services.tool.application.tool_presentation_service import (
    ToolPresentationService,
)
from app.services.tool.application.tool_query_service import (
    ToolQueryService,
)
from app.services.tool.domain.schemas import (
    PagedToolSummaryResponse,
    SourceState,
    SourceType,
    ToolClassificationUpdate,
    ToolDefinitionDetail,
    ToolDocumentation,
    ToolDocumentationUpdate,
    ToolEnabledState,
    ToolEnabledUpdate,
    ToolPresentation,
    ToolPresentationState,
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
        source_ref: str | None = Query(
            default=None,
            description=(
                "来源对象过滤：MCP 场景下是 provider 名"
                "（见 GET /mcp/providers）。"
            ),
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
            source_ref=source_ref,
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
# Tool Runtime State
# ============================================================


@router.patch(
    "/tools/{tool_id}/enabled",
    response_model=ToolEnabledState,
    tags=["Tools"],
)
async def set_tool_enabled(
        tool_id: UUID,
        request: ToolEnabledUpdate,
        service: ToolDefinitionStateService = Service[
            ToolDefinitionStateService
        ],
):
    """
    显式禁用 / 恢复启用一个工具。

    **禁用是对下一轮生效**：该工具不再进入下一次工具解析（主 agent /
    planner / skill 共用同一条路）；正在跑的这一轮不受影响。

    持久层语义是"有行 = 禁用"：enabled=false 写入（或刷新）一行，
    enabled=true 删除那一行。同步不会碰这张表，所以代码变更不会
    把开关冲掉 —— 只有这个接口能改。
    """

    try:
        return await service.set_enabled(tool_id, request.enabled)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# Tool Presentation
# ============================================================


@router.put(
    "/tools/{tool_id}/presentation",
    response_model=ToolPresentationState,
    tags=["Tools"],
)
async def set_tool_presentation(
        tool_id: UUID,
        request: ToolPresentation,
        service: ToolPresentationService = Service[
            ToolPresentationService
        ],
):
    """
    设置工具的展示描述符（整份替换）。

    **它只决定「显示什么」**：这条工具调用出不出现在事件流、以什么
    形态出现、带哪些字段。不决定模型能不能调用（那是
    PATCH /tools/{tool_id}/enabled），也不控制时机 / 合并 / 消失
    （那些属于客户端）。

    持久层语义是"有行 = 显示"：写入即开启。没有 presenter 行时，
    该工具的 tool.called / tool.completed 一条都不发，名字也不进
    message.completed.tool_calls —— 默认不显示。

    关闭展示走 DELETE，而不是写入某个 hidden 取值。
    """

    try:
        return await service.set_presentation(tool_id, request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.delete(
    "/tools/{tool_id}/presentation",
    response_model=ToolPresentationState,
    tags=["Tools"],
)
async def clear_tool_presentation(
        tool_id: UUID,
        service: ToolPresentationService = Service[
            ToolPresentationService
        ],
):
    """
    删除展示描述符，回到默认隐藏。

    工具**仍然可以**被模型调用 —— 「不显示」不等于「下线」。
    没有 presenter 时也成功（幂等）。
    """

    try:
        return await service.clear_presentation(tool_id)

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
    response_model=list[Capability],
    tags=["Tool Classifications"],
)
async def get_tool_classifications(
        tool_id: UUID,
        service: ToolClassificationService = Service[
            ToolClassificationService
        ],
):
    """
    取某个工具已关联的能力。

    返回的是能力对象（Capability），定义见 capability.yaml ——
    但路径与名字保持 classifications：主语是工具，这一侧的术语不跟着
    能力改名走。
    """

    try:
        return await service.get_tool_classifications(tool_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/tools/{tool_id}/classifications",
    response_model=list[Capability],
    tags=["Tool Classifications"],
)
async def replace_tool_classifications(
        tool_id: UUID,
        request: ToolClassificationUpdate,
        service: ToolClassificationService = Service[
            ToolClassificationService
        ],
):
    """
    整体替换某个工具的能力关联。

    **替换**语义：没列出的能力会被解除关联。传空数组即全部解除。
    """

    try:
        return await service.replace_tool_classifications(
            tool_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
