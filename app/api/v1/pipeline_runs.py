"""
流水线可观测性与重新处理接口。

前端消费面
----------
    GET  /documents/{id}/versions/{vid}/pipeline-runs   某版本的运行历史
    GET  /pipeline-runs/{rid}                           运行详情（含节点明细）
    GET  /pipeline-runs/{rid}/nodes                     节点行分页
    GET  /document-pipeline-nodes                       节点目录
    GET  /document-pipeline-nodes/{node_id}             单个节点
    POST /document-versions/{vid}/reprocess             重新处理

前端契约（必读）
----------------
1. **详情接口返回该次运行自己的 topology_snapshot**，
   按它渲染，不要用你本地写死的节点列表 ——
   这是历史运行在图演进的今天仍能正确展示的主机制。

2. **node_id 是稳定键**，展示名走 node_label（写入时冻结）。

3. **遇到不认识的 node_id 不要丢弃该行**：用 node_label
   渲染通用节点，按 sequence 排序。详情里的 unknown_nodes
   为 true 时提示「此运行使用旧版流水线」。

4. **不要假设 sequence 连续或从 0 开始**，它是冻结的历史值。

5. 扇出阶段的节点行数不是固定的：每个参与本次运行的
   RetrievalDefinition 各产生一行，靠 branch_key 区分。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import (
    APIRouter,
    Query,
    status,
)

from app.common.exceptions import NotFoundError
from app.common.utils import to_http_error

from app.services.document.application.document_service import (
    DocumentRuntimeService,
)
from app.services.document.application.pipeline_query_service import (
    PipelineQueryService,
)
from app.services.dependencies import Service
from app.services.document.domain.schemas.observability import (
    PipelineNodeCatalogEntry,
    PipelineNodeCatalogResponse,
    PipelineNodeRunListResponse,
    PipelineRunDetailResponse,
    PipelineRunListResponse,
    ReprocessDocumentVersionRequest,
    ReprocessDocumentVersionResponse,
)

router = APIRouter(
    tags=["Document Pipeline"],
)


# ============================================================
# Reprocess
# ============================================================


@router.post(
    "/document-versions/{document_version_id}/reprocess",
    response_model=ReprocessDocumentVersionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_document_version(
        document_version_id: UUID,
        request: ReprocessDocumentVersionRequest,
        service: DocumentRuntimeService = Service[DocumentRuntimeService],
):
    """
    重新处理某个 DocumentVersion（HTTP 202）。

    document_processor_definition_ids：
        留空 -> 按现有规则重跑全部适用的 retrieval
        非空 -> 只重跑这些 processor 映射到的 retrieval

    force：
        true 时绕过"已完成即跳过"的短路。

    返回的 job_id 就是本次运行的 pipeline_run_id，
    拿它去 GET /pipeline-runs/{id} 查进度。
    """

    try:
        return await service.reprocess_version(
            document_version_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# Runs
# ============================================================


@router.get(
    "/documents/{document_id}/versions/{document_version_id}"
    "/pipeline-runs",
    response_model=PipelineRunListResponse,
)
async def list_version_pipeline_runs(
        document_id: UUID,
        document_version_id: UUID,
        status_: str | None = Query(
            default=None, alias="status",
        ),
        trigger_source: str | None = Query(default=None),
        started_from: datetime | None = Query(default=None),
        started_to: datetime | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        service: PipelineQueryService = Service[PipelineQueryService],
):
    """
    某个版本的运行历史，按开始时间倒序。

    reprocess 会产生多条记录，靠 parent_pipeline_run_id
    串成链。
    """

    try:
        return await service.list_runs(
            document_id=document_id,
            document_version_id=document_version_id,
            status=status_,
            trigger_source=trigger_source,
            started_from=started_from,
            started_to=started_to,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/pipeline-runs/{pipeline_run_id}",
    response_model=PipelineRunDetailResponse,
)
async def get_pipeline_run(
        pipeline_run_id: UUID,
        service: PipelineQueryService = Service[PipelineQueryService],
):
    """
    运行详情：run 行 + 全部节点行 + 该次运行的拓扑快照。

    unknown_nodes 为 true 表示本次运行用了当前目录里
    没有的节点（图已演进），前端应提示并靠每行的
    node_label + sequence 兜底渲染。
    """

    try:
        return await service.get_run(pipeline_run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/pipeline-runs/{pipeline_run_id}/nodes",
    response_model=PipelineNodeRunListResponse,
)
async def list_pipeline_node_runs(
        pipeline_run_id: UUID,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=200, ge=1, le=500),
        service: PipelineQueryService = Service[PipelineQueryService],
):
    """
    节点行分页。

    扇出阶段的行数 = 参与的 RetrievalDefinition 数，
    retrieval 很多时详情接口可能被截断，用本接口翻页。
    """

    try:
        return await service.list_node_runs(
            pipeline_run_id,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# Node catalog
# ============================================================


@router.get(
    "/document-pipeline-nodes",
    response_model=PipelineNodeCatalogResponse,
)
async def get_pipeline_node_catalog(
        service: PipelineQueryService = Service[PipelineQueryService],
):
    """
    当前图版本的节点目录。

    node_id 是契约键。退役节点永久保留并标 deprecated，
    因此历史运行里的旧 node_id 永远能在这里查到解释。
    """

    return await service.get_node_catalog()


@router.get(
    "/document-pipeline-nodes/{node_id}",
    response_model=PipelineNodeCatalogEntry,
)
async def get_pipeline_node(
        node_id: str,
        service: PipelineQueryService = Service[PipelineQueryService],
):
    """
    单个节点。

    给前端解析一个不认识的 node_id 用 —— 不必为了
    一个节点拉整个目录。
    """

    catalog = await service.get_node_catalog()

    for node in catalog.nodes:
        if node.node_id == node_id:
            return node

    raise to_http_error(
        NotFoundError(
            f"Unknown pipeline node: {node_id!r}. "
            f"Graph version {catalog.graph_version}."
        )
    )
