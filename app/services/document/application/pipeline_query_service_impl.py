"""
PipelineQueryService 的默认实现。

事务边界由调用方（dependencies）持有的 session 决定，
本服务内部不 commit / rollback。
"""

from __future__ import annotations

import math
from datetime import datetime
from uuid import UUID

from app.common.exceptions import NotFoundError
from app.infra.graph import GraphRegistry

from app.common.service.base import ServiceContext

from ..domain.entities.pipeline import (
    PipelineNodeRun,
    PipelineRun,
)
from ..domain.schemas.observability import (
    PipelineNodeCatalogEntry,
    PipelineNodeCatalogResponse,
    PipelineNodeRunListResponse,
    PipelineNodeRunResponse,
    PipelineRunDetailResponse,
    PipelineRunListResponse,
    PipelineRunResponse,
)
from ..repository.pipeline_run import PipelineRunRepository
from .pipeline_query_service import PipelineQueryService

__all__ = ["PipelineQueryServiceImpl"]


class PipelineQueryServiceImpl(PipelineQueryService):

    def __init__(
            self,
            *,
            run_repository: PipelineRunRepository,
            graph_registry: GraphRegistry,
            graph_name: str,
            ctx: ServiceContext | None = None,
    ) -> None:
        """
        graph_name 指明本服务查询的是哪个图的运行 ——
        注册表里可能有多个图，节点目录必须按图区分。
        """

        super().__init__(ctx)

        self._runs = run_repository
        self._registry = graph_registry
        self._graph_name = graph_name

    # ========================================================
    # Runs
    # ========================================================

    async def list_runs(
            self,
            *,
            document_id: UUID | None = None,
            document_version_id: UUID | None = None,
            status: str | None = None,
            trigger_source: str | None = None,
            triggered_by: UUID | None = None,
            started_from: datetime | None = None,
            started_to: datetime | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> PipelineRunListResponse:

        offset = (page - 1) * page_size

        runs, total = await self._runs.list_runs(
            document_id=document_id,
            document_version_id=document_version_id,
            status=status,
            trigger_source=trigger_source,
            triggered_by=triggered_by,
            started_from=started_from,
            started_to=started_to,
            offset=offset,
            limit=page_size,
        )

        return PipelineRunListResponse(
            items=[self._to_run_response(r) for r in runs],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._pages(total, page_size),
        )

    async def get_run(
            self,
            pipeline_run_id: UUID,
    ) -> PipelineRunDetailResponse:

        run = await self._runs.get(pipeline_run_id)

        if run is None:
            raise NotFoundError(
                f"PipelineRun not found: {pipeline_run_id}"
            )

        nodes = await self._runs.list_all_node_runs(
            pipeline_run_id
        )

        catalog_ids = self._current_node_ids()

        snapshot = run.topology_snapshot or {}
        snapshot_ids = {
            node.get("node_id")
            for node in snapshot.get("nodes", [])
        }

        # 该次运行用到的节点里，有不在当前目录中的 —— 说明图在
        # 这次运行之后演进过。前端据此展示"旧版流水线"提示，
        # 并依靠每行冻结的 node_label + sequence 兜底渲染。
        unknown = bool(snapshot_ids - catalog_ids)

        return PipelineRunDetailResponse(
            **self._to_run_response(run).model_dump(),
            graph_name=run.graph_name,
            graph_version=run.graph_version,
            topology_fingerprint=run.topology_fingerprint,
            topology_snapshot=snapshot,
            run_config=run.run_config,
            nodes=[self._to_node_response(n) for n in nodes],
            unknown_nodes=unknown,
        )

    async def list_node_runs(
            self,
            pipeline_run_id: UUID,
            *,
            page: int = 1,
            page_size: int = 200,
    ) -> PipelineNodeRunListResponse:

        run = await self._runs.get(pipeline_run_id)

        if run is None:
            raise NotFoundError(
                f"PipelineRun not found: {pipeline_run_id}"
            )

        offset = (page - 1) * page_size

        nodes, total = await self._runs.list_node_runs(
            pipeline_run_id,
            offset=offset,
            limit=page_size,
        )

        return PipelineNodeRunListResponse(
            items=[self._to_node_response(n) for n in nodes],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._pages(total, page_size),
        )

    # ========================================================
    # Catalog
    # ========================================================

    async def get_node_catalog(
            self,
    ) -> PipelineNodeCatalogResponse:

        definition = self._registry.get(self._graph_name)

        return PipelineNodeCatalogResponse(
            graph_name=definition.name,
            graph_version=definition.version,
            topology_fingerprint=(
                definition.topology_fingerprint
            ),
            nodes=[
                PipelineNodeCatalogEntry(
                    node_id=spec.node_id,
                    label=spec.label,
                    group=spec.group,
                    sequence=spec.sequence,
                    description=spec.description,
                    deprecated=spec.deprecated,
                    replaced_by=spec.replaced_by,
                )
                for spec in definition.catalog
            ],
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _current_node_ids(self) -> set[str]:
        definition = self._registry.get_optional(
            self._graph_name
        )
        if definition is None:
            return set()
        return set(definition.node_ids)

    @staticmethod
    def _pages(total: int, page_size: int) -> int:
        if page_size <= 0:
            return 0
        return max(math.ceil(total / page_size), 1) if total else 0

    @staticmethod
    def _to_run_response(run: PipelineRun) -> PipelineRunResponse:
        return PipelineRunResponse(
            pipeline_run_id=run.pipeline_run_id,
            document_id=run.document_id,
            document_version_id=run.document_version_id,
            document_type_definition_id=(
                run.document_type_definition_id
            ),
            trigger_source=run.trigger_source,
            triggered_by=run.triggered_by,
            parent_pipeline_run_id=(
                run.parent_pipeline_run_id
            ),
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_ms=run.duration_ms,
            resumed_from_checkpoint_id=(
                run.resumed_from_checkpoint_id
            ),
            error_type=run.error_type,
            error_node_id=run.error_node_id,
            error_message=run.error_message,
            node_count=run.node_count,
            failed_node_count=run.failed_node_count,
            created_at=run.created_at,
        )

    @staticmethod
    def _to_node_response(
            node: PipelineNodeRun,
    ) -> PipelineNodeRunResponse:
        return PipelineNodeRunResponse(
            pipeline_node_run_id=node.pipeline_node_run_id,
            pipeline_run_id=node.pipeline_run_id,
            node_id=node.node_id,
            node_label=node.node_label,
            node_group=node.node_group,
            sequence=node.sequence,
            document_retrieval_definition_id=(
                node.document_retrieval_definition_id
            ),
            branch_key=node.branch_key,
            attempt=node.attempt,
            status=node.status,
            started_at=node.started_at,
            finished_at=node.finished_at,
            duration_ms=node.duration_ms,
            input_summary=node.input_summary,
            output_summary=node.output_summary,
            metrics=node.metrics,
            error_type=node.error_type,
            error_message=node.error_message,
        )
