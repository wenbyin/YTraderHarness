"""
PostgreSQL implementation of PipelineRunRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.common.utils import to_uuid, to_uuid_or_none
from app.infra.graph.recorder import (
    NodeRunStatus,
    NodeRunUpdate,
    RunOutcome,
    RunSnapshot,
    RunStatus,
    RunSummary,
)

from ..domain.entities.pipeline import (
    PipelineNodeRun,
    PipelineRun,
)
from ..models.sqlalchemy.document_pipeline_node_run import (
    DocumentPipelineNodeRun as NodeRunModel,
)
from ..models.sqlalchemy.document_pipeline_run import (
    DocumentPipelineRun as RunModel,
)
from .pipeline_run import PipelineRunRepository

_ORPHAN_ERROR_TYPE = "Interrupted"

def _uuid_from(ref: dict, key: str) -> UUID:
    """
    从通用 subject_ref 里取出一个必填 UUID。

    用 KeyError 式的显式失败而不是静默填 None ——
    主体信息缺失会让运行记录挂到一个错误的版本上，
    比直接报错难查得多。
    """

    value = ref.get(key)

    if value is None:
        raise ValueError(
            f"RunSnapshot.subject_ref is missing required key "
            f"{key!r}. Got keys: {sorted(ref)}."
        )

    return to_uuid(value)

def _uuid_or_none(ref: dict, key: str) -> UUID | None:
    return to_uuid_or_none(ref.get(key))

class PipelineRunRepositoryImpl(PipelineRunRepository):
    """
    节点行的 upsert 依赖表上的
    uq_document_pipeline_node_run_identity
        (pipeline_run_id, node_id, branch_key, attempt)
    约束。
    """

    # ========================================================
    # Run lifecycle
    # ========================================================

    #: 本表的主体类型。infra 的 RunSnapshot 用 subject_type 表达，
    #: 本仓储把它落成 document 语义的具体列。
    SUBJECT_TYPE = "document_version"

    async def add(
            self,
            snapshot: RunSnapshot,
    ) -> PipelineRun:
        """
        把通用的 RunSnapshot 落成 document 语义的行。

        这是 infra 与 document 之间的**唯一映射缝**：

            infra (通用)              document (具体)
            subject_type          ->  （断言为 document_version）
            subject_id            ->   document_version_id
            subject_ref[...]      ->   document_id /
                                       document_type_definition_id /
                                       file_id / tenant_id

        表结构保持 document 语义的精确列名是刻意的 ——
        FK 级联、按版本查历史的索引、前端的筛选参数都依赖它。
        通用化只发生在 infra 的类型层，不往库里渗。
        """

        if snapshot.subject_type != self.SUBJECT_TYPE:
            raise ValueError(
                "This repository only persists "
                f"{self.SUBJECT_TYPE!r} runs, got "
                f"{snapshot.subject_type!r}."
            )

        ref = snapshot.subject_ref

        model = RunModel(
            pipeline_run_id=snapshot.pipeline_run_id,
            thread_id=snapshot.thread_id,
            graph_name=snapshot.graph_name,
            graph_version=snapshot.graph_version,
            topology_fingerprint=snapshot.topology_fingerprint,
            topology_snapshot=dict(snapshot.topology_snapshot),
            trigger_source=snapshot.trigger_source,
            triggered_by=snapshot.triggered_by,
            parent_pipeline_run_id=(
                snapshot.parent_pipeline_run_id
            ),
            root_pipeline_run_id=(
                snapshot.root_pipeline_run_id
            ),
            document_id=_uuid_from(ref, "document_id"),
            document_version_id=snapshot.subject_id,
            document_type_definition_id=_uuid_from(
                ref, "document_type_definition_id"
            ),
            file_id=_uuid_or_none(ref, "file_id"),
            tenant_id=_uuid_or_none(ref, "tenant_id"),
            run_config=dict(snapshot.run_config),
            status=RunStatus.PENDING,
            resumed_from_checkpoint_id=(
                snapshot.resumed_from_checkpoint_id
            ),
            node_count=0,
            failed_node_count=0,
        )

        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)

        return self._to_run(model)

    async def get(
            self,
            pipeline_run_id: UUID,
    ) -> PipelineRun | None:

        model = (
            await self._session.execute(
                select(RunModel).where(
                    RunModel.pipeline_run_id == pipeline_run_id
                )
            )
        ).scalar_one_or_none()

        return None if model is None else self._to_run(model)

    async def get_by_thread_id(
            self,
            thread_id: str,
    ) -> PipelineRun | None:

        model = (
            await self._session.execute(
                select(RunModel).where(
                    RunModel.thread_id == thread_id
                )
            )
        ).scalar_one_or_none()

        return None if model is None else self._to_run(model)

    async def mark_running(
            self,
            pipeline_run_id: UUID,
            *,
            started_at: datetime,
    ) -> None:

        await self._session.execute(
            update(RunModel)
            .where(RunModel.pipeline_run_id == pipeline_run_id)
            .values(
                status=RunStatus.RUNNING,
                started_at=started_at,
            )
        )
        await self._session.flush()

    async def finish(
            self,
            pipeline_run_id: UUID,
            outcome: RunOutcome,
    ) -> None:

        counts = await self._session.execute(
            select(
                func.count(NodeRunModel.pipeline_node_run_id),
                func.count(NodeRunModel.pipeline_node_run_id).filter(
                    NodeRunModel.status == NodeRunStatus.FAILED
                ),
            ).where(
                NodeRunModel.pipeline_run_id == pipeline_run_id
            )
        )

        total, failed = counts.one()

        await self._session.execute(
            update(RunModel)
            .where(RunModel.pipeline_run_id == pipeline_run_id)
            .values(
                status=outcome.status,
                finished_at=outcome.finished_at,
                duration_ms=outcome.duration_ms,
                error_type=outcome.error_type,
                error_node_id=outcome.error_node_id,
                error_message=(
                    None
                    if outcome.error_message is None
                    else outcome.error_message[:4000]
                ),
                node_count=int(total or 0),
                failed_node_count=int(failed or 0),
            )
        )
        await self._session.flush()

    # ========================================================
    # Node runs
    # ========================================================

    async def upsert_node_run(
            self,
            update_: NodeRunUpdate,
    ) -> None:

        stmt = insert(NodeRunModel).values(
            pipeline_run_id=update_.pipeline_run_id,
            node_id=update_.node_id,
            node_label=update_.node_label,
            node_group=update_.node_group,
            sequence=update_.sequence,
            # infra 用通用的 branch_scope_id；本表是 document 的表，
            # 列名保持精确语义（含指向 retrieval definition 的 FK）。
            document_retrieval_definition_id=(
                update_.branch_scope_id
            ),
            branch_key=update_.branch_key,
            attempt=update_.attempt,
            langgraph_task_id=update_.langgraph_task_id,
            langgraph_run_id=update_.langgraph_run_id,
            parent_run_id=update_.parent_run_id,
            status=update_.status,
            started_at=update_.started_at,
            finished_at=update_.finished_at,
            duration_ms=update_.duration_ms,
            input_summary=update_.input_summary,
            output_summary=update_.output_summary,
            metrics=update_.metrics,
            error_type=update_.error_type,
            error_message=(
                None
                if update_.error_message is None
                else update_.error_message[:4000]
            ),
            error_detail=update_.error_detail,
        )

        # 只有非空值才覆盖既有列：node_span 先写 RUNNING（此时
        # 还没有 metrics / output_summary），随后写 COMPLETED 才带上。
        # 用 COALESCE 语义，避免后写的事件把先写的数据抹成 NULL。
        stmt = stmt.on_conflict_do_update(
            constraint="uq_document_pipeline_node_run_identity",
            set_={
                "node_label": stmt.excluded.node_label,
                "node_group": stmt.excluded.node_group,
                "sequence": stmt.excluded.sequence,
                "status": stmt.excluded.status,
                "langgraph_task_id": func.coalesce(
                    stmt.excluded.langgraph_task_id,
                    NodeRunModel.langgraph_task_id,
                ),
                "langgraph_run_id": func.coalesce(
                    stmt.excluded.langgraph_run_id,
                    NodeRunModel.langgraph_run_id,
                ),
                "started_at": func.coalesce(
                    stmt.excluded.started_at,
                    NodeRunModel.started_at,
                ),
                "finished_at": func.coalesce(
                    stmt.excluded.finished_at,
                    NodeRunModel.finished_at,
                ),
                "duration_ms": func.coalesce(
                    stmt.excluded.duration_ms,
                    NodeRunModel.duration_ms,
                ),
                "input_summary": func.coalesce(
                    stmt.excluded.input_summary,
                    NodeRunModel.input_summary,
                ),
                "output_summary": func.coalesce(
                    stmt.excluded.output_summary,
                    NodeRunModel.output_summary,
                ),
                "metrics": func.coalesce(
                    stmt.excluded.metrics,
                    NodeRunModel.metrics,
                ),
                "error_type": func.coalesce(
                    stmt.excluded.error_type,
                    NodeRunModel.error_type,
                ),
                "error_message": func.coalesce(
                    stmt.excluded.error_message,
                    NodeRunModel.error_message,
                ),
                "error_detail": func.coalesce(
                    stmt.excluded.error_detail,
                    NodeRunModel.error_detail,
                ),
            },
        )

        await self._session.execute(stmt)
        await self._session.flush()

    async def list_node_runs(
            self,
            pipeline_run_id: UUID,
            *,
            offset: int = 0,
            limit: int = 200,
    ) -> tuple[list[PipelineNodeRun], int]:

        total = (
            await self._session.execute(
                select(
                    func.count(NodeRunModel.pipeline_node_run_id)
                ).where(
                    NodeRunModel.pipeline_run_id
                    == pipeline_run_id
                )
            )
        ).scalar_one()

        models = (
            await self._session.execute(
                select(NodeRunModel)
                .where(
                    NodeRunModel.pipeline_run_id
                    == pipeline_run_id
                )
                .order_by(
                    NodeRunModel.sequence.asc(),
                    NodeRunModel.started_at.asc().nulls_last(),
                    NodeRunModel.node_id.asc(),
                )
                .offset(offset)
                .limit(limit)
            )
        ).scalars().all()

        return [self._to_node_run(m) for m in models], int(total)

    async def list_all_node_runs(
            self,
            pipeline_run_id: UUID,
    ) -> list[PipelineNodeRun]:

        models = (
            await self._session.execute(
                select(NodeRunModel)
                .where(
                    NodeRunModel.pipeline_run_id
                    == pipeline_run_id
                )
                .order_by(
                    NodeRunModel.sequence.asc(),
                    NodeRunModel.started_at.asc().nulls_last(),
                    NodeRunModel.node_id.asc(),
                )
            )
        ).scalars().all()

        return [self._to_node_run(m) for m in models]

    async def summarize(
            self,
            pipeline_run_id: UUID,
    ) -> RunSummary:

        stmt = (
            select(
                NodeRunModel.node_id,
                NodeRunModel.status,
            )
            .where(
                NodeRunModel.pipeline_run_id == pipeline_run_id
            )
            .order_by(
                NodeRunModel.sequence.asc(),
                NodeRunModel.node_id.asc(),
            )
        )

        rows = (await self._session.execute(stmt)).all()

        completed = 0
        failed_ids: list[str] = []

        for node_id, status in rows:
            if status == NodeRunStatus.COMPLETED:
                completed += 1
            elif status == NodeRunStatus.FAILED:
                # 同一节点可能有多个失败分支，去重后保留顺序
                if node_id not in failed_ids:
                    failed_ids.append(node_id)

        return RunSummary(
            node_count=len(rows),
            failed_node_count=sum(
                1
                for _, status in rows
                if status == NodeRunStatus.FAILED
            ),
            failed_node_ids=tuple(failed_ids),
            completed_node_count=completed,
        )

    # ========================================================
    # Queries
    # ========================================================

    async def list_runs(
            self,
            *,
            document_version_id: UUID | None = None,
            document_id: UUID | None = None,
            status: str | None = None,
            trigger_source: str | None = None,
            triggered_by: UUID | None = None,
            started_from: datetime | None = None,
            started_to: datetime | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[PipelineRun], int]:

        conditions = []

        if document_version_id is not None:
            conditions.append(
                RunModel.document_version_id == document_version_id
            )

        if document_id is not None:
            conditions.append(
                RunModel.document_id == document_id
            )

        if status is not None:
            conditions.append(RunModel.status == status)

        if trigger_source is not None:
            conditions.append(
                RunModel.trigger_source == trigger_source
            )

        if triggered_by is not None:
            conditions.append(
                RunModel.triggered_by == triggered_by
            )

        if started_from is not None:
            conditions.append(RunModel.started_at >= started_from)

        if started_to is not None:
            conditions.append(RunModel.started_at <= started_to)

        total = (
            await self._session.execute(
                select(func.count(RunModel.pipeline_run_id))
                .where(*conditions)
            )
        ).scalar_one()

        models = (
            await self._session.execute(
                select(RunModel)
                .where(*conditions)
                .order_by(
                    RunModel.started_at.desc().nulls_last(),
                    RunModel.created_at.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
        ).scalars().all()

        return [self._to_run(m) for m in models], int(total)

    # ========================================================
    # Maintenance
    # ========================================================

    async def mark_orphans_failed(
            self,
            *,
            older_than: datetime,
    ) -> int:

        now = datetime.now(timezone.utc)

        stale_runs = (
            await self._session.execute(
                select(RunModel.pipeline_run_id).where(
                    RunModel.status.in_([
                        RunStatus.PENDING,
                        RunStatus.RUNNING,
                    ]),
                    RunModel.created_at < older_than,
                )
            )
        ).scalars().all()

        if not stale_runs:
            return 0

        run_ids = list(stale_runs)

        await self._session.execute(
            update(NodeRunModel)
            .where(
                NodeRunModel.pipeline_run_id.in_(run_ids),
                NodeRunModel.status.in_([
                    NodeRunStatus.PENDING,
                    NodeRunStatus.RUNNING,
                ]),
            )
            .values(
                status=NodeRunStatus.FAILED,
                finished_at=now,
                error_type=_ORPHAN_ERROR_TYPE,
                error_message=(
                    "Process terminated before this node "
                    "finished. Marked as failed on startup."
                ),
            )
        )

        result = await self._session.execute(
            update(RunModel)
            .where(RunModel.pipeline_run_id.in_(run_ids))
            .values(
                status=RunStatus.FAILED,
                finished_at=now,
                error_type=_ORPHAN_ERROR_TYPE,
                error_message=(
                    "Process terminated mid-run. Marked as failed "
                    "during startup reconciliation."
                ),
            )
        )

        await self._session.flush()

        return int(result.rowcount or 0)

    # ========================================================
    # Mapping
    # ========================================================

    @staticmethod
    def _to_run(model: RunModel) -> PipelineRun:

        return PipelineRun(
            pipeline_run_id=model.pipeline_run_id,
            thread_id=model.thread_id,
            langgraph_run_id=model.langgraph_run_id,
            graph_name=model.graph_name,
            graph_version=model.graph_version,
            topology_fingerprint=model.topology_fingerprint,
            topology_snapshot=dict(model.topology_snapshot or {}),
            trigger_source=model.trigger_source,
            triggered_by=model.triggered_by,
            parent_pipeline_run_id=(
                model.parent_pipeline_run_id
            ),
            root_pipeline_run_id=model.root_pipeline_run_id,
            # 出口是 **document 语义** —— PipelineRun 是 document 的
            # 领域实体，查询服务与 DTO 都按 document 字段消费。
            #
            # 通用化只发生在入口（add 收的是 infra 的 RunSnapshot），
            # 本仓储就是那条缝。不对称是刻意的：进来要通用，
            # 出去要给 document 用。
            document_id=model.document_id,
            document_version_id=model.document_version_id,
            document_type_definition_id=(
                model.document_type_definition_id
            ),
            file_id=model.file_id,
            tenant_id=model.tenant_id,
            run_config=dict(model.run_config or {}),
            status=model.status,
            started_at=model.started_at,
            finished_at=model.finished_at,
            duration_ms=model.duration_ms,
            resumed_from_checkpoint_id=(
                model.resumed_from_checkpoint_id
            ),
            error_type=model.error_type,
            error_node_id=model.error_node_id,
            error_message=model.error_message,
            node_count=model.node_count,
            failed_node_count=model.failed_node_count,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _to_node_run(
            model: NodeRunModel,
    ) -> PipelineNodeRun:

        return PipelineNodeRun(
            pipeline_node_run_id=model.pipeline_node_run_id,
            pipeline_run_id=model.pipeline_run_id,
            node_id=model.node_id,
            node_label=model.node_label,
            node_group=model.node_group,
            sequence=model.sequence,
            document_retrieval_definition_id=(
                model.document_retrieval_definition_id
            ),  # 本 entity 是 document 的，故保留精确名
            branch_key=model.branch_key,
            attempt=model.attempt,
            langgraph_task_id=model.langgraph_task_id,
            langgraph_run_id=model.langgraph_run_id,
            parent_run_id=model.parent_run_id,
            status=model.status,
            started_at=model.started_at,
            finished_at=model.finished_at,
            duration_ms=model.duration_ms,
            input_summary=model.input_summary,
            output_summary=model.output_summary,
            metrics=model.metrics,
            error_type=model.error_type,
            error_message=model.error_message,
            error_detail=model.error_detail,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
