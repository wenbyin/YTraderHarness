"""
PostgreSQL implementation of RetrievalEvalRunRepository.

Transaction rule
----------------
This repository NEVER calls session.commit() / session.rollback().
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select, tuple_
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import (
    RetrievalEvalCaseResult,
    RetrievalEvalRun,
    RetrievalEvalStageCandidate,
    RetrievalEvalStageObservation,
)
from ..domain.schemas.enum import EvalCaseResultStatus, EvalJobStatus
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalCase as RetrievalEvalCaseModel,
)
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalCaseResult as RetrievalEvalCaseResultModel,
)
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalRun as RetrievalEvalRunModel,
)
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalStageCandidate as RetrievalEvalStageCandidateModel,
)
from ..models.sqlalchemy.retrieval_eval import (
    RetrievalEvalStageObservation as RetrievalEvalStageObservationModel,
)
from .retrieval_eval_case_impl import _omit_unset_server_defaults
from .retrieval_eval_run import RetrievalEvalRunRepository

__all__ = ["RetrievalEvalRunRepositoryImpl"]


class RetrievalEvalRunRepositoryImpl(RetrievalEvalRunRepository):
    """PostgreSQL implementation of RetrievalEvalRunRepository."""

    # ========================================================
    # Run
    # ========================================================

    async def add_run(self, run: RetrievalEvalRun) -> RetrievalEvalRun:

        model = RetrievalEvalRunModel(
            eval_run_id=run.eval_run_id,
            bundle_id=run.bundle_id,
            status=run.status,
            max_k=run.max_k,
            query_processor_override=dict(run.query_processor_override),
            effective_config_snapshot=dict(run.effective_config_snapshot),
            include_stage_trace=run.include_stage_trace,
            total_cases=run.total_cases,
            completed_cases=run.completed_cases,
            failed_cases=run.failed_cases,
            metrics=dict(run.metrics),
            created_by=run.created_by,
            started_at=run.started_at,
            completed_at=run.completed_at,
            created_at=run.created_at,
        )

        self._session.add(model)
        await self._session.flush()

        return _to_run(model)

    async def get_run(self, eval_run_id: UUID) -> RetrievalEvalRun | None:

        stmt = select(RetrievalEvalRunModel).where(
            RetrievalEvalRunModel.eval_run_id == eval_run_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        return None if model is None else _to_run(model)

    async def update_run(
            self,
            run: RetrievalEvalRun,
    ) -> RetrievalEvalRun:

        stmt = select(RetrievalEvalRunModel).where(
            RetrievalEvalRunModel.eval_run_id == run.eval_run_id
        )

        model = (
            await self._session.execute(stmt)
        ).scalar_one_or_none()

        if model is None:
            raise ValueError(
                f"RetrievalEvalRun not found: {run.eval_run_id}"
            )

        model.status = run.status
        model.max_k = run.max_k
        model.query_processor_override = dict(run.query_processor_override)
        model.effective_config_snapshot = dict(
            run.effective_config_snapshot
        )
        model.include_stage_trace = run.include_stage_trace
        model.total_cases = run.total_cases
        model.completed_cases = run.completed_cases
        model.failed_cases = run.failed_cases
        model.metrics = dict(run.metrics)
        model.started_at = run.started_at
        model.completed_at = run.completed_at

        # `bundle_id` / `created_at` 不更新：它们是这次轮次的**身份**。
        # 而 `eval_run` 表没有 `updated_at` 列（只有 bundle 有）——
        # 所以这里不存在"onupdate 让属性 expired"那个坑。

        await self._session.flush()

        return _to_run(model)

    async def list_runs(
            self,
            *,
            bundle_id: UUID | None = None,
            status: EvalJobStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[RetrievalEvalRun], int]:

        conditions = []

        if bundle_id is not None:
            conditions.append(
                RetrievalEvalRunModel.bundle_id == bundle_id
            )

        if status is not None:
            conditions.append(
                RetrievalEvalRunModel.status == status.value
            )

        count_stmt = (
            select(func.count())
            .select_from(RetrievalEvalRunModel)
        )

        if conditions:
            count_stmt = count_stmt.where(*conditions)

        total = int(
            (await self._session.execute(count_stmt)).scalar_one()
        )

        page_stmt = select(RetrievalEvalRunModel)

        if conditions:
            page_stmt = page_stmt.where(*conditions)

        page_stmt = (
            page_stmt
            .order_by(
                RetrievalEvalRunModel.created_at.desc(),
                RetrievalEvalRunModel.eval_run_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return [_to_run(model) for model in models], total

    # ========================================================
    # Case result
    # ========================================================

    async def add_case_results(
            self,
            results: list[RetrievalEvalCaseResult],
    ) -> int:

        if not results:
            return 0

        rows = [
            {
                "eval_case_result_id": result.eval_case_result_id,
                "eval_run_id": result.eval_run_id,
                "eval_case_id": result.eval_case_id,
                "attempt_no": result.attempt_no,
                "query_id": result.query_id,
                "query_text": result.query_text,
                "gold_hit": result.gold_hit,
                "gold_rank": result.gold_rank,
                "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
                "latency_ms": result.latency_ms,
                "degraded_routes": list(result.degraded_routes),
                "status": result.status.value,
                "error_code": result.error_code,
                "error_message": result.error_message,
                "created_at": result.created_at,
            }
            for result in results
        ]

        rows = _omit_unset_server_defaults(rows)

        stmt = (
            insert(RetrievalEvalCaseResultModel)
            .values(rows)
            .on_conflict_do_nothing()
        )

        result = await self._session.execute(stmt)

        return result.rowcount or 0

    async def list_case_results(
            self,
            *,
            eval_run_id: UUID,
            attempt_no: int | None = None,
            offset: int = 0,
            limit: int = 50,
    ) -> tuple[list[RetrievalEvalCaseResult], int]:

        conditions = [
            RetrievalEvalCaseResultModel.eval_run_id == eval_run_id
        ]

        if attempt_no is not None:
            conditions.append(
                RetrievalEvalCaseResultModel.attempt_no == attempt_no
            )

        count_stmt = (
            select(func.count())
            .select_from(RetrievalEvalCaseResultModel)
            .where(*conditions)
        )

        total = int(
            (await self._session.execute(count_stmt)).scalar_one()
        )

        # 顺序按 **case 身份**（DTD、源 chunk、题号）而不是主键：结果行本身
        # 是按轮次分组的，按行序读会混着不同 DTD 的 case。这里要 join case
        # 表取那三列。
        #
        # ⚠️ 不提供 `hit` / DTD 过滤。逐轮放宽下一条 case 有多行，而「这条
        # case 命中了吗」只有**折叠之后**才答得出来 —— 在 SQL 里筛 `hit`
        # 会捞出一条 case 第 1 轮 miss 的行，与它第 2 轮命中的结论打架。
        # 筛选归调用方，在 `collapse_by_case` 之后做。
        page_stmt = (
            select(RetrievalEvalCaseResultModel)
            .join(
                RetrievalEvalCaseModel,
                RetrievalEvalCaseModel.eval_case_id
                == RetrievalEvalCaseResultModel.eval_case_id,
            )
            .where(*conditions)
            .order_by(
                RetrievalEvalCaseModel.document_type_definition_id,
                RetrievalEvalCaseModel.source_chunk_id,
                RetrievalEvalCaseModel.question_order,
                RetrievalEvalCaseResultModel.attempt_no,
            )
            .offset(offset)
            .limit(limit)
        )

        models = (
            await self._session.execute(page_stmt)
        ).scalars().all()

        return [_to_case_result(model) for model in models], total

    async def list_case_result_briefs(
            self,
            eval_run_id: UUID,
    ) -> list[RetrievalEvalCaseResult]:
        """
        本轮**全部**结果行，但只取标量列（不取重的那几列）。
        """

        stmt = select(
            RetrievalEvalCaseResultModel.eval_case_result_id,
            RetrievalEvalCaseResultModel.eval_run_id,
            RetrievalEvalCaseResultModel.eval_case_id,
            RetrievalEvalCaseResultModel.attempt_no,
            RetrievalEvalCaseResultModel.gold_hit,
            RetrievalEvalCaseResultModel.gold_rank,
            RetrievalEvalCaseResultModel.status,
            RetrievalEvalCaseResultModel.latency_ms,
        ).where(RetrievalEvalCaseResultModel.eval_run_id == eval_run_id)

        rows = (await self._session.execute(stmt)).all()

        return [
            RetrievalEvalCaseResult(
                eval_case_result_id=row.eval_case_result_id,
                eval_run_id=row.eval_run_id,
                eval_case_id=row.eval_case_id,
                attempt_no=row.attempt_no,
                gold_hit=row.gold_hit,
                gold_rank=row.gold_rank,
                status=EvalCaseResultStatus(row.status),
                latency_ms=row.latency_ms,
            )
            for row in rows
        ]

    async def list_results_by_identity(
            self,
            keys: list[tuple[UUID, int]],
    ) -> list[RetrievalEvalCaseResult]:
        """
        按 `(eval_case_id, attempt_no)` 精确取若干行（整行）。
        """

        if not keys:
            return []

        stmt = select(RetrievalEvalCaseResultModel).where(
            tuple_(
                RetrievalEvalCaseResultModel.eval_case_id,
                RetrievalEvalCaseResultModel.attempt_no,
            ).in_(keys)
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_case_result(model) for model in models]

    async def list_results_for_trace(
            self,
            eval_run_id: UUID,
            eval_case_id: UUID,
    ) -> list[RetrievalEvalCaseResult]:

        # 按 attempt_no 升序：调用方按轮次取（下一轮 / 代表行）都依赖这个
        # 顺序，而不是库的返回顺序。
        stmt = (
            select(RetrievalEvalCaseResultModel)
            .where(
                RetrievalEvalCaseResultModel.eval_run_id == eval_run_id,
                RetrievalEvalCaseResultModel.eval_case_id == eval_case_id,
            )
            .order_by(RetrievalEvalCaseResultModel.attempt_no)
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_case_result(model) for model in models]

    # ========================================================
    # 逐阶段
    # ========================================================

    async def add_stage_observations(
            self,
            observations: list[RetrievalEvalStageObservation],
    ) -> int:

        if not observations:
            return 0

        rows = [
            {
                "stage_observation_id": item.stage_observation_id,
                "eval_case_result_id": item.eval_case_result_id,
                "stage_key": item.stage_key,
                "source_key": item.source_key,
                "executed": item.executed,
                "input_count": item.input_count,
                "output_count": item.output_count,
                "gold_present_before": item.gold_present_before,
                "gold_present_after": item.gold_present_after,
                "gold_rank_before": item.gold_rank_before,
                "gold_rank_after": item.gold_rank_after,
                "gold_score_before": item.gold_score_before,
                "gold_score_after": item.gold_score_after,
                "gold_score_name": item.gold_score_name,
                "gold_score_direction": item.gold_score_direction,
                "drop_reason": item.drop_reason,
                "details": dict(item.details),
                "created_at": item.created_at,
            }
            for item in observations
        ]

        rows = _omit_unset_server_defaults(rows)

        stmt = (
            insert(RetrievalEvalStageObservationModel)
            .values(rows)
            .on_conflict_do_nothing()
        )

        result = await self._session.execute(stmt)

        return result.rowcount or 0

    async def add_stage_candidates(
            self,
            candidates: list[RetrievalEvalStageCandidate],
    ) -> int:

        if not candidates:
            return 0

        rows = [
            {
                "stage_candidate_id": item.stage_candidate_id,
                "eval_case_result_id": item.eval_case_result_id,
                "stage_key": item.stage_key,
                "source_key": item.source_key,
                "chunk_id": item.chunk_id,
                "rank_before": item.rank_before,
                "rank_after": item.rank_after,
                "score_before": item.score_before,
                "score_after": item.score_after,
                "score_name_before": item.score_name_before,
                "score_name_after": item.score_name_after,
                "score_direction_before": item.score_direction_before,
                "score_direction_after": item.score_direction_after,
                "selected": item.selected,
                "drop_reason": item.drop_reason,
                "metadata_": dict(item.metadata),
                "created_at": item.created_at,
            }
            for item in candidates
        ]

        rows = _omit_unset_server_defaults(rows)

        stmt = (
            insert(RetrievalEvalStageCandidateModel)
            .values(rows)
            .on_conflict_do_nothing()
        )

        result = await self._session.execute(stmt)

        return result.rowcount or 0

    async def list_stage_observations(
            self,
            eval_case_result_id: UUID,
    ) -> list[RetrievalEvalStageObservation]:

        stmt = (
            select(RetrievalEvalStageObservationModel)
            .where(
                RetrievalEvalStageObservationModel.eval_case_result_id
                == eval_case_result_id
            )
            .order_by(
                RetrievalEvalStageObservationModel.stage_key,
                RetrievalEvalStageObservationModel.source_key,
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_stage_observation(model) for model in models]

    async def list_stage_candidates(
            self,
            eval_case_result_id: UUID,
    ) -> list[RetrievalEvalStageCandidate]:

        stmt = (
            select(RetrievalEvalStageCandidateModel)
            .where(
                RetrievalEvalStageCandidateModel.eval_case_result_id
                == eval_case_result_id
            )
            .order_by(
                RetrievalEvalStageCandidateModel.stage_key,
                RetrievalEvalStageCandidateModel.source_key,
                RetrievalEvalStageCandidateModel.rank_after,
                RetrievalEvalStageCandidateModel.rank_before,
            )
        )

        models = (
            await self._session.execute(stmt)
        ).scalars().all()

        return [_to_stage_candidate(model) for model in models]

    async def aggregate_stage_stats(
            self,
            eval_run_id: UUID,
    ) -> list[dict]:
        """
        按 (stage_key, source_key) 聚合。**在 SQL 里分组**，不把行捞进内存。

        `drop_reason_counts` 用 `FILTER (WHERE ...)` 逐个数：理由的取值是
        受控词表（`EvalDropReason`），而它会长（已经加过一次
        `ownership_filter`）。用 `jsonb_object_agg` 之类做成动态的会让这条
        查询难读又难改；列出当前的取值反而是**看得见**的 —— 新加一个原因
        时这里会漏掉，而漏掉的表现是"那个原因的计数不见了"，比"多一个
        不认识的键"更容易发现。

        ⚠️ 但"漏掉"仍然是这个写法的代价。所以最后一列是**总数**：
        `gold_drop_count` 与逐项之和不等时，说明词表加了新值而这里没跟上。
        调用方可以据此发现，而不是默默丢一个桶。
        """

        observations = RetrievalEvalStageObservationModel
        results = RetrievalEvalCaseResultModel

        # 按原因分桶的计数。列在下面，新增原因要同时加一行。
        reason_columns = {
            "not_recalled": "not_recalled",
            "top_k_cutoff": "top_k_cutoff",
            "score_threshold": "score_threshold",
            "metadata_filter": "metadata_filter",
            "ownership_filter": "ownership_filter",
            "rerank_cutoff": "rerank_cutoff",
            "final_top_k_cutoff": "final_top_k_cutoff",
            "deduplicated": "deduplicated",
            "permission_filter": "permission_filter",
            "processor_error": "processor_error",
            "unknown": "unknown",
            "not_applicable": "not_applicable",
        }

        columns = [
            observations.stage_key.label("stage_key"),
            observations.source_key.label("source_key"),
            func.count().label("observed_cases"),
            func.count()
            .filter(observations.executed.is_(True))
            .label("executed_cases"),
            func.avg(observations.input_count).label("average_input_count"),
            func.avg(observations.output_count).label("average_output_count"),
            func.count()
            .filter(observations.gold_present_before.is_(True))
            .label("gold_present_before_count"),
            func.count()
            .filter(observations.gold_present_after.is_(True))
            .label("gold_present_after_count"),
            # 丢弃 = 有原因的那些。归因判断在写入时就做完了（见实体说明），
            # 这里只数。
            func.count()
            .filter(observations.drop_reason.is_not(None))
            .label("gold_drop_count"),
        ]

        columns.extend(
            func.count()
            .filter(observations.drop_reason == value)
            .label(f"reason__{name}")
            for name, value in reason_columns.items()
        )

        stmt = (
            select(*columns)
            .join(
                results,
                results.eval_case_result_id
                == observations.eval_case_result_id,
            )
            .where(results.eval_run_id == eval_run_id)
            .group_by(observations.stage_key, observations.source_key)
            .order_by(observations.stage_key, observations.source_key)
        )

        rows = (await self._session.execute(stmt)).all()

        aggregated: list[dict] = []

        for row in rows:

            mapping = row._mapping

            aggregated.append({
                "stage_key": mapping["stage_key"],
                "source_key": mapping["source_key"],
                "observed_cases": int(mapping["observed_cases"]),
                "executed_cases": int(mapping["executed_cases"]),
                "average_input_count": (
                    None
                    if mapping["average_input_count"] is None
                    else float(mapping["average_input_count"])
                ),
                "average_output_count": (
                    None
                    if mapping["average_output_count"] is None
                    else float(mapping["average_output_count"])
                ),
                "gold_present_before_count": int(
                    mapping["gold_present_before_count"]
                ),
                "gold_present_after_count": int(
                    mapping["gold_present_after_count"]
                ),
                "gold_drop_count": int(mapping["gold_drop_count"]),
                "drop_reason_counts": {
                    name: int(mapping[f"reason__{name}"])
                    for name in reason_columns
                    if int(mapping[f"reason__{name}"]) > 0
                },
            })

        return aggregated


# ============================================================
# Mapping
# ============================================================


def _to_run(model: RetrievalEvalRunModel) -> RetrievalEvalRun:

    return RetrievalEvalRun(
        eval_run_id=model.eval_run_id,
        bundle_id=model.bundle_id,
        status=model.status,
        max_k=model.max_k,
        query_processor_override=dict(
            model.query_processor_override or {}
        ),
        effective_config_snapshot=dict(
            model.effective_config_snapshot or {}
        ),
        include_stage_trace=model.include_stage_trace,
        total_cases=model.total_cases,
        completed_cases=model.completed_cases,
        failed_cases=model.failed_cases,
        metrics=dict(model.metrics or {}),
        created_by=model.created_by,
        started_at=model.started_at,
        completed_at=model.completed_at,
        created_at=model.created_at,
    )


def _to_case_result(
        model: RetrievalEvalCaseResultModel,
) -> RetrievalEvalCaseResult:

    return RetrievalEvalCaseResult(
        eval_case_result_id=model.eval_case_result_id,
        eval_run_id=model.eval_run_id,
        eval_case_id=model.eval_case_id,
        attempt_no=model.attempt_no,
        query_id=model.query_id,
        query_text=model.query_text,
        gold_hit=model.gold_hit,
        gold_rank=model.gold_rank,
        retrieved_chunk_ids=[
            dict(item) for item in (model.retrieved_chunk_ids or [])
        ],
        latency_ms=model.latency_ms,
        degraded_routes=[
            dict(item) for item in (model.degraded_routes or [])
        ],
        status=model.status,
        error_code=model.error_code,
        error_message=model.error_message,
        created_at=model.created_at,
    )


def _to_stage_observation(
        model: RetrievalEvalStageObservationModel,
) -> RetrievalEvalStageObservation:

    return RetrievalEvalStageObservation(
        stage_observation_id=model.stage_observation_id,
        eval_case_result_id=model.eval_case_result_id,
        stage_key=model.stage_key,
        source_key=model.source_key,
        executed=model.executed,
        input_count=model.input_count,
        output_count=model.output_count,
        gold_present_before=model.gold_present_before,
        gold_present_after=model.gold_present_after,
        gold_rank_before=model.gold_rank_before,
        gold_rank_after=model.gold_rank_after,
        gold_score_before=model.gold_score_before,
        gold_score_after=model.gold_score_after,
        gold_score_name=model.gold_score_name,
        gold_score_direction=model.gold_score_direction,
        drop_reason=model.drop_reason,
        details=dict(model.details or {}),
        created_at=model.created_at,
    )


def _to_stage_candidate(
        model: RetrievalEvalStageCandidateModel,
) -> RetrievalEvalStageCandidate:

    return RetrievalEvalStageCandidate(
        stage_candidate_id=model.stage_candidate_id,
        eval_case_result_id=model.eval_case_result_id,
        stage_key=model.stage_key,
        source_key=model.source_key,
        chunk_id=model.chunk_id,
        rank_before=model.rank_before,
        rank_after=model.rank_after,
        score_before=model.score_before,
        score_after=model.score_after,
        score_name_before=model.score_name_before,
        score_name_after=model.score_name_after,
        score_direction_before=model.score_direction_before,
        score_direction_after=model.score_direction_after,
        selected=model.selected,
        drop_reason=model.drop_reason,
        metadata=dict(model.metadata_ or {}),
        created_at=model.created_at,
    )
