"""
实体 -> 对外契约的组装。

纯函数、不碰数据库 —— 与其余模块的 _assembly.py 同一分工。

为什么集中在这里
----------------
同一份数据在列表与详情里必须长得一样。若 list 与 get 各写一遍字段
映射，迟早出现「列表里的 status 与详情里的 status 含义不同」这种事，
而那种不一致没有任何测试会抓到。

⚠️ 指标是从 JSONB 透传的
----------------------
StressRunMetrics 的分组形状（sessions / turns / model / ...）与表列
不是一一对应：表里是 session_latency、turn_latency、model_metrics
这类 JSONB，响应里是 sessions.duration_ms、turns.latency_ms、
model.duration_ms。映射只在这里做一次。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..domain.entities import StressRunMetricsRecord, StressRunRecord
from ..domain.schemas.stress import (
    AggregationStatus,
    Distribution,
    FanoutMetrics,
    IngestionMetrics,
    ModelMetrics,
    SessionMetrics,
    StressRun,
    StressRunCreate,
    StressRunMetrics,
    StressRunPage,
    TurnMetrics,
)

__all__ = [
    "build_stress_run",
    "build_stress_run_metrics",
    "build_stress_run_page",
    "empty_stress_run_metrics",
]


def build_stress_run(
        run: StressRunRecord,
) -> StressRun:
    """
    运行记录 -> 响应。

    config 在这里才解释成 StressRunCreate：库里存的是 JSON 快照，
    解释失败说明那一行不是本服务写进去的，应当以 500 暴露，而不是
    悄悄少几个字段。
    """

    return StressRun(
        run_id=run.run_id,
        name=run.name,
        status=run.status,
        config=StressRunCreate.model_validate(run.config),
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_code=run.error_code,
        error_message=run.error_message,
    )


def build_stress_run_page(
        items: list[StressRunRecord],
        next_cursor: str | None,
) -> StressRunPage:

    return StressRunPage(
        items=[build_stress_run(item) for item in items],
        next_cursor=next_cursor,
    )


def _distribution(
        raw: Any,
) -> Distribution:
    """
    JSONB -> Distribution。

    None / 空 dict 都合法（还没采到样本）；非 dict 的脏数据让 Pydantic
    报错，不在这里吞掉 —— 那说明写入侧已经违约了。
    """

    return Distribution.model_validate(raw or {})


def build_stress_run_metrics(
        metrics: StressRunMetricsRecord,
) -> StressRunMetrics:
    """快照记录 -> 响应。"""

    model_raw = metrics.model_metrics or {}
    fanout_raw = metrics.fanout_metrics or {}

    completion_rate = (
        metrics.sessions_completed / metrics.sessions_started
        if metrics.sessions_started > 0
        else None
    )

    sessions = SessionMetrics(
        started=metrics.sessions_started,
        completed=metrics.sessions_completed,
        failed=metrics.sessions_failed,
        abandoned=metrics.sessions_abandoned,
        completion_rate=completion_rate,
        achieved_sessions_per_second=(
            metrics.achieved_sessions_per_second
        ),
        duration_ms=_distribution(metrics.session_latency),
        sampled_turns_per_session=_distribution(
            metrics.sampled_turns_per_session
        ),
        turns_per_session=_distribution(metrics.turns_per_session),
    )

    turns = TurnMetrics(
        started=metrics.turns_started,
        completed=metrics.turns_completed,
        succeeded=metrics.turns_succeeded,
        failed=metrics.turns_failed,
        timed_out=metrics.turns_timed_out,
        cancelled=metrics.turns_cancelled,
        achieved_turns_per_second=(
            metrics.achieved_turns_per_second
        ),
        error_rate=metrics.turn_error_rate,
        latency_ms=_distribution(metrics.turn_latency),
        time_to_first_output_ms=_distribution(
            metrics.time_to_first_output
        ),
        first_turn_latency_ms=_distribution(
            metrics.first_turn_latency
        ),
        later_turn_latency_ms=_distribution(
            metrics.later_turn_latency
        ),
        by_index={
            str(index): _distribution(value)
            for index, value in (
                metrics.turn_latency_by_index or {}
            ).items()
        },
    )

    model = ModelMetrics(
        call_count=int(model_raw.get("call_count", 0)),
        retry_count=int(model_raw.get("retry_count", 0)),
        error_count=int(model_raw.get("error_count", 0)),
        input_tokens_sum=metrics.model_input_tokens_sum,
        output_tokens_sum=metrics.model_output_tokens_sum,
        duration_ms=_distribution(model_raw.get("duration_ms")),
        input_tokens_per_turn=_distribution(
            model_raw.get("input_tokens_per_turn")
        ),
        output_tokens_per_turn=_distribution(
            model_raw.get("output_tokens_per_turn")
        ),
    )

    return StressRunMetrics(
        run_id=metrics.run_id,
        aggregation_status=metrics.aggregation_status,
        aggregation_algorithm=metrics.aggregation_algorithm,
        finalized_at=metrics.finalized_at,
        sessions=sessions,
        turns=turns,
        trace_counts={
            str(key): int(value)
            for key, value in (metrics.trace_counts or {}).items()
        },
        model=model,
        nodes={
            str(name): _distribution(value)
            for name, value in (metrics.node_metrics or {}).items()
        },
        tools={
            str(name): _distribution(value)
            for name, value in (metrics.tool_metrics or {}).items()
        },
        fanout=FanoutMetrics(
            count=int(fanout_raw.get("count", 0)),
            branch_count=int(fanout_raw.get("branch_count", 0)),
            failed_branch_count=int(
                fanout_raw.get("failed_branch_count", 0)
            ),
        ),
        ingestion=IngestionMetrics(
            events_seen=metrics.events_seen,
            events_deduplicated=metrics.events_deduplicated,
            events_unmatched=metrics.events_unmatched,
            events_late=metrics.events_late,
        ),
    )


def empty_stress_run_metrics(
        run_id: UUID,
) -> StressRunMetrics:
    """
    还没有快照行时的合法响应。

    正常路径下建运行就会建快照行，所以这是防御性分支。这里返回
    collecting 而不是 404：运行存在、只是还没有样本，两者不能混为
    一谈 —— 否则前端会把「还没开始」显示成「运行不存在」。
    """

    return StressRunMetrics(
        run_id=run_id,
        aggregation_status=AggregationStatus.COLLECTING,
    )
