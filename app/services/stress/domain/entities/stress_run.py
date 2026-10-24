"""
stress_runs / stress_run_metrics 的 Python 形态。

与其余模块的实体一样，这一层**不认识 ORM**：仓储把 Core 查询返回的
行映射成这里的模型，服务层只处理实体与 DTO，不 import SQLAlchemy。

为什么 metrics 的分布字段是裸 dict
--------------------------------
``session_latency`` / ``turn_latency`` / ``turns_per_session`` 等在库里
是 JSONB，聚合器可以在不改变表结构的情况下往里加字段。实体只负责
搬运，解释成 ``Distribution`` 是 application/_assembly.py 的事。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.stress import AggregationStatus, StressRunStatus

__all__ = [
    "StressRunCursor",
    "StressRunMetricsRecord",
    "StressRunRecord",
]


class StressRunRecord(BaseModel):
    """stress_runs 的一行。"""

    model_config = ConfigDict(from_attributes=True)

    run_id: UUID

    name: str | None = None

    status: StressRunStatus

    config: dict[str, Any]

    created_at: datetime

    started_at: datetime | None = None

    finished_at: datetime | None = None

    error_code: str | None = None

    error_message: str | None = None

    created_by: str | None = None


class StressRunMetricsRecord(BaseModel):
    """
    stress_run_metrics 的一行。

    列名与 design_agent_eval/schema.sql 一一对应；响应里的分组形状
    （sessions / turns / model / nodes / ...）由 application/_assembly.py
    组装。
    """

    model_config = ConfigDict(from_attributes=True)

    run_id: UUID

    aggregation_status: AggregationStatus = AggregationStatus.COLLECTING

    aggregation_algorithm: str | None = None

    finalized_at: datetime | None = None

    # --------------------------------------------------------
    # 会话级
    # --------------------------------------------------------

    sessions_started: int = 0

    sessions_completed: int = 0

    sessions_failed: int = 0

    sessions_abandoned: int = 0

    achieved_sessions_per_second: float | None = None

    session_latency: dict[str, Any] = Field(default_factory=dict)

    sampled_turns_per_session: dict[str, Any] = Field(
        default_factory=dict,
    )

    turns_per_session: dict[str, Any] = Field(default_factory=dict)

    # --------------------------------------------------------
    # 轮次级
    # --------------------------------------------------------

    turns_started: int = 0

    turns_completed: int = 0

    turns_succeeded: int = 0

    turns_failed: int = 0

    turns_timed_out: int = 0

    turns_cancelled: int = 0

    achieved_turns_per_second: float | None = None

    turn_error_rate: float | None = None

    turn_latency: dict[str, Any] = Field(default_factory=dict)

    time_to_first_output: dict[str, Any] = Field(
        default_factory=dict,
    )

    turn_latency_by_index: dict[str, Any] = Field(
        default_factory=dict,
    )

    first_turn_latency: dict[str, Any] = Field(
        default_factory=dict,
    )

    later_turn_latency: dict[str, Any] = Field(
        default_factory=dict,
    )

    # --------------------------------------------------------
    # Trace 派生
    # --------------------------------------------------------

    trace_counts: dict[str, Any] = Field(default_factory=dict)

    model_metrics: dict[str, Any] = Field(default_factory=dict)

    node_metrics: dict[str, Any] = Field(default_factory=dict)

    tool_metrics: dict[str, Any] = Field(default_factory=dict)

    fanout_metrics: dict[str, Any] = Field(default_factory=dict)

    model_input_tokens_sum: int = 0

    model_output_tokens_sum: int = 0

    # --------------------------------------------------------
    # 聚合消费质量
    # --------------------------------------------------------

    events_seen: int = 0

    events_deduplicated: int = 0

    events_unmatched: int = 0

    events_late: int = 0


@dataclass(frozen=True)
class StressRunCursor:
    """
    列表分页游标。

    ⚠️ 用 (created_at, run_id) 两列而不是 offset：这是一个会持续插入
    新行的列表页。offset 分页在翻页时有新记录落库的情况下会漏行或
    重复行，而且不会报错。
    """

    created_at: datetime

    run_id: UUID
