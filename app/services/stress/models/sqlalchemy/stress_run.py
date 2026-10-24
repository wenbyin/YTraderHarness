"""
压测运行与聚合快照两张表。

表结构以 ``design_agent_eval/schema.sql`` 为准。这里刻意**不持久化**
Trace Event / Span / 逐会话 / 逐轮明细 —— 它们只在 Redis 或 Runner
内存里短暂存在，本模块只保存运行参数、生命周期状态与最终聚合快照。

两张表为什么拆开
----------------
stress_runs 是**状态机**：每次状态推进都要做条件 UPDATE（CAS），
而且要按 status / created_at 建索引做列表查询。

stress_run_metrics 是**大块 JSONB**：聚合器可以反复覆盖快照，直到
定稿；它与运行状态的写入频率、并发语义都不同。

没有迁移工具
------------
与仓库其余模块一样，建表靠 initializer 的 ``Base.metadata.create_all()``
（幂等但**不会改已有表**）。已有库上要改列必须手工 ALTER，否则 ORM
声明了新列、SELECT 会带上它，症状是每次查询都 500。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

__all__ = [
    "StressRun",
    "StressRunMetrics",
    "StressRunMetricsModel",
]


class StressRun(Base):
    """一次压测运行。"""

    __tablename__ = "stress_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="运行 ID。每次压测独立一份。",
    )

    name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="可读名称，仅用于列表展示。",
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'queued'"),
        comment=(
            "queued / running / completed / failed / cancelled。"
            "取值定义在 domain/schemas/stress.py::StressRunStatus。"
        ),
    )

    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment=(
            "创建时固化的完整参数（StressRunCreate 的 JSON 视图）。"
            "包含 target / auth / session / turn / load / mock_model。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。列表按它倒序。",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="真正开始执行的时刻。",
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="进入终态的时刻。",
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="失败摘要的机器可读代码。",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败摘要。不得存放 Trace / 会话 / 轮次明细。",
    )

    created_by: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment=(
            "创建者标识。当前接口不填 —— 鉴权主体与业务身份如何映射"
            "尚未定案。"
        ),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', "
            "'cancelled')",
            name="ck_stress_runs_status",
        ),
        Index("idx_stress_runs_created_at", "created_at"),
        Index(
            "idx_stress_runs_status_created",
            "status",
            "created_at",
        ),
    )


class StressRunMetrics(Base):
    """一次运行的聚合快照。一行对应一次运行。"""

    __tablename__ = "stress_run_metrics"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stress_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
        comment="所属运行。运行删除时快照一并删除。",
    )

    aggregation_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'collecting'"),
        comment="collecting / final / incomplete。",
    )

    aggregation_algorithm: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "分位数算法与精度配置。当前模拟器写 simulated-v1；"
            "真实聚合器接入后写实际算法。"
        ),
    )

    finalized_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="快照定稿时刻。",
    )

    # --------------------------------------------------------
    # 会话级指标
    # --------------------------------------------------------

    sessions_started: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    sessions_completed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    sessions_failed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    sessions_abandoned: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    achieved_sessions_per_second: Mapped[float | None] = mapped_column(
        Numeric(14, 4),
        nullable=True,
        comment="completed sessions / measurement duration。",
    )

    session_latency: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="会话端到端时延的 Distribution 快照。",
    )

    sampled_turns_per_session: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="每个会话创建时抽到的 N 的分布。",
    )

    turns_per_session: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment=(
            "每个会话实际发起的轮数分布；正常完成时等于抽到的 N，"
            "abort / 取消 / 截断时小于 N。"
        ),
    )

    # --------------------------------------------------------
    # 轮次级指标
    # --------------------------------------------------------

    turns_started: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    turns_completed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
        comment="收到终态的轮次（succeeded + failed）。",
    )

    turns_succeeded: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    turns_failed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    turns_timed_out: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    turns_cancelled: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    achieved_turns_per_second: Mapped[float | None] = mapped_column(
        Numeric(14, 4),
        nullable=True,
        comment="completed turns / measurement duration。",
    )

    turn_error_rate: Mapped[float | None] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        comment=(
            "(turns_failed + turns_timed_out) / turns_started；"
            "不含取消。"
        ),
    )

    turn_latency: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment=(
            "单轮端到端时延分布；仅统计收到终态的轮次，"
            "不把超时阈值混进真实时延。"
        ),
    )

    time_to_first_output: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="单轮首段输出时延分布（仅 streaming）。",
    )

    turn_latency_by_index: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment=(
            "按 turn_index（\"1\" ~ \"30\"）分组的时延分布。"
        ),
    )

    first_turn_latency: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="每个会话第 1 轮的时延分布。",
    )

    later_turn_latency: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="第 2 轮及以后的时延分布。",
    )

    # --------------------------------------------------------
    # Trace 聚合指标
    # --------------------------------------------------------

    trace_counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment=(
            "事件类别计数。键是有限枚举，不用 trace_id / span_id / "
            "session_id 这类高基数标识。"
        ),
    )

    model_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment=(
            "模型调用计数、重试、错误、耗时分布、每轮 token 分布。"
        ),
    )

    node_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="按 node_name 分组的耗时分布。",
    )

    tool_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="按 tool_name 分组的耗时分布。",
    )

    fanout_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="并行分支计数。",
    )

    model_input_tokens_sum: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    model_output_tokens_sum: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    # --------------------------------------------------------
    # 聚合消费质量
    # --------------------------------------------------------

    events_seen: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    events_deduplicated: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    events_unmatched: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    events_late: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "aggregation_status IN ('collecting', 'final', 'incomplete')",
            name="ck_stress_metrics_status",
        ),
        Index(
            "idx_stress_metrics_aggregation_status",
            "aggregation_status",
            "updated_at",
        ),
    )


#: 跨模块引用时的别名。
StressRunMetricsModel = StressRunMetrics
