"""
多轮会话压测的对外契约。

对应 ``design_agent_eval/openapi.yaml`` 的 components/schemas。这里的
字段名、必填性、取值范围与那份规格一一对应；不要在这里加规格之外的
字段，前端是按那份 yaml 生成的。

核心概念
--------
一次 Stress Run 由一组**虚拟会话**组成。每个虚拟会话对应 Agent 的
一个真实 ``session_id``，在其上串行进行 N 轮对话；N 从截断正态分布
抽样，范围 ``[1, 30]``。

    session.agent_id     POST /sessions 的 agent_id
    session.turns        截断正态参数
    turn.*               每一轮 POST /sessions/{id}/runs 的输入
    load                 并发会话数或新会话速率

两类模型
--------
    StressRunCreate  进站请求。``extra=forbid``，未知字段直接 422
    StressRun / StressRunMetrics  出站响应。由实体组装

``config`` 的建站与回读
-----------------------
``stress_runs.config`` 是 JSONB，写入时用 ``model_dump(mode="json")``，
读出时再用 ``StressRunCreate.model_validate``。因此请求里的默认值
（turn.streaming / think_time_ms / ...）会被**固化**进快照。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from statistics import NormalDist
from typing import Annotated, Any, Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.common.schemas import APIModel

__all__ = [
    "AggregationStatus",
    "AuthConfig",
    "ConcurrencyLoad",
    "Distribution",
    "FanoutMetrics",
    "IngestionMetrics",
    "MockModelConfig",
    "ModelMetrics",
    "SessionConfig",
    "SessionMetrics",
    "SessionRateLoad",
    "StressRun",
    "StressRunCreate",
    "StressRunLoad",
    "StressRunMetrics",
    "StressRunPage",
    "StressRunStatus",
    "TargetConfig",
    "TurnConfig",
    "TurnCountConfig",
    "TurnMetrics",
]


# ============================================================
# 枚举
# ============================================================


class StressRunStatus(str, Enum):
    """一次压测运行的终态机。取值与 schema.sql 的 CHECK 一致。"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """终态：状态不会再变，取消也改不动。"""

        return self.value in {"completed", "failed", "cancelled"}


class AggregationStatus(str, Enum):
    """聚合快照的生命周期。"""

    COLLECTING = "collecting"
    FINAL = "final"
    INCOMPLETE = "incomplete"


# ============================================================
# 请求：目标与认证
# ============================================================


class TargetConfig(APIModel):
    """被测 Agent 服务的地址与端点路径。"""

    base_url: str = Field(
        description="被测 Agent 服务的 base URL（http/https）。",
    )

    create_session_path: str = Field(
        default="/api/v1/sessions",
        description="建立会话的路径。",
    )

    create_run_path: str = Field(
        default="/api/v1/sessions/{session_id}/runs",
        description="在会话下建 Run 的路径模板。",
    )

    run_events_path: str = Field(
        default="/api/v1/runs/{run_id}/events",
        description="单轮 Run 的 SSE 事件流路径模板。",
    )

    cancel_run_path: str = Field(
        default="/api/v1/runs/{run_id}/cancel",
        description="取消单轮 Run 的路径模板。",
    )

    resume_run_path: str = Field(
        default="/api/v1/runs/{run_id}/resume",
        description="HITL 恢复路径模板；仅 auto_approve/reject 时使用。",
    )

    delete_session_path: str = Field(
        default="/api/v1/sessions/{session_id}",
        description="清理会话的路径模板。",
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        """
        只接受 http/https 绝对地址。

        ⚠️ 这不等于 SSRF 防护。第一阶段不发起真实请求；接真实 Runner
        时必须另做 allowlist，否则一个内网地址就能把压测器变成探测器。
        """

        parsed = urlparse(value)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "target.base_url 必须是 http/https 绝对地址，"
                f"当前是 {value!r}。"
            )

        return value


class AuthConfig(APIModel):
    """Runner 访问被测服务时使用的凭据。"""

    type: Literal["bearer"] = Field(
        description="认证类型。第一版只支持 bearer。",
    )

    token_env: str = Field(
        min_length=1,
        description=(
            "保存 Bearer token 的环境变量名。"
            "**不要**在这里传明文 token：stress_runs.config 会落库并回显。"
        ),
    )


# ============================================================
# 请求：会话与轮次
# ============================================================


class TurnCountConfig(APIModel):
    """
    会话轮数 N 的截断正态配置。

    抽样一次，写入该会话的 Runner 内存状态；之后不再变化。
    ``mean`` / ``stddev`` 描述的是**截断前**的正态；实际分布记录在
    指标 ``sampled_turns_per_session`` / ``turns_per_session`` 里。
    """

    mean: float = Field(
        ge=1,
        le=30,
        description="截断前正态的均值。",
    )

    stddev: float = Field(
        ge=0,
        le=30,
        description="截断前正态的标准差；0 表示固定轮数，便于测试。",
    )

    min: int = Field(
        default=1,
        ge=1,
        le=30,
        description="截断下界（含）。",
    )

    max: int = Field(
        default=30,
        ge=1,
        le=30,
        description="截断上界（含）。",
    )

    rounding: Literal["round", "floor", "ceil"] = Field(
        default="round",
        description="连续样本的取整方式。",
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "TurnCountConfig":
        if self.min > self.max:
            raise ValueError(
                f"session.turns.min 不能大于 max：{self.min} > {self.max}。"
            )

        if not (self.min <= self.mean <= self.max):
            raise ValueError(
                "session.turns.mean 必须落在 [min, max] 内："
                f"mean={self.mean}, min={self.min}, max={self.max}。"
            )

        if self.stddev > 0:
            # 拒绝采样的接受率太低时，配置实际上不可运行。
            # 这里用正态 CDF 提前算出来，而不是等 Runner 抽样一千次后才报错。
            distribution = NormalDist(self.mean, self.stddev)
            acceptance = (
                distribution.cdf(self.max)
                - distribution.cdf(self.min)
            )

            if acceptance < 0.05:
                raise ValueError(
                    "session.turns 的截断区间只覆盖不到 5% 的正态概率，"
                    "拒绝采样会几乎全部失败；请调小 stddev 或让均值更靠近区间中心。"
                )

        return self


class SessionConfig(APIModel):
    """一个虚拟会话的策略。"""

    agent_id: str = Field(
        min_length=1,
        description="建立会话时传给 POST /sessions 的 agent_id。",
    )

    turns: TurnCountConfig = Field(
        description="轮数 N 的分布配置。",
    )

    think_time_ms: int = Field(
        default=0,
        ge=0,
        le=600_000,
        description="同一会话两轮之间的等待时间。",
    )

    on_turn_failure: Literal["abort", "continue"] = Field(
        default="abort",
        description="某轮失败后：终止会话，还是继续下一轮。",
    )

    on_hitl: Literal["abort", "auto_approve", "reject"] = Field(
        default="abort",
        description="遇到 waiting_input 时的策略。",
    )

    cleanup: Literal["keep", "delete"] = Field(
        default="keep",
        description="会话结束后是否 DELETE /sessions/{session_id}。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="建立会话时携带的 metadata。",
    )


class TurnConfig(APIModel):
    """每一轮对话的输入配置。"""

    input_template: dict[str, Any] | None = Field(
        default=None,
        description=(
            "每轮 CreateRunRequest.input 的模板，例如 {\"text\": \"...\"}。"
            "缺省时由 input_size_chars 生成合成文本。"
        ),
    )

    input_size_chars: int = Field(
        default=512,
        ge=1,
        le=100_000,
        description="未提供 input_template 时的合成文本长度。",
    )

    streaming: bool = Field(
        default=True,
        description=(
            "是否消费 SSE 正文。第一版必须为 true：Agent 只提供 "
            "GET /runs/{run_id}/events，没有 GET /runs/{run_id}。"
        ),
    )

    @model_validator(mode="after")
    def _require_streaming(self) -> "TurnConfig":
        if not self.streaming:
            raise ValueError(
                "第一版只支持 turn.streaming=true：当前 Agent 没有 "
                "GET /runs/{run_id}，非流式无法可靠判断单轮终态。"
            )

        return self


# ============================================================
# 请求：负载
# ============================================================


class ConcurrencyLoad(APIModel):
    """固定并发会话数。"""

    mode: Literal["concurrency"] = Field(
        description="负载模式。本对象固定为 concurrency。",
    )

    concurrency: int = Field(
        ge=1,
        le=10_000,
        description="同时在途的虚拟会话数。",
    )


class SessionRateLoad(APIModel):
    """目标新会话速率。"""

    mode: Literal["session_rate"] = Field(
        description="负载模式。本对象固定为 session_rate。",
    )

    session_rate: float = Field(
        gt=0,
        le=10_000,
        description=(
            "每秒新建的虚拟会话数。轮次速率约为 "
            "session_rate × E[N]。"
        ),
    )

    max_concurrency: int = Field(
        default=1000,
        ge=1,
        le=10_000,
        description=(
            "同时在途的虚拟会话数上限。session_rate 只控制**接纳速率**，"
            "不限制慢会话堆积；没有这个上限时，会话一慢，在途会话数会"
            "无限增长。"
        ),
    )


StressRunLoad = Annotated[
    ConcurrencyLoad | SessionRateLoad,
    Field(discriminator="mode"),
]


# ============================================================
# 请求：Mock Model
# ============================================================


class MockModelConfig(APIModel):
    """
    Mock Model 的延迟 / Token / 故障注入参数。

    参数按**每次模型调用**生效；同一会话后续轮次的输入 Token 会随
    历史增长，但生成行为仍按这份配置执行。
    """

    profile: str | None = Field(
        default=None,
        description="profile 名称。第一版只存不用。",
    )

    base_latency_ms: int = Field(
        ge=0,
        le=600_000,
        description="模型请求的基准延迟。",
    )

    latency_jitter_ms: int = Field(
        default=0,
        ge=0,
        le=600_000,
        description="基准延迟上的随机波动范围。",
    )

    ttft_ms: int = Field(
        ge=0,
        le=600_000,
        description="首个 Token 的延迟。",
    )

    output_tokens_min: int = Field(
        ge=0,
        le=100_000,
        description="输出 Token 数下界。",
    )

    output_tokens_max: int = Field(
        ge=0,
        le=100_000,
        description="输出 Token 数上界。",
    )

    tokens_per_second: float = Field(
        gt=0,
        le=100_000,
        description="模拟输出生成速率。",
    )

    error_rate: float = Field(
        default=0,
        ge=0,
        le=1,
        description="模型请求模拟失败概率。",
    )

    timeout_rate: float = Field(
        default=0,
        ge=0,
        le=1,
        description="模型请求模拟超时概率。",
    )

    stream_chunk_tokens: int = Field(
        default=1,
        ge=1,
        le=10_000,
        description="SSE 每个 chunk 模拟输出的 Token 数。",
    )

    max_inflight_requests: int | None = Field(
        default=None,
        ge=1,
        description="Mock Model Server 允许的最大并发请求数。",
    )

    @model_validator(mode="after")
    def _validate_token_range(self) -> "MockModelConfig":
        if self.output_tokens_max < self.output_tokens_min:
            raise ValueError(
                "output_tokens_max 不能小于 output_tokens_min："
                f"当前是 {self.output_tokens_max} < "
                f"{self.output_tokens_min}。"
            )

        return self


# ============================================================
# 请求：创建
# ============================================================


class StressRunCreate(APIModel):
    """创建一次多轮会话压测所需的全部参数。"""

    name: str | None = Field(
        default=None,
        max_length=200,
        description="可读名称，仅用于列表展示。",
    )

    target: TargetConfig = Field(
        description="被测 Agent 服务地址与端点路径。",
    )

    auth: AuthConfig | None = Field(
        default=None,
        description="Runner 访问被测服务时的认证；不传则由服务端配置提供。",
    )

    session: SessionConfig = Field(
        description="虚拟会话策略。",
    )

    turn: TurnConfig = Field(
        default_factory=TurnConfig,
        description="每轮对话的输入配置。",
    )

    load: StressRunLoad = Field(
        description="负载模式：并发会话数或新会话速率，二选一。",
    )

    duration_seconds: int = Field(
        ge=1,
        le=86_400,
        description="整个 Run 的硬性墙钟上限。",
    )

    warmup_seconds: int = Field(
        default=0,
        ge=0,
        le=3_600,
        description=(
            "在该窗口内开始创建的会话不计入聚合，但仍正常执行。"
        ),
    )

    ramp_up_seconds: int = Field(
        default=0,
        ge=0,
        le=3_600,
        description="会话接纳速率/并发从 0 增长到目标值的时间。",
    )

    total_sessions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "有限工作负载：接纳满该数量的会话后不再创建新会话。"
            "不传则由 duration 与负载模式决定。"
        ),
    )

    max_turns: int | None = Field(
        default=None,
        ge=1,
        description=(
            "整个 Run 的轮次硬上限。达到后不再发起新轮次，"
            "未完成会话计为 abandoned。"
        ),
    )

    request_timeout_ms: int = Field(
        ge=100,
        le=600_000,
        description=(
            "单轮从提交 Run 到收到终态事件的最长等待时间，"
            "包含 HITL 处理时间。"
        ),
    )

    seed: int | None = Field(
        default=None,
        description=(
            "N 抽样与 Mock Model 随机行为的种子。"
            "同一 seed + 同一配置必须复现同一串会话轮数。"
        ),
    )

    mock_model: MockModelConfig = Field(
        description="Mock Model 行为配置。",
    )

    @model_validator(mode="after")
    def _validate_warmup(self) -> "StressRunCreate":
        if self.warmup_seconds >= self.duration_seconds:
            raise ValueError(
                "warmup_seconds 必须小于 duration_seconds："
                "否则整个运行都落在预热窗口里，没有任何可聚合样本。"
            )

        return self


# ============================================================
# 响应：运行
# ============================================================


class StressRun(APIModel):
    """一次压测运行的参数与状态。"""

    run_id: UUID = Field(description="运行 ID。")

    name: str | None = Field(
        default=None,
        description="可读名称。",
    )

    status: StressRunStatus = Field(description="运行状态。")

    config: StressRunCreate = Field(
        description="创建时固化的完整参数快照。",
    )

    created_at: datetime = Field(description="创建时间。")

    started_at: datetime | None = Field(
        default=None,
        description="真正开始执行的时刻。",
    )

    finished_at: datetime | None = Field(
        default=None,
        description="进入终态的时刻。",
    )

    error_code: str | None = Field(
        default=None,
        description="失败摘要的机器可读代码。",
    )

    error_message: str | None = Field(
        default=None,
        description="失败摘要。不存放 Trace / 会话 / 轮次明细。",
    )


class StressRunPage(APIModel):
    """运行记录的一页。"""

    items: list[StressRun] = Field(description="当前页的运行记录。")

    next_cursor: str | None = Field(
        default=None,
        description="下一页游标；null 表示没有更多。",
    )


# ============================================================
# 响应：聚合指标
# ============================================================


class Distribution(BaseModel):
    """
    一组可合并统计的最终快照。

    ⚠️ 用 BaseModel 而不是 APIModel：这是**透传**存储侧 JSONB 的响应，
    未来聚合器往里加字段时不该让整个 metrics 接口 500。因此这里
    ``extra=ignore``。
    """

    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
    )

    count: int = Field(default=0, description="样本数。")

    mean: float | None = Field(default=None, description="均值。")

    min: float | None = Field(default=None, description="最小值。")

    max: float | None = Field(default=None, description="最大值。")

    p50: float | None = Field(default=None, description="中位数。")

    p90: float | None = Field(default=None, description="P90。")

    p95: float | None = Field(default=None, description="P95。")

    p99: float | None = Field(default=None, description="P99。")


class SessionMetrics(APIModel):
    """Runner 侧的会话级指标。"""

    started: int = Field(default=0, description="尝试创建会话的次数。")

    completed: int = Field(
        default=0,
        description=(
            "计划 N 轮全部执行完毕的会话数。"
            "on_turn_failure=continue 时个别轮失败不改变完成状态。"
        ),
    )

    failed: int = Field(
        default=0,
        description=(
            "建会话失败，或因轮次失败而终止的会话数。"
        ),
    )

    abandoned: int = Field(
        default=0,
        description=(
            "因时长、取消、max_turns 等提前结束的会话数。"
        ),
    )

    completion_rate: float | None = Field(
        default=None,
        description="completed / started。",
    )

    achieved_sessions_per_second: float | None = Field(
        default=None,
        description="completed sessions / measurement duration。",
    )

    duration_ms: Distribution = Field(
        default_factory=Distribution,
        description="会话端到端时延分布。",
    )

    sampled_turns_per_session: Distribution = Field(
        default_factory=Distribution,
        description="每个会话创建时抽到的 N 的分布。",
    )

    turns_per_session: Distribution = Field(
        default_factory=Distribution,
        description=(
            "每个会话实际发起的轮数分布；正常完成时等于抽到的 N，"
            "abort / 取消 / 截断时小于 N。"
        ),
    )


class TurnMetrics(APIModel):
    """轮次级指标。"""

    started: int = Field(default=0)

    completed: int = Field(
        default=0,
        description="收到终态的轮次（succeeded + failed）。",
    )

    succeeded: int = Field(default=0)

    failed: int = Field(default=0)

    timed_out: int = Field(default=0)

    cancelled: int = Field(default=0)

    achieved_turns_per_second: float | None = Field(
        default=None,
        description="completed turns / measurement duration。",
    )

    error_rate: float | None = Field(
        default=None,
        description=(
            "(failed + timed_out) / started；不含取消。"
        ),
    )

    latency_ms: Distribution = Field(
        default_factory=Distribution,
        description="单轮端到端时延分布（仅含收到终态的轮次）。",
    )

    time_to_first_output_ms: Distribution = Field(
        default_factory=Distribution,
        description="单轮首段输出时延分布（仅 streaming）。",
    )

    first_turn_latency_ms: Distribution = Field(
        default_factory=Distribution,
        description="每个会话第 1 轮的时延分布。",
    )

    later_turn_latency_ms: Distribution = Field(
        default_factory=Distribution,
        description="第 2 轮及以后的时延分布。",
    )

    by_index: dict[str, Distribution] = Field(
        default_factory=dict,
        description=(
            "按 turn_index（\"1\" ~ \"30\"）分组的时延分布；"
            "只保留实际出现过的 index。"
        ),
    )


class ModelMetrics(APIModel):
    """Trace 聚合出的模型调用指标。"""

    call_count: int = Field(default=0)

    retry_count: int = Field(default=0)

    error_count: int = Field(default=0)

    input_tokens_sum: int = Field(default=0)

    output_tokens_sum: int = Field(default=0)

    duration_ms: Distribution = Field(default_factory=Distribution)

    input_tokens_per_turn: Distribution = Field(
        default_factory=Distribution,
    )

    output_tokens_per_turn: Distribution = Field(
        default_factory=Distribution,
    )


class FanoutMetrics(APIModel):
    """并行分支指标。"""

    count: int = Field(default=0)

    branch_count: int = Field(default=0)

    failed_branch_count: int = Field(default=0)


class IngestionMetrics(APIModel):
    """聚合消费质量。"""

    events_seen: int = Field(default=0)

    events_deduplicated: int = Field(default=0)

    events_unmatched: int = Field(default=0)

    events_late: int = Field(default=0)


class StressRunMetrics(APIModel):
    """一次运行的最终聚合快照。"""

    run_id: UUID = Field(description="运行 ID。")

    aggregation_status: AggregationStatus = Field(
        description="collecting / final / incomplete。",
    )

    aggregation_algorithm: str | None = Field(
        default=None,
        description=(
            "分位数算法与精度标识。"
            "当前模拟器写 simulated-v1 —— 看到它就知道这不是压测结果。"
        ),
    )

    finalized_at: datetime | None = Field(
        default=None,
        description="快照定稿时刻。",
    )

    sessions: SessionMetrics = Field(
        default_factory=SessionMetrics,
        description="会话级指标。",
    )

    turns: TurnMetrics = Field(
        default_factory=TurnMetrics,
        description="轮次级指标。",
    )

    trace_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "按事件类别汇总的计数。键是有限枚举（run / node / model / "
            "tool / context / fanout），不是 trace_id 这类高基数标识。"
        ),
    )

    model: ModelMetrics = Field(
        default_factory=ModelMetrics,
        description="模型调用指标。",
    )

    nodes: dict[str, Distribution] = Field(
        default_factory=dict,
        description="按 node_name 分组的耗时分布。",
    )

    tools: dict[str, Distribution] = Field(
        default_factory=dict,
        description="按 tool_name 分组的耗时分布。",
    )

    fanout: FanoutMetrics = Field(
        default_factory=FanoutMetrics,
        description="并行分支指标。",
    )

    ingestion: IngestionMetrics = Field(
        default_factory=IngestionMetrics,
        description="聚合消费质量。",
    )
