"""
本地模拟执行器：**不发出任何网络请求**的多轮会话压测占位实现。

它存在的唯一理由是「让前端能开发」。真实方案里，Runner 会按 §3 的
流程建立真实会话、逐轮提交 Agent Run、消费 SSE 事件；Aggregator 会
消费 Redis Trace 并按 run 聚合。这些当前都不做；本模块只做两件事：

    1. 把运行从 queued 推进到 running -> completed；
    2. 按配置推导一份**多轮会话**的聚合快照。

⚠️ 快照不是压测结果
------------------
``aggregation_algorithm`` 固定写 ``simulated-v1``。看到它就知道这些
数字来自配置推导，而不是真实请求。

模拟的口径
----------
- 每个会话在创建时抽一次 N（截断正态 `[1,30]`），之后不再改变。
- 会话内逐轮执行：每轮有成功/失败/超时三种结果，失败后按
  ``on_turn_failure`` 决定是否终止会话。
- 会话按负载模式被“接纳”：
  - ``concurrency``：按波次接纳，每波 `concurrency` 个会话；
  - ``session_rate``：按 `session_rate` 个/秒均匀铺开。
- 预热窗口内创建的会话不进入聚合。
- 达到 ``duration_seconds`` / ``max_turns`` 时停止发起新轮次，未完成
  会话计为 abandoned。
- 每个会话的随机源由 ``seed + session_index`` 派生，避免并发顺序
  影响抽样结果。

真实 Runner 接入时替换 ``schedule()`` 与 ``_simulate()`` 即可；API、
表结构与 DTO 不需要跟着改。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from app.common.repository.base import SessionFactory
from app.common.repository.factory import RepositoryFactory

from ..domain.entities import StressRunMetricsRecord
from ..domain.schemas.stress import (
    AggregationStatus,
    ConcurrencyLoad,
    StressRunCreate,
    TurnCountConfig,
)
from ..repository.stress_run import StressRunStore

__all__ = [
    "SIMULATED_ALGORITHM",
    "SimulatedStressRunner",
    "build_simulated_metrics",
    "sample_turn_count",
    "simulation_enabled",
]

logger = logging.getLogger(__name__)

#: 快照里记录的分位算法标识。
SIMULATED_ALGORITHM = "simulated-v1"

#: 关闭开关。默认开启 —— 这是给前端联调的开发默认值。
SIMULATION_ENABLED_ENV = "STRESS_SIMULATION_ENABLED"

#: 合成负载的上限。10000 并发 × 86400 秒会推出百万级会话，
#: 前端图表没有可读性，模拟器也没有必要真的跑那么多。
MAX_PLANNED_SESSIONS = 10_000

#: 截断正态拒绝采样的最大尝试次数。DTO 已经校验了接受率 >= 5%，
#: 这里的上限只是防御性兜底。
REJECTION_SAMPLE_LIMIT = 10_000

#: 每个 Agent Run 的模型调用数（分类 + 推理）。与容量文档一致。
MODEL_CALLS_PER_TURN = 2

#: 每轮除模型调用外的框架开销（ms）。
TURN_OVERHEAD_MS = 50.0

#: 后续轮次每多一轮，输入 Token 的近似增长量（上下文累积）。
INPUT_TOKEN_GROWTH_PER_TURN = 40

#: 单轮输入 Token 的基础值。
INPUT_TOKENS_BASE = 200


def simulation_enabled() -> bool:
    """环境变量解析。拼错了按开启处理。"""

    raw = os.environ.get(SIMULATION_ENABLED_ENV, "1").strip().lower()

    return raw not in {"0", "false", "no", "off"}


# ============================================================
# 统计小工具
# ============================================================


def _empty_distribution() -> dict[str, Any]:
    return {
        "count": 0,
        "mean": None,
        "min": None,
        "max": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
    }


def _percentile(
        ordered: list[float],
        fraction: float,
) -> float:
    """线性插值分位数。ordered 必须已排序。"""

    count = len(ordered)

    if count == 1:
        return ordered[0]

    position = (count - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * weight
    )


def _distribution(
        values: list[float],
) -> dict[str, Any]:
    """样本列表 -> Distribution 快照。"""

    if not values:
        return _empty_distribution()

    ordered = sorted(float(value) for value in values)

    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p90": round(_percentile(ordered, 0.90), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
    }


# ============================================================
# N 的抽样
# ============================================================


def sample_turn_count(
        config: TurnCountConfig,
        rng: random.Random,
) -> int:
    """
    从截断正态抽一个整数 N。

    截断用拒绝采样；DTO 已校验区间内概率 >= 5%，因此几乎不会走到
    兜底分支。取整按 ``rounding``，再夹到 ``[min, max]``。
    """

    if config.stddev == 0:
        value = float(config.mean)
    else:

        for _ in range(REJECTION_SAMPLE_LIMIT):
            sample = rng.gauss(config.mean, config.stddev)

            if config.min <= sample <= config.max:
                value = sample
                break

        else:
            value = min(
                max(config.mean, config.min),
                config.max,
            )

    if config.rounding == "floor":
        turns = math.floor(value)
    elif config.rounding == "ceil":
        turns = math.ceil(value)
    else:
        # 用 floor(x + 0.5) 而不是 Python 的 round()：
        # 后者是银行家舍入，15.5 -> 16、16.5 -> 16，难以向使用者解释。
        turns = math.floor(value + 0.5)

    return max(config.min, min(config.max, int(turns)))


# ============================================================
# 模拟执行器
# ============================================================


class SimulatedStressRunner:
    """推进运行状态并写一份合成的多轮会话快照。"""

    def __init__(
            self,
            session_factory: SessionFactory,
            *,
            start_delay_seconds: float = 0.3,
            min_run_seconds: float = 2.0,
            max_run_seconds: float = 15.0,
    ) -> None:

        self._repositories = RepositoryFactory(
            session_factory=session_factory,
        )

        self._start_delay = max(0.0, start_delay_seconds)
        self._min_run = max(0.0, min_run_seconds)
        self._max_run = max(self._min_run, max_run_seconds)

        #: 持有引用，避免任务被 GC；done callback 里移出。
        self._tasks: set[asyncio.Task[None]] = set()

        #: run_id -> task，供 cancel / stop 使用。
        self._task_by_run: dict[UUID, asyncio.Task[None]] = {}

    # --------------------------------------------------------
    # 调度
    # --------------------------------------------------------

    def schedule(
            self,
            run_id: UUID,
            config: StressRunCreate,
    ) -> None:
        """
        启动一个后台模拟任务。

        同步非阻塞：post-commit 钩子里调用，不能在这里 await 一段真实
        时长。真正的工作在 ``_simulate`` 里。
        """

        if not simulation_enabled():
            logger.info(
                "[Stress] %s 未启动模拟：%s=0，运行将停在 queued。",
                run_id,
                SIMULATION_ENABLED_ENV,
            )
            return

        task = asyncio.create_task(
            self._simulate(run_id, config),
            name=f"stress-sim:{run_id}",
        )

        self._tasks.add(task)
        self._task_by_run[run_id] = task
        task.add_done_callback(
            lambda finished, rid=run_id: self._on_done(rid, finished)
        )

    def cancel(self, run_id: UUID) -> None:
        """协作式取消的模拟版：直接取消后台任务。"""

        task = self._task_by_run.get(run_id)

        if task is None:
            return

        task.cancel()

    def stop(self) -> None:
        """进程关闭：取消全部模拟任务。"""

        for task in list(self._tasks):
            task.cancel()

    def _on_done(
            self,
            run_id: UUID,
            task: asyncio.Task[None],
    ) -> None:

        self._tasks.discard(task)
        self._task_by_run.pop(run_id, None)

        if task.cancelled():
            return

        exc = task.exception()

        if exc is not None:
            # 模拟器是旁路：失败不能让请求路径炸（请求早已返回 202），
            # 但也不能静默吞掉，否则运行会永远停在 running。
            logger.error(
                "[Stress] 模拟执行失败：%s",
                task.get_name(),
                exc_info=exc,
            )

    # --------------------------------------------------------
    # 状态机
    # --------------------------------------------------------

    async def _simulate(
            self,
            run_id: UUID,
            config: StressRunCreate,
    ) -> None:

        await asyncio.sleep(self._start_delay)

        async with self._repositories.scope() as repos:

            started = await repos.create(StressRunStore).mark_running(
                run_id
            )

        if started is None:
            logger.info(
                "[Stress] %s 已不在 queued，模拟不启动。",
                run_id,
            )
            return

        await asyncio.sleep(self._wall_clock_duration(config))

        snapshot = build_simulated_metrics(run_id, config)

        async with self._repositories.scope() as repos:

            completed = await repos.create(
                StressRunStore
            ).complete_run(run_id, snapshot)

        if completed is None:
            logger.info(
                "[Stress] %s 在模拟完成前已离开 running（多为取消），"
                "快照不写 final。",
                run_id,
            )
            return

        logger.info(
            "[Stress] %s 模拟完成：%d 会话 / %d 轮，algorithm=%s。",
            run_id,
            snapshot.sessions_started,
            snapshot.turns_started,
            SIMULATED_ALGORITHM,
        )

    def _wall_clock_duration(
            self,
            config: StressRunCreate,
    ) -> float:
        """
        真实等待多久。

        与 duration_seconds 成比例但不照搬：前端联调不需要等一小时。
        上限保证长压测也不会把一个后台任务挂一天。
        """

        scaled = config.duration_seconds / 10.0

        return min(max(scaled, self._min_run), self._max_run)


# ============================================================
# 合成快照
# ============================================================


def _avg_call_ms(config: StressRunCreate) -> float:
    """单次模型调用的平均耗时。"""

    model = config.mock_model

    avg_output = (
        model.output_tokens_min + model.output_tokens_max
    ) / 2.0

    generation_ms = (
        avg_output / model.tokens_per_second * 1000.0
        if model.tokens_per_second > 0
        else 0.0
    )

    return max(
        1.0,
        model.base_latency_ms
        + model.latency_jitter_ms / 2.0
        + model.ttft_ms
        + generation_ms,
    )


def _avg_turn_seconds(config: StressRunCreate) -> float:
    """一轮的平均墙钟耗时（模型调用 + 框架开销 + think time）。"""

    model_ms = _avg_call_ms(config) * MODEL_CALLS_PER_TURN

    return (
        model_ms + TURN_OVERHEAD_MS
    ) / 1000.0 + config.session.think_time_ms / 1000.0


def _sample_call_ms(
        config: StressRunCreate,
        rng: random.Random,
) -> float:
    """一次模型调用的合成耗时。"""

    model = config.mock_model

    return max(
        1.0,
        model.base_latency_ms
        + rng.uniform(0.0, float(model.latency_jitter_ms))
        + model.ttft_ms
        + (
            (
                model.output_tokens_min + model.output_tokens_max
            ) / 2.0
            / model.tokens_per_second
            * 1000.0
            if model.tokens_per_second > 0
            else 0.0
        ),
    )


def _sample_ttft_ms(
        config: StressRunCreate,
        rng: random.Random,
) -> float:
    """首 Token 延迟（从本轮开始算）。"""

    model = config.mock_model

    return max(
        0.0,
        model.base_latency_ms
        + rng.uniform(0.0, float(model.latency_jitter_ms))
        + model.ttft_ms,
    )


def _planned_sessions(config: StressRunCreate) -> int:
    """
    估算本次模拟要接纳多少会话。

    真实 Runner 的接纳由 duration / 负载模式 / total_sessions 共同
    决定；这里用同一个口径估算，只是不真的 sleep 那么久。
    """

    if config.total_sessions is not None:
        planned = config.total_sessions
    elif isinstance(config.load, ConcurrencyLoad):
        avg_session_seconds = max(
            1.0,
            config.session.turns.mean * _avg_turn_seconds(config),
        )
        waves = max(
            1,
            math.ceil(config.duration_seconds / avg_session_seconds),
        )
        planned = config.load.concurrency * waves
    else:
        planned = max(
            1,
            math.ceil(
                config.load.session_rate * config.duration_seconds
            ),
        )

    return max(1, min(planned, MAX_PLANNED_SESSIONS))


def _session_offset_seconds(
        config: StressRunCreate,
        index: int,
        avg_session_seconds: float,
) -> float:
    """第 index 个会话的模拟接纳时刻。"""

    if isinstance(config.load, ConcurrencyLoad):
        wave = index // config.load.concurrency
        return wave * avg_session_seconds

    return index / config.load.session_rate


def build_simulated_metrics(
        run_id: UUID,
        config: StressRunCreate,
) -> StressRunMetricsRecord:
    """
    从配置推导一份确定性的多轮会话聚合快照。

    推导链是「配置 -> 单轮平均耗时 -> 会话时长 -> 接纳波次 ->
    逐会话逐轮模拟」。这只是让数字之间**互相自洽**，不是对真实吞吐
    的预测；真正的结果必须等 Runner 跑出来。
    """

    base_seed = (
        config.seed
        if config.seed is not None
        else run_id.int % (2 ** 32)
    )

    model = config.mock_model

    avg_turn_seconds = _avg_turn_seconds(config)
    avg_session_seconds = max(
        1.0,
        config.session.turns.mean * avg_turn_seconds,
    )

    planned_sessions = _planned_sessions(config)

    # --------------------------------------------------------
    # 会话 / 轮次计数
    # --------------------------------------------------------

    sessions_started = 0
    sessions_completed = 0
    sessions_failed = 0
    sessions_abandoned = 0

    turns_started = 0
    turns_completed = 0
    turns_succeeded = 0
    turns_failed = 0
    turns_timed_out = 0
    turns_cancelled = 0

    global_turns_started = 0

    sampled_turns: list[float] = []
    turns_per_session: list[float] = []
    session_latencies: list[float] = []

    turn_latencies: list[float] = []
    ttft_samples: list[float] = []
    first_turn_latencies: list[float] = []
    later_turn_latencies: list[float] = []
    turn_latencies_by_index: dict[str, list[float]] = {}

    input_tokens_per_turn_values: list[float] = []
    output_tokens_per_turn_values: list[float] = []
    model_call_durations: list[float] = []

    model_input_tokens_sum = 0
    model_output_tokens_sum = 0

    finish_offset_seconds = 0.0

    for session_index in range(planned_sessions):

        offset_seconds = _session_offset_seconds(
            config,
            session_index,
            avg_session_seconds,
        )

        # 时长到点：后续会话不再接纳。
        if offset_seconds >= config.duration_seconds:
            break

        # 预热窗口内的会话不进入聚合（真实 Runner 仍会执行它们）。
        if offset_seconds < config.warmup_seconds:
            continue

        sessions_started += 1

        rng = random.Random(f"{base_seed}:{session_index}")

        sampled_n = sample_turn_count(config.session.turns, rng)

        sampled_turns.append(float(sampled_n))

        actual_turns = 0
        session_latency_ms = 0.0
        session_failed = False
        session_aborted = False

        for turn_index in range(1, sampled_n + 1):

            if (
                config.max_turns is not None
                and global_turns_started >= config.max_turns
            ):
                session_aborted = True
                break

            if (
                offset_seconds
                + session_latency_ms / 1000.0
                + avg_turn_seconds
                > config.duration_seconds
            ):
                session_aborted = True
                break

            global_turns_started += 1
            actual_turns += 1
            turns_started += 1

            call_durations = [
                _sample_call_ms(config, rng)
                for _ in range(MODEL_CALLS_PER_TURN)
            ]
            model_call_durations.extend(call_durations)

            turn_latency_ms = (
                sum(call_durations) + rng.uniform(5.0, 80.0)
            )
            ttft_ms = _sample_ttft_ms(config, rng)

            input_tokens = (
                INPUT_TOKENS_BASE
                + config.turn.input_size_chars // 4
                + (turn_index - 1) * INPUT_TOKEN_GROWTH_PER_TURN
            )
            output_tokens = sum(
                rng.randint(
                    model.output_tokens_min,
                    model.output_tokens_max,
                )
                for _ in range(MODEL_CALLS_PER_TURN)
            )

            input_tokens_per_turn_values.append(float(input_tokens))
            output_tokens_per_turn_values.append(float(output_tokens))
            model_input_tokens_sum += input_tokens
            model_output_tokens_sum += output_tokens

            is_timeout = rng.random() < model.timeout_rate
            is_failed = (
                not is_timeout
                and rng.random() < model.error_rate
            )

            if is_timeout:
                turns_timed_out += 1
                session_latency_ms += float(config.request_timeout_ms)

                if config.session.on_turn_failure == "abort":
                    session_failed = True
                    break

                continue

            turns_completed += 1

            if is_failed:
                turns_failed += 1
            else:
                turns_succeeded += 1

            session_latency_ms += (
                turn_latency_ms + config.session.think_time_ms
            )

            # 时延分布只统计收到终态的轮次：超时阈值不是“真实时延”。
            turn_latencies.append(turn_latency_ms)
            ttft_samples.append(ttft_ms)

            if turn_index == 1:
                first_turn_latencies.append(turn_latency_ms)
            else:
                later_turn_latencies.append(turn_latency_ms)

            key = str(turn_index)
            turn_latencies_by_index.setdefault(key, []).append(
                turn_latency_ms
            )

            if is_failed and config.session.on_turn_failure == "abort":
                session_failed = True
                break

        turns_per_session.append(float(actual_turns))
        session_latencies.append(session_latency_ms)

        finish_offset_seconds = max(
            finish_offset_seconds,
            offset_seconds + session_latency_ms / 1000.0,
        )

        if session_failed:
            sessions_failed += 1
        elif session_aborted or actual_turns < sampled_n:
            sessions_abandoned += 1
        else:
            sessions_completed += 1

    # --------------------------------------------------------
    # 速率与错误率
    # --------------------------------------------------------

    measurement_seconds = max(
        1.0,
        min(
            float(config.duration_seconds),
            finish_offset_seconds,
        )
        - config.warmup_seconds,
    )

    achieved_sessions_per_second = round(
        sessions_completed / measurement_seconds,
        4,
    )

    achieved_turns_per_second = round(
        turns_completed / measurement_seconds,
        4,
    )

    turn_error_rate = (
        round(
            (turns_failed + turns_timed_out) / turns_started,
            6,
        )
        if turns_started > 0
        else None
    )

    model_call_count = turns_started * MODEL_CALLS_PER_TURN

    # --------------------------------------------------------
    # Trace 派生的合成计数
    # --------------------------------------------------------

    trace_counts = {
        "run_span_count": turns_started,
        "node_span_count": turns_started * 2,
        "model_call_count": model_call_count,
        "tool_call_count": 0,
        "context_build_count": turns_started,
        "fanout_count": 0,
    }

    events_seen = turns_started * 4 + model_call_count * 2

    return StressRunMetricsRecord(
        run_id=run_id,
        aggregation_status=AggregationStatus.FINAL,
        aggregation_algorithm=SIMULATED_ALGORITHM,
        finalized_at=datetime.now(timezone.utc),
        sessions_started=sessions_started,
        sessions_completed=sessions_completed,
        sessions_failed=sessions_failed,
        sessions_abandoned=sessions_abandoned,
        achieved_sessions_per_second=achieved_sessions_per_second,
        session_latency=_distribution(session_latencies),
        sampled_turns_per_session=_distribution(sampled_turns),
        turns_per_session=_distribution(turns_per_session),
        turns_started=turns_started,
        turns_completed=turns_completed,
        turns_succeeded=turns_succeeded,
        turns_failed=turns_failed,
        turns_timed_out=turns_timed_out,
        turns_cancelled=turns_cancelled,
        achieved_turns_per_second=achieved_turns_per_second,
        turn_error_rate=turn_error_rate,
        turn_latency=_distribution(turn_latencies),
        time_to_first_output=_distribution(ttft_samples),
        turn_latency_by_index={
            key: _distribution(values)
            for key, values in turn_latencies_by_index.items()
        },
        first_turn_latency=_distribution(first_turn_latencies),
        later_turn_latency=_distribution(later_turn_latencies),
        trace_counts=trace_counts,
        model_metrics={
            "call_count": model_call_count,
            "retry_count": int(
                model_call_count * model.error_rate * 0.5
            ),
            "error_count": int(
                model_call_count * model.error_rate
            ),
            "duration_ms": _distribution(model_call_durations),
            "input_tokens_per_turn": _distribution(
                input_tokens_per_turn_values
            ),
            "output_tokens_per_turn": _distribution(
                output_tokens_per_turn_values
            ),
        },
        node_metrics={
            "main_agent": _distribution(turn_latencies),
            "classification": _distribution(model_call_durations),
        },
        tool_metrics={},
        fanout_metrics={
            "count": 0,
            "branch_count": 0,
            "failed_branch_count": 0,
        },
        model_input_tokens_sum=model_input_tokens_sum,
        model_output_tokens_sum=model_output_tokens_sum,
        events_seen=events_seen,
        events_deduplicated=int(events_seen * 0.001),
        events_unmatched=0,
        events_late=int(events_seen * 0.0005),
    )
