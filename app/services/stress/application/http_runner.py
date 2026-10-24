"""
真实 HTTP/SSE Stress Driver（Phase A）。

运行模型
--------
    POST /api/v1/stress-runs
      -> StressRunServiceImpl.create()
      -> post_commit: HttpStressRunner.schedule(run_id, config)
      -> 独立守护线程（自己的 asyncio loop + aiohttp）
      -> 每个虚拟会话：建会话 -> 抽 N -> 串行 N 轮 -> SSE 等终态
      -> Reporter 把状态/指标投回主 loop 写库

为什么是独立线程
------------
主 loop 已经被 API、图调度、SSE 订阅占着。压测 Driver 会同时持有大量
HTTP/SSE 连接和定时器；把它放在主 loop 里会让"被测系统"与"压测器"
共享同一个 event loop 的调度预算，数字没有意义。

为什么 DB 写回主 loop
------------------
主应用的 asyncpg / SQLAlchemy async engine 是在主 loop 上创建的。
Driver 线程不能直接使用；见 stress_reporter.py。

Phase A 范围
------------
- 支持 concurrency 模式；
- session_rate 也实现了基本令牌桶 + max_concurrency；
- warmup / duration / total_sessions / max_turns；
- SSE 终态、TTFT、超时、取消、on_turn_failure；
- cleanup=keep/delete；
- HITL：abort / auto_approve / reject；
- Runner 侧指标 + 周期 collecting 快照；
- Trace 聚合不在本文件（Phase 2）。

不做的
------
- `StressRunCreate.mock_model` 暂不下发到 Mock Server（v1 忽略）；
- 不保存逐会话/逐轮明细；
- 不做断点续跑。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from uuid import UUID

import aiohttp

from app.common.event import stream_key
from app.services.stress.domain.entities import StressRunMetricsRecord
from app.services.stress.domain.schemas.stress import (
    AggregationStatus,
    ConcurrencyLoad,
    SessionRateLoad,
    StressRunCreate,
    StressRunStatus,
)

from .config import StressDriverConfig, load_stress_eval_config
from .metrics import DistributionBuilder
from .simulated_runner import sample_turn_count
from .stress_reporter import StressRunReporter
from .trace_aggregator import TraceAggregate, TraceAggregator

#: 兼容旧名字；配置现在由 app/config/stress_eval.yaml 提供。
StressDriverSettings = StressDriverConfig

__all__ = [
    "HttpStressRunner",
    "StressDriverConfig",
    "StressDriverSettings",
]

logger = logging.getLogger(__name__)

#: 快照算法标识。看到它就知道指标来自真实 Runner，但不含 Trace 聚合。
RUNNER_ALGORITHM = "runner-exact-capped-v1"

_TURN_SUCCEEDED = "succeeded"
_TURN_FAILED = "failed"
_TURN_TIMED_OUT = "timed_out"
_TURN_CANCELLED = "cancelled"

_SESSION_COMPLETED = "completed"
_SESSION_FAILED = "failed"
_SESSION_ABANDONED = "abandoned"


# ============================================================
# 配置
# ============================================================
#
# 所有 Driver 参数在 app/config/stress_eval.yaml；这里只保留小的
# 读取辅助函数，避免第二套默认值。


def _token_file_path(configured: str) -> Path:
    """配置里的 token 文件路径（支持 ``~`` 展开）。"""

    return Path(configured).expanduser()


def _read_token_file(path: Path) -> str | None:
    """读本地 token 文件；不存在/空/读失败返回 None。"""

    try:
        value = path.read_text(encoding="utf-8").strip()

    except FileNotFoundError:
        return None

    except OSError as exc:
        logger.warning(
            "[Stress] 读取 token 文件 %s 失败：%s",
            path,
            exc,
        )
        return None

    return value or None


# ============================================================
# 小结果类型
# ============================================================


@dataclass(frozen=True, slots=True)
class _TurnResult:
    status: str
    latency_ms: float
    ttft_ms: float | None = None
    error: str | None = None
    #: create_run 是否真的成功。False = 这一轮没有对应的 agent_run 行，
    #: 不能计入 turns_started / turns_per_session。
    submitted: bool = True


@dataclass(frozen=True, slots=True)
class _SseOutcome:
    kind: str  # terminal | hitl | closed
    status: str | None = None
    interrupts: tuple[str, ...] = ()
    error: str | None = None


class _HttpStatusError(RuntimeError):
    """被测服务返回的 HTTP 错误；带 status 供 create_run 重试判断。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# ============================================================
# 指标
# ============================================================


class _CampaignMetrics:
    """单 Campaign 的内存指标。只在 Driver loop 里读写。"""

    def __init__(self, *, sample_limit: int) -> None:
        self.sessions_started = 0
        self.sessions_completed = 0
        self.sessions_failed = 0
        self.sessions_abandoned = 0

        self.turns_started = 0
        self.turns_completed = 0
        self.turns_succeeded = 0
        self.turns_failed = 0
        self.turns_timed_out = 0
        self.turns_cancelled = 0

        self.session_latency = DistributionBuilder(limit=sample_limit)
        self.sampled_turns = DistributionBuilder(limit=sample_limit)
        self.actual_turns = DistributionBuilder(limit=sample_limit)
        self.turn_latency = DistributionBuilder(limit=sample_limit)
        self.ttft = DistributionBuilder(limit=sample_limit)
        self.first_turn_latency = DistributionBuilder(limit=sample_limit)
        self.later_turn_latency = DistributionBuilder(limit=sample_limit)
        self.by_index: dict[str, DistributionBuilder] = {}

    # --------------------------------------------------------
    # 会话
    # --------------------------------------------------------

    def session_started(self) -> None:
        self.sessions_started += 1

    def session_finished(
            self,
            outcome: str,
            *,
            sampled_n: int,
            actual_turns: int,
            latency_ms: float,
    ) -> None:
        self.sampled_turns.add(float(sampled_n))
        self.actual_turns.add(float(actual_turns))
        self.session_latency.add(latency_ms)

        if outcome == _SESSION_COMPLETED:
            self.sessions_completed += 1
        elif outcome == _SESSION_FAILED:
            self.sessions_failed += 1
        else:
            self.sessions_abandoned += 1

    # --------------------------------------------------------
    # 轮次
    # --------------------------------------------------------

    def turn_started(self) -> None:
        self.turns_started += 1

    def turn_finished(
            self,
            turn_index: int,
            result: _TurnResult,
    ) -> None:
        if result.status in {
            _TURN_SUCCEEDED,
            _TURN_FAILED,
        }:
            self.turns_completed += 1
            self.turn_latency.add(result.latency_ms)

            if turn_index == 1:
                self.first_turn_latency.add(result.latency_ms)
            else:
                self.later_turn_latency.add(result.latency_ms)

            key = str(turn_index)
            self.by_index.setdefault(
                key,
                DistributionBuilder(limit=200_000),
            ).add(result.latency_ms)

        if result.ttft_ms is not None:
            self.ttft.add(result.ttft_ms)

        if result.status == _TURN_SUCCEEDED:
            self.turns_succeeded += 1
        elif result.status == _TURN_FAILED:
            self.turns_failed += 1
        elif result.status == _TURN_TIMED_OUT:
            self.turns_timed_out += 1
        else:
            self.turns_cancelled += 1

    # --------------------------------------------------------
    # 快照
    # --------------------------------------------------------

    def snapshot(
            self,
            *,
            status: AggregationStatus,
            finalized_at: datetime | None,
            started_monotonic: float,
            duration_seconds: int,
            warmup_seconds: int,
            now_monotonic: float,
            trace: TraceAggregate | None = None,
    ) -> StressRunMetricsRecord:
        measurement_end = min(
            now_monotonic,
            started_monotonic + duration_seconds,
        )
        measurement_start = started_monotonic + warmup_seconds
        measurement_seconds = max(
            0.001,
            measurement_end - measurement_start,
        )

        turn_error_rate = (
            (self.turns_failed + self.turns_timed_out)
            / self.turns_started
            if self.turns_started
            else None
        )

        trace_data = (
            trace.snapshot()
            if trace is not None and trace.has_data
            else {}
        )
        algorithm = (
            f"{RUNNER_ALGORITHM}+trace-v1"
            if trace_data
            else RUNNER_ALGORITHM
        )

        return StressRunMetricsRecord(
            run_id=UUID(int=0),  # 调用方替换
            aggregation_status=status,
            aggregation_algorithm=algorithm,
            finalized_at=finalized_at,
            sessions_started=self.sessions_started,
            sessions_completed=self.sessions_completed,
            sessions_failed=self.sessions_failed,
            sessions_abandoned=self.sessions_abandoned,
            achieved_sessions_per_second=round(
                self.sessions_completed / measurement_seconds,
                4,
            ),
            session_latency=self.session_latency.snapshot(),
            sampled_turns_per_session=self.sampled_turns.snapshot(),
            turns_per_session=self.actual_turns.snapshot(),
            turns_started=self.turns_started,
            turns_completed=self.turns_completed,
            turns_succeeded=self.turns_succeeded,
            turns_failed=self.turns_failed,
            turns_timed_out=self.turns_timed_out,
            turns_cancelled=self.turns_cancelled,
            achieved_turns_per_second=round(
                self.turns_completed / measurement_seconds,
                4,
            ),
            turn_error_rate=(
                None
                if turn_error_rate is None
                else round(turn_error_rate, 6)
            ),
            turn_latency=self.turn_latency.snapshot(),
            time_to_first_output=self.ttft.snapshot(),
            turn_latency_by_index={
                key: builder.snapshot()
                for key, builder in self.by_index.items()
            },
            first_turn_latency=self.first_turn_latency.snapshot(),
            later_turn_latency=self.later_turn_latency.snapshot(),
            trace_counts=trace_data.get("trace_counts", {}),
            model_metrics=trace_data.get("model_metrics", {}),
            node_metrics=trace_data.get("node_metrics", {}),
            tool_metrics=trace_data.get("tool_metrics", {}),
            fanout_metrics=trace_data.get("fanout_metrics", {}),
            model_input_tokens_sum=trace_data.get(
                "model_input_tokens_sum", 0
            ),
            model_output_tokens_sum=trace_data.get(
                "model_output_tokens_sum", 0
            ),
            events_seen=trace_data.get("events_seen", 0),
            events_deduplicated=trace_data.get(
                "events_deduplicated", 0
            ),
            events_unmatched=trace_data.get("events_unmatched", 0),
            events_late=trace_data.get("events_late", 0),
        )


# ============================================================
# Campaign
# ============================================================


class _Campaign:
    """一次 Stress Run 的执行状态。"""

    def __init__(
            self,
            *,
            run_id: UUID,
            config: StressRunCreate,
            reporter: StressRunReporter,
            session: aiohttp.ClientSession,
            settings: StressDriverSettings,
            redis_url: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self._reporter = reporter
        self._session = session
        self._settings = settings

        #: create session 的准入限流。
        #:
        #: 后端每个 create session 会占用一条业务 DB 连接，而 Driver 的
        #: 会话并发可能远大于后端的连接池（默认 30）。不限流时，多出来的
        #: 请求会在池里排队 30 秒，最终以 500（pool checkout timeout）
        #: 失败 —— 那并不是被测 Agent 的容量结论，只是压测器自己把准备
        #: 阶段打爆了。建会话只是准备动作，收窄它不影响正式轮次的并发。
        self._session_create_slots = asyncio.Semaphore(
            self._settings.session_create_concurrency
        )

        #: cancel 的准入限流。
        #:
        #: Campaign 收尾时可能有数百个在途 run 同时被取消。全部并发会让
        #: 后端的业务 DB 池、HTTP 连接和客户端 socket 一起过载，产生大量
        #: 500 与 WinError 10054。取消是尽力而为，8 路足够。
        self._cancel_slots = asyncio.Semaphore(
            self._settings.cancel_concurrency
        )

        self._redis_url = redis_url

        #: redis 事件模式下的共享客户端（惰性创建，每次 XRANGE 借还连接）。
        self._poll_redis: Any | None = None

        #: run_id -> 已消费到的 Redis Stream entry id。
        #: **跨 resume 保留**：恢复后继续增量拉取，不重读整条流。
        self._run_cursors: dict[str, str] = {}

        self._stream_prefix = "agent_run"

        # 后端 SSE 上限来自 event_stream.yaml；Driver 自己的配置在
        # stress_eval.yaml。有效并发取两者的较小值。
        try:
            from app.agent.event.config import load_stream_config

            stream_config = load_stream_config()

            self._backend_stream_limit = int(
                stream_config.max_active_run_streams
            )

            self._stream_prefix = stream_config.key_prefix

        except Exception:
            self._backend_stream_limit = self._settings.max_concurrency

        self._trace_ids: set[str] = set()
        self._trace_aggregate: TraceAggregate | None = None
        self._trace_aggregator = (
            TraceAggregator(redis_url)
            if (
                redis_url
                and self._settings.trace_aggregation_enabled
            )
            else None
        )

        #: 失败 turn 的 (status, error) 样本。内存里只留前 200 条，
        #: 但计数不封顶 —— Campaign 收尾时会打一条汇总，方便确认
        #: "失败到底来自读取路径还是运行本身"。
        self._failure_count = 0
        self._failure_errors: Counter[str] = Counter()
        self._failure_samples: list[tuple[str, str]] = []

        self.cancel_event = asyncio.Event()
        self.started_at = time.monotonic()
        self.started_wall = datetime.now(timezone.utc)
        self.deadline = self.started_at + config.duration_seconds
        self.warmup_deadline = self.started_at + config.warmup_seconds

        base_seed = (
            config.seed
            if config.seed is not None
            else run_id.int % (2 ** 32)
        )
        self._base_seed = base_seed

        self._metrics = _CampaignMetrics(
            sample_limit=settings.sample_limit,
        )

        self._next_index = 0
        self._global_turns = 0
        self._index_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._snapshot_task: asyncio.Task[None] | None = None
        self._stopped = False

        self._headers: dict[str, str] = {}
        self._base_url = config.target.base_url.rstrip("/")

    # --------------------------------------------------------
    # 启动 / 收尾
    # --------------------------------------------------------

    async def run(self) -> None:
        try:
            self._validate_target()
            self._headers = self._build_headers()

            logger.info(
                "[Stress] %s event_transport=%s poll_interval=%dms "
                "poll_pool=%d (failure logging active)",
                self.run_id,
                self._settings.event_transport,
                self._settings.event_poll_interval_ms,
                self._settings.event_poll_max_connections,
            )

        except Exception as exc:  # noqa: BLE001 - 配置错误要落库
            logger.warning(
                "[Stress] %s 配置校验失败：%s",
                self.run_id,
                exc,
            )
            await self._reporter.mark_failed(
                self.run_id,
                error_code="invalid_stress_config",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            return

        started = await self._reporter.mark_running(self.run_id)

        if started is None:
            logger.info(
                "[Stress] %s 已不在 queued（取消或已终态），不启动 Campaign。",
                self.run_id,
            )
            return

        self._snapshot_task = asyncio.create_task(
            self._snapshot_loop(),
            name=f"stress-metrics:{self.run_id}",
        )

        try:
            await self._run_workers()

            # Trace 是旁路：等一个很短的 flush 窗口，再一次性读全部
            # trace stream。失败不影响 Runner 指标。
            await self._collect_trace()
            await self._finalize()

        except asyncio.CancelledError:
            self._stopped = True
            await self._finalize_incomplete()
            raise

        except Exception as exc:  # noqa: BLE001 - Runner 自身故障要落库
            logger.exception("[Stress] %s Campaign 失败。", self.run_id)
            self._stopped = True

            try:
                await self._reporter.mark_failed(
                    self.run_id,
                    error_code="runner_error",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                await self._finalize_incomplete()
            except Exception:
                logger.exception(
                    "[Stress] %s 失败状态落库也失败了。",
                    self.run_id,
                )

        finally:
            if self._snapshot_task is not None:
                self._snapshot_task.cancel()
                await asyncio.gather(
                    self._snapshot_task,
                    return_exceptions=True,
                )

            await self.aclose()

    async def aclose(self) -> None:
        if self._poll_redis is not None:
            try:
                await self._poll_redis.aclose()
            except Exception:  # noqa: BLE001 —— 收尾不抛
                logger.exception(
                    "[Stress] %s 关闭事件轮询 Redis 客户端失败。",
                    self.run_id,
                )
            self._poll_redis = None

        if self._trace_aggregator is not None:
            await self._trace_aggregator.aclose()

    async def _collect_trace(self) -> None:
        """Campaign 结束后一次性聚合所有会话的 Trace Stream。"""

        if self._trace_aggregator is None or not self._trace_ids:
            return

        delay = self._settings.trace_flush_delay_seconds

        if delay > 0:
            await asyncio.sleep(delay)

        try:
            self._trace_aggregate = (
                await self._trace_aggregator.collect_many(
                    sorted(self._trace_ids)
                )
            )
        except Exception:
            logger.exception(
                "[Stress] %s Trace 聚合失败（Runner 指标照常保留）。",
                self.run_id,
            )

    def _log_failure_summary(self) -> None:
        """Campaign 收尾时打印失败 turn 的类型/错误汇总。"""

        if not self._failure_count:
            return

        logger.warning(
            "[Stress] %s failure summary: %d failed turn(s); top errors: %s",
            self.run_id,
            self._failure_count,
            self._failure_errors.most_common(5),
        )

        for status, error in self._failure_samples[:5]:
            logger.warning(
                "[Stress] %s failure sample status=%s：%s",
                self.run_id,
                status,
                error,
            )

    async def _finalize(self) -> None:
        self._stopped = True

        self._log_failure_summary()

        latest = await self._reporter.get_run(self.run_id)
        cancelled = (
            self.cancel_event.is_set()
            or (latest is not None and latest.status is StressRunStatus.CANCELLED)
        )

        if cancelled:
            await self._finalize_incomplete()
            return

        metrics = self._metrics.snapshot(
            status=AggregationStatus.FINAL,
            finalized_at=datetime.now(timezone.utc),
            started_monotonic=self.started_at,
            duration_seconds=self.config.duration_seconds,
            warmup_seconds=self.config.warmup_seconds,
            now_monotonic=time.monotonic(),
            trace=self._trace_aggregate,
        )
        metrics.run_id = self.run_id

        completed = await self._reporter.complete_run(
            self.run_id,
            metrics,
        )

        if completed is None:
            # 取消/失败先到；保留已经采到的计数。
            await self._finalize_incomplete()

    async def _finalize_incomplete(self) -> None:
        self._log_failure_summary()

        metrics = self._metrics.snapshot(
            status=AggregationStatus.INCOMPLETE,
            finalized_at=datetime.now(timezone.utc),
            started_monotonic=self.started_at,
            duration_seconds=self.config.duration_seconds,
            warmup_seconds=self.config.warmup_seconds,
            now_monotonic=time.monotonic(),
            trace=self._trace_aggregate,
        )
        metrics.run_id = self.run_id

        try:
            await self._reporter.update_metrics(
                self.run_id,
                metrics,
            )
        except Exception:
            logger.exception(
                "[Stress] %s incomplete 快照写入失败。",
                self.run_id,
            )

    async def _snapshot_loop(self) -> None:
        interval = self._settings.metrics_flush_interval_seconds

        while not self._stopped:
            await asyncio.sleep(interval)

            if self._stopped:
                return

            snapshot = self._metrics.snapshot(
                status=AggregationStatus.COLLECTING,
                finalized_at=None,
                started_monotonic=self.started_at,
                duration_seconds=self.config.duration_seconds,
                warmup_seconds=self.config.warmup_seconds,
                now_monotonic=time.monotonic(),
                trace=self._trace_aggregate,
            )
            snapshot.run_id = self.run_id

            try:
                await self._reporter.update_metrics(
                    self.run_id,
                    snapshot,
                )
            except Exception:
                logger.exception(
                    "[Stress] %s 周期快照写入失败。",
                    self.run_id,
                )

    # --------------------------------------------------------
    # Worker 池
    # --------------------------------------------------------

    async def _run_workers(self) -> None:
        if isinstance(self.config.load, ConcurrencyLoad):
            await self._run_concurrency_workers(self.config.load)
        else:
            await self._run_session_rate(self.config.load)

    async def _run_concurrency_workers(
            self,
            load: ConcurrencyLoad,
    ) -> None:
        workers = load.concurrency

        if self.config.total_sessions is not None:
            workers = min(workers, self.config.total_sessions)

        workers = self._clamp_concurrency(workers, source="concurrency")

        if workers <= 0:
            return

        tasks = [
            asyncio.create_task(
                self._worker(index, workers),
                name=f"stress-worker:{self.run_id}:{index}",
            )
            for index in range(workers)
        ]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                logger.error(
                    "[Stress] %s worker 退出异常：%s",
                    self.run_id,
                    result,
                    exc_info=result,
                )

    async def _worker(
            self,
            worker_index: int,
            worker_count: int,
    ) -> None:
        ramp = max(0, self.config.ramp_up_seconds)

        if ramp > 0:
            delay = ramp * worker_index / max(1, worker_count)
            if not await self._sleep_cancellable(delay):
                return

        while not self._should_stop():
            session_index = await self._claim_session()

            if session_index is None:
                return

            try:
                await self._run_session(session_index)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "[Stress] %s session %d 执行异常。",
                    self.run_id,
                    session_index,
                )

    async def _run_session_rate(
            self,
            load: SessionRateLoad,
    ) -> None:
        """
        单准入循环 + 有界在途会话。

        旧实现给 `max_concurrency` 个 worker 逐个等令牌：1000 个 worker
        会预占未来 100 秒的令牌槽，即使 `total_sessions` 只有 135；
        Campaign 还要等这些 sleep 结束才收尾。现在只保留一个准入循环：

            acquire 并发槽 -> 按速率等到下一个接纳时刻 -> 建一个 session task

        连接/任务数始终不超过 max_concurrency，速率由同一个 `next_at`
        控制。
        """

        max_concurrency = self._clamp_concurrency(
            max(1, load.max_concurrency),
            source="max_concurrency",
        )
        semaphore = asyncio.Semaphore(max_concurrency)
        active: set[asyncio.Task[None]] = set()

        rate = max(0.0001, load.session_rate)
        ramp = max(0.0, self.config.ramp_up_seconds)
        next_at = time.monotonic()

        def _release(task: asyncio.Task[None]) -> None:
            active.discard(task)
            semaphore.release()

        try:
            while not self._should_stop():
                if (
                    self.config.total_sessions is not None
                    and self._next_index >= self.config.total_sessions
                ):
                    break

                await semaphore.acquire()

                now = time.monotonic()

                if ramp > 0:
                    progress = min(
                        1.0,
                        max(0.0, (now - self.started_at) / ramp),
                    )
                    effective = rate * progress
                else:
                    effective = rate

                if effective <= 0:
                    wait = max(
                        0.0,
                        self.started_at + ramp - now,
                    )
                    next_at = max(next_at, now + wait)
                else:
                    next_at = max(next_at, now)
                    wait = max(0.0, next_at - now)
                    next_at += 1.0 / effective

                    # ⚠️ ramp 刚开始时 effective 接近 0，1/effective 可能是
                    # 几千秒。若直接按这个间隔睡，整个 Campaign 只会接纳
                    # 第一个会话（实测 ramp=5s 时 60s 内只有 1 个）。
                    # 把等待截到 ramp 结束，并把 next_at 拉回醒来时刻，
                    # 让 ramp 结束后重新按目标速率排。
                    if ramp > 0 and effective < rate:
                        remaining = max(
                            0.0,
                            self.started_at + ramp - now,
                        )

                        if wait > remaining:
                            wait = remaining
                            next_at = now + wait

                if wait > 0 and not await self._sleep_cancellable(wait):
                    semaphore.release()
                    break

                session_index = await self._claim_session()

                if session_index is None:
                    semaphore.release()
                    break

                task = asyncio.create_task(
                    self._run_session(session_index),
                    name=(
                        f"stress-session:{self.run_id}:{session_index}"
                    ),
                )
                active.add(task)
                task.add_done_callback(_release)

        except asyncio.CancelledError:
            for task in active:
                task.cancel()

            await asyncio.gather(*active, return_exceptions=True)
            raise

        if active:
            await asyncio.gather(*active, return_exceptions=True)

    def _clamp_concurrency(
            self,
            requested: int,
            *,
            source: str,
    ) -> int:
        """
        Driver 侧并发硬上限。

        Windows 的 SelectorEventLoop 受 select() 512 socket 限制。即使
        后端 max_active_run_streams 调低了，前端仍可能传
        max_concurrency=1000；Driver 自己也会打开 1000 条出站连接。
        这里按 event_stream 的 max_active_run_streams 留 20% 余量后
        截断，避免把后端和 Driver 一起推过 512。
        """

        driver_limit = max(1, int(self._settings.max_concurrency))
        backend_limit = max(
            1,
            int(
                self._backend_stream_limit
                * self._settings.concurrency_safety_ratio
            ),
        )
        limit = min(driver_limit, backend_limit)

        if requested > limit:
            logger.warning(
                "[Stress] %s %s=%d 超过安全并发上限 %d"
                "（driver.max_concurrency=%d，"
                "backend max_active_run_streams=%d × %.2f），"
                "本次按 %d 运行。",
                self.run_id,
                source,
                requested,
                limit,
                driver_limit,
                self._backend_stream_limit,
                self._settings.concurrency_safety_ratio,
                limit,
            )
            return limit

        return requested

    async def _claim_session(self) -> int | None:
        async with self._index_lock:
            if self._should_stop():
                return None

            total = self.config.total_sessions

            if total is not None and self._next_index >= total:
                return None

            index = self._next_index
            self._next_index += 1

            return index

    # --------------------------------------------------------
    # 会话
    # --------------------------------------------------------

    async def _run_session(self, session_index: int) -> None:
        session_started = time.monotonic()
        counted = session_started >= self.warmup_deadline

        if counted:
            self._metrics.session_started()

        session_id: UUID | None = None
        sampled_n = 0
        actual_turns = 0
        outcome = _SESSION_COMPLETED

        try:
            async with self._session_create_slots:
                session_id = await self._create_session_with_retry()

            # 与 _create_run 里的 trace_id 完全一致：同一会话的 N 个 Run
            # 共享一个 trace 聚合键。
            self._trace_ids.add(
                f"stress:{self.run_id}:{session_index}"
            )

            rng = random.Random(
                f"{self._base_seed}:{session_index}"
            )
            sampled_n = sample_turn_count(
                self.config.session.turns,
                rng,
            )

            for turn_index in range(1, sampled_n + 1):
                if self._should_stop():
                    outcome = _SESSION_ABANDONED
                    break

                if (
                    turn_index > 1
                    and self.config.session.think_time_ms > 0
                ):
                    if not await self._sleep_cancellable(
                        self.config.session.think_time_ms / 1000.0
                    ):
                        outcome = _SESSION_ABANDONED
                        break

                if self._should_stop():
                    outcome = _SESSION_ABANDONED
                    break

                if not await self._reserve_turn():
                    outcome = _SESSION_ABANDONED
                    break

                result = await self._run_turn(
                    session_id,
                    session_index,
                    turn_index,
                    rng,
                    counted=counted,
                )

                if result.submitted:
                    actual_turns += 1

                    if counted:
                        self._metrics.turn_finished(
                            turn_index,
                            result,
                        )

                    # 诊断：已提交但失败的 turn 之前**不落日志**，导致
                    # "降轮询间隔就大面积失败"却看不到原因。这里把错误
                    # 原文打出来，并在 Campaign 收尾时再打一条汇总
                    # （指标表没有 per-turn error 列）。
                    if result.status in {_TURN_FAILED, _TURN_TIMED_OUT}:

                        error_text = result.error or "(empty)"

                        self._failure_count += 1
                        self._failure_errors[error_text] += 1

                        if len(self._failure_samples) < 200:
                            self._failure_samples.append(
                                (result.status, error_text)
                            )

                        logger.warning(
                            "[Stress] %s session %d turn %d %s：%s",
                            self.run_id,
                            session_index,
                            turn_index,
                            result.status,
                            error_text,
                        )
                else:
                    # create_run 失败：没有 agent_run 行，不能计
                    # turns_started / turns_per_session。
                    logger.warning(
                        "[Stress] %s session %d turn %d 未提交：%s",
                        self.run_id,
                        session_index,
                        turn_index,
                        result.error,
                    )

                    if (
                        self.config.session.on_turn_failure
                        == "abort"
                    ):
                        outcome = _SESSION_FAILED
                        break

                    continue

                if result.status == _TURN_CANCELLED:
                    outcome = _SESSION_ABANDONED
                    break

                if (
                    result.status in {_TURN_FAILED, _TURN_TIMED_OUT}
                    and self.config.session.on_turn_failure == "abort"
                ):
                    outcome = _SESSION_FAILED
                    break

        except asyncio.CancelledError:
            raise

        except Exception as exc:  # noqa: BLE001 - 会话失败要计数
            logger.warning(
                "[Stress] %s session %d 失败：%s",
                self.run_id,
                session_index,
                exc,
            )
            outcome = _SESSION_FAILED

        finally:
            if session_id is not None and (
                self.config.session.cleanup == "delete"
            ):
                await self._delete_session(session_id)

            if counted:
                self._metrics.session_finished(
                    outcome,
                    sampled_n=sampled_n,
                    actual_turns=actual_turns,
                    latency_ms=(time.monotonic() - session_started) * 1000.0,
                )

    async def _reserve_turn(self) -> bool:
        async with self._turn_lock:
            max_turns = self.config.max_turns

            if (
                max_turns is not None
                and self._global_turns >= max_turns
            ):
                return False

            self._global_turns += 1
            return True

    # --------------------------------------------------------
    # 单轮
    # --------------------------------------------------------

    async def _run_turn(
            self,
            session_id: UUID,
            session_index: int,
            turn_index: int,
            rng: random.Random,
            *,
            counted: bool,
    ) -> _TurnResult:
        started = time.monotonic()
        ttft_holder: list[float | None] = [None]

        if self._should_stop():
            return _TurnResult(
                status=_TURN_CANCELLED,
                latency_ms=0.0,
                submitted=False,
            )

        try:
            run_id = await self._create_run_with_retry(
                session_id,
                session_index,
                turn_index,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Stress] %s create_run 失败 session=%d turn=%d：%s",
                self.run_id,
                session_index,
                turn_index,
                exc,
            )
            return _TurnResult(
                status=_TURN_FAILED,
                latency_ms=(time.monotonic() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
                submitted=False,
            )

        # 只有 create_run 成功才计入 turns_started；这样指标里的
        # turns_started 与 agent_run 行数对得上。
        if counted:
            self._metrics.turn_started()

        timeout_seconds = self.config.request_timeout_ms / 1000.0

        consume = asyncio.create_task(
            self._consume_turn(
                run_id,
                started,
                ttft_holder,
            ),
            name=f"stress-turn:{self.run_id}:{session_index}:{turn_index}",
        )
        cancel_wait = asyncio.create_task(
            self.cancel_event.wait(),
            name=f"stress-cancel:{self.run_id}:{session_index}:{turn_index}",
        )

        try:
            done, pending = await asyncio.wait(
                {consume, cancel_wait},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # timeout
                consume.cancel()
                cancel_wait.cancel()
                await asyncio.gather(
                    consume,
                    cancel_wait,
                    return_exceptions=True,
                )
                await self._cancel_run(run_id)
                self._run_cursors.pop(str(run_id), None)
                return _TurnResult(
                    status=_TURN_TIMED_OUT,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    ttft_ms=ttft_holder[0],
                )

            if cancel_wait in done and not consume.done():
                consume.cancel()
                await asyncio.gather(consume, return_exceptions=True)
                await self._cancel_run(run_id)
                self._run_cursors.pop(str(run_id), None)
                return _TurnResult(
                    status=_TURN_CANCELLED,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    ttft_ms=ttft_holder[0],
                )

            cancel_wait.cancel()
            await asyncio.gather(cancel_wait, return_exceptions=True)

            outcome = consume.result()

            self._run_cursors.pop(str(run_id), None)

            def _result(status: str, error: str | None = None) -> _TurnResult:
                return _TurnResult(
                    status=status,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    ttft_ms=ttft_holder[0],
                    error=error,
                )

            if outcome.kind == "terminal":
                if outcome.status == "cancelled":
                    return _result(_TURN_CANCELLED, outcome.error)

                return _result(
                    _TURN_SUCCEEDED
                    if outcome.status == "succeeded"
                    else _TURN_FAILED,
                    outcome.error,
                )

            return _result(
                _TURN_FAILED,
                outcome.error or "事件流在终态事件之前关闭",
            )

        except asyncio.CancelledError:
            consume.cancel()
            cancel_wait.cancel()
            await asyncio.gather(
                consume,
                cancel_wait,
                return_exceptions=True,
            )
            await self._cancel_run(run_id)
            self._run_cursors.pop(str(run_id), None)
            raise

        except Exception as exc:  # noqa: BLE001
            self._run_cursors.pop(str(run_id), None)
            return _TurnResult(
                status=_TURN_FAILED,
                latency_ms=(time.monotonic() - started) * 1000.0,
                ttft_ms=ttft_holder[0],
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _consume_turn(
            self,
            run_id: UUID,
            turn_started: float,
            ttft_holder: list[float | None],
    ) -> _SseOutcome:
        resumes = 0

        while True:
            outcome = await self._consume_once(
                run_id,
                turn_started,
                ttft_holder,
            )

            if outcome.kind == "terminal":
                return outcome

            if outcome.kind == "hitl":
                if self.config.session.on_hitl == "abort":
                    await self._cancel_run(run_id)
                    return _SseOutcome(
                        kind="terminal",
                        status="failed",
                        error="hitl requested and on_hitl=abort",
                    )

                if resumes >= self._settings.max_hitl_resumes:
                    await self._cancel_run(run_id)
                    return _SseOutcome(
                        kind="terminal",
                        status="failed",
                        error="too many HITL resumes",
                    )

                await self._resume_run(
                    run_id,
                    outcome.interrupts,
                    approve=self.config.session.on_hitl == "auto_approve",
                )
                resumes += 1
                continue

            return outcome

    async def _consume_once(
            self,
            run_id: UUID,
            turn_started: float,
            ttft_holder: list[float | None],
    ) -> _SseOutcome:
        """按配置选择事件来源：Redis 轮询 / HTTP SSE。"""

        if (
                self._settings.event_transport == "redis"
                and self._redis_url
        ):
            return await self._consume_once_redis(
                run_id,
                turn_started,
                ttft_holder,
            )

        return await self._consume_once_sse(
            run_id,
            turn_started,
            ttft_holder,
        )

    async def _consume_once_sse(
            self,
            run_id: UUID,
            turn_started: float,
            ttft_holder: list[float | None],
    ) -> _SseOutcome:
        url = self._url(
            self.config.target.run_events_path,
            run_id=run_id,
        )
        headers = {
            **self._headers,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }

        interrupts: list[str] = []

        try:
            async with self._session.get(
                url,
                headers=headers,
            ) as response:

                if response.status >= 400:
                    body = await response.text()
                    logger.warning(
                        "[Stress] %s SSE HTTP %d（run=%s）：%s",
                        self.run_id,
                        response.status,
                        run_id,
                        body[:200],
                    )
                    return _SseOutcome(
                        kind="closed",
                        error=f"SSE HTTP {response.status}: {body[:300]}",
                    )

                event_name: str | None = None
                data_lines: list[str] = []

                async for raw in response.content:
                    line = raw.decode("utf-8").rstrip("\r\n")

                    if line == "":
                        outcome = self._handle_frame(
                            event_name,
                            data_lines,
                            turn_started,
                            ttft_holder,
                            interrupts,
                        )
                        event_name = None
                        data_lines = []

                        if outcome is not None:
                            return outcome

                        continue

                    if line.startswith(":"):
                        continue

                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                        continue

                    if line.startswith("data:"):
                        data_lines.append(
                            line[len("data:"):].lstrip()
                        )
                        continue

                # 流关闭前可能还有最后一帧（正常实现不会，但别丢）
                outcome = self._handle_frame(
                    event_name,
                    data_lines,
                    turn_started,
                    ttft_holder,
                    interrupts,
                )

                if outcome is not None:
                    return outcome

        except asyncio.CancelledError:
            raise

        except Exception as exc:  # noqa: BLE001
            return _SseOutcome(
                kind="closed",
                error=f"{type(exc).__name__}: {exc}",
            )

        if interrupts:
            return _SseOutcome(
                kind="hitl",
                interrupts=tuple(interrupts),
            )

        return _SseOutcome(
            kind="closed",
            error="SSE 关闭且没有终态事件",
        )

    async def _consume_once_redis(
            self,
            run_id: UUID,
            turn_started: float,
            ttft_holder: list[float | None],
    ) -> _SseOutcome:
        """
        直接从 Redis Stream 增量拉取运行事件。

        每 `event_poll_interval_ms`（默认 1000ms）做一次 `XRANGE`，游标存在
        `self._run_cursors` 里；只取新条目，读到终态事件就返回。

        与 SSE 的差别：**不占用长连接**。每次 XRANGE 从 Driver 的 Redis
        连接池借一条连接、查完即还，因此不会给服务端每个在途 run 压一条
        SSE + 一条 Redis 长连接。
        """

        key = stream_key(self._stream_prefix, str(run_id))

        cursor = self._run_cursors.get(str(run_id), "0")

        interrupts: list[str] = []

        interval = self._settings.event_poll_interval_ms / 1000.0

        try:
            redis_client = self._redis_client()

            while True:

                lower = "-" if cursor in ("", "0") else f"({cursor}"

                rows = await redis_client.xrange(
                    key,
                    lower,
                    "+",
                    count=500,
                )

                for entry_id, fields in rows:

                    cursor = (
                        entry_id
                        if isinstance(entry_id, str)
                        else entry_id.decode()
                    )

                    outcome = self._handle_redis_entry(
                        fields,
                        turn_started,
                        ttft_holder,
                        interrupts,
                    )

                    if outcome is not None:
                        self._run_cursors[str(run_id)] = cursor

                        if outcome.kind == "terminal":
                            self._run_cursors.pop(str(run_id), None)

                        return outcome

                self._run_cursors[str(run_id)] = cursor

                if interrupts:
                    return _SseOutcome(
                        kind="hitl",
                        interrupts=tuple(interrupts),
                    )

                # 这一批已经排干净：等下一个轮询周期。取满 500 条说明
                # 后面可能还有积压，立刻继续拉，不 sleep。
                if len(rows) >= 500:
                    continue

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            raise

        except Exception as exc:  # noqa: BLE001 —— 与 SSE 路径同一语义
            return _SseOutcome(
                kind="closed",
                error=(
                    "Redis 事件轮询失败："
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def _redis_client(self) -> Any:
        """Driver 自己的 Redis 客户端（共享池，惰性创建）。"""

        if self._poll_redis is None:

            from redis.asyncio import Redis

            self._poll_redis = Redis.from_url(
                self._redis_url,
                max_connections=self._settings.event_poll_max_connections,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30,
                decode_responses=True,
            )

        return self._poll_redis

    @staticmethod
    def _handle_redis_entry(
            fields: Mapping[str, Any],
            turn_started: float,
            ttft_holder: list[float | None],
            interrupts: list[str],
    ) -> _SseOutcome | None:
        """一条 Redis Stream 记录 -> 与 SSE 帧相同的 outcome。"""

        event = str(fields.get("event") or "")

        if not event:
            return None

        try:
            data = json.loads(str(fields.get("data") or "{}"))

        except ValueError:
            data = {}

        data = data if isinstance(data, Mapping) else {}

        if event == "message.delta":
            if ttft_holder[0] is None and data.get("delta"):
                ttft_holder[0] = (
                    time.monotonic() - turn_started
                ) * 1000.0
            return None

        if event == "message.completed":
            if ttft_holder[0] is None and data.get("content"):
                ttft_holder[0] = (
                    time.monotonic() - turn_started
                ) * 1000.0
            return None

        if event == "hitl.requested":
            interrupt_id = data.get("interrupt_id")
            if interrupt_id:
                interrupts.append(str(interrupt_id))
            return None

        if event == "run.completed":
            return _SseOutcome(kind="terminal", status="succeeded")

        if event == "run.failed":
            status = str(data.get("status") or "")

            error = (
                data.get("error_message")
                or data.get("error_type")
                or data
            )

            return _SseOutcome(
                kind="terminal",
                status="cancelled" if status == "cancelled" else "failed",
                error=str(error),
            )

        return None

    def _handle_frame(
            self,
            event_name: str | None,
            data_lines: list[str],
            turn_started: float,
            ttft_holder: list[float | None],
            interrupts: list[str],
    ) -> _SseOutcome | None:
        if not data_lines:
            return None

        raw = "\n".join(data_lines)

        try:
            payload = json.loads(raw)
        except ValueError:
            return None

        if not isinstance(payload, Mapping):
            return None

        event = str(payload.get("event") or event_name or "")
        data = payload.get("data")
        data = data if isinstance(data, Mapping) else {}

        if event == "message.delta":
            delta = data.get("delta")
            if ttft_holder[0] is None and delta:
                ttft_holder[0] = (
                    time.monotonic() - turn_started
                ) * 1000.0
            return None

        if event == "message.completed":
            if ttft_holder[0] is None and data.get("content"):
                ttft_holder[0] = (
                    time.monotonic() - turn_started
                ) * 1000.0
            return None

        if event == "hitl.requested":
            interrupt_id = data.get("interrupt_id")
            if interrupt_id:
                interrupts.append(str(interrupt_id))
            return None

        if event == "run.completed":
            return _SseOutcome(
                kind="terminal",
                status="succeeded",
            )

        if event == "run.failed":
            error = payload.get("error")
            return _SseOutcome(
                kind="terminal",
                status="failed",
                error=str(error or data or "run.failed"),
            )

        return None

    # --------------------------------------------------------
    # HTTP 调用
    # --------------------------------------------------------

    async def _create_session(self) -> UUID:
        url = self._url(self.config.target.create_session_path)
        payload = {
            "agent_id": self.config.session.agent_id,
            "metadata": self.config.session.metadata,
        }

        async with self._session.post(
            url,
            json=payload,
            headers=self._headers,
        ) as response:

            text = await response.text()

            if response.status >= 400:
                raise _HttpStatusError(
                    response.status,
                    f"create session HTTP {response.status}: {text[:300]}",
                )

            data = json.loads(text)
            session_id = data.get("session_id") or data.get("id")

            if not session_id:
                # 这不是可恢复错误：响应形状不对，重试只会得到同一个结果。
                raise RuntimeError(
                    f"create session 响应里没有 session_id：{text[:300]}"
                )

            return UUID(str(session_id))

    async def _create_session_with_retry(self) -> UUID:
        """
        建会话，并对可恢复错误重试。

        500 最常见的原因是后端业务 DB 连接池耗尽（pool checkout
        timeout）。它不是业务错误，等池里的连接归还后重试通常就能成功；
        4xx 不重试。
        """

        attempts = self._settings.session_create_max_attempts
        base_delay = self._settings.session_create_backoff_seconds

        for attempt in range(1, attempts + 1):

            try:
                return await self._create_session()

            except _HttpStatusError as exc:

                retryable = exc.status in {
                    408, 425, 429, 500, 502, 503, 504,
                }

                if not retryable or attempt >= attempts:
                    raise

                delay = base_delay * (2 ** (attempt - 1))
                delay *= random.uniform(0.75, 1.25)

                if exc.status == 500:
                    logger.warning(
                        "[Stress] %s create session HTTP 500"
                        "（常见原因：后端业务 DB 连接池耗尽 / "
                        "pool checkout timeout），%.2fs 后重试 "
                        "attempt=%d/%d。",
                        self.run_id,
                        delay,
                        attempt,
                        attempts,
                    )
                else:
                    logger.info(
                        "[Stress] %s create session HTTP %d，"
                        "%.2fs 后重试 attempt=%d/%d。",
                        self.run_id,
                        exc.status,
                        delay,
                        attempt,
                        attempts,
                    )

                await asyncio.sleep(delay)

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:

                if attempt >= attempts:
                    raise

                delay = base_delay * (2 ** (attempt - 1))
                delay *= random.uniform(0.75, 1.25)

                logger.info(
                    "[Stress] %s create session 网络错误（%s），"
                    "%.2fs 后重试 attempt=%d/%d。",
                    self.run_id,
                    type(exc).__name__,
                    delay,
                    attempt,
                    attempts,
                )

                await asyncio.sleep(delay)

        raise RuntimeError("create session 重试耗尽")

    async def _create_run_with_retry(
            self,
            session_id: UUID,
            session_index: int,
            turn_index: int,
    ) -> UUID:
        """
        创建 Run，并对可恢复错误重试。

        409 是上一轮终态事件已到、但 dispatch handler 还没放掉 thread
        lock 的竞态；等待几百毫秒重试即可。503/429/5xx 也按短退避重试。
        """

        attempts = 3

        for attempt in range(1, attempts + 1):
            try:
                return await self._create_run(
                    session_id,
                    session_index,
                    turn_index,
                )

            except _HttpStatusError as exc:
                retryable = exc.status in {
                    409, 425, 429, 500, 502, 503, 504,
                }

                if not retryable or attempt >= attempts:
                    raise

                delay = 0.2 * attempt
                logger.info(
                    "[Stress] %s create_run HTTP %d，%.1fs 后重试"
                    "（session=%d turn=%d attempt=%d/%d）",
                    self.run_id,
                    exc.status,
                    delay,
                    session_index,
                    turn_index,
                    attempt,
                    attempts,
                )
                await asyncio.sleep(delay)

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= attempts:
                    raise

                delay = 0.2 * attempt
                logger.info(
                    "[Stress] %s create_run 网络错误（%s），"
                    "%.1fs 后重试（attempt=%d/%d）",
                    self.run_id,
                    type(exc).__name__,
                    delay,
                    attempt,
                    attempts,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("create_run 重试耗尽")

    async def _create_run(
            self,
            session_id: UUID,
            session_index: int,
            turn_index: int,
    ) -> UUID:
        url = self._url(
            self.config.target.create_run_path,
            session_id=session_id,
        )
        payload = {
            "input": self._build_input(
                session_index,
                turn_index,
            ),
            "trace_id": (
                f"stress:{self.run_id}:{session_index}"
            ),
        }

        async with self._session.post(
            url,
            json=payload,
            headers=self._headers,
        ) as response:

            text = await response.text()

            if response.status >= 400:
                raise _HttpStatusError(
                    response.status,
                    f"create run HTTP {response.status}: {text[:300]}",
                )

            data = json.loads(text)
            run_id = data.get("run_id") or data.get("id")

            if not run_id:
                raise RuntimeError(
                    f"create run 响应里没有 run_id：{text[:300]}"
                )

            return UUID(str(run_id))

    async def _resume_run(
            self,
            run_id: UUID,
            interrupt_ids: tuple[str, ...],
            *,
            approve: bool,
    ) -> None:
        url = self._url(
            self.config.target.resume_run_path,
            run_id=run_id,
        )
        decision = "approve" if approve else "reject"
        payload = {
            "answers": [
                {
                    "interrupt_id": interrupt_id,
                    "data": {"decision": decision},
                }
                for interrupt_id in interrupt_ids
            ]
        }

        async with self._session.post(
            url,
            json=payload,
            headers=self._headers,
        ) as response:

            text = await response.text()

            if response.status >= 400:
                raise RuntimeError(
                    f"resume HTTP {response.status}: {text[:300]}"
                )

    async def _cancel_run(self, run_id: UUID) -> None:
        """
        尽力取消一个在途 run。

        限流 + 对 5xx / 网络错误做短重试：
        Campaign 收尾时可能同时取消数百个 run，全部并发会把后端打到
        500（包括 DB 连接被服务端关闭后的重连失败）。取消失败本身不影响
        指标，但大量 500 会污染日志、拖慢收尾。
        """

        url = self._url(
            self.config.target.cancel_run_path,
            run_id=run_id,
        )

        attempts = 3

        for attempt in range(1, attempts + 1):

            try:
                async with self._cancel_slots:
                    async with self._session.post(
                        url,
                        headers=self._headers,
                    ) as response:
                        await response.read()
                        status = response.status

            except asyncio.CancelledError:
                raise

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:

                if attempt >= attempts:
                    logger.debug(
                        "[Stress] 取消 run %s 失败（%s，已忽略）。",
                        run_id,
                        type(exc).__name__,
                    )
                    return

                await asyncio.sleep(0.2 * attempt)
                continue

            if status < 500:
                if status >= 400:
                    logger.debug(
                        "[Stress] 取消 run %s 返回 HTTP %d（忽略）。",
                        run_id,
                        status,
                    )
                return

            if attempt >= attempts:
                logger.debug(
                    "[Stress] 取消 run %s 连续 HTTP %d（已忽略）。",
                    run_id,
                    status,
                )
                return

            await asyncio.sleep(0.2 * attempt)

    async def _delete_session(self, session_id: UUID) -> None:
        url = self._url(
            self.config.target.delete_session_path,
            session_id=session_id,
        )

        try:
            async with self._session.delete(
                url,
                headers=self._headers,
            ) as response:
                await response.read()
        except Exception:
            logger.debug(
                "[Stress] 删除 session %s 失败（忽略）。",
                session_id,
                exc_info=True,
            )

    # --------------------------------------------------------
    # 工具
    # --------------------------------------------------------

    def _build_input(
            self,
            session_index: int,
            turn_index: int,
    ) -> dict[str, Any]:
        template = self.config.turn.input_template

        if template is not None:
            return copy.deepcopy(template)

        prefix = (
            f"[stress {self.run_id} session={session_index} "
            f"turn={turn_index}] "
        )
        size = self.config.turn.input_size_chars
        fill = ("mock input " * ((size // 12) + 1))[
            : max(0, size - len(prefix))
        ]

        return {"text": prefix + fill}

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self._resolve_token()

        if token:
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def _resolve_token(self) -> str | None:
        auth = self.config.auth
        env_name = (
            auth.token_env
            if auth is not None
            else self._settings.default_token_env
        )

        token = os.environ.get(env_name)
        source = "env" if token else None
        token_path = _token_file_path(self._settings.token_file)

        if not token:
            token = _read_token_file(token_path)
            source = "file" if token else None

        if token:
            logger.info(
                "[Stress] token 已加载：来源=%s，env=%s，backend pid=%s。",
                source,
                env_name,
                os.getpid(),
            )

        else:
            logger.warning(
                "[Stress] token 环境变量 %r 为空，文件兜底 %s 也不存在。"
                "backend pid=%s；环境变量必须在启动后端进程之前注入。",
                env_name,
                token_path,
                os.getpid(),
            )

        if auth is not None and not token:
            raise ValueError(
                f"Stress Runner 需要 Bearer token，但环境变量 "
                f"{env_name!r} 为空，文件兜底 {token_path} 也不存在"
                f"（backend pid={os.getpid()}）。"
                "请确认 token 注入到了启动后端的那个进程。"
            )

        return token

    def _validate_target(self) -> None:
        host = urlparse(self.config.target.base_url).hostname

        if not host:
            raise ValueError(
                f"target.base_url 无法解析出 host："
                f"{self.config.target.base_url!r}"
            )

        allowed = self._settings.target_allowlist

        if allowed and host not in allowed:
            raise ValueError(
                f"target.base_url 的 host {host!r} 不在 allowlist "
                f"{allowed} 内。通过 app/config/stress_eval.yaml 的 "
                "driver.target_allowlist 显式放行。"
            )

    def _url(
            self,
            path: str,
            **values: Any,
    ) -> str:
        return self._base_url + path.format(**values)

    def _should_stop(self) -> bool:
        return (
            self._stopped
            or self.cancel_event.is_set()
            or time.monotonic() >= self.deadline
        )

    async def _sleep_cancellable(self, seconds: float) -> bool:
        """睡 seconds；取消返回 False。"""

        if seconds <= 0:
            return not self.cancel_event.is_set()

        try:
            await asyncio.wait_for(
                self.cancel_event.wait(),
                timeout=seconds,
            )
            return False

        except asyncio.TimeoutError:
            return not self._should_stop()


# ============================================================
# Runner
# ============================================================


class HttpStressRunner:
    """真实 Driver 的线程安全门面。"""

    def __init__(
            self,
            reporter: StressRunReporter,
            *,
            redis_url: str | None = None,
            settings: StressDriverSettings | None = None,
    ) -> None:

        self._reporter = reporter
        self._redis_url = redis_url
        self._settings = (
            settings
            or load_stress_eval_config().driver
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._campaigns: dict[UUID, _Campaign] = {}
        self._cancel_requested: set[UUID] = set()
        self._stopping = False

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------

    def start(self) -> None:
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_thread,
                name="stress-driver",
                daemon=True,
            )
            self._thread.start()

            if not self._ready.wait(timeout=10.0):
                raise RuntimeError("Stress Driver 启动超时。")

    def schedule(
            self,
            run_id: UUID,
            config: StressRunCreate,
    ) -> None:
        self.start()

        with self._state_lock:
            if self._stopping:
                logger.warning(
                    "[Stress] Driver 正在停止，忽略新运行 %s。",
                    run_id,
                )
                return

            if run_id in self._cancel_requested:
                self._cancel_requested.discard(run_id)
                logger.info(
                    "[Stress] %s 在启动前已被取消，不创建 Campaign。",
                    run_id,
                )
                return

        loop = self._loop

        if loop is None:
            raise RuntimeError("Stress Driver loop 不可用。")

        loop.call_soon_threadsafe(
            self._spawn_campaign,
            run_id,
            config.model_copy(deep=True),
        )

    def cancel(self, run_id: UUID) -> None:
        with self._state_lock:
            campaign = self._campaigns.get(run_id)

            if campaign is None:
                self._cancel_requested.add(run_id)
                return

        loop = self._loop

        if loop is None:
            return

        loop.call_soon_threadsafe(campaign.cancel_event.set)

    def stop(self) -> None:
        with self._state_lock:
            self._stopping = True

        loop = self._loop

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._shutdown)

        thread = self._thread

        if thread is not None:
            thread.join(timeout=15.0)

        self._thread = None
        self._loop = None

    # --------------------------------------------------------
    # 线程
    # --------------------------------------------------------

    def _run_thread(self) -> None:
        # ⚠️ Driver 线程**不需要** SelectorEventLoop
        # ------------------------------------------
        # 主应用必须用 Selector 是因为 psycopg 异步不支持 Proactor；但
        # Driver 线程只跑 aiohttp + redis.asyncio，所有数据库写入都通过
        # StressRunReporter 投回主 loop（见 stress_reporter.py 的说明）。
        # 在 Windows 上继续用 Selector 会白白继承 select() 的 512 fd 上限：
        # 并发会话一多，这个线程会先于服务端崩掉，Campaign 停在 collecting。
        # Proactor（IOCP）没有这个限制。
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.SelectorEventLoop()

        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()

        try:
            loop.run_forever()

        finally:
            pending = asyncio.all_tasks(loop)

            for task in pending:
                task.cancel()

            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            loop.close()

    def _shutdown(self) -> None:
        async def _stop() -> None:
            with self._state_lock:
                campaigns = list(self._campaigns.values())

            for campaign in campaigns:
                campaign.cancel_event.set()

            current = asyncio.current_task()
            tasks = [
                task
                for task in asyncio.all_tasks()
                if task is not current
            ]

            if tasks:
                # 给 Campaign 一点时间把 incomplete 快照写回主 loop。
                await asyncio.wait(tasks, timeout=10.0)

            asyncio.get_running_loop().stop()

        asyncio.ensure_future(_stop())

    # --------------------------------------------------------
    # Campaign 入口
    # --------------------------------------------------------

    def _spawn_campaign(
            self,
            run_id: UUID,
            config: StressRunCreate,
    ) -> None:
        asyncio.ensure_future(
            self._campaign_entry(run_id, config)
        )

    async def _campaign_entry(
            self,
            run_id: UUID,
            config: StressRunCreate,
    ) -> None:
        connector = aiohttp.TCPConnector(
            limit=max(64, self._settings.http_max_connections),
            limit_per_host=max(64, self._settings.http_max_connections),
            ttl_dns_cache=30,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self._settings.connect_timeout_seconds,
            sock_read=None,
        )

        async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                trust_env=False,
        ) as session:

            with self._state_lock:
                if run_id in self._cancel_requested:
                    self._cancel_requested.discard(run_id)
                    logger.info(
                        "[Stress] %s 在 Campaign 创建前已取消。",
                        run_id,
                    )
                    return

            try:
                campaign = _Campaign(
                    run_id=run_id,
                    config=config,
                    reporter=self._reporter,
                    session=session,
                    settings=self._settings,
                    redis_url=self._redis_url,
                )

            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[Stress] Campaign %s 构造失败。",
                    run_id,
                )
                await self._reporter.mark_failed(
                    run_id,
                    error_code="runner_bootstrap_failed",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                return

            with self._state_lock:
                self._campaigns[run_id] = campaign

            try:
                await campaign.run()

            except asyncio.CancelledError:
                campaign.cancel_event.set()
                raise

            except Exception:
                logger.exception(
                    "[Stress] Campaign %s 未捕获异常。",
                    run_id,
                )

            finally:
                try:
                    await campaign.aclose()
                except Exception:
                    logger.exception(
                        "[Stress] Campaign %s 清理失败。",
                        run_id,
                    )

                with self._state_lock:
                    self._campaigns.pop(run_id, None)
                self._cancel_requested.discard(run_id)
