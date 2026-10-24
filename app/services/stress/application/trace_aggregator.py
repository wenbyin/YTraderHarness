"""
Trace Event 聚合器：把 `trace:{trace_id}:events` 汇总成运行级指标。

边界
----
- 只读 Redis Trace Stream，不写 Trace。
- 不保存逐会话 / 逐轮明细；输出只包含有限枚举维度（node_name /
  tool_name）与聚合计数/分布。
- 在 Driver 线程里运行，使用**独立的 redis.asyncio 客户端**，不碰主
  loop 的连接池。

聚合时机
--------
Phase 2 采用“Campaign 结束后一次性读取”。同一虚拟会话的 trace_id 是
`stress:{run_id}:{session_index}`；Campaign 记录全部 trace_id，结束后
等一个很短的 flush 窗口，再逐个 XRANGE。

为什么不是逐事件消费
------------------
Trace 事件本身是旁路：数量大、TTL 短。一次性读取让实现简单、可重放，
也不会让 Aggregator 的消费速度反过来影响 Agent 主链路。代价是运行中
的 `/metrics` 只有 Runner 侧指标；Trace 字段在 final 快照里出现。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Iterable, Mapping

import redis.asyncio as aioredis

from app.common.event import EventType, trace_stream_key

from .metrics import DistributionBuilder, distribution

__all__ = ["TraceAggregate", "TraceAggregator"]

logger = logging.getLogger(__name__)

#: 同时读取的 trace stream 数。读取本身很轻，但不要一次开上千条。
_READ_CONCURRENCY = 20


class TraceAggregate:
    """一组 Trace Event 的聚合结果。"""

    def __init__(self) -> None:
        self.events_seen = 0
        self.events_deduplicated = 0
        self.events_unmatched = 0
        self.events_late = 0

        self._event_ids: set[str] = set()

        self.counts: dict[str, int] = {
            "run_span_count": 0,
            "node_span_count": 0,
            "model_call_count": 0,
            "tool_call_count": 0,
            "context_build_count": 0,
            "fanout_count": 0,
        }

        self.model_retry_count = 0
        self.model_error_count = 0
        self.tool_retry_count = 0
        self.tool_error_count = 0

        self.model_duration = DistributionBuilder()
        self.node_duration: dict[str, DistributionBuilder] = {}
        self.tool_duration: dict[str, DistributionBuilder] = {}

        self.model_input_tokens_sum = 0
        self.model_output_tokens_sum = 0

        #: run_id -> [input_tokens, output_tokens]
        self._tokens_by_run: dict[str, list[int]] = {}

        self.fanout_branch_count = 0
        self.fanout_failed_branch_count = 0

    @property
    def has_data(self) -> bool:
        return self.events_seen > 0

    # --------------------------------------------------------
    # 合并事件
    # --------------------------------------------------------

    def merge_events(
            self,
            events: Iterable[Mapping[str, Any]],
            *,
            expected_trace_id: str | None = None,
    ) -> None:
        """把一批已解析的 TraceEvent 合并进来。"""

        for event in events:
            self.events_seen += 1

            event_id = str(event.get("event_id") or "")

            if event_id:
                if event_id in self._event_ids:
                    self.events_deduplicated += 1
                    continue

                self._event_ids.add(event_id)

            if (
                expected_trace_id is not None
                and event.get("trace_id") != expected_trace_id
            ):
                self.events_unmatched += 1
                continue

            self._merge_one(event)

    def _merge_one(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        run_id = str(event.get("run_id") or "")
        duration_ms = _as_float(event.get("duration_ms"))

        if event_type == EventType.RUN_STARTED.value:
            self.counts["run_span_count"] += 1
            return

        if event_type == EventType.NODE_STARTED.value:
            self.counts["node_span_count"] += 1
            return

        if event_type == EventType.CONTEXT_BUILD_STARTED.value:
            self.counts["context_build_count"] += 1
            return

        if event_type == EventType.MODEL_STARTED.value:
            self.counts["model_call_count"] += 1
            return

        if event_type == EventType.MODEL_COMPLETED.value:
            if duration_ms is not None:
                self.model_duration.add(duration_ms)

            input_tokens = _as_int(payload.get("input_tokens"))
            output_tokens = _as_int(payload.get("output_tokens"))

            self.model_input_tokens_sum += input_tokens
            self.model_output_tokens_sum += output_tokens

            bucket = self._tokens_by_run.setdefault(run_id, [0, 0])
            bucket[0] += input_tokens
            bucket[1] += output_tokens
            return

        if event_type == EventType.MODEL_FAILED.value:
            self.model_error_count += 1
            return

        if event_type in {
            EventType.MODEL_RETRY_SCHEDULED.value,
            EventType.MODEL_FALLBACK.value,
        }:
            self.model_retry_count += 1
            return

        if event_type in {
            EventType.NODE_COMPLETED.value,
            EventType.NODE_FAILED.value,
        }:
            name = str(
                payload.get("node_name")
                or event.get("span_name")
                or "unknown"
            )

            if duration_ms is not None:
                self.node_duration.setdefault(
                    name,
                    DistributionBuilder(),
                ).add(duration_ms)
            return

        if event_type == EventType.TOOL_STARTED.value:
            self.counts["tool_call_count"] += 1
            return

        if event_type in {
            EventType.TOOL_COMPLETED.value,
            EventType.TOOL_FAILED.value,
        }:
            name = str(
                payload.get("tool_name")
                or event.get("span_name")
                or "unknown"
            )

            if event_type == EventType.TOOL_FAILED.value:
                self.tool_error_count += 1

            if duration_ms is not None:
                self.tool_duration.setdefault(
                    name,
                    DistributionBuilder(),
                ).add(duration_ms)
            return

        if event_type == EventType.TOOL_RETRY_SCHEDULED.value:
            self.tool_retry_count += 1
            return

        if event_type == EventType.FANOUT_STARTED.value:
            self.counts["fanout_count"] += 1
            return

        if event_type == EventType.FANOUT_BRANCH_CREATED.value:
            self.fanout_branch_count += 1
            return

        if event_type == EventType.FANOUT_COMPLETED.value:
            self.fanout_failed_branch_count += _as_int(
                payload.get("failed_branch_count")
            )
            return

        if event_type == EventType.FANOUT_FAILED.value:
            self.fanout_failed_branch_count += 1
            return

    # --------------------------------------------------------
    # 快照
    # --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """输出与 StressRunMetricsRecord / _assembly 对齐的字段。"""

        trace_counts = dict(self.counts)
        trace_counts.update({
            "model_retry_count": self.model_retry_count,
            "model_error_count": self.model_error_count,
            "tool_retry_count": self.tool_retry_count,
            "tool_error_count": self.tool_error_count,
        })

        input_per_turn = [
            float(value[0]) for value in self._tokens_by_run.values()
        ]
        output_per_turn = [
            float(value[1]) for value in self._tokens_by_run.values()
        ]

        return {
            "trace_counts": trace_counts,
            "model_metrics": {
                "call_count": self.counts["model_call_count"],
                "retry_count": self.model_retry_count,
                "error_count": self.model_error_count,
                "duration_ms": self.model_duration.snapshot(),
                "input_tokens_per_turn": distribution(input_per_turn),
                "output_tokens_per_turn": distribution(output_per_turn),
            },
            "node_metrics": {
                name: builder.snapshot()
                for name, builder in self.node_duration.items()
            },
            "tool_metrics": {
                name: builder.snapshot()
                for name, builder in self.tool_duration.items()
            },
            "fanout_metrics": {
                "count": self.counts["fanout_count"],
                "branch_count": self.fanout_branch_count,
                "failed_branch_count": self.fanout_failed_branch_count,
            },
            "model_input_tokens_sum": self.model_input_tokens_sum,
            "model_output_tokens_sum": self.model_output_tokens_sum,
            "events_seen": self.events_seen,
            "events_deduplicated": self.events_deduplicated,
            "events_unmatched": self.events_unmatched,
            "events_late": self.events_late,
        }


# ============================================================
# Redis 读取
# ============================================================


class TraceAggregator:
    """按 trace_id 读取 Redis Trace Stream。"""

    def __init__(
            self,
            redis_url: str | None,
            *,
            read_concurrency: int = _READ_CONCURRENCY,
    ) -> None:
        self._redis_url = redis_url
        self._client: aioredis.Redis | None = None
        self._read_concurrency = max(1, read_concurrency)

    @property
    def enabled(self) -> bool:
        return bool(self._redis_url)

    async def collect_many(
            self,
            trace_ids: Iterable[str],
    ) -> TraceAggregate:
        aggregate = TraceAggregate()

        if not self.enabled:
            return aggregate

        ids = [str(trace_id) for trace_id in trace_ids if trace_id]

        if not ids:
            return aggregate

        semaphore = asyncio.Semaphore(self._read_concurrency)

        async def _read(trace_id: str) -> tuple[str, list[dict[str, Any]]]:
            async with semaphore:
                return trace_id, await self._read_trace(trace_id)

        results = await asyncio.gather(
            *(_read(trace_id) for trace_id in ids),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                logger.warning(
                    "[Stress] 读取 Trace Stream 失败：%s",
                    result,
                )
                continue

            trace_id, events = result
            aggregate.merge_events(
                events,
                expected_trace_id=trace_id,
            )

        return aggregate

    async def _client_instance(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30,
            )

        return self._client

    async def _read_trace(
            self,
            trace_id: str,
    ) -> list[dict[str, Any]]:
        client = await self._client_instance()
        entries = await client.xrange(
            trace_stream_key(trace_id),
            min="-",
            max="+",
        )

        events: list[dict[str, Any]] = []

        for _, fields in entries:
            raw = fields.get("data")

            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    continue
            elif isinstance(raw, Mapping):
                parsed = dict(raw)
            else:
                continue

            if isinstance(parsed, Mapping):
                events.append(dict(parsed))

        return events

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ============================================================
# 小工具
# ============================================================


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
