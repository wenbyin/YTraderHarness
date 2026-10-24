"""
多轮会话模拟器的纯函数测试：N 抽样、失败策略、预热与 max_turns。

这些用例不碰数据库、不发出网络请求 —— 验的是 snapshots 的推导逻辑，
不是 FastAPI 的序列化（那在 tests/api/test_stress_runs_routes.py）。
"""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from app.services.stress.application.simulated_runner import (
    build_simulated_metrics,
    sample_turn_count,
)
from app.services.stress.domain.schemas.stress import (
    ConcurrencyLoad,
    MockModelConfig,
    SessionConfig,
    SessionRateLoad,
    StressRunCreate,
    TargetConfig,
    TurnCountConfig,
)


def _mock_model(**overrides: Any) -> MockModelConfig:
    payload: dict[str, Any] = {
        "base_latency_ms": 100,
        "latency_jitter_ms": 20,
        "ttft_ms": 50,
        "output_tokens_min": 20,
        "output_tokens_max": 40,
        "tokens_per_second": 50,
    }
    payload.update(overrides)

    return MockModelConfig(**payload)


def _config(
        *,
        turns: TurnCountConfig | None = None,
        session: dict[str, Any] | None = None,
        load: Any = None,
        total_sessions: int = 3,
        duration_seconds: int = 60,
        warmup_seconds: int = 0,
        max_turns: int | None = None,
        mock_model: MockModelConfig | None = None,
) -> StressRunCreate:

    session_payload: dict[str, Any] = {
        "agent_id": "main",
        "turns": turns or TurnCountConfig(mean=3, stddev=0),
    }
    session_payload.update(session or {})

    return StressRunCreate(
        target=TargetConfig(base_url="http://127.0.0.1:8080"),
        session=SessionConfig(**session_payload),
        load=load or ConcurrencyLoad(
            mode="concurrency", concurrency=5,
        ),
        duration_seconds=duration_seconds,
        warmup_seconds=warmup_seconds,
        total_sessions=total_sessions,
        max_turns=max_turns,
        request_timeout_ms=30_000,
        seed=42,
        mock_model=mock_model or _mock_model(),
    )


# ============================================================
# N 抽样
# ============================================================


def test_truncated_normal_stays_in_range_and_is_reproducible() -> None:
    turns = TurnCountConfig(mean=15.5, stddev=7.0)

    first = random.Random("42:0")
    second = random.Random("42:0")

    samples_a = [sample_turn_count(turns, first) for _ in range(200)]
    samples_b = [sample_turn_count(turns, second) for _ in range(200)]

    assert samples_a == samples_b
    assert all(1 <= value <= 30 for value in samples_a)
    # 分布不能退化成常数（除非配置本来如此）。
    assert len(set(samples_a)) > 5


def test_stddev_zero_is_deterministic() -> None:
    assert sample_turn_count(
        TurnCountConfig(mean=3, stddev=0),
        random.Random(1),
    ) == 3

    # 15.5 按四舍五入取 16，而不是银行家舍入的 16/16 交替。
    assert sample_turn_count(
        TurnCountConfig(mean=15.5, stddev=0),
        random.Random(1),
    ) == 16

    assert sample_turn_count(
        TurnCountConfig(mean=15.5, stddev=0, rounding="floor"),
        random.Random(1),
    ) == 15

    assert sample_turn_count(
        TurnCountConfig(mean=15.5, stddev=0, rounding="ceil"),
        random.Random(1),
    ) == 16


# ============================================================
# 失败策略
# ============================================================


def test_abort_on_turn_failure_fails_the_session() -> None:
    config = _config(
        session={"on_turn_failure": "abort"},
        mock_model=_mock_model(error_rate=1.0),
    )

    metrics = build_simulated_metrics(uuid4(), config)

    assert metrics.sessions_started == 3
    assert metrics.sessions_failed == 3
    assert metrics.sessions_completed == 0
    # 每个会话都在第 1 轮失败后终止。
    assert metrics.turns_started == 3
    assert metrics.turns_failed == 3
    assert metrics.turns_succeeded == 0
    assert metrics.turns_completed == 3
    assert metrics.turn_error_rate == 1.0
    assert metrics.turns_per_session["min"] == 1.0


def test_continue_on_turn_failure_completes_planned_turns() -> None:
    config = _config(
        session={"on_turn_failure": "continue"},
        mock_model=_mock_model(error_rate=1.0),
    )

    metrics = build_simulated_metrics(uuid4(), config)

    assert metrics.sessions_started == 3
    assert metrics.sessions_completed == 3
    assert metrics.sessions_failed == 0
    # 3 个会话 × 3 轮，全部失败但都跑满。
    assert metrics.turns_started == 9
    assert metrics.turns_failed == 9
    assert metrics.turns_completed == 9
    assert metrics.turn_error_rate == 1.0


# ============================================================
# 预热与 max_turns
# ============================================================


def test_warmup_excludes_sessions_started_inside_the_window() -> None:
    config = _config(
        load=SessionRateLoad(mode="session_rate", session_rate=0.1),
        total_sessions=5,
        duration_seconds=100,
        warmup_seconds=25,
    )

    metrics = build_simulated_metrics(uuid4(), config)

    # 接纳时刻为 0/10/20/30/40 秒；前三个落在 25 秒预热窗口内。
    assert metrics.sessions_started == 2
    assert metrics.sampled_turns_per_session["count"] == 2


def test_max_turns_stops_starting_new_turns() -> None:
    config = _config(
        turns=TurnCountConfig(mean=5, stddev=0),
        total_sessions=5,
        max_turns=7,
    )

    metrics = build_simulated_metrics(uuid4(), config)

    assert metrics.sessions_started == 5
    assert metrics.sessions_completed == 1
    assert metrics.sessions_abandoned == 4
    assert metrics.turns_started == 7
    assert metrics.turns_completed == 7
    assert metrics.turns_per_session["count"] == 5
