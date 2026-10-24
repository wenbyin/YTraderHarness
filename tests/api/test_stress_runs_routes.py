"""
压测运行端点的线上形状（多轮会话版）。

⚠️ 为什么必须真打一次 HTTP
------------------------
服务层用例拿回来的已经是 DTO；字段名写错、类型不对、被 response_model
挡掉，服务层用例都不会红。决定前端看到什么的是 FastAPI 的序列化这一步。

装置刻意不碰数据库：``Service[契约]`` 只在请求时调一次
get_service_factory()，替成一个不需要事务的假工厂即可。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

import app.services.dependencies as dependencies_module
from app.api.v1.router import protected_router
from app.api.v1.stress_runs import router as stress_runs_router
from app.common.auth import current_user
from app.common.exceptions import ConflictError, NotFoundError
from app.services.stress.domain.schemas.stress import (
    AggregationStatus,
    ConcurrencyLoad,
    Distribution,
    ModelMetrics,
    MockModelConfig,
    SessionConfig,
    SessionMetrics,
    StressRun,
    StressRunCreate,
    StressRunMetrics,
    StressRunPage,
    StressRunStatus,
    TargetConfig,
    TurnCountConfig,
    TurnMetrics,
)

_RUN_ID = uuid4()

_NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)

#: 五个端点：(方法, 路径)。全部必须挂在受保护的路由树上。
_PROTECTED_PATHS = (
    ("post", "/api/v1/stress-runs"),
    ("get", "/api/v1/stress-runs"),
    ("get", "/api/v1/stress-runs/{run_id}"),
    ("get", "/api/v1/stress-runs/{run_id}/metrics"),
    ("post", "/api/v1/stress-runs/{run_id}/cancel"),
)


def _config(**overrides: Any) -> StressRunCreate:
    payload: dict[str, Any] = {
        "name": "前端联调压测",
        "target": TargetConfig(
            base_url="http://127.0.0.1:8080",
        ),
        "session": SessionConfig(
            agent_id="main",
            turns=TurnCountConfig(mean=15.5, stddev=7.0),
        ),
        "load": ConcurrencyLoad(mode="concurrency", concurrency=10),
        "duration_seconds": 600,
        "request_timeout_ms": 120_000,
        "total_sessions": 20,
        "seed": 42,
        "mock_model": MockModelConfig(
            base_latency_ms=100,
            ttft_ms=50,
            output_tokens_min=20,
            output_tokens_max=40,
            tokens_per_second=50,
        ),
    }
    payload.update(overrides)

    return StressRunCreate(**payload)


def _run(
        status: StressRunStatus = StressRunStatus.QUEUED,
        config: StressRunCreate | None = None,
) -> StressRun:

    return StressRun(
        run_id=_RUN_ID,
        name="前端联调压测",
        status=status,
        config=config or _config(),
        created_at=_NOW,
    )


def _metrics() -> StressRunMetrics:

    return StressRunMetrics(
        run_id=_RUN_ID,
        aggregation_status=AggregationStatus.FINAL,
        aggregation_algorithm="simulated-v1",
        finalized_at=_NOW,
        sessions=SessionMetrics(
            started=10,
            completed=9,
            failed=0,
            abandoned=1,
            completion_rate=0.9,
            achieved_sessions_per_second=0.015,
            duration_ms=Distribution(
                count=10,
                mean=155_000.0,
                p50=155_000.0,
                p95=294_500.0,
            ),
            sampled_turns_per_session=Distribution(
                count=10,
                mean=15.5,
                min=2.0,
                max=29.0,
                p50=15.5,
            ),
            turns_per_session=Distribution(
                count=10,
                mean=15.0,
                min=2.0,
                max=28.0,
                p50=15.0,
            ),
        ),
        turns=TurnMetrics(
            started=150,
            completed=149,
            succeeded=149,
            failed=0,
            timed_out=0,
            cancelled=0,
            achieved_turns_per_second=1.5,
            error_rate=0.0,
            latency_ms=Distribution(
                count=149,
                mean=10_250.0,
                min=8_000.0,
                max=20_000.0,
                p50=10_250.0,
                p90=14_500.0,
                p95=16_500.0,
                p99=20_000.0,
            ),
            time_to_first_output_ms=Distribution(
                count=149,
                mean=1_050.0,
                p95=1_700.0,
            ),
            first_turn_latency_ms=Distribution(
                count=10,
                mean=12_000.0,
            ),
            later_turn_latency_ms=Distribution(
                count=139,
                mean=10_100.0,
            ),
            by_index={
                "1": Distribution(count=10, mean=12_000.0),
                "2": Distribution(count=10, mean=10_500.0),
            },
        ),
        trace_counts={
            "run_span_count": 150,
            "model_call_count": 300,
        },
        model=ModelMetrics(
            call_count=300,
            retry_count=1,
            error_count=0,
            input_tokens_sum=96_000,
            output_tokens_sum=18_000,
            duration_ms=Distribution(count=300, mean=5_100.0),
            input_tokens_per_turn=Distribution(
                count=150,
                mean=640.0,
            ),
            output_tokens_per_turn=Distribution(
                count=150,
                mean=120.0,
            ),
        ),
        nodes={"main_agent": Distribution(count=149, mean=10_250.0)},
        tools={},
        ingestion={
            "events_seen": 1_200,
            "events_deduplicated": 1,
            "events_unmatched": 0,
            "events_late": 0,
        },
    )


class _FakeService:
    """
    只实现这些用例走到的方法。

    ``list_calls`` 用来回头看路由有没有把查询参数漏掉 —— 那只有这一面
    看得见。
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.list_calls: list[tuple[Any, Any, Any]] = []

    def _raise_if_needed(self) -> None:
        if self._error is not None:
            raise self._error

    async def create(self, request: StressRunCreate) -> StressRun:
        self._raise_if_needed()
        return _run(config=request)

    async def list_runs(
            self,
            *,
            status: StressRunStatus | None = None,
            limit: int = 20,
            cursor: str | None = None,
    ) -> StressRunPage:
        self._raise_if_needed()
        self.list_calls.append((status, limit, cursor))
        return StressRunPage(items=[_run()], next_cursor="next-page")

    async def get(self, run_id: UUID) -> StressRun:
        self._raise_if_needed()
        return _run()

    async def get_metrics(self, run_id: UUID) -> StressRunMetrics:
        self._raise_if_needed()
        return _metrics()

    async def cancel(self, run_id: UUID) -> StressRun:
        self._raise_if_needed()
        return _run(status=StressRunStatus.CANCELLED)


class _FakeFactory:
    """needs_session=False 走 create(contract) 分支，不打开事务。"""

    def __init__(self, service: Any) -> None:
        self._service = service

    def needs_session(self, contract: Any) -> bool:
        return False

    def create(self, contract: Any) -> Any:
        return self._service


def _app(
        monkeypatch: pytest.MonkeyPatch,
        *,
        error: Exception | None = None,
        service: _FakeService | None = None,
) -> FastAPI:

    app = FastAPI()
    app.include_router(stress_runs_router, prefix="/api/v1")

    app.dependency_overrides[current_user] = lambda: SimpleNamespace(
        subject="user-1"
    )

    fake = service if service is not None else _FakeService(error=error)

    monkeypatch.setattr(
        dependencies_module,
        "get_service_factory",
        lambda: _FakeFactory(fake),
    )

    return app


async def _request(
        app: FastAPI,
        method: str,
        path: str,
        **kwargs: Any,
) -> httpx.Response:

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        return await client.request(method, path, **kwargs)


def _create_payload() -> dict[str, Any]:
    return {
        "name": "前端联调压测",
        "target": {
            "base_url": "http://127.0.0.1:8080",
        },
        "session": {
            "agent_id": "main",
            "turns": {
                "mean": 15.5,
                "stddev": 7.0,
            },
        },
        "load": {"mode": "concurrency", "concurrency": 10},
        "duration_seconds": 600,
        "request_timeout_ms": 120000,
        "total_sessions": 20,
        "seed": 42,
        "mock_model": {
            "base_latency_ms": 100,
            "ttft_ms": 50,
            "output_tokens_min": 20,
            "output_tokens_max": 40,
            "tokens_per_second": 50,
        },
    }


# ============================================================
# 创建
# ============================================================


async def test_create_accepts_with_202_and_echoes_config(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    202 而不是 201：记录建出来了，但它还没跑完。

    响应里的 config 必须是完整的多轮会话参数 —— 前端详情页直接渲染
    它，缺字段会让「这次压测是怎么配的」永远说不清。
    """

    app = _app(monkeypatch)

    response = await _request(
        app,
        "POST",
        "/api/v1/stress-runs",
        json=_create_payload(),
    )

    assert response.status_code == 202

    body = response.json()

    assert body["run_id"] == str(_RUN_ID)
    assert body["status"] == "queued"
    assert body["config"]["session"]["agent_id"] == "main"
    assert body["config"]["session"]["turns"]["mean"] == 15.5
    assert body["config"]["load"] == {
        "mode": "concurrency",
        "concurrency": 10,
    }
    assert body["config"]["target"]["base_url"] == (
        "http://127.0.0.1:8080"
    )
    # 请求里没传的字段由默认值固化进快照。
    assert body["config"]["turn"]["streaming"] is True
    assert body["config"]["session"]["think_time_ms"] == 0


@pytest.mark.parametrize(
    "patch",
    [
        # 负载模式判别错：concurrency 分支不接受 session_rate。
        {
            "load": {
                "mode": "concurrency",
                "concurrency": 10,
                "session_rate": 5,
            },
        },
        # 并发不得为 0。
        {"load": {"mode": "concurrency", "concurrency": 0}},
        # Token 区间上下界颠倒。
        {
            "mock_model": {
                "base_latency_ms": 100,
                "ttft_ms": 50,
                "output_tokens_min": 100,
                "output_tokens_max": 10,
                "tokens_per_second": 50,
            },
        },
        # target 不是 http/https。
        {"target": {"base_url": "127.0.0.1:8080"}},
        # 第一版不支持非流式终态检测。
        {"turn": {"streaming": False}},
        # 均值必须落在 [min, max] 内。
        {
            "session": {
                "agent_id": "main",
                "turns": {
                    "mean": 30,
                    "stddev": 1,
                    "min": 1,
                    "max": 10,
                },
            },
        },
        # 截断区间覆盖的概率太低（约 1.3%），拒绝采样几乎必然失败。
        {
            "session": {
                "agent_id": "main",
                "turns": {
                    "mean": 1,
                    "stddev": 30,
                    "min": 1,
                    "max": 2,
                },
            },
        },
        # 预热窗口不能吃掉整个运行。
        {"warmup_seconds": 600},
        # 未知字段必须被拒绝，而不是悄悄丢掉。
        {"unexpected": 1},
    ],
)
async def test_invalid_config_is_rejected_with_422(
        monkeypatch: pytest.MonkeyPatch,
        patch: dict[str, Any],
) -> None:
    """非法配置在进服务层之前就被 Pydantic 挡下（422，不是 400）。"""

    app = _app(monkeypatch)

    payload = _create_payload()
    payload.update(patch)

    response = await _request(
        app, "POST", "/api/v1/stress-runs", json=payload,
    )

    assert response.status_code == 422


# ============================================================
# 查询
# ============================================================


async def test_list_passes_filters_limit_and_cursor(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询参数必须原样到达服务层。"""

    fake = _FakeService()
    app = _app(monkeypatch, service=fake)

    response = await _request(
        app,
        "GET",
        "/api/v1/stress-runs",
        params={"status": "running", "limit": 7, "cursor": "abc"},
    )

    assert response.status_code == 200
    assert fake.list_calls == [
        (StressRunStatus.RUNNING, 7, "abc"),
    ]

    body = response.json()

    assert body["items"][0]["run_id"] == str(_RUN_ID)
    assert body["next_cursor"] == "next-page"


async def test_metrics_uses_contract_field_names(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    指标响应的键名以规格为准：sessions / turns，而不是旧的 requests。

    表里叫 session_latency、turn_latency、model_metrics，那是存储形态；
    直接把这些列名透给前端就是一次静默的接口违约。
    """

    app = _app(monkeypatch)

    response = await _request(
        app,
        "GET",
        f"/api/v1/stress-runs/{_RUN_ID}/metrics",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["aggregation_status"] == "final"
    assert body["aggregation_algorithm"] == "simulated-v1"
    assert body["sessions"]["completed"] == 9
    assert body["sessions"]["turns_per_session"]["count"] == 10
    assert body["turns"]["latency_ms"]["p95"] == 16_500.0
    assert body["turns"]["by_index"]["1"]["mean"] == 12_000.0
    assert body["model"]["input_tokens_per_turn"]["mean"] == 640.0
    assert "requests" not in body
    assert "request_latency" not in body


async def test_missing_run_is_404(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务层的 NotFoundError 必须映射成 404。"""

    app = _app(
        monkeypatch,
        error=NotFoundError("压测运行不存在"),
    )

    response = await _request(
        app, "GET", f"/api/v1/stress-runs/{_RUN_ID}",
    )

    assert response.status_code == 404


# ============================================================
# 取消
# ============================================================


async def test_cancel_terminal_run_is_409(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已终结的运行再取消是 409，不是假装成功。"""

    app = _app(
        monkeypatch,
        error=ConflictError("运行已经终结"),
    )

    response = await _request(
        app,
        "POST",
        f"/api/v1/stress-runs/{_RUN_ID}/cancel",
    )

    assert response.status_code == 409


# ============================================================
# 路由树
# ============================================================


def test_all_paths_live_on_the_protected_router() -> None:
    """
    五个端点都必须挂在 protected_router 上。

    新接口忘了注册到受保护的树上，后果是匿名就能创建压测运行。
    """

    app = FastAPI()
    app.include_router(protected_router)

    paths = app.openapi()["paths"]

    for method, path in _PROTECTED_PATHS:
        assert path in paths, f"缺少路径 {path}"
        assert method in paths[path], f"{path} 缺少方法 {method}"
