"""
Stress Eval 的端到端冒烟：HTTP -> Service -> Repository -> PostgreSQL。

跑法：
    .venv/Scripts/python.exe -m app.services.stress.stress_http_smoke

与两个邻居的分工
----------------
    tests/api/test_stress_runs_routes.py   假服务，只验路由/序列化
    stress_store_smoke.py                  真库，但不经过 FastAPI
    本文件                                 真库 + 真路由 + 真 response_model

⚠️ 本文件不启动完整 lifespan（不碰 Redis/Milvus/模型），只装配
ServiceFactory 与一个最小的 FastAPI。它仍然不发出任何真实压力请求：
运行由本地模拟器推进。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

import httpx
from fastapi import FastAPI
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.services.dependencies as dependencies_module
from app.api.v1.stress_runs import router as stress_runs_router
from app.common.repository.factory import RepositoryFactory
from app.common.service.factory import ServiceFactory
from app.services.stress.application.simulated_runner import (
    SimulatedStressRunner,
)
from app.services.stress.application.stress_run_service import (
    StressRunService,
)
from app.services.stress.domain.schemas.stress import (
    ConcurrencyLoad,
    MockModelConfig,
    SessionConfig,
    StressRunCreate,
    TargetConfig,
    TurnCountConfig,
)
from app.services.stress.models.sqlalchemy.stress_run import (
    StressRun as StressRunModel,
)
from app.services.stress.models.sqlalchemy.stress_run import (
    StressRunMetrics as StressRunMetricsModel,
)

#: 与 app/config/datasource/postgre.yaml 里的 dsn 一致。
DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

FAILS: list[str] = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        FAILS.append(label)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def build_payload(
        *,
        name: str,
        total_sessions: int = 3,
        turns: int = 3,
) -> dict[str, Any]:

    config = StressRunCreate(
        name=name,
        target=TargetConfig(base_url="http://127.0.0.1:8080"),
        session=SessionConfig(
            agent_id="main",
            turns=TurnCountConfig(mean=turns, stddev=0),
        ),
        load=ConcurrencyLoad(mode="concurrency", concurrency=5),
        duration_seconds=60,
        request_timeout_ms=30_000,
        total_sessions=total_sessions,
        seed=42,
        mock_model=MockModelConfig(
            base_latency_ms=100,
            latency_jitter_ms=20,
            ttft_ms=50,
            output_tokens_min=20,
            output_tokens_max=40,
            tokens_per_second=50,
        ),
    )

    return config.model_dump(mode="json")


async def main() -> int:

    logging.basicConfig(level=logging.WARNING)

    engine = create_async_engine(DSN)
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    repositories = RepositoryFactory(session_factory=session_factory)
    runner = SimulatedStressRunner(
        session_factory,
        start_delay_seconds=0.1,
        min_run_seconds=0.05,
        max_run_seconds=0.1,
    )
    services = ServiceFactory(
        repositories,
        pool={"stress_runner": runner},
    )

    app = FastAPI()
    app.include_router(stress_runs_router, prefix="/api/v1")

    original_get_factory = dependencies_module.get_service_factory
    dependencies_module.get_service_factory = lambda: services

    created: list[UUID] = []

    transport = httpx.ASGITransport(app=app)

    try:
        await _ensure_tables(engine)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:

            # =================================================
            section("HTTP 创建 -> 查询 -> 指标")

            response = await client.post(
                "/api/v1/stress-runs",
                json=build_payload(name="http-冒烟-A"),
            )

            check("POST 返回 202", response.status_code == 202)
            body = response.json()
            run_id = UUID(body["run_id"])
            created.append(run_id)

            check(
                "响应 config 是会话结构",
                body["config"]["session"]["agent_id"] == "main"
                and body["config"]["session"]["turns"]["mean"] == 3,
            )
            check("初始状态 queued", body["status"] == "queued")

            detail = await client.get(
                f"/api/v1/stress-runs/{run_id}"
            )
            check("GET 详情 200", detail.status_code == 200)
            check(
                "详情带 turn 默认值",
                detail.json()["config"]["turn"]["streaming"] is True,
            )

            final = await _wait_for_final(client, run_id)
            check(
                "运行在合理时间内 final",
                final is not None,
                "超时" if final is None else "",
            )

            if final is not None:
                sessions = final["sessions"]
                turns = final["turns"]

                check(
                    "sessions 3/3 完成",
                    sessions["started"] == 3
                    and sessions["completed"] == 3,
                )
                check(
                    "turns 9/9 成功",
                    turns["started"] == 9
                    and turns["succeeded"] == 9,
                )
                check(
                    "by_index 有 1/2/3",
                    set(turns["by_index"]) == {"1", "2", "3"},
                )
                check(
                    "响应里没有旧 requests 分组",
                    "requests" not in final,
                )

            # =================================================
            section("HTTP 列表分页")

            # 先造第二条记录；否则库里只有一条，limit=1 时不该有下一页。
            seed_response = await client.post(
                "/api/v1/stress-runs",
                json=build_payload(
                    name="http-冒烟-A2",
                    total_sessions=1,
                ),
            )
            seed_id = UUID(seed_response.json()["run_id"])
            created.append(seed_id)

            # 立刻取消，避免后台模拟任务与清理阶段抢跑。
            await client.post(
                f"/api/v1/stress-runs/{seed_id}/cancel"
            )

            listing = await client.get(
                "/api/v1/stress-runs",
                params={"limit": 1},
            )
            listing_body = listing.json()

            check("GET 列表 200", listing.status_code == 200)
            check(
                "limit=1 时给 next_cursor",
                len(listing_body["items"]) == 1
                and listing_body["next_cursor"] is not None,
            )

            # =================================================
            section("HTTP 取消")

            cancel_payload = build_payload(name="http-冒烟-B")
            cancel_create = await client.post(
                "/api/v1/stress-runs",
                json=cancel_payload,
            )
            cancel_id = UUID(cancel_create.json()["run_id"])
            created.append(cancel_id)

            cancelled = await client.post(
                f"/api/v1/stress-runs/{cancel_id}/cancel"
            )
            check("取消返回 202", cancelled.status_code == 202)
            check(
                "取消后状态 cancelled",
                cancelled.json()["status"] == "cancelled",
            )

            # 等模拟器醒来，确认它不会覆盖取消结果。
            await asyncio.sleep(0.4)

            after = await client.get(
                f"/api/v1/stress-runs/{cancel_id}"
            )
            check(
                "取消不会被模拟执行覆盖",
                after.json()["status"] == "cancelled",
            )

            metrics_after = await client.get(
                f"/api/v1/stress-runs/{cancel_id}/metrics"
            )
            check(
                "取消后的快照为 incomplete",
                metrics_after.json()["aggregation_status"]
                == "incomplete",
            )

            late_cancel = await client.post(
                f"/api/v1/stress-runs/{cancel_id}/cancel"
            )
            check(
                "终态再取消返回 409",
                late_cancel.status_code == 409,
            )

            missing = await client.get(
                "/api/v1/stress-runs/00000000-0000-0000-0000-000000000000"
            )
            check("不存在的运行返回 404", missing.status_code == 404)

    finally:
        dependencies_module.get_service_factory = original_get_factory

        if created:
            async with session_factory() as session:
                await session.execute(
                    delete(StressRunMetricsModel).where(
                        StressRunMetricsModel.run_id.in_(created)
                    )
                )
                await session.execute(
                    delete(StressRunModel).where(
                        StressRunModel.run_id.in_(created)
                    )
                )
                await session.commit()

            print(f"\n  已清理 {len(created)} 条探针运行及其快照")

        await engine.dispose()

    if FAILS:
        print(f"\n失败 {len(FAILS)} 项：")
        for item in FAILS:
            print(f"  - {item}")
        return 1

    print("\n全部通过。")
    return 0


async def _wait_for_final(
        client: httpx.AsyncClient,
        run_id: UUID,
        *,
        timeout_seconds: float = 5.0,
) -> dict[str, Any] | None:
    """轮询 metrics 直到 final；超时返回 None。"""

    deadline = asyncio.get_running_loop().time() + timeout_seconds

    while asyncio.get_running_loop().time() < deadline:

        response = await client.get(
            f"/api/v1/stress-runs/{run_id}/metrics"
        )

        if response.status_code == 200:
            body = response.json()

            if body.get("aggregation_status") == "final":
                return body

        await asyncio.sleep(0.05)

    return None


async def _ensure_tables(engine: Any) -> None:
    from app.common.models import Base
    from app.infra.datasource.initializer import import_all_models

    import_all_models()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
