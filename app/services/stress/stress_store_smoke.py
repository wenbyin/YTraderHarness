"""
StressRun 全链路冒烟：**真 Postgres** 上的建、查、分页、多轮完成与取消。

跑法：
    .venv/Scripts/python.exe -m app.services.stress.stress_store_smoke

它验的是「API 背后的那一整条 DB 链路」，但**不经过 HTTP、不发出任何
真实压力请求**：路由层另有 tests/api/test_stress_runs_routes.py。
模拟执行器的时间尺度在这里被压到毫秒级。

⚠️ 本文件只碰自己造出来的行
------------------------
探针数据在结尾按 run_id 精确删除。表由 initialize_database 的
create_all 建（幂等）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.common.exceptions import ConflictError, NotFoundError
from app.common.repository.factory import RepositoryFactory
from app.common.service.factory import ServiceFactory
from app.services.stress.application.simulated_runner import (
    SimulatedStressRunner,
)
from app.services.stress.application.stress_run_service import (
    StressRunService,
)
from app.services.stress.domain.schemas.stress import (
    AggregationStatus,
    ConcurrencyLoad,
    MockModelConfig,
    SessionConfig,
    StressRunCreate,
    StressRunStatus,
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


def build_request(
        *,
        name: str,
        seed: int,
        turns: int = 3,
        total_sessions: int = 3,
) -> StressRunCreate:

    return StressRunCreate(
        name=name,
        target=TargetConfig(
            base_url="http://127.0.0.1:8080",
        ),
        session=SessionConfig(
            agent_id="main",
            # stddev=0 -> 固定 N，冒烟断言稳定。
            turns=TurnCountConfig(mean=turns, stddev=0),
        ),
        load=ConcurrencyLoad(mode="concurrency", concurrency=5),
        duration_seconds=60,
        request_timeout_ms=30_000,
        total_sessions=total_sessions,
        seed=seed,
        mock_model=MockModelConfig(
            base_latency_ms=100,
            latency_jitter_ms=20,
            ttft_ms=50,
            output_tokens_min=20,
            output_tokens_max=40,
            tokens_per_second=50,
        ),
    )


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
        start_delay_seconds=0.15,
        min_run_seconds=0.05,
        max_run_seconds=0.1,
    )

    services = ServiceFactory(
        repositories,
        pool={"stress_runner": runner},
    )

    created: list[UUID] = []

    try:
        await _ensure_tables(engine)

        # =====================================================
        section("创建：queued + collecting 快照")

        async with services.scope(StressRunService) as service:
            run_a = await service.create(
                build_request(name="冒烟-A", seed=1)
            )

        created.append(run_a.run_id)

        check("创建后是 queued", run_a.status is StressRunStatus.QUEUED)
        check("started_at 为空", run_a.started_at is None)
        check(
            "config 固化了会话参数",
            run_a.config.session.agent_id == "main"
            and run_a.config.session.turns.mean == 3,
        )
        check(
            "config 固化了轮次默认值",
            run_a.config.turn.streaming is True,
        )

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            metrics_now = await service.get_metrics(run_a.run_id)

        check(
            "建运行即有 metrics 行（不会 404）",
            metrics_now.run_id == run_a.run_id,
        )
        check(
            "模拟完成前是 collecting 或已 final",
            metrics_now.aggregation_status
            in {
                AggregationStatus.COLLECTING,
                AggregationStatus.FINAL,
            },
            metrics_now.aggregation_status.value,
        )

        # =====================================================
        section("模拟执行：queued -> running -> completed")

        completed_a = await _wait_for_status(
            services, run_a.run_id, StressRunStatus.COMPLETED
        )

        check(
            "运行最终是 completed",
            completed_a is not None,
            "超时仍是 queued/running" if completed_a is None else "",
        )

        if completed_a is not None:
            check(
                "started_at / finished_at 都有值",
                completed_a.started_at is not None
                and completed_a.finished_at is not None,
            )

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            final_metrics = await service.get_metrics(run_a.run_id)

        check(
            "快照定稿为 final",
            final_metrics.aggregation_status is AggregationStatus.FINAL,
        )
        check(
            "算法标识为 simulated-v1（不是真实压测）",
            final_metrics.aggregation_algorithm == "simulated-v1",
        )
        check(
            "3 个会话全部完成",
            final_metrics.sessions.started == 3
            and final_metrics.sessions.completed == 3,
            f"started={final_metrics.sessions.started} "
            f"completed={final_metrics.sessions.completed}",
        )
        check(
            "抽样 N 与实际 N 都是 3",
            final_metrics.sessions.sampled_turns_per_session.min == 3
            and final_metrics.sessions.sampled_turns_per_session.max == 3
            and final_metrics.sessions.turns_per_session.min == 3
            and final_metrics.sessions.turns_per_session.max == 3,
        )
        check(
            "轮次 9 次全部成功",
            final_metrics.turns.started == 9
            and final_metrics.turns.succeeded == 9
            and final_metrics.turns.completed == 9,
            f"started={final_metrics.turns.started} "
            f"succeeded={final_metrics.turns.succeeded}",
        )
        check(
            "时延分布样本数等于终态轮次",
            final_metrics.turns.latency_ms.count == 9,
        )
        check(
            "按 turn_index 分组存在 1/2/3",
            set(final_metrics.turns.by_index) == {"1", "2", "3"},
            str(sorted(final_metrics.turns.by_index)),
        )
        check(
            "首轮与后续轮时延都有样本",
            final_metrics.turns.first_turn_latency_ms.count == 3
            and final_metrics.turns.later_turn_latency_ms.count == 6,
        )
        check(
            "会话时长分布有 3 个样本",
            final_metrics.sessions.duration_ms.count == 3,
        )
        check(
            "achieved_turns_per_second > 0",
            (final_metrics.turns.achieved_turns_per_second or 0) > 0,
        )
        check(
            "每轮输入/输出 token 分布有样本",
            final_metrics.model.input_tokens_per_turn.count == 9
            and final_metrics.model.output_tokens_per_turn.count == 9,
        )
        check(
            "模型 token 汇总 > 0",
            final_metrics.model.input_tokens_sum > 0
            and final_metrics.model.output_tokens_sum > 0,
        )
        check(
            "trace 计数只有有限枚举键",
            set(final_metrics.trace_counts)
            == {
                "run_span_count",
                "node_span_count",
                "model_call_count",
                "tool_call_count",
                "context_build_count",
                "fanout_count",
            },
        )

        # =====================================================
        section("列表：倒序 + keyset 分页")

        async with services.scope(StressRunService) as service:
            run_b = await service.create(
                build_request(name="冒烟-B", seed=2)
            )
            run_c = await service.create(
                build_request(name="冒烟-C", seed=3)
            )

        created += [run_b.run_id, run_c.run_id]

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            page1 = await service.list_runs(limit=2)

        check(
            "第一页正好两条",
            len(page1.items) == 2,
            str(len(page1.items)),
        )
        check(
            "第一页有 next_cursor",
            page1.next_cursor is not None,
        )

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            page2 = await service.list_runs(
                limit=2,
                cursor=page1.next_cursor,
            )

        page1_ids = [item.run_id for item in page1.items]
        page2_ids = [item.run_id for item in page2.items]

        check(
            "两页没有重复行",
            not (set(page1_ids) & set(page2_ids)),
        )
        check(
            "倒序：最新创建的在第一页",
            page1_ids[0] == run_c.run_id
            and page1_ids[1] == run_b.run_id,
            f"{page1_ids}",
        )
        check(
            "第二页能拿到更早的行",
            run_a.run_id in page2_ids,
            f"{page2_ids}",
        )

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            try:
                await service.list_runs(cursor="not-a-cursor")
            except ValueError:
                rejected = True
            else:
                rejected = False

        check("非法 cursor 被拒绝（400，而不是从头开始）", rejected)

        # =====================================================
        section("取消：与模拟执行竞争")

        async with services.scope(StressRunService) as service:
            run_d = await service.create(
                build_request(name="冒烟-D", seed=4)
            )

        created.append(run_d.run_id)

        async with services.scope(StressRunService) as service:
            cancelled_d = await service.cancel(run_d.run_id)

        check(
            "queued 的运行取消后是 cancelled",
            cancelled_d.status is StressRunStatus.CANCELLED,
            cancelled_d.status.value,
        )

        # 等模拟器醒来。它应当发现状态已不是 queued 并安静放弃。
        await asyncio.sleep(0.4)

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            after_d = await service.get(run_d.run_id)
            metrics_d = await service.get_metrics(run_d.run_id)

        check(
            "取消结果不会被模拟执行覆盖",
            after_d.status is StressRunStatus.CANCELLED,
            after_d.status.value,
        )
        check(
            "取消后的快照是 incomplete（永远不会再定稿）",
            metrics_d.aggregation_status is AggregationStatus.INCOMPLETE,
            metrics_d.aggregation_status.value,
        )

        async with services.scope(StressRunService) as service:
            try:
                await service.cancel(run_a.run_id)
            except ConflictError:
                conflict = True
            else:
                conflict = False

        check("已完成的运行再取消 -> ConflictError（路由 409）", conflict)

        # =====================================================
        section("不存在与筛选")

        missing = uuid4()

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            try:
                await service.get(missing)
            except NotFoundError:
                not_found = True
            else:
                not_found = False

        check("不存在的 run_id -> NotFoundError（路由 404）", not_found)

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            cancelled_page = await service.list_runs(
                status=StressRunStatus.CANCELLED,
                limit=10,
            )

        check(
            "按状态筛选只返回 cancelled",
            all(
                item.status is StressRunStatus.CANCELLED
                for item in cancelled_page.items
            )
            and any(
                item.run_id == run_d.run_id
                for item in cancelled_page.items
            ),
        )

        # =====================================================
        # 等后台模拟任务收尾，避免 engine.dispose() 与它们抢连接。
        await _wait_for_status(
            services, run_b.run_id, StressRunStatus.COMPLETED
        )
        await _wait_for_status(
            services, run_c.run_id, StressRunStatus.COMPLETED
        )

    finally:
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


async def _wait_for_status(
        services: ServiceFactory,
        run_id: UUID,
        expected: StressRunStatus,
        *,
        timeout_seconds: float = 5.0,
) -> Any:
    """轮询到目标状态；超时返回 None。"""

    deadline = asyncio.get_running_loop().time() + timeout_seconds

    while asyncio.get_running_loop().time() < deadline:

        async with services.scope(
            StressRunService, begin=False
        ) as service:
            run = await service.get(run_id)

        if run.status is expected:
            return run

        await asyncio.sleep(0.05)

    return None


async def _ensure_tables(engine: Any) -> None:
    """正常启动路径由 lifespan 建；冒烟跳过 lifespan，补一次。"""

    from app.common.models import Base
    from app.infra.datasource.initializer import import_all_models

    import_all_models()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
