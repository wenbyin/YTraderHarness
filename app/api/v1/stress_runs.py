"""
Mock Model Stress Eval 的端点。

契约权威是 ``design_agent_eval/openapi.yaml`` —— 本文件是它的路由
实现，路径、状态码、参数名与那份 yaml 一一对应。

五个端点，两个主语
-----------------
    /stress-runs               运行本身：建、列、看、取消
    /stress-runs/{id}/metrics  运行的聚合结果

一次运行 = 一组**虚拟会话**；每个虚拟会话在一个真实 `session_id` 上
串行进行 N 轮对话，N 从截断正态分布抽样，范围 `[1, 30]`。参数结构见
`design_agent_eval/openapi.yaml` 与 `services/stress/domain/schemas`。

⚠️ 这里**没有** trace / span / 逐请求明细的端点
--------------------------------------------
Trace 实体只在 Redis 里按 TTL 短暂存在，PostgreSQL 只保存运行参数、
生命周期与最终聚合快照（设计 README §1）。要加明细接口，先改设计。

⚠️ 当前返回的指标是本地模拟的
---------------------------
后端还没有接真实 Runner / Aggregator，运行状态由
``application/simulated_runner.py`` 推进。指标的
``aggregation_algorithm`` 为 ``simulated-v1``；前端可以据此打上
「模拟数据」标记，避免被当成真实压测结论。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, status

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.stress.application.stress_run_service import (
    StressRunService,
)
from app.services.stress.domain.schemas.stress import (
    StressRun,
    StressRunCreate,
    StressRunMetrics,
    StressRunPage,
    StressRunStatus,
)

router = APIRouter()


# ============================================================
# 运行
# ============================================================


@router.post(
    "/stress-runs",
    response_model=StressRun,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Stress Runs"],
)
async def create_stress_run(
        request: StressRunCreate,
        service: StressRunService = Service[StressRunService],
) -> StressRun:
    """
    创建并排队一次压测运行。

    **202 而不是 201**：记录确实建出来了，但它还没跑完 —— 202 表达
    的是「受理了一个还没完成的动作」。之后用 GET 看状态，用
    ``/metrics`` 看聚合结果。

    参数非法由 Pydantic / FastAPI 直接回 422（与规格一致）。
    """

    try:
        return await service.create(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/stress-runs",
    response_model=StressRunPage,
    tags=["Stress Runs"],
)
async def list_stress_runs(
        status_: StressRunStatus | None = Query(
            default=None,
            alias="status",
            description="按状态过滤。不传表示不过滤。",
        ),
        limit: int = Query(
            default=20,
            ge=1,
            le=100,
            description="本页条数上限。",
        ),
        cursor: str | None = Query(
            default=None,
            description=(
                "上一页返回的 next_cursor。不传表示第一页。"
                "游标是不透明字符串，不要自己拼。"
            ),
        ),
        service: StressRunService = Service[StressRunService],
) -> StressRunPage:
    """
    列出运行记录，最近创建的在前。

    next_cursor 为 null 表示没有下一页；传 null 或省略都表示从头开始。
    """

    try:
        return await service.list_runs(
            status=status_,
            limit=limit,
            cursor=cursor,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/stress-runs/{run_id}",
    response_model=StressRun,
    tags=["Stress Runs"],
)
async def get_stress_run(
        run_id: UUID,
        service: StressRunService = Service[StressRunService],
) -> StressRun:
    """
    取一次运行的完整参数与当前状态。

    config 是**创建时固化的快照**：之后默认值改了，历史运行仍然按
    它当时的样子回放。
    """

    try:
        return await service.get(run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/stress-runs/{run_id}/cancel",
    response_model=StressRun,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Stress Runs"],
)
async def cancel_stress_run(
        run_id: UUID,
        service: StressRunService = Service[StressRunService],
) -> StressRun:
    """
    请求取消。

    取消是协作式的，但返回体给出的是**取消之后真实所处的位置**：
    queued / running 的运行会立即变成 cancelled；已经终结的运行回 409。
    """

    try:
        return await service.cancel(run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 聚合指标
# ============================================================


@router.get(
    "/stress-runs/{run_id}/metrics",
    response_model=StressRunMetrics,
    tags=["Metrics"],
)
async def get_stress_run_metrics(
        run_id: UUID,
        service: StressRunService = Service[StressRunService],
) -> StressRunMetrics:
    """
    取运行级聚合指标。

    运行中返回 ``aggregation_status=collecting`` 的当前快照；结束后
    是 ``final``；被取消 / 执行侧失败导致采集中断时为 ``incomplete``。

    **不返回** Trace / Span 明细 —— 那些只在 Redis 里按 TTL 存在。
    """

    try:
        return await service.get_metrics(run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc
