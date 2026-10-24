"""
StressRunService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback —— 与其余模块
一致。创建时的两次写入（stress_runs + stress_run_metrics）落在同一个
ServiceFactory.scope() 事务里，因此不可能出现「运行写了、快照行没写」。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import ServiceContext

from ..domain.entities import StressRunCursor, StressRunRecord
from ..domain.schemas.stress import (
    StressRun,
    StressRunCreate,
    StressRunMetrics,
    StressRunPage,
    StressRunStatus,
)
from ..repository.stress_run import StressRunStore
from ._assembly import (
    build_stress_run,
    build_stress_run_metrics,
    build_stress_run_page,
    empty_stress_run_metrics,
)
from ._pagination import decode_cursor, encode_cursor
from .stress_run_service import StressRunService
from .stress_runner import StressRunner

__all__ = ["StressRunServiceImpl"]


class StressRunServiceImpl(StressRunService):

    def __init__(
            self,
            *,
            stress_run_store: StressRunStore,
            stress_runner: StressRunner | None = None,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._store = stress_run_store

        #: 可选。没有它时运行会停在 queued —— 那是**正确的**降级：
        #: 冒烟与单元测试不该因为一次 create 就触发后台协程。
        #:
        #: 生产装配在 main.py 的服务工厂池子里注入它
        #: （键名 stress_runner，与参数名一致）。
        self._runner = stress_runner

    # ========================================================
    # 创建
    # ========================================================

    async def create(
            self,
            request: StressRunCreate,
    ) -> StressRun:

        run_id = uuid4()

        record = StressRunRecord(
            run_id=run_id,
            name=request.name,
            status=StressRunStatus.QUEUED,
            #: mode="json" 让 datetime / enum / 嵌套模型都落成 JSON
            #: 可写进 JSONB 的形态；读回时再由 StressRunCreate 解释。
            config=request.model_dump(mode="json"),
            created_at=datetime.now(timezone.utc),
        )

        saved = await self._store.add_run(record)

        # 空快照与运行同事务落库：不存在「运行建了、metrics 接口 404」
        # 的中间态。
        await self._store.add_metrics(run_id)

        self._schedule_runner(saved.run_id, request)

        return build_stress_run(saved)

    # ========================================================
    # 查询
    # ========================================================

    async def list_runs(
            self,
            *,
            status: StressRunStatus | None = None,
            limit: int = 20,
            cursor: str | None = None,
    ) -> StressRunPage:

        decoded = decode_cursor(cursor) if cursor else None

        # 多取一条用来判断「还有没有下一页」。只看 len == limit 是不行
        # 的：恰好整除时会给客户端一个必然为空的 next_cursor。
        records = await self._store.list_runs(
            status=status,
            limit=limit + 1,
            cursor=decoded,
        )

        has_more = len(records) > limit

        items = records[:limit]

        next_cursor = (
            encode_cursor(
                StressRunCursor(
                    created_at=items[-1].created_at,
                    run_id=items[-1].run_id,
                )
            )
            if has_more
            else None
        )

        return build_stress_run_page(items, next_cursor)

    async def get(
            self,
            run_id: UUID,
    ) -> StressRun:

        return build_stress_run(await self._require(run_id))

    async def get_metrics(
            self,
            run_id: UUID,
    ) -> StressRunMetrics:

        # 先确认运行存在：不存在才是 404。运行存在而快照行缺失
        # 只可能是历史数据或人工删行，这时返回 collecting 空快照。
        await self._require(run_id)

        metrics = await self._store.get_metrics(run_id)

        if metrics is None:
            return empty_stress_run_metrics(run_id)

        return build_stress_run_metrics(metrics)

    # ========================================================
    # 取消
    # ========================================================

    async def cancel(
            self,
            run_id: UUID,
    ) -> StressRun:

        current = await self._require(run_id)

        if current.status.is_terminal:
            raise ConflictError(
                f"运行已经终结（{current.status.value}），不能取消。"
            )

        updated = await self._store.mark_cancelled(run_id)

        if updated is None:
            # 读与写之间被别的路径推进到了终态。回读一次，用真实的
            # 状态回答 —— 不要假装取消成功。
            latest = await self._store.get_run(run_id)

            if latest is None:
                raise NotFoundError(f"压测运行不存在：{run_id}")

            if latest.status.is_terminal:
                raise ConflictError(
                    f"运行已经终结（{latest.status.value}），不能取消。"
                )

            # 理论上到不了：mark_cancelled 只在非终态失败。保底返回
            # 刚读到的那一行，而不是抛一个无法解释的错误。
            return build_stress_run(latest)

        # DB 已经置为 cancelled；通知 Runner 停止接纳/发起新工作。
        # 接口仍以 DB 为权威：即使 Driver 没接住信号，它在下一轮检查
        # 或最终写入时也会看到 cancelled。
        if self._runner is not None:
            self._runner.cancel(run_id)

        return build_stress_run(updated)

    # ========================================================
    # internals
    # ========================================================

    async def _require(
            self,
            run_id: UUID,
    ) -> StressRunRecord:

        run = await self._store.get_run(run_id)

        if run is None:
            raise NotFoundError(f"压测运行不存在：{run_id}")

        return run

    def _schedule_runner(
            self,
            run_id: UUID,
            request: StressRunCreate,
    ) -> None:
        """
        登记「事务提交之后再启动执行」的钩子。

        钩子不在事务里执行的理由见契约 docstring；执行时机由
        ServiceFactory.scope() 在 commit 之后的那一段保证。
        """

        runner = self._runner

        if runner is None:
            return

        async def _hook() -> None:

            runner.schedule(run_id, request)

        self.post_commit.append(_hook)
