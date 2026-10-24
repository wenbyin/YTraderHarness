"""
Driver 线程 -> 主事件循环的 DB 写入桥。

为什么需要这一层
----------------
Stress Driver 跑在独立守护线程里（自己的 asyncio loop），而应用的
asyncpg / SQLAlchemy async engine 和连接池是在 FastAPI 主 loop 上创建的。
**跨 event loop 使用 asyncpg 连接会失败**，而且这种失败通常不是
"拒绝连接"，而是更难查的 loop 绑定时序问题。

所以 Driver 线程不 import 任何主 loop 的 session/engine；它只通过本类：

    await reporter.update_metrics(run_id, snapshot)

本类把真正的工作 `run_coroutine_threadsafe` 投回主 loop，Driver 线程
用 `asyncio.wrap_future` 等结果。写入频率是启动 / 周期快照 / 结束，
不在每轮热路径上，因此这点跨线程调度成本可以接受。

替代方案（不采用）
------------------
Driver 线程自己另建一套 DB 池：连接数翻倍、生命周期翻倍，DSN / 池参数
两处漂移；而 Stress 的写入本来就很稀疏。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from app.common.repository.base import SessionFactory
from app.common.repository.factory import RepositoryFactory

from ..domain.entities import StressRunMetricsRecord, StressRunRecord
from ..repository.stress_run import StressRunStore

__all__ = ["StressRunReporter"]

logger = logging.getLogger(__name__)


class StressRunReporter:
    """把 Driver 的写入请求投回主 loop 执行。"""

    def __init__(
            self,
            session_factory: SessionFactory,
            main_loop: asyncio.AbstractEventLoop,
            *,
            timeout_seconds: float = 15.0,
    ) -> None:

        self._repositories = RepositoryFactory(
            session_factory=session_factory,
        )
        self._main_loop = main_loop
        self._timeout = max(1.0, timeout_seconds)

    # ========================================================
    # 生命周期状态
    # ========================================================

    async def mark_running(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:

        return await self._call(self._mark_running(run_id))

    async def complete_run(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunRecord | None:

        return await self._call(
            self._complete_run(run_id, metrics)
        )

    async def mark_failed(
            self,
            run_id: UUID,
            *,
            error_code: str | None,
            error_message: str | None,
    ) -> StressRunRecord | None:

        return await self._call(
            self._mark_failed(
                run_id,
                error_code=error_code,
                error_message=error_message,
            )
        )

    # ========================================================
    # 指标
    # ========================================================

    async def update_metrics(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunMetricsRecord | None:

        return await self._call(
            self._update_metrics(run_id, metrics)
        )

    async def get_run(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:

        return await self._call(self._get_run(run_id))

    # ========================================================
    # 主 loop 上的实际实现
    # ========================================================

    async def _mark_running(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:

        async with self._repositories.scope() as repos:
            return await repos.create(StressRunStore).mark_running(run_id)

    async def _complete_run(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunRecord | None:

        async with self._repositories.scope() as repos:
            return await repos.create(
                StressRunStore
            ).complete_run(run_id, metrics)

    async def _mark_failed(
            self,
            run_id: UUID,
            *,
            error_code: str | None,
            error_message: str | None,
    ) -> StressRunRecord | None:

        async with self._repositories.scope() as repos:
            return await repos.create(StressRunStore).mark_failed(
                run_id,
                error_code=error_code,
                error_message=error_message,
            )

    async def _update_metrics(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunMetricsRecord | None:

        async with self._repositories.scope() as repos:
            return await repos.create(
                StressRunStore
            ).update_metrics(run_id, metrics)

    async def _get_run(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:

        async with self._repositories.scope() as repos:
            return await repos.create(StressRunStore).get_run(run_id)

    # ========================================================
    # 跨线程调度
    # ========================================================

    async def _call(self, coro):
        """
        把 coroutine 投回主 loop 并等待结果。

        coroutine 在 Driver 线程里创建、在主 loop 上执行；它捕获的只有
        ``RepositoryFactory``（主 loop 的对象），不碰 Driver loop 的
        aiohttp/session。
        """

        if not self._main_loop.is_running():
            raise RuntimeError(
                "主事件循环没有在运行，Stress Reporter 无法写库。"
            )

        future = asyncio.run_coroutine_threadsafe(
            coro,
            self._main_loop,
        )

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=self._timeout,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "Stress Reporter 写库超时（%.0fs），主 loop 可能被阻塞。",
                self._timeout,
            )
            raise


def utcnow() -> datetime:
    """统一用带时区的 UTC；快照的 finalized_at 需要它。"""

    return datetime.now(timezone.utc)
