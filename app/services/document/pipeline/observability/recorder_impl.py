"""
GraphRunRecorder 的 SQLAlchemy 实现。

事务策略（与业务仓储刻意不同）
------------------------------
业务仓储**从不**自己 commit，由组合根的请求级事务决定边界。

本类**每个方法自成一段已提交的短事务**。这不是疏忽：

    - 节点失败时，观测行不会被业务事务一起回滚掉
      （"部分失败不丢观测"就靠这条）
    - 观测写入不会在图运行的全程持有锁
    - node_span 因此完全不需要感知 session

代价是观测写入与业务写入之间没有原子性 —— 这正是要的：
观测是旁路，不该有资格让业务回滚。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.common.repository.factory import RepositoryFactory
from app.infra.graph.recorder import (
    NodeRunUpdate,
    RunOutcome,
    RunSnapshot,
    RunSummary,
)

from ...repository.pipeline_run import PipelineRunRepository

__all__ = ["SqlAlchemyRunRecorder"]


class SqlAlchemyRunRecorder:
    """把观测写入落到 PostgreSQL。"""

    def __init__(
            self,
            session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

        # 每个方法自成一段短事务（见模块 docstring 的事务策略），
        # 因此仓储工厂处在 session_factory 模式，用 scope() 取会话。
        self._repositories = RepositoryFactory(
            session_factory=session_factory
        )

    # ========================================================
    # Run lifecycle
    # ========================================================

    async def create_run(
            self,
            snapshot: RunSnapshot,
    ) -> None:

        async with self._repositories.scope() as repos:
            await repos.create(PipelineRunRepository).add(snapshot)

    async def mark_run_running(
            self,
            pipeline_run_id: UUID,
            *,
            started_at: datetime,
    ) -> None:

        async with self._repositories.scope() as repos:
            await repos.create(PipelineRunRepository).mark_running(
                pipeline_run_id,
                started_at=started_at,
            )

    async def finish_run(
            self,
            pipeline_run_id: UUID,
            outcome: RunOutcome,
    ) -> None:

        async with self._repositories.scope() as repos:
            await repos.create(PipelineRunRepository).finish(
                pipeline_run_id, outcome
            )

    # ========================================================
    # Node runs
    # ========================================================

    async def upsert_node_run(
            self,
            update: NodeRunUpdate,
    ) -> None:

        async with self._repositories.scope() as repos:
            await repos.create(
                PipelineRunRepository
            ).upsert_node_run(update)

    # ========================================================
    # Read-back
    # ========================================================

    async def summarize(
            self,
            pipeline_run_id: UUID,
    ) -> RunSummary:

        # 只读：不开显式事务。
        async with self._repositories.scope(begin=False) as repos:
            return await repos.create(
                PipelineRunRepository
            ).summarize(pipeline_run_id)

    # ========================================================
    # Maintenance
    # ========================================================

    async def mark_orphans_failed(
            self,
            *,
            older_than: datetime,
    ) -> int:

        async with self._repositories.scope() as repos:
            return await repos.create(
                PipelineRunRepository
            ).mark_orphans_failed(older_than=older_than)
