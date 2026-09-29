"""
Repository contract for pipeline run observability.

Transaction rule
----------------
实现不得调用 session.commit() / session.rollback()。

但注意 GraphRunRecorder 的实现（pipeline/observability/
recorder_impl.py）**会**自行开短会话并提交 —— 那是刻意的，
见 infra/graph/recorder.py 的模块 docstring：
观测写入必须自成事务，否则节点失败会把观测行一起回滚掉。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities.pipeline import (
    PipelineNodeRun,
    PipelineRun,
)
from ..domain.schemas.enum import ProcessingStatus
from app.infra.graph.recorder import (
    NodeRunUpdate,
    RunOutcome,
    RunSnapshot,
    RunSummary,
)


class PipelineRunRepository(BaseRepository):
    """运行记录与节点记录的持久化契约。"""

    # ========================================================
    # Run lifecycle
    # ========================================================

    @abstractmethod
    async def add(
            self,
            snapshot: RunSnapshot,
    ) -> PipelineRun:
        """插入 run 行。必须早于图开始运行。"""
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            pipeline_run_id: UUID,
    ) -> PipelineRun | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_thread_id(
            self,
            thread_id: str,
    ) -> PipelineRun | None:
        raise NotImplementedError

    @abstractmethod
    async def mark_running(
            self,
            pipeline_run_id: UUID,
            *,
            started_at: datetime,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def finish(
            self,
            pipeline_run_id: UUID,
            outcome: RunOutcome,
    ) -> None:
        """
        写回终态、耗时、错误归属与节点计数。

        计数由实现方在同一个事务里从节点表聚合，
        避免调用方多跑一次查询、也避免计数与状态不一致。
        """
        raise NotImplementedError

    # ========================================================
    # Node runs
    # ========================================================

    @abstractmethod
    async def upsert_node_run(
            self,
            update: NodeRunUpdate,
    ) -> None:
        """
        按 (pipeline_run_id, node_id, branch_key, attempt)
        插入或更新。两个写入方共用，必须幂等。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_node_runs(
            self,
            pipeline_run_id: UUID,
            *,
            offset: int = 0,
            limit: int = 200,
    ) -> tuple[list[PipelineNodeRun], int]:
        raise NotImplementedError

    @abstractmethod
    async def list_all_node_runs(
            self,
            pipeline_run_id: UUID,
    ) -> list[PipelineNodeRun]:
        """不分页取全部节点行，供详情接口组装。"""
        raise NotImplementedError

    @abstractmethod
    async def summarize(
            self,
            pipeline_run_id: UUID,
    ) -> RunSummary:
        """
        汇总节点行，供 runner 推导运行终态。

        必须是**从库里读真实结果**，不能靠调用方传入的推测 ——
        "一路成功一路失败"这种 partial 场景只能这样识别出来。
        """
        raise NotImplementedError

    # ========================================================
    # Queries
    # ========================================================

    @abstractmethod
    async def list_runs(
            self,
            *,
            document_version_id: UUID | None = None,
            document_id: UUID | None = None,
            status: str | None = None,
            trigger_source: str | None = None,
            triggered_by: UUID | None = None,
            started_from: datetime | None = None,
            started_to: datetime | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[PipelineRun], int]:
        """按 started_at 倒序分页。"""
        raise NotImplementedError

    # ========================================================
    # Maintenance
    # ========================================================

    @abstractmethod
    async def mark_orphans_failed(
            self,
            *,
            older_than: datetime,
    ) -> int:
        """
        把卡在 running/pending 的 run 行与 node 行标为 failed。

        进程中途被杀时行会永远停在 running，UI 一直转圈。
        启动时调用一次收敛。返回修正的行数。
        """
        raise NotImplementedError

    # ========================================================
    # Aggregation
    # ========================================================

    @staticmethod
    def derive_run_status(
            node_statuses: list[str],
    ) -> str:
        """
        由节点状态推导运行状态（纯函数，供实现方复用）。

        规则：
            无节点行                  -> failed（图根本没跑起来）
            任一 failed               -> partial（有成功的分支）
                                         或 failed（全挂）
            全 completed              -> completed
            否则                      -> running

        partial 与 failed 的区分是本次设计的重点：
            ES 挂了但 Milvus 成了 —— 文档是**部分可检索**的，
            这个信息必须保留下来，而不是笼统报失败。
        """

        from app.infra.graph.recorder import (
            NodeRunStatus,
            RunStatus,
        )

        if not node_statuses:
            return RunStatus.FAILED.value

        statuses = set(node_statuses)

        failed = NodeRunStatus.FAILED.value

        if failed not in statuses:
            if statuses == {NodeRunStatus.COMPLETED.value}:
                return RunStatus.COMPLETED.value
            return RunStatus.RUNNING.value

        if statuses == {failed}:
            return RunStatus.FAILED.value

        return RunStatus.PARTIAL.value

    @staticmethod
    def map_to_business_status(
            run_status: str,
    ) -> ProcessingStatus:
        """
        运行状态映射回业务状态。

        partial 在业务侧没有对应值（业务只回答"能用/不能用"），
        因此按 failed 处理。运行状态永不直接写业务字段。
        """

        from app.infra.graph.recorder import RunStatus

        if run_status == RunStatus.COMPLETED.value:
            return ProcessingStatus.COMPLETED

        if run_status in (
                RunStatus.FAILED.value,
                RunStatus.PARTIAL.value,
                RunStatus.CANCELLED.value,
        ):
            return ProcessingStatus.FAILED

        if run_status == RunStatus.RUNNING.value:
            return ProcessingStatus.PROCESSING

        return ProcessingStatus.PENDING
