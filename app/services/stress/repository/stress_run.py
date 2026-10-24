"""
``stress_runs`` / ``stress_run_metrics`` 的仓储契约。

两条贯穿全契约的规则
------------------
1. **从不 commit / rollback**（仓库通则）。事务边界由调用方决定：
   请求路径是 ServiceFactory.scope()，模拟执行器是自己开的 scope。

2. **状态推进一律是条件 UPDATE（CAS）**，不是「读出来、改内存、写回去」。
   典型竞态是取消与完成同时发生：执行侧算完了要写 completed，用户同时
   点了取消。只有 CAS 能保证**取消赢** —— 否则界面上显示已取消，而实际
   那次执行又把状态改回了 completed。

为什么 metrics 与 run 分两个方法
-------------------------------
运行状态是高频、小行；聚合快照是大块 JSONB，定稿前可能反复覆盖。
两者分开更新，避免状态 CAS 拖着整行 JSONB 走。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import (
    StressRunCursor,
    StressRunMetricsRecord,
    StressRunRecord,
)
from ..domain.schemas.stress import StressRunStatus

__all__ = ["StressRunStore"]


class StressRunStore(BaseRepository):

    # ========================================================
    # 写入
    # ========================================================

    @abstractmethod
    async def add_run(
            self,
            run: StressRunRecord,
    ) -> StressRunRecord:
        """插入一条运行记录，返回落库后的样子。"""

        raise NotImplementedError

    @abstractmethod
    async def add_metrics(
            self,
            run_id: UUID,
    ) -> StressRunMetricsRecord:
        """
        为一条运行建一份**空的 collecting 快照**。

        建运行时就建它，而不是等跑完再插：运行中的 metrics 接口要能
        返回「正在采集」的合法响应，而 404 会让前端把「还没有结果」
        误读成「这次运行不存在」。列默认值由数据库填。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_running(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:
        """
        queued -> running，写 ``started_at``。

        只在当前是 queued 时生效。已取消 / 已完成时返回 None —— 调用方
        应当安静地放弃，重复推进不是错误。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_cancelled(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:
        """
        把一条**未终结**的运行置为 cancelled。

        已终结时返回 None（取消来晚了不是错误，由服务层翻译成 409）。
        同时把还处于 collecting 的快照标成 incomplete：运行被取消了，
        那份快照永远不会再定稿。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_failed(
            self,
            run_id: UUID,
            *,
            error_code: str | None,
            error_message: str | None,
    ) -> StressRunRecord | None:
        """
        把一条**未终结**的运行置为 failed，并记录失败摘要。

        错误摘要只放「为什么没跑成」，不放 Trace / Span 明细。
        快照同时标成 incomplete。
        """

        raise NotImplementedError

    @abstractmethod
    async def update_metrics(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunMetricsRecord | None:
        """
        覆盖一份**未 final** 的聚合快照（collecting / incomplete）。

        运行中的 ``/metrics`` 必须能读到实时计数，而不是等到结束时才有
        数据。``final`` 快照不允许覆盖 —— 终态不可逆。

        Returns:
            更新后的快照；快照已是 final 或行不存在时返回 None。
        """

        raise NotImplementedError

    @abstractmethod
    async def complete_run(
            self,
            run_id: UUID,
            metrics: StressRunMetricsRecord,
    ) -> StressRunRecord | None:
        """
        running -> completed，并把聚合快照写入 metrics 表。

        **只从 running 进入**。这样取消赢在竞态里：取消先把状态改成
        cancelled 之后，这里的影响行数为 0，快照也不会被写成 final。

        Returns:
            完成后的运行记录；状态已不是 running 时返回 None，且
            不写任何快照列。
        """

        raise NotImplementedError

    # ========================================================
    # 读取
    # ========================================================

    @abstractmethod
    async def get_run(
            self,
            run_id: UUID,
    ) -> StressRunRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(
            self,
            run_id: UUID,
    ) -> StressRunMetricsRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def list_runs(
            self,
            *,
            status: StressRunStatus | None = None,
            limit: int = 20,
            cursor: StressRunCursor | None = None,
    ) -> list[StressRunRecord]:
        """
        按创建时间倒序列出运行。

        ``cursor`` 是上一页最后一条的 (created_at, run_id)。传它时只取
        严格排在它之后的记录 —— 用两列做 keyset 分页，而不是 offset：
        列表持续有新行插入，offset 会在翻页时漏行或重复行，且不报错。
        """

        raise NotImplementedError
