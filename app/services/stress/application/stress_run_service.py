"""
StressRunService：压测运行的应用服务契约。

它只负责「运行记录」这一层：参数校验、落库、状态查询、取消。**不负责**
产生负载，也不负责聚合 Trace —— 那两条分别是 Runner 与 Aggregator 的
职责（设计 README §2）。当前阶段 Runner 由本地模拟器占位。

为什么 create 是 202 而不是 201
-----------------------------
运行记录确实建出来了，但它**还没跑**（queued）。202 表达的是「受理了
一个还没完成的动作」；201 会让调用方以为可以立刻去读最终指标。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas.stress import (
    StressRun,
    StressRunCreate,
    StressRunMetrics,
    StressRunPage,
    StressRunStatus,
)

__all__ = ["StressRunService"]


class StressRunService(BaseService):

    @abstractmethod
    async def create(
            self,
            request: StressRunCreate,
    ) -> StressRun:
        """
        建一条 queued 运行，并在事务提交后把执行交给 Runner。

        事务里做三件事：写 stress_runs、写空的 stress_run_metrics
        （collecting）、登记一个提交后调度钩子。

        为什么调度要等提交之后
        --------------------
        在事务里就启动执行的话，Runner 的第一次 UPDATE（queued ->
        running）看不到尚未提交的行，影响 0 行，于是这次运行永远停在
        queued，而调用方看到的是创建成功。钩子必须在 commit 之后跑。

        Raises:
            ValueError: 参数之间自相矛盾（Pydantic 已在 HTTP 层挡掉
                大部分；到这里的是跨字段语义问题）。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_runs(
            self,
            *,
            status: StressRunStatus | None = None,
            limit: int = 20,
            cursor: str | None = None,
    ) -> StressRunPage:
        """
        按创建时间倒序分页。

        Raises:
            ValueError: cursor 不是本服务签发过的形状 -> 400。
        """

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            run_id: UUID,
    ) -> StressRun:
        """
        取一条运行的参数与状态。

        Raises:
            NotFoundError: 运行不存在 -> 404。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_metrics(
            self,
            run_id: UUID,
    ) -> StressRunMetrics:
        """
        取聚合结果。

        运行存在但还没采到样本时返回 aggregation_status=collecting，
        **不是** 404 —— 「还没结果」与「没有这次运行」是两件事。

        Raises:
            NotFoundError: 运行不存在 -> 404。
        """

        raise NotImplementedError

    @abstractmethod
    async def cancel(
            self,
            run_id: UUID,
    ) -> StressRun:
        """
        请求取消。

        取消是协作式的：queued 的运行直接置 cancelled；running 的运行
        也置 cancelled，执行侧下一次写回会因为状态已不是 running 而
        无效 —— 于是最终停在 cancelled。

        返回的是**取消之后真实所处的位置**，不是一个无脑的成功。

        Raises:
            NotFoundError: 运行不存在 -> 404。
            ConflictError: 运行已经终结（完成 / 失败 / 已取消）-> 409。
        """

        raise NotImplementedError
