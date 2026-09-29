"""
图调度端口。

为什么需要这个端口
------------------
`create_document` 跑在请求级事务里（dependencies.py 的
`session.begin()` 包住整个请求）。流水线绝不能在里面跑：

    - 后台任务在事务提交前读到的是**未提交**的版本，必然查不到
    - 一次运行可能几分钟，压在请求上会超时

所以业务侧只负责**登记**一次调度，由组合根在事务提交后触发。

将来换真队列时，把组合根的 post-commit 钩子换成
"在请求事务内写入 outbox 行 + worker 轮询" 即可 ——
那是唯一能跨网络跳保持原子的模式。端口本身不用动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

__all__ = [
    "DispatchEnvelope",
    "DispatchHandle",
    "GraphDispatcher",
]


@dataclass(frozen=True, slots=True)
class DispatchEnvelope:
    """
    一次调度请求。

    pipeline_run_id 与 thread_id 由**调用方**先生成：
    调用方需要在同一个请求里把它返回给客户端（job_id）
    并写进业务绑定，不能等调度器回调才知道。
    """

    graph_name: str

    pipeline_run_id: UUID

    thread_id: str

    #: 业务自定义载荷。调度器不解释它，原样交给已注册的 handler。
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DispatchHandle:
    """受理结果。"""

    pipeline_run_id: UUID

    thread_id: str

    accepted: bool

    message: str = ""


class GraphDispatcher(ABC):
    """
    调度器契约。

    实现必须是**立即返回**的：调用方在 HTTP 请求内调用它，
    不能被一次完整的图运行阻塞。
    """

    @abstractmethod
    async def dispatch(
            self,
            envelope: DispatchEnvelope,
    ) -> DispatchHandle:
        """受理一次运行，立即返回。"""
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """停止接收新任务并等待在跑的任务结束。"""
        raise NotImplementedError
