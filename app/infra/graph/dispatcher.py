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
    async def cancel(self, pipeline_run_id: UUID) -> bool:
        """
        请求取消一次运行。返回"这个 run 我认得、已经通知它了"。

        ⚠️ 返回值是"**受理**了"，不是"取消**完成**了"
        ------------------------------------------
        真正的取消是异步的：任务要跑到下一个 await 点才真的抛出来，而收尾
        （rollback、记 cancelled、放锁）是 handler 的事。所以 True 只意味着
        "已经通知它了"。

        **False 的含义要分清**：要么这个 run 早就跑完/从没跑过（无事可做），
        要么它不归这个调度器管（别的进程）。调用方据此决定报 409 还是当作
        "已经结束了" —— 两者都不该被当成成功。

        实现必须**立即返回**（与 dispatch 同一条理由：调用方在 HTTP 请求里）。
        """
        raise NotImplementedError

    @abstractmethod
    async def is_active(self, pipeline_run_id: UUID) -> bool:
        """
        这个 run 归本调度器管、而且**还没跑完**吗（排队中 **或** 正在跑）。

        用途：回答"这条事件流还有必要挂着吗"。它与数据库里那份状态回答的
        **不是同一个问题**：

            数据库的 status   "这个 run 最后被记成了什么"
            本方法            "本进程手里还有它吗"

        两者合起来才能把「跑得慢」和「执行器没了」分开。数据库停在 running
        而这里说 False，意味着**跑它的那个东西已经不在了**（进程崩过、或
        重启过）—— 那种流如果不主动关，就再也不会有人来关它，连接白挂到
        天荒地老。

        ⚠️ 判据必须包含**排队中**，不能只报"正在跑"
        ----------------------------------------
        "建完运行立刻开 SSE"是最常见的时序：那一刻 run 已经落库（queued），
        但 handler 还没开工。只按"正在跑"回答的话，这个刚创建的 run 会被
        判成死的、连接当场被关掉 —— 恰好关在 run.started 发布之前。

        ⚠️ 它是**进程内**判断，不是集群内的
        ---------------------------------
        多 worker 部署时，跑在别的 worker 上的 run 在这里一律是 False，
        于是"DB 还是 running + 本进程说 False"会被误判成执行器死了。本项目
        当前 workers=1（同 thread_lock.py 的前提）；换多 worker 时这一条要
        跟着改成查共享存储。

        实现必须**立即返回**（调用方在事件流循环里）。
        """
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """停止接收新任务并等待在跑的任务结束。"""
        raise NotImplementedError
