"""
Stress Runner 的最小协议。

让服务层同时接受真实 ``HttpStressRunner`` 与本地 ``SimulatedStressRunner``，
而不把实现细节写进服务契约。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from ..domain.schemas.stress import StressRunCreate

__all__ = ["StressRunner"]


class StressRunner(Protocol):

    def schedule(
            self,
            run_id: UUID,
            config: StressRunCreate,
    ) -> None:
        """事务提交后启动一次运行。同步、非阻塞。"""
        ...

    def cancel(self, run_id: UUID) -> None:
        """协作式取消；接口返回不代表在途请求已停止。"""
        ...

    def stop(self) -> None:
        """进程关闭时停止所有 Campaign。"""
        ...
