"""
Service contract for ToolExecution.

一次工具调用的账本。「先查后执行」这件事的**判定**在这里，
判定所依据的那份集合在 domain/schemas/enum.py。

为什么调用方递的是 `arguments` 而不是 `param_hash`
-----------------------------------------------
判重键是"这次调用是不是同一件事"的定义，**属于本服务**。让调用方
自己算摘要，等于把定义复制到每一个调用点 —— 而它错了不会报错，
只会静静地每次都判成"新调用"，也就是这张表完全失效。

事务边界
--------
由调用方决定，本服务内部不 commit / rollback（与其余服务同一条）。
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import ToolExecutionStatus

__all__ = ["ToolExecutionService"]


class ToolExecutionService(BaseService):

    # ========================================================
    # 抢占
    # ========================================================

    @abstractmethod
    async def claim(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            tool_key: str,
            arguments: Mapping[str, Any],
    ) -> ToolExecution | None:
        """
        原子抢占这个键。返回 None = **没抢到，不许执行**。

        调用方的用法是：`claim` 返回 None 就**放弃执行**，并用
        `find` 取回那一行以便说清"为什么不执行"。两件事分开是因为
        **判定只能在 claim 里做**（它才有原子性），而 find 只负责
        措辞。

        ⚠️ 不要把这里写成"先 find 再 claim"
        ----------------------------------
        `find` 的返回值**不能**用来决定要不要执行 —— 那样两个并发
        分支（本图是 Send 扇出，同一轮模型可以发出两个相同调用）
        会都读到"没有记录"，于是一个副作用执行两遍。

        ⚠️ `arguments` 是**模型给的原始参数**，见 _execution.param_hash_of。
        """

        raise NotImplementedError

    # ========================================================
    # 读
    # ========================================================

    @abstractmethod
    async def find(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            tool_key: str,
            arguments: Mapping[str, Any],
    ) -> ToolExecution | None:
        """
        取这个键上的那一行，没有就 None。

        用途只有两个：拒绝时写清"它现在是什么状态"、以及排查。
        **不参与判定**（见 claim 的说明）。
        """

        raise NotImplementedError

    # ========================================================
    # 落终态
    # ========================================================

    @abstractmethod
    async def settle(
            self,
            execution_id: UUID,
            *,
            status: ToolExecutionStatus,
            result: Any = None,
            error: dict[str, Any] | None = None,
            error_exc: BaseException | None = None,
    ) -> ToolExecution:
        """
        把一行推进到某个状态。

        ⚠️ 只允许推进到终态或 `RUNNING`，**不接受 PENDING**
        ------------------------------------------------
        PENDING 是"刚抢到、还没跑"，它是 claim 的产物。允许 settle 到
        PENDING 等于开一条把"正在跑的记录"退回起点的路，而那条路一旦
        被走到，别的分支就会以为"没人持有"从而抢走执行权。

        error 与 error_exc 二选一
        ------------------------
        `error_exc` 是给"手上正好有个异常对象"的调用方用的便捷入口，
        本服务会把它经 _execution.execution_error 规范成存库的形状。
        两个都传以 `error` 为准（它是已经规范好的那份）。
        """

        raise NotImplementedError
