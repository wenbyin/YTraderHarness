"""
Repository contract for ToolExecution.

只管 `tool_execution` 一张表。它不做任何策略判断 ——
「这次让不让做」是服务层的 decide_existing / decide_new 说了算，
仓储只负责把查询与写入变成 SQL。

⚠️ 为什么 `lock` 单独一个方法
--------------------------
守卫的「查最近一次 + 数次数 + 插入」必须先拿一把按 `match_key`
的 advisory lock，否则并发分支（本图是 Send 扇出）会同时读到
「还没满」，于是都执行。锁的作用域是**事务**，所以它必须和后面的
读写在同一个事务里 —— 仓储不自己 commit，事务边界留给调用方。

⚠️ 为什么没有 `get_or_create`
--------------------------
判定不是两个操作。调用方拿锁之后读到的 latest / count 与随后的
insert 必须在同一个临界区里；把「查」和「插」拆给两个调用点，
早晚会有人只调其中一个。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import ToolExecutionStatus

__all__ = ["ToolExecutionRepository"]


class ToolExecutionRepository(BaseRepository):

    # ========================================================
    # 并发
    # ========================================================

    @abstractmethod
    async def lock(self, match_key: str) -> None:
        """
        取得按 `match_key` 的事务级 advisory lock。

        同一个 match_key 的并发调用会被串行化；不同的 match_key
        即使哈希碰撞也只会多一次排队，不会影响正确性。

        ⚠️ 必须在同一个事务里调用本方法与后面的读 / 写：事务一结束
        锁就释放。
        """

        raise NotImplementedError

    # ========================================================
    # 读
    # ========================================================

    @abstractmethod
    async def find_by_call_id(
            self,
            *,
            run_id: UUID,
            tool_call_id: str,
    ) -> ToolExecution | None:
        """
        按 `(run_id, tool_call_id)` 取那一行。

        这是 replay / HITL resume 的去重入口：同一次模型调用在图中
        重跑时，应该复用这一行，而不是再消耗一次额度。

        没有 tool_call_id 的图外直调不走这里。
        """

        raise NotImplementedError

    @abstractmethod
    async def status_counts(
            self,
            *,
            match_key: str,
            window_start: datetime | None,
    ) -> dict[ToolExecutionStatus, int]:
        """
        窗口内该 match_key 按状态分组的 `counted=True` 记录数。

        返回的是**原始状态计数**，不做任何策略合并：

            SUCCESS / PENDING / RUNNING  调用方按 success 配额合并
            FAILED                       调用方按 error 配额判定
            UNKNOWN                      调用方按 unknown 配额判定

        `PENDING/RUNNING` 也返回，是因为它们在途、可能变成成功；
        调用方会把它们算进 success 名额，避免并发分支同时通过。

        只数 counted：被拒绝 / 中断 / 取消 / 守卫拦下的行没有真正
        执行，不该占额度。`window_start=None` 表示整个生命周期。
        """

        raise NotImplementedError

    # ========================================================
    # 写
    # ========================================================

    @abstractmethod
    async def insert(
            self,
            execution: ToolExecution,
    ) -> ToolExecution:
        """
        插入一条新记录。调用方负责填好所有字段。

        预占时 status=PENDING、counted=True —— 让并发的其它分支
        在计数里看见「有人正在跑」。
        """

        raise NotImplementedError

    @abstractmethod
    async def reopen(
            self,
            execution_id: UUID,
    ) -> ToolExecution:
        """
        把同一次工具调用的旧记录重置成 PENDING，准备重新执行。

        用于 resume（旧状态 INTERRUPT）与按策略放行的重试
        （REJECTED / CANCELLED / BLOCKED / FAILED / UNKNOWN）。
        `counted` 回到 True，上一轮的 result / error 清空 ——
        这一行表达**当前这次尝试**，与旧账本的 claim 重置同一条立场。

        ⚠️ 不是所有状态都能 reopen：SUCCESS 走 REPLAY，
        PENDING/RUNNING 直接拒绝。判定在服务层。
        """

        raise NotImplementedError

    @abstractmethod
    async def settle(
            self,
            execution_id: UUID,
            *,
            status: ToolExecutionStatus,
            counted: bool,
            result: Any = None,
            error: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """
        把一行推进到终态。

        按主键更新，不是按判重键 —— 此时那一行已经是我们预占的。

        `result` / `error` 的默认 None 是**覆盖**语义：每次 settle
        都把两列一起写成给定值，不做"只改非空的那个"。理由与 reopen
        的重置一致 —— "分不清是没传还是真的要清空"是 COALESCE 式
        写法最难查的一类错。

        `counted` 也在这一步修正：预占时是 True，落终态时按结果
        改成 False（拒绝 / 中断 / 取消 / 缓存命中）。
        """

        raise NotImplementedError
