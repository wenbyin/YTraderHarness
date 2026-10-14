"""
Repository contract for ToolExecution.

只管 tool_execution 一张表。它不做任何策略判断 ——
"这个状态能不能重跑"是 domain/schemas/enum.py 的
RETRYABLE_EXECUTION_STATUSES 说了算，实现只是把那份集合翻译成
SQL 的 WHERE 条件。

⚠️ 为什么没有 `insert` 也没有 `先查再插`
--------------------------------------
判重这件事**不是**两个操作。调用方能做的、且唯一安全的动作是
`claim`：一条 INSERT ... ON CONFLICT 语句里同时完成"查"与"占"。

    先查再插的窗口里，另一个并发分支（本图是 Send 扇出，同一轮
    模型可以发出两个完全相同的调用）会看到"没有记录"，于是两个都
    执行 —— 而这张表存在的全部意义就是挡住这件事。

`find` 因此**不参与判定**，只用来写拒绝时的文案（"它现在是什么
状态"），见 application 层的说明。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import ToolExecutionStatus

__all__ = ["ToolExecutionRepository"]


class ToolExecutionRepository(BaseRepository):

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
            param_hash: str,
    ) -> ToolExecution | None:
        """
        原子地抢占这个键，写下一行 `PENDING`。返回 None = **没抢到**。

        "没抢到"只有一种含义：那一行已经存在，**且它当前的状态不在
        RETRYABLE_EXECUTION_STATUSES 里** —— 也就是它要么正被另一个
        分支跑着（PENDING / RUNNING），要么已经跑成功了（SUCCESS），
        要么不知道跑没跑（UNKNOWN）。调用于是必须**放弃执行**。

        抢占成功时，这一行会被**重置**：上一轮尝试的 result / error
        一起清空。一行只表达"当前这次尝试"，历史尝试的痕迹不保留
        —— 要保留就得另开一张带 attempt 维度的表，而那与
        「(session, run, tool, param) 一个键一行」的判重模型冲突。

        ⚠️ 实现必须是一条语句
        --------------------
        见模块 docstring。任何"先 select 再 insert/update"的写法都会
        把并发窗口重新打开，而症状是"两个相同调用都执行了"，没有任何
        报错。
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
            param_hash: str,
    ) -> ToolExecution | None:
        """
        按判重键取那一行，没有就 None。

        ⚠️ **不要**用它来做"要不要执行"的决定 —— 那是 claim 的事
        （理由见模块 docstring）。它只服务于两件事：

            拒绝时的文案      告诉模型/运维它现在是什么状态
            观测与排查        这一次调用之前发生过什么
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
    ) -> ToolExecution:
        """
        把一行推进到终态（或 `RUNNING` 这个"不确定"态）。

        按主键更新，不是按判重键 —— 此时那一行已经是我们抢到的，
        再按判重键找一次只是多一次索引扫描，且会掩盖"id 传错了"。

        `result` / `error` 的默认 None 是**覆盖**语义：每次 settle 都
        把两列一起写成给定的值，不做"只改非空的那个"。理由与 claim
        的重置一致 —— 一行只表达当前这次尝试，而"分不清是没传还是
        真的要清空"正是 COALESCE 式写法最难查的一类错。
        """

        raise NotImplementedError
