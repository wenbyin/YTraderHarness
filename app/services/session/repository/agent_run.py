"""
Repository contract for AgentRun.

只管 agent_run 一张表。与 AgentSessionRepository 分开的理由见那个文件。

⚠️ 现在有写、单条读，以及会话删除需要的一条窄查询
----------------------------------------------
"某个会话发生过什么"的完整历史仍然没有接口；删除会话只需要回答一个
更窄的问题：**这个会话还有没有没结束的运行**。因此这里加的是
`list_unfinished_for_session`，不是通用分页列表 —— 等事件流 / 历史接口
真的需要按会话翻运行记录时再加，那时也能顺手确定要不要分页。

那条 `idx_agent_run_session(session_id, created_at DESC)` 仍然保留：
它是表设计的一部分（这张表天然按会话+时间查）。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import AgentRun

__all__ = ["AgentRunRepository"]


class AgentRunRepository(BaseRepository):

    @abstractmethod
    async def add(
            self,
            run: AgentRun,
    ) -> AgentRun:
        """新建一次运行。"""

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            run_id: UUID,
    ) -> AgentRun | None:
        raise NotImplementedError

    @abstractmethod
    async def mark_running(
            self,
            run_id: UUID,
    ) -> None:
        """
        把运行置为 running。

        派发 handler 在真正调图**之前**调它 —— 排在运行开始前是刻意的：
        图跑挂了也留下"它确实开始过"这个事实，而不是一直停在 queued
        让"没跑"和"跑挂了"看起来一样。

        Raises:
            ValueError: 运行不存在（后台路径上只有"接线错了"会走到）。
        """

        raise NotImplementedError

    @abstractmethod
    async def finish(
            self,
            run_id: UUID,
            status: str,
            *,
            error_type: str | None = None,
            error_message: str | None = None,
    ) -> None:
        """
        写终态：completed / failed / cancelled。

        error_type / error_message 只在失败时有值 —— 规格的 Run 不回它们，
        **落库是它们唯一的去处**，否则一次失败的运行在库里只剩 failed
        一个词。

        Raises:
            ValueError: 运行不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_unfinished_for_session(
            self,
            session_id: UUID,
    ) -> list[AgentRun]:
        """
        列出会话里**还没结束**的运行：queued / running / waiting_input。

        调用方是会话删除（归档）—— 它要先把这些运行结束掉，否则归档后的
        会话里还挂着一件“没做完的事”。终态（completed / failed /
        cancelled）不返回。

        正常路径上同一个会话最多一条（进程内 thread 锁保证），但这里按
        列表返回：锁是进程内的，重启 / 多进程下可能出现孤儿，删除逻辑
        不该假定“只有一条”。
        """

        raise NotImplementedError

    @abstractmethod
    async def cancel_if_unfinished(
            self,
            run_id: UUID,
    ) -> bool:
        """
        条件更新：只有**未终态**的运行才改成 cancelled。

        返回是否改到了行。False 表示它已经结束（或不存在）—— 调用方据此
        当成“无事可做”，而不是报错。

        ⚠️ 只用于**没有进程内任务**的运行（waiting_input、调度器尚未受理的
        queued、进程重启后的孤儿）。有任务的 queued/running 必须走
        `GraphDispatcher.cancel`：那样 handler 才会做 checkpoint 回滚、
        写中断记录并放锁；直接改库会让图继续跑。
        """

        raise NotImplementedError
