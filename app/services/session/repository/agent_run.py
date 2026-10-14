"""
Repository contract for AgentRun.

只管 agent_run 一张表。与 AgentSessionRepository 分开的理由见那个文件。

⚠️ 现在只有写与单条读，**没有** list_for_session
----------------------------------------------
"某个会话发生过什么"迟早要查（表上那条
`idx_agent_run_session(session_id, created_at DESC)` 就是为它留的），
但**当前没有任何接口或调用方需要它**。加一个没人调的方法等于把"以后大概
要"写进代码 —— 等 ④ 事件流 / ⑤ 恢复 / 会话历史这几个里有一个真的需要时
再加，那时也能顺手确定要不要分页。

保留那条索引而不是一起砍掉：索引是**表设计**的一部分（这张表天然按
会话+时间查），删了将来要补一次迁移；而一个方法删了只是删几行。
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
        写终态：completed / failed（将来还有 cancelled）。

        error_type / error_message 只在失败时有值 —— 规格的 Run 不回它们，
        **落库是它们唯一的去处**，否则一次失败的运行在库里只剩 failed
        一个词。

        Raises:
            ValueError: 运行不存在。
        """

        raise NotImplementedError
