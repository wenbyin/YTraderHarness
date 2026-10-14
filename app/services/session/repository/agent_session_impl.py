"""
PostgreSQL implementation of AgentSessionRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, or_, select, update

from ..domain.entities import AgentSession
from ..models.sqlalchemy.agent_session import (
    AgentSession as AgentSessionModel,
)
from .agent_session import AgentSessionRepository


class AgentSessionRepositoryImpl(AgentSessionRepository):

    async def add(
            self,
            session: AgentSession,
    ) -> AgentSession:

        model = AgentSessionModel(
            session_id=session.session_id,
            agent_id=session.agent_id,
            status=session.status,
            meta=dict(session.meta),
            user_id=session.user_id,
            # ⚠️ 后加的列：ORM 声明 + models.sql 之后，**这里也要加一行**。
            # 漏了不会报错（列可空），只会静默写 NULL —— 与 agent_run
            # 那侧漏掉 caller_identity 是同一种坑（那里留过一段警告）。
            org_id=session.org_id,
            # 同上。它当前恒为 None（没有输入来源），但**这一行不能省** ——
            # 省了的话，将来有人开始填它时，值会在这一句被静默丢掉，
            # 而那时没人会想起来看这里。
            task_id=session.task_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    async def get(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSession | None:
        """
        归属过滤必须进 SQL，理由与语义见契约的 docstring。

        ⚠️ 刻意**不用** `session.get(AgentSessionModel, session_id)`
        -------------------------------------------------------
        那是按主键直取，加不上 WHERE —— 要么先取出来再在 Python 里比对
        （多一次全表读取的机会，且过滤条件藏在服务层，看仓储看不出
        权限），要么就得放弃过滤。这里用显式 SELECT，让"能不能看见"
        这件事留在数据层，任何调用方绕不过去。

        `or_(... IS NULL)` 是**存量兼容**，不是给将来留的口子：加这一列
        之前的会话没有归属，按约定"对谁都可见"（见 models.sql 末尾）。
        """

        model = await self._session.scalar(
            select(AgentSessionModel).where(
                AgentSessionModel.session_id == session_id,
                or_(
                    AgentSessionModel.user_id == user_id,
                    AgentSessionModel.user_id.is_(None),
                ),
            )
        )

        return None if model is None else self._to_entity(model)

    async def get_any(
            self,
            session_id: UUID,
    ) -> AgentSession | None:
        """
        不过滤归属。为什么这样是安全的，见契约的 docstring ——
        一句话：调用方（memory.extract）要的是**这一行的归属值**，
        而不是"这个人能不能看"，而写出去的索引行自己带 user_id 过滤。
        """

        model = await self._session.scalar(
            select(AgentSessionModel).where(
                AgentSessionModel.session_id == session_id,
            )
        )

        return None if model is None else self._to_entity(model)

    async def touch(
            self,
            session_id: UUID,
    ) -> None:
        """
        语义与理由见契约。显式写 updated_at，不依赖列的 onupdate ——
        这里没有任何别的字段变动可供它依附。

        改到 0 行（会话不存在）就到此为止：契约说明了为什么不抛。
        """

        await self._session.execute(
            update(AgentSessionModel)
            .where(AgentSessionModel.session_id == session_id)
            .values(updated_at=func.now())
        )

    async def ensure_thread_id(
            self,
            session_id: UUID,
            candidate: str,
    ) -> str:
        """
        条件更新 + 回读。语义与理由见契约的 docstring。

        ⚠️ 刻意用 `update()` 语句而不是"取实例 -> 赋值 -> flush"
        ----------------------------------------------------------
        后者不是原子的：两个事务都读到 NULL，各自赋值，最后写的赢 ——
        而两边都返回自己那个值，于是同一个会话有了两条 thread。
        `WHERE thread_id IS NULL` 让并发在下游变成"只有一行被改到"。

        并发安全依据（READ COMMITTED）：输的那个 UPDATE 会**阻塞**
        在赢家持有的行锁上，等它提交后重新求值 WHERE，此时
        thread_id 已非 NULL，于是改到 0 行 —— 不是读旧快照，是重读。
        """

        result = await self._session.execute(
            update(AgentSessionModel)
            .where(
                AgentSessionModel.session_id == session_id,
                AgentSessionModel.thread_id.is_(None),
            )
            .values(thread_id=candidate)
        )

        if result.rowcount == 1:
            return candidate

        # 改到 0 行有两种可能：已经有值了（并发输了），或会话不存在。
        # 用一句 SELECT 把两者分开 —— 不复用实例，避免读到身份映射里
        # 那个尚未刷新的旧值。
        stored = await self._session.scalar(
            select(AgentSessionModel.thread_id).where(
                AgentSessionModel.session_id == session_id
            )
        )

        if stored is None:
            raise ValueError(f"会话不存在：{session_id}")

        return stored

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: AgentSessionModel,
    ) -> AgentSession:

        return AgentSession.model_validate(model)
