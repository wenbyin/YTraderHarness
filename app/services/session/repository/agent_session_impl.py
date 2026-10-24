"""
PostgreSQL implementation of AgentSessionRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, or_, select, update

from ..domain.entities import AgentSession
from ..domain.schemas.enum import SessionStatus
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

    async def get_for_update(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSession | None:
        """
        可见性与 `get` 相同，额外加行锁。语义见契约。

        `with_for_update()` 生成 `SELECT ... FOR UPDATE`：两个事务同时读到
        active 时，后一个会阻塞到前一个提交，然后重读 —— 这正是
        `create_run` 与归档需要的串行化点。
        """

        model = await self._session.scalar(
            select(AgentSessionModel)
            .where(
                AgentSessionModel.session_id == session_id,
                or_(
                    AgentSessionModel.user_id == user_id,
                    AgentSessionModel.user_id.is_(None),
                ),
            )
            .with_for_update()
        )

        return None if model is None else self._to_entity(model)

    async def get_owned_for_update(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSession | None:
        """
        严格本人 + 行锁。理由见契约：NULL 存量会话不能被任何用户删除。
        """

        model = await self._session.scalar(
            select(AgentSessionModel)
            .where(
                AgentSessionModel.session_id == session_id,
                AgentSessionModel.user_id == user_id,
            )
            .with_for_update()
        )

        return None if model is None else self._to_entity(model)

    async def list_page(
            self,
            *,
            user_id: str,
            status: str | None,
            agent_id: str | None,
            updated_after: datetime | None,
            updated_before: datetime | None,
            offset: int,
            limit: int,
    ) -> tuple[list[AgentSession], int]:
        """
        语义与排序见契约。

        ⚠️ 条件只在 `_conditions` 里构造一次
        ---------------------------------
        count 与 items 必须是**同一组**筛选条件。这里不复制粘贴那几行 ——
        复制出来的第二份迟早会少一个条件，而症状是“总数比列表多”，很难
        从接口响应上看出是哪一边错了。

        两条语句在同一请求事务里执行。READ COMMITTED 下两者之间可能有极小
        的并发写入窗口（total 与 items 来自不同快照），与仓库其余列表接口
        的取舍一致；要严格同快照需要改写成一条带窗口函数的语句。
        """

        conditions = self._conditions(
            user_id=user_id,
            status=status,
            agent_id=agent_id,
            updated_after=updated_after,
            updated_before=updated_before,
        )

        total = (await self._session.execute(
            select(func.count())
            .select_from(AgentSessionModel)
            .where(*conditions)
        )).scalar_one()

        result = await self._session.execute(
            select(AgentSessionModel)
            .where(*conditions)
            .order_by(
                # updated_at 会在“发消息”时被 touch 推到当下，同一时刻可能
                # 有多行；补 session_id 让翻页稳定（不漏、不重）。
                AgentSessionModel.updated_at.desc(),
                AgentSessionModel.session_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return (
            [self._to_entity(model) for model in result.scalars()],
            total,
        )

    async def archive(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> bool:
        """
        严格本人的条件更新。语义见契约。
        """

        result = await self._session.execute(
            update(AgentSessionModel)
            .where(
                AgentSessionModel.session_id == session_id,
                AgentSessionModel.user_id == user_id,
            )
            .values(
                status=SessionStatus.ARCHIVED.value,
                updated_at=func.now(),
            )
        )

        return result.rowcount == 1

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
    def _conditions(
            *,
            user_id: str,
            status: str | None,
            agent_id: str | None,
            updated_after: datetime | None,
            updated_before: datetime | None,
    ) -> list:
        """
        列表与计数共用的 WHERE 条件。

        `user_id == user_id` 是**严格等号**，刻意没有 `OR user_id IS NULL`：
        列表只属于某个具体的人，见契约里那条说明。
        """

        conditions = [AgentSessionModel.user_id == user_id]

        if status is not None:
            conditions.append(AgentSessionModel.status == status)

        if agent_id is not None:
            conditions.append(AgentSessionModel.agent_id == agent_id)

        if updated_after is not None:
            conditions.append(
                AgentSessionModel.updated_at >= updated_after
            )

        if updated_before is not None:
            conditions.append(
                AgentSessionModel.updated_at <= updated_before
            )

        return conditions

    @staticmethod
    def _to_entity(
            model: AgentSessionModel,
    ) -> AgentSession:

        return AgentSession.model_validate(model)
