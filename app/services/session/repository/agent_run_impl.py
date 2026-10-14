"""
PostgreSQL implementation of AgentRunRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。

两个调用场景因此形状不同：

    请求路径（建运行）      组合根给的请求级 session，随请求提交
    后台路径（翻状态）      RepositoryFactory.scope() 自开的短事务，
                            见 application/agent_run_dispatch.py

第二条是必须自己开会话的：派发 handler 跑在调度器的后台任务里
（事务早已提交），而且**不能**借用请求的 session —— 那个连接已经还回池子了。

状态改写走 ORM 的取行+赋值（而不是 Core 的 UPDATE），与 add / get 一致：
updated_at 的 onupdate 因此照常生效，不必手动写时间。
"""

from __future__ import annotations

from uuid import UUID

from ..domain.entities import AgentRun
from ..domain.schemas.enum import RunStatus
from ..models.sqlalchemy.agent_run import AgentRun as AgentRunModel
from .agent_run import AgentRunRepository


class AgentRunRepositoryImpl(AgentRunRepository):

    async def add(
            self,
            run: AgentRun,
    ) -> AgentRun:

        model = AgentRunModel(
            run_id=run.run_id,
            session_id=run.session_id,
            status=run.status,
            input=dict(run.input),
            meta=dict(run.meta),
            error_type=run.error_type,
            error_message=run.error_message,
            user_id=run.user_id,
            # ⚠️ 这一行**曾经漏掉**，症状极具迷惑性：工具级 checkauth 永远
            # 看到"调用者持有 []"，也就是"没有身份"，而服务层明明设了、
            # ORM 模型也明明声明了 —— 断的正是中间这一跳。
            #
            # 列有 `DEFAULT '{}'::jsonb`，所以漏传**不会报错**，只会静默写
            # 成空对象。而空的 `{}` 与"有身份但没角色"在拦截器里长得一模
            # 一样（都走 `caller.get("roles") or ()` -> `[]`），于是排查
            # 会一路往 token / Casdoor claim 那边找，离原因很远。
            #
            # 新增「后加的列」时，除了 ORM 模型与 models.sql，**这个 add()
            # 也要一起改** —— 这是三处声明里唯一一处没有测试盯着的，
            # 而 `_to_entity`（`model_validate(model)` 按属性名映射）那侧
            # 是自动的，看不出漏没漏。
            caller_identity=dict(run.caller_identity),
            # 后加的列。**这里漏掉不会报错**（列可空）—— 只会静默写 NULL，
            # 症状是"运行行上永远看不到这次用了哪份职责"，而服务层明明设了。
            # 上面那段注释就是当年 caller_identity 漏掉这一跳留下的。
            duty_key=run.duty_key,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    async def get(
            self,
            run_id: UUID,
    ) -> AgentRun | None:

        model = await self._session.get(AgentRunModel, run_id)

        return None if model is None else self._to_entity(model)

    # ========================================================
    # Status transitions
    # ========================================================

    async def mark_running(
            self,
            run_id: UUID,
    ) -> None:

        model = await self._require(run_id)

        model.status = RunStatus.RUNNING.value

        await self._session.flush()

    async def finish(
            self,
            run_id: UUID,
            status: str,
            *,
            error_type: str | None = None,
            error_message: str | None = None,
    ) -> None:

        model = await self._require(run_id)

        model.status = status
        model.error_type = error_type
        model.error_message = error_message

        await self._session.flush()

    # ========================================================
    # internals
    # ========================================================

    async def _require(
            self,
            run_id: UUID,
    ) -> AgentRunModel:
        """
        取运行行，取不到就抛。

        这里**不能**返回 None 了事：状态改写路径上"行不存在"只有一种
        原因 —— 接线错了（handler 拿了个不属于本表的 id）。静默不写会
        让运行永远停在中间态，而没人知道为什么。
        """

        model = await self._session.get(AgentRunModel, run_id)

        if model is None:
            raise ValueError(f"运行不存在：{run_id}")

        return model

    @staticmethod
    def _to_entity(model: AgentRunModel) -> AgentRun:

        return AgentRun.model_validate(model)
