"""
AgentSessionService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext
from app.infra.graph import GraphDispatcher, get_thread_run_lock

from ..domain.entities import AgentSession as AgentSessionEntity
from ..domain.schemas import (
    CreateSessionRequest,
    RunStatus,
    Session,
    SessionPage,
    SessionStatus,
)
from ..repository.agent_run import AgentRunRepository
from ..repository.agent_session import AgentSessionRepository
from ._assembly import build_session, build_session_page
from .agent_session_service import AgentSessionService

logger = logging.getLogger(__name__)


class AgentSessionServiceImpl(AgentSessionService):

    def __init__(
            self,
            *,
            agent_session_repository: AgentSessionRepository,
            agent_run_repository: AgentRunRepository,
            dispatcher: GraphDispatcher | None = None,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._sessions = agent_session_repository

        #: 删除会话时要先结束它的未终态运行。同模块的两个仓储，直接注入；
        #: 不复用 AgentRunService —— 它有自己的准入/恢复语义，而这里只需要
        #: “列出未终态 + 条件取消”。
        self._runs = agent_run_repository

        #: 判断一条运行**在本进程里还有没有任务**。工厂按参数名从池子注入；
        #: 单元测试直接构造时为 None，那时所有未终态运行都走数据库硬取消。
        self._dispatcher = dispatcher

    async def create_session(
            self,
            request: CreateSessionRequest,
            *,
            user_id: str | None,
            org_id: str | None = None,
    ) -> Session:

        now = datetime.now(timezone.utc)

        entity = await self._sessions.add(
            AgentSessionEntity(
                session_id=uuid4(),
                agent_id=request.agent_id,
                # 新建一律 active。archived 只能通过 DELETE /sessions/{id}
                # 到达（见本类的 delete_session）。
                status=SessionStatus.ACTIVE.value,
                meta=dict(request.metadata),
                # 归属来自 access token，不从请求体取（见契约的说明）。
                user_id=user_id,
                # 组织同理：来自 token 的 claim，取不到就是 None。
                # 它只是归属的记录，**不参与**归属过滤（那是 user_id）。
                org_id=org_id,
                created_at=now,
                updated_at=now,
            )
        )

        return build_session(entity)

    async def get_session(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> Session:

        # 归属过滤在仓储里做（进 SQL，不在这里事后比对）：取不到就一并
        # 当作不存在 —— 别人的会话与不存在的会话拿到完全一样的 404。
        entity = await self._sessions.get(session_id, user_id=user_id)

        if entity is None:
            raise NotFoundError(f"会话不存在：{session_id}")

        return build_session(entity)

    async def list_sessions(
            self,
            *,
            user_id: str,
            status: SessionStatus | None = SessionStatus.ACTIVE,
            agent_id: str | None = None,
            updated_after: datetime | None = None,
            updated_before: datetime | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> SessionPage:

        # 没有身份就没有“我的会话”。显式短路，而不是让一条必然为空的 SQL
        # 去回答这个问题。
        if not user_id:
            return SessionPage(items=[], total=0)

        after = self._as_utc(updated_after, "updated_after")
        before = self._as_utc(updated_before, "updated_before")

        if after is not None and before is not None and after > before:
            raise ValueError(
                "updated_after 不能晚于 updated_before。"
            )

        entities, total = await self._sessions.list_page(
            user_id=user_id,
            status=None if status is None else status.value,
            agent_id=agent_id,
            updated_after=after,
            updated_before=before,
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        return build_session_page(entities, total)

    async def delete_session(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> None:

        # 先锁住会话行：与 create_run / resume_run 的 `get_for_update` 配对，
        # 保证“读到 active 之后归档插进来”这种交错不会发生。严格本人 ——
        # NULL 存量会话在这里得到与不存在完全一样的 404。
        session = await self._sessions.get_owned_for_update(
            session_id, user_id=user_id,
        )

        if session is None:
            raise NotFoundError(f"会话不存在：{session_id}")

        runs = await self._runs.list_unfinished_for_session(session_id)

        #: 有进程内任务、要在提交后请求取消的运行。
        to_cancel: list[UUID] = []

        #: 没有进程内任务、已经被数据库直接记为 cancelled 的运行。
        #: 提交后要为它们放掉 thread 锁 —— 尤其是 waiting_input：那个
        #: handler 刻意留着锁等 resume，归档之后不会再有 resume 了。
        hard_cancelled: list[UUID] = []

        for run in runs:

            # waiting_input 没有任务可 cancel（它停在那里等人）。其它的先问
            # 调度器：本进程有它才算“还在跑”，否则可能是刚结束、尚未被
            # post-commit 受理，或进程重启留下的孤儿。
            active = (
                run.status != RunStatus.WAITING_INPUT.value
                and self._dispatcher is not None
                and await self._dispatcher.is_active(run.run_id)
            )

            if active:
                # 有任务的走 dispatcher：handler 负责回滚 checkpoint、写
                # 中断记录、放锁。直接改库会让图继续在后台跑。
                to_cancel.append(run.run_id)
                continue

            # 条件更新只改未终态的；改到 0 行说明它刚好结束了，无事可做。
            if await self._runs.cancel_if_unfinished(run.run_id):
                hard_cancelled.append(run.run_id)

        # 已归档时跳过 UPDATE：重复删除不应把 updated_at 推前，否则一条
        # 早已删除的会话会跳到列表顶部。
        if session.status != SessionStatus.ARCHIVED.value:

            archived = await self._sessions.archive(
                session_id, user_id=user_id,
            )

            # 锁是本人行的、归属不会在事务里改变，理论上必为 True。
            # 真为 False 说明行在这一刻消失了（例如运维直接删库），
            # 按不存在处理。
            if not archived:
                raise NotFoundError(f"会话不存在：{session_id}")

        if not to_cancel and not hard_cancelled:
            return

        # ⚠️ 取消与放锁都放在 post-commit
        # ---------------------------------
        # 事务可能回滚（上面的 archive 返回 False、或提交失败）。在事务里
        # 发取消请求的话，回滚后会话还活着，运行却已经被要了命。
        async def _after_commit() -> None:
            self._release_thread_locks(session, hard_cancelled)
            await self._request_cancellations(to_cancel)

        self.post_commit.append(_after_commit)

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _as_utc(
            value: datetime | None,
            name: str,
    ) -> datetime | None:
        """
        Query 解析出来的 datetime 必须是 aware 的。

        naive 时间没有唯一含义（服务器本地时区？UTC？），而 `updated_at`
        是 TIMESTAMPTZ。静默按某个时区补上，会让“差 8 小时”这种错误一直
        到用户发现列表少了数据才暴露。因此直接 400。
        """

        if value is None:
            return None

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{name} 必须带时区偏移，例如 2026-10-18T00:00:00Z。"
            )

        return value.astimezone(timezone.utc)

    @staticmethod
    def _release_thread_locks(
            session: AgentSessionEntity,
            run_ids: list[UUID],
    ) -> None:
        """
        放掉被硬取消的运行持有的 thread 锁。

        `waiting_input` 的 handler 在 interrupt 那条路径上**刻意不放锁**
        （它继承给 resume），而归档之后不会再有 resume —— 不补放就会
        永久泄漏这个会话的锁。release 对“不持有的键”是 no-op，因此对
        孤儿运行、以及已经由 handler 放过的锁重复调用都安全。

        同步调用是刻意的：它只碰进程内字典，没有 I/O。
        """

        if not run_ids:
            return

        lock = get_thread_run_lock()

        for run_id in run_ids:
            try:
                lock.release(session.thread_id, run_id=run_id)
            except Exception:  # noqa: BLE001 —— 提交后，异常只能记日志
                logger.exception(
                    "Failed to release thread lock for run %s.", run_id
                )

    async def _request_cancellations(
            self,
            run_ids: list[UUID],
    ) -> None:
        """
        提交后请求调度器取消。返回 False 只记 warning：通常是它刚好跑完，
        或不在本进程（workers>1 的已知前提，见 dispatcher 契约）。
        """

        if self._dispatcher is None or not run_ids:
            return

        for run_id in run_ids:
            try:
                accepted = await self._dispatcher.cancel(run_id)

                if not accepted:
                    logger.warning(
                        "Cancel for run %s was not accepted after archiving "
                        "its session; it may have just finished or be owned "
                        "by another process.",
                        run_id,
                    )

            except Exception:  # noqa: BLE001 —— 事务已提交，不能因此 500
                logger.exception(
                    "Failed to request cancellation for run %s.", run_id
                )
