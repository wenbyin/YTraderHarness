"""
AgentSessionService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext

from ..domain.entities import AgentSession as AgentSessionEntity
from ..domain.schemas import (
    CreateSessionRequest,
    Session,
    SessionStatus,
)
from ..repository.agent_session import AgentSessionRepository
from ._assembly import build_session
from .agent_session_service import AgentSessionService


class AgentSessionServiceImpl(AgentSessionService):

    def __init__(
            self,
            *,
            agent_session_repository: AgentSessionRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._sessions = agent_session_repository

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
                # 新建一律 active。archived 需要一个归档动作才能到达，
                # 而那个端点还没有 —— 见 enum.SessionStatus 的说明。
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
