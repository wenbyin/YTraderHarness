"""
会话的读写契约。

对应 session.yaml：
    POST /sessions                  建会话
    GET  /sessions/{session_id}     查会话

⚠️ 没有 delete / archive
----------------------
规格里没有这两个端点（status 有 archived 这个取值，但产生不出来）。
因此本契约也不提供 —— 契约不该比规格多出接口。归档落地时，
"archived 的会话不许再开新运行"这条判断要一起加在
AgentRunService 上（见 domain/schemas/enum.py 的 SessionStatus.ARCHIVED）。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import CreateSessionRequest, Session

__all__ = ["AgentSessionService"]


class AgentSessionService(BaseService):

    @abstractmethod
    async def create_session(
            self,
            request: CreateSessionRequest,
            *,
            user_id: str | None,
            org_id: str | None = None,
    ) -> Session:
        """
        建一个会话，初始状态 active。

        user_id 是会话**归属**（Casdoor 的 sub），由 api 层从 access token
        取 —— 刻意不从请求体来，见 domain/schemas/session.py 的说明。
        允许 None（token 里没有 sub 时归一成 NULL = 公共，见 api 层）。

        org_id 是会话所属的**组织**（JWT 的 org claim），同样由 api 层从
        token 取。它与 user_id 有两处不同，别把两者当成一回事：

            它不是过滤键   —— 可见性只由 user_id 决定（见仓储的 get）
            空置是正常的   —— token 里没有组织信息就存 NULL，
                              Casdoor 各版本 claim 配置不一

        Raises:
            ValueError: agent_id 不合规（由 DTO 校验，见 CreateSessionRequest）。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_session(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> Session:
        """
        Raises:
            NotFoundError: 会话不存在，**或它不属于 user_id** ——
                           两者是同一个响应，不泄露 id 是否存在。
        """

        raise NotImplementedError
