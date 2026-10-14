"""
MessageService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext

from ..domain.schemas import MessagePage, decode_cursor
from ..repository.agent_session import AgentSessionRepository
from ..repository.message import MessageRepository
from ._assembly import build_message_page
from .message_service import DEFAULT_LIMIT, MessageService


class MessageServiceImpl(MessageService):

    def __init__(
            self,
            *,
            message_repository: MessageRepository,
            agent_session_repository: AgentSessionRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._messages = message_repository
        self._sessions = agent_session_repository

    async def list_messages(
            self,
            session_id: UUID,
            *,
            user_id: str,
            limit: int = DEFAULT_LIMIT,
            before: str | None = None,
            after: str | None = None,
    ) -> MessagePage:

        # ⚠️ 校验顺序：先会话（404），再游标（400）
        # -------------------------------------
        # 两步都是"请求有问题"，谁先谁后决定了同时踩中时回哪一个码。
        # 选会话优先，与 AgentRunServiceImpl.create_run 的排法一致
        # （那里也是先查会话存在性再校验 input）—— 同一张 API 上，
        # "路径指向的东西不存在"比"参数写错了"更接近根因。
        #
        # 归属过滤在仓储里做：读别人的会话拿到的是 None，与不存在的会话
        # 走同一条 404 —— 会话的历史消息因此不会漏给非本人。
        session = await self._sessions.get(session_id, user_id=user_id)

        if session is None:
            raise NotFoundError(f"会话不存在：{session_id}")

        # 两个游标同时给是无解的：一个说"往更旧取"，一个说"往更新取"。
        # 挑一个执行等于静默丢掉另一个 —— 客户端会拿到一页它没要的数据，
        # 而且看不出为什么。因此明确拒掉。
        if before is not None and after is not None:
            raise ValueError(
                "before 与 after 不能同时给：游标只能指向一个方向。"
            )

        # 解析失败会抛 ValueError（-> 400），理由见 decode_cursor 的说明：
        # 坏游标不能退化成"当作没给"，那会让客户端在死循环里翻同一页。
        before_seq = None if before is None else decode_cursor(before)
        after_seq = None if after is None else decode_cursor(after)

        entities, has_more_before, has_more_after = (
            await self._messages.list_page(
                session_id,
                limit=limit,
                before_seq=before_seq,
                after_seq=after_seq,
            )
        )

        return build_message_page(
            session_id=session_id,
            entities=entities,
            limit=limit,
            has_more_before=has_more_before,
            has_more_after=has_more_after,
        )
