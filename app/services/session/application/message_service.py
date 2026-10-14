"""
会话消息的读契约。

对应 messages.yaml：
    GET /sessions/{session_id}/messages     取一页历史

⚠️ 只有读，没有写
--------------
消息的写入**不经过本服务**，也不该经过：两个写入点各自在自己的上下文里
（建运行的请求事务 / 派发 handler 的短事务），需要的依赖与事务边界都与
"读一页"完全不同。把它们塞进同一个服务，服务就得同时回答"我现在在哪个
事务里"—— 那正是 application/ 这一层刻意不碰的东西。

写入逻辑在：
    application/agent_run_service_impl.py   用户消息（接收时）
    application/message_recorder.py          assistant 消息（终局确定时）

⚠️ 这里也没有"取单条 / 删除 / 编辑"
---------------------------------
规格里没有这些端点，消息也是不可变的（没有 updated_at）。契约不该比规格
多出接口 —— 与 AgentSessionService 里那条"没有 delete / archive"同一条规矩。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import MessagePage

__all__ = ["MessageService"]

#: 每页条数的默认值。与 messages.yaml 的 default 一致 —— 两处都是 50，
#: 改一处要改两处（规格决定取值，实现跟着走）。
DEFAULT_LIMIT = 50


class MessageService(BaseService):

    @abstractmethod
    async def list_messages(
            self,
            session_id: UUID,
            *,
            user_id: str,
            limit: int = DEFAULT_LIMIT,
            before: str | None = None,
            after: str | None = None,
    ) -> MessagePage:
        """
        取一页会话历史，按 seq 升序（旧 -> 新）。

        Args:
            session_id: 会话。**不存在时是 404，不是空页** —— 见下。
            user_id: 当前用户（Casdoor 的 sub），由 api 层从 access token
                     取。会话若不属于他就走与"不存在"同一条 404 ——
                     历史消息不因为猜中一个 session_id 就漏出去。
            limit: 每页条数。
            before: 不透明游标，取它**之前**（更旧）的一页。
            after: 不透明游标，取它**之后**（更新）的一页。
                   两者不能同时给。

        Raises:
            NotFoundError: 会话不存在，**或不属于 user_id**（-> 404）。
            ValueError: `before` 与 `after` 同时给了，或游标解析不出来
                        （-> 400）。

        ⚠️ 会话不存在为什么是 404 而不是空数组
        -----------------------------------
        空数组的意思是"这个会话还没有消息"——那是一个**正常的**状态（刚建
        的会话就是这样）。把它和"这个会话压根不存在"混成同一种响应，客户端
        就分不清"还没聊过"和"我拿了个错的 session_id"，而后者通常是它自己
        的 bug，越早炸越好。

        会话存在性因此要查一次库 —— 值得，它换来的是两种失败不会长得一样。
        """

        raise NotImplementedError
