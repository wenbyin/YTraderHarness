"""
实体 -> 对外契约的组装。

纯函数、不碰数据库。放在这里而不是各服务里各写一份：会话与运行各只有一个
读点，但"契约长什么样"仍然只该有一处 —— 将来加了列表接口，两处各拼一份
必然会漂移（一个带 status 一个不带那种）。

消息那一组已经印证了这条：读路径在 MessageService（列表）与将来的
"按 id 取单条"之间共享同一个 build_message，改字段只改这里。
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from ..domain.entities import AgentMessage as AgentMessageEntity
from ..domain.entities import AgentRun as AgentRunEntity
from ..domain.entities import AgentSession as AgentSessionEntity
from ..domain.schemas import (
    AgentMessage,
    MessagePage,
    Pagination,
    Run,
    Session,
    encode_cursor,
)

__all__ = [
    "build_message",
    "build_message_page",
    "build_run",
    "build_session",
]


def build_session(entity: AgentSessionEntity) -> Session:
    """
    会话实体 -> 契约。

    ⚠️ 这里**丢掉**了 meta 与 created_at —— 不是漏写，是契约里没有它们
    （session.yaml 的 Session 只有 session_id / agent_id / status /
    user_id / updated_at）。契约本身的不对称记在
    domain/schemas/session.py 里。
    """

    return Session(
        session_id=entity.session_id,
        agent_id=entity.agent_id,
        status=entity.status,
        user_id=entity.user_id,
        updated_at=entity.updated_at,
    )


def build_run(entity: AgentRunEntity) -> Run:
    """
    运行实体 -> 契约。同样丢掉 input / meta / 时间戳 / caller_identity，见上。

    ⚠️ `caller_identity` 是**刻意丢掉**的，不是漏写：它是服务端鉴权用的
    身份快照（groups / roles），契约里没有这个字段。逐字段构造让"新增
    一个实体字段"默认不会外泄 —— 别为了省事改成 `**entity.model_dump()`，
    那会把内部字段一起送出去。
    """

    return Run(
        run_id=entity.run_id,
        session_id=entity.session_id,
        status=entity.status,
    )


def build_message(entity: AgentMessageEntity) -> AgentMessage:
    """
    消息实体 -> 契约。

    ⚠️ 这里做了两次改名，方向相反，别搞混：

        session_id   实体有、契约**没有**   -> 丢掉（见下）
        meta         实体有、契约叫 metadata -> 改名

    丢掉 session_id 与上面两个 builder 丢掉 meta / 时间戳是同一条规矩：契约
    里没有的东西不硬塞（messages.yaml 的 Message 不含它）。它是查询的输入
    —— 一整页里每条都相同，逐条回显只是噪声。那一份在 MessagePage 上。
    """

    return AgentMessage(
        message_id=entity.message_id,
        seq=entity.seq,
        run_id=entity.run_id,
        role=entity.role,
        kind=entity.kind,
        business_agent_id=entity.business_agent_id,
        content=entity.content,
        # 逐字段构造（不是 **entity.model_dump()）的代价：漏一行就是
        # "库里有、接口没有"，而且不报错。见模块 docstring。
        reasoning_content=entity.reasoning_content,
        status=entity.status,
        reply_to_message_id=entity.reply_to_message_id,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        metadata=dict(entity.meta),
    )


def build_message_page(
        *,
        session_id: UUID,
        entities: Sequence[AgentMessageEntity],
        limit: int,
        has_more_before: bool,
        has_more_after: bool,
) -> MessagePage:
    """
    一页实体 -> 契约。

    两个游标由**本页的首尾**编出来（`entities` 按 seq 升序，仓储保证）：

        before_cursor -> 最旧那条   拿它往旧翻
        after_cursor  -> 最新那条   拿它往新翻

    空页时两个都是 None（连同两个 false）—— 客户端据此收手，而不是拿着
    null 再请求一次。这与仓储那条"空页两个标志都回 False"是配套的。
    """

    return MessagePage(
        session_id=session_id,
        messages=[build_message(entity) for entity in entities],
        pagination=Pagination(
            limit=limit,
            has_more_before=has_more_before,
            before_cursor=(
                encode_cursor(entities[0].seq) if entities else None
            ),
            has_more_after=has_more_after,
            after_cursor=(
                encode_cursor(entities[-1].seq) if entities else None
            ),
        ),
    )
