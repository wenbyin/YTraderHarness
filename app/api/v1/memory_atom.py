"""
L1（memory_atom）记忆检索接口。

对应 memory_atom.yaml 的 tag：Memory Atom。规格与实现在
`app/services/memory/` 下，本文件只做 HTTP 层的转发 —— 与 memory.py
（L0）同一套分层。

⚠️ 身份从这里进，**请求体里没有**
-----------------------------
与 memory.py 逐条同理：请求体（`MemoryAtomSearchRequest`）**同时是暴露
给模型的工具入参 schema**，身份字段一旦进去，模型就能自己填一个别人的
id。所以 `session_id` 走 query 参数、`user_id` 从 access token 取。

这条路径上还有一个 token 带来的好处：**归属校验不需要额外查一次**。
`session_id` 给出来时，服务会用 `user_id` 去取会话
（`AgentSessionRepository.get` 本身就带可见性条件），取不到就是 404。

⚠️ 与 L0 的一条差别：scope=user 才是 L1 的主场
----------------------------------------
L0 的记忆是"一次对话里的原话"，按会话搜是常态；L1 的记忆是**跨会话**的
用户事实（memory.yaml 里那句 "the scope L1 facts live in"），所以调用方
多数时候该传 `scope=user`。默认值仍然是 `session`（不传就只看当前会话，
范围最小、最不容易读到意外的东西）。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.common.auth import AuthenticatedUser, current_user
from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.memory.application.memory_atom_service import (
    MemoryAtomService,
)
from app.services.memory.domain.schemas.memory_atom import (
    MemoryAtomSearchRequest,
    MemoryAtomSearchResponse,
)

router = APIRouter()


@router.post(
    "/memory-atom/search",
    response_model=MemoryAtomSearchResponse,
    tags=["Memory Atom"],
)
async def search_memory_atom(
        request: MemoryAtomSearchRequest,
        session_id: UUID | None = Query(
            default=None,
            description=(
                "限定单个会话。省略时由请求体里的 scope 决定。"
                "不属于调用者的会话返回 404 —— 与「不存在」同一个响应，"
                "不泄露某个 id 是否存在。"
            ),
        ),
        user: AuthenticatedUser = Depends(current_user),
        service: MemoryAtomService = Service[MemoryAtomService],
):
    """
    对 L1 记忆做一轮混合召回（关键词 + 向量）。

    返回的是**完整的记忆陈述**，不是片段 —— 与 L0 不同（那边截断，因为
    原始对话的长度是会话的性质）。这里截断可能把「喜欢 X，但**不能** Y」
    切掉后半句，比不给还糟。

    ⚠️ `user.subject or None` 里的那个 or
    -----------------------------------
    与 memory.py 同一条：空串归一成 None，而服务对 None 的态度是**拒绝**
    （两种 scope 都拒）—— "token 有效但里面没有 sub"会被挡在检索之外，
    而不是退化成"不过滤"。宁可解释成一次 400，也不能让它变成一次跨用户读。
    """

    try:
        return await service.search(
            request,
            session_id=session_id,
            user_id=user.subject or None,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
