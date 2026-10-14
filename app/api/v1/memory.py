"""
L0 记忆检索接口。

对应 memory.yaml 的 tag：Memory。规格与实现都在 app/services/memory/ 下，
本文件只做 HTTP 层的转发 —— 与 capabilities.py / tools.py 等其余路由同一套
分层。

⚠️ 身份从这里进，**请求体里没有**
-----------------------------
`session_id` 走 query 参数，`user_id` 从 access token 取。这不是风格问题：
请求体（`MemorySearchRequest`）**同时是暴露给模型的工具入参 schema**，
身份字段一旦进去，模型就能自己填一个别人的 id。理由写在 memory.yaml 的
info.description 与 domain/schemas/memory.py 的模块 docstring。

这条路径上还有一个 token 带来的好处：**归属校验不需要额外查一次**。
`session_id` 给出来时，服务会用 `user_id` 去取会话（`AgentSessionRepository
.get` 本身就带可见性条件），取不到就是 404。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.common.auth import AuthenticatedUser, current_user
from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.memory.application.memory_service import MemoryService
from app.services.memory.domain.schemas import (
    MemorySearchRequest,
    MemorySearchResponse,
)

router = APIRouter()


@router.post(
    "/memory/search",
    response_model=MemorySearchResponse,
    tags=["Memory"],
)
async def search_memory(
        request: MemorySearchRequest,
        session_id: UUID | None = Query(
            default=None,
            description=(
                "限定单个会话。省略时由请求体里的 scope 决定。"
                "不属于调用者的会话返回 404 —— 与「不存在」同一个响应，"
                "不泄露某个 id 是否存在。"
            ),
        ),
        user: AuthenticatedUser = Depends(current_user),
        service: MemoryService = Service[MemoryService],
):
    """
    对 L0 原文做一轮混合召回（关键词 + 向量）。

    返回的是**片段**而不是整条消息：要全文就拿 hit 里的 message_id 去
    messages 接口取，见 memory.yaml 的说明。

    ⚠️ `user.subject or None` 里的那个 or
    -----------------------------------
    `subject` 是 Casdoor 的 sub，可能是空串（见 sessions.py 里同名处理的
    说明：auth 层是纯机制，不替业务决定空 sub 该不该放行）。空串在这里
    归一成 None，而服务对 None 的态度是**拒绝**（两条 scope 都拒）——
    也就是说"token 有效但里面没有 sub"会被挡在检索之外，而不是退化成
    "不过滤"。

    这条宁可解释成一次 400，也不能让它变成一次跨用户读。
    """

    try:
        return await service.search(
            request,
            session_id=session_id,
            user_id=user.subject or None,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
