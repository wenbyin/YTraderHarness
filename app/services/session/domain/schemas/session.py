"""
会话的对外契约。

逐字段对应 session.yaml 的 components/schemas：
    CreateSessionRequest / Session

⚠️ 三处刻意的**不对称**，都是照规格来的，不是漏写
-----------------------------------------------
1. `CreateSessionRequest` 收 `metadata`，而 `Session` **不返回它**。
   能写不能读。规格如此，这里跟着，但这是契约本身值得改的地方 ——
   调用方塞进去的东西自己拿不回来，只能靠别处存一份。

2. `Session` 带 `updated_at`，但**没有 created_at**。
   库里两个都有（仓库每张表都有）。只暴露 updated_at，是因为新建会话时
   它**就等于 created_at**（见 create_session），而前端真正要的是"最后
   活动时间"——发一条消息它就会前进。真需要区分"建了多久"与"多久没动"
   时再加 created_at。

3. `Session` 带 `user_id`，但**只读** —— `CreateSessionRequest` 里没有它。
   归属由后端从 access token 取，理由见 user_id 字段的说明。

三处都记在这里而不是"顺手补上"：契约先行的工作流里，规格是决定，
实现不该单方面扩它。要加就改 session.yaml，两边一起。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "CreateSessionRequest",
    "Session",
]


class CreateSessionRequest(APIModel):
    """POST /sessions 的请求体。"""

    agent_id: str = Field(
        min_length=1,
        max_length=255,
        description="这个会话交给哪种 Agent。",
        examples=["stock_research"],
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "调用方附加信息。原样存下，**当前没有读回的接口**"
            "（规格里 Session 不含该字段）。"
        ),
    )


class Session(APIModel):
    """一个会话。"""

    session_id: UUID = Field(description="会话 ID。")

    agent_id: str = Field(description="这个会话属于哪种 Agent。")

    status: str = Field(
        description="active / archived。",
    )

    user_id: str | None = Field(
        default=None,
        description=(
            "会话归属的用户 ID（Casdoor 的 sub）。\n\n"
            "⚠️ **只读，且不可指定**：它由后端从 access token 取，"
            "CreateSessionRequest 里刻意没有这个字段 —— 让调用方声明归属"
            "等于允许把会话挂到别人名下。\n\n"
            "为 null 的是加这列之前建的存量会话，它们对**所有**登录用户"
            "可见（刻意的向后兼容，不是漏洞）。"
        ),
    )

    updated_at: datetime = Field(
        description=(
            "最后活动时间。**每次往这个会话发消息都会前进** —— 消息写在"
            " agent_message 表，会话这一行本身没有字段变动，所以是"
            " create_run 显式把这一列推到当下"
            "（见 repository/agent_session.py 的 touch）。\n\n"
            "新建时它等于创建时刻，之后每次发送都刷新。前端按"
            "「最后活动」排序就用它（会话列表接口还没有，这一列先备着）。"
        ),
    )
