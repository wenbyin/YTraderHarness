"""
Domain entity for AgentSession.

会话是**长期存在的主体**：一次对话可以开很多次运行，运行结束了会话还在。
它与 `agent_run` 的关系是"一对多"，且方向单向 —— 运行派生自会话，会话
不聚合运行（不需要在会话实体上挂一个 runs 列表：那会让"读一个会话"顺便
把它的全部历史拖出来）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AgentSession"]


class AgentSession(BaseModel):
    """
    一个 Agent 会话。

    ⚠️ `meta` 而不是 `metadata`
    --------------------------
    领域实体这一层其实没有 SQLAlchemy 那个限制，但这里仍叫 meta ——
    一个值从契约到库里要穿过三层（DTO 的 metadata → 实体 → 列），
    中间改一次名，读代码的人就得在脑子里做一次映射。统一叫 meta 更省事，
    转换只在**契约边界**（DTO）发生一次。

    agent_id 是字符串而不是枚举
    -------------------------
    取值由 Agent 目录决定，而那个目录**还不存在**（见 domain/schemas/
    enum.py 的说明）。在它落地之前把取值钉成枚举，等于在这里替产品做
    一个"有哪些 Agent"的决定；做成字符串则只是把它记下来。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    session_id: UUID = Field(description="会话 ID。")

    agent_id: str = Field(
        description="这个会话属于哪种 Agent。",
    )

    status: str = Field(
        description="active / archived，见 schemas.enum.SessionStatus。",
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="调用方随会话携带的附加信息。",
    )

    thread_id: str | None = Field(
        default=None,
        description=(
            "本会话在 LangGraph checkpointer 里的线程键，全库唯一。"
            "首轮运行时生成（形如 dev:react:<uuid>），此后不变 —— "
            "同一会话的多轮历史靠它累积。"
            "None 表示还没发过运行（历史行就是这种）。"
        ),
    )

    user_id: str | None = Field(
        default=None,
        description=(
            "会话归属的用户 ID —— Casdoor 的 sub（app/common/auth），"
            "字符串，与 file.uploaded_by 同一口径。"
            "所有 session 接口按它过滤：等于当前用户、或为 None"
            "（加这列之前的存量会话，对谁都可见）才可见。"
        ),
    )

    org_id: str | None = Field(
        default=None,
        description=(
            "会话所属的**组织** —— JWT 的 org claim（claim 名可配，见 "
            "app/common/auth 的 org_claim）。\n\n"
            "用途只有一个：团队 / 组织归属的**记录**。它不参与鉴权"
            "（那是 groups 的事），也不参与可见性过滤（过滤键仍然是"
            " user_id）。\n\n"
            "None 表示这次登录的 token 里没有组织信息 —— 这是允许的"
            "（Casdoor 各版本与各应用的 Token fields 配置不一样），"
            "不是漏写。"
        ),
    )

    task_id: str | None = Field(
        default=None,
        description=(
            "预留位：这个会话是为哪一件**工作任务**服务的。\n\n"
            "⚠️ **当前恒为 None** —— 还没有任何地方写它（没有输入来源）。\n\n"
            "⚠️ 别与 `tasks.task_id`（后台任务主键）混起来：同名不同物。"
        ),
    )

    created_at: datetime = Field(description="创建时间。")

    updated_at: datetime = Field(description="最后更新时间。")
