"""
Domain entity for ToolExecution.

一次工具调用的执行记录。它是 `tool_execution` 表的领域形态 ——
表结构与约束的说明在 models/sqlalchemy/tool_execution.py，
状态语义在 domain/schemas/enum.py 的 ToolExecutionStatus。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import ToolExecutionStatus

__all__ = ["ToolExecution"]


class ToolExecution(BaseModel):
    """
    一条执行记录。

    四条**判重键**字段（session_id / run_id / tool_key / param_hash）
    是不可变的：它们共同决定「这是不是同一次调用」。状态与结果则是
    一行生命里会变的部分 —— 从 PENDING 走到某个终态。

    ⚠️ `session_id` 是 `agent_session.session_id`，不是 langgraph 的
    `thread_id` —— 见 ORM 类 docstring 里那段理由。

    ⚠️ `result` 是**脱敏之后**的那一份
    --------------------------------
    写库发生在处理器链跑完之后，privacy_transformer 已生效，因此这里
    存的就是模型当时看到的内容。见 ORM 类 docstring 的同一段说明。

    ⚠️ `result` 与 `error` 的"空"不是一回事
    -------------------------------------
    工具正常返回 `None` 是**合法成功**（`ToolOutcome.value` 可以为
    None），因此 SUCCESS 时 `result` 可以是 None。要判断"这次有没有
    说到结果"，看 status，不要看 result 是不是 None。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    execution_id: UUID = Field(
        description="执行记录 ID。",
    )

    session_id: UUID = Field(
        description=(
            "所属会话（agent_session.session_id）。"
            "⚠️ 不是 langgraph 的 thread_id。"
        ),
    )

    run_id: UUID = Field(
        description="本次运行（agent_run 主键，即 pipeline_run_id）。",
    )

    tool_key: str = Field(
        description="被调用的工具，形如 document.retrieve_documents。",
    )

    param_hash: str = Field(
        description="调用参数的 sha256 十六进制摘要。",
    )

    status: ToolExecutionStatus = Field(
        description=(
            "执行状态。⚠️ 「能不能重跑」的判据在枚举那边，"
            "不在这个字段的**名字**上 —— 见 ToolExecutionStatus。"
        ),
    )

    result: Any = Field(
        default=None,
        description=(
            "工具返回值（规范化后的 JSON 形态）。"
            "None 可能是「工具正常返回了 None」，也可能是「还没走到"
            "那一步」—— 以 status 为准。"
        ),
    )

    error: dict[str, Any] | None = Field(
        default=None,
        description=(
            "失败 / 拒绝的说明，形状 "
            '{"type": 异常类名, "message": 给模型看的文案}。'
        ),
    )

    created_at: datetime = Field(
        description="创建时间（claim 那一刻）。",
    )

    updated_at: datetime = Field(
        description="最后一次状态转移的时间。",
    )
