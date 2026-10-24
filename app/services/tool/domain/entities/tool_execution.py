"""
Domain entity for ToolExecution.

**一次执行一条记录**的 append-only 账本 —— 与旧版「一个判重键一行、
重试时重置」的幂等账本不同。表结构与约束的说明在
models/sqlalchemy/tool_execution.py（沿用 `tool_execution` 表名，
在原表上扩展），状态语义在
domain/schemas/enum.py 的 ToolExecutionStatus。

为什么要改成 append-only
------------------------
守卫要回答的问题从「这件事是不是已经做过」变成了「这个维度上
已经发生过几次」。后者必须保留每一次尝试，否则「重试一次」会把
上一次的痕迹抹掉，计数永远停在 1。

判重（同一 tool_call_id 不重复执行）仍然保留，但它落在
`(run_id, tool_call_id)` 的唯一索引上，而不是靠覆盖同一行。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import (
    ExecutionMatch,
    ExecutionScope,
    ToolExecutionStatus,
)

__all__ = ["ToolExecution"]


class ToolExecution(BaseModel):
    """
    一条执行记录。

    身份、维度、状态是三组字段：

        身份      execution_id / run_id / tool_call_id
        维度      scope_type / scope_key / execution_type / param_hash
        状态      status / counted / result / error

    ⚠️ `counted` 与 `status` 不是一回事
    ---------------------------------
    `status` 描述发生了什么，`counted` 描述这次尝试是否占用守卫额度。
    被拒绝、中断、取消、守卫拦下的记录 `counted=False` —— 它们没有
    真正执行，不该把下一次调用挡在门外。

    ⚠️ `session_id` / `run_id` 可以为空
    ---------------------------------
    图外直调（冒烟、脚本）没有会话与运行。守卫在 scope 需要它们时
    会 fail-closed，但记录本身允许为空 —— 否则图外调用连「被拒绝」
    都记不下来。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    execution_id: UUID = Field(
        description="执行记录 ID。",
    )

    tool_key: str = Field(
        description="被调用的工具，形如 document.retrieve_documents。",
    )

    tool_id: UUID | None = Field(
        default=None,
        description="tool_definition 主键；还没同步进库的工具为空。",
    )

    session_id: UUID | None = Field(
        default=None,
        description=(
            "所属会话（agent_session.session_id）。"
            "⚠️ 不是 langgraph 的 thread_id。图外直调时为空。"
        ),
    )

    run_id: UUID | None = Field(
        default=None,
        description="本次运行（agent_run 主键，即 pipeline_run_id）。",
    )

    user_id: str | None = Field(
        default=None,
        description=(
            "调用者（caller_identity.user_id 的字符串形态）。"
            "拿不到时为空 —— 不要回落到会话反查。"
        ),
    )

    scope_type: ExecutionScope = Field(
        description="计数维度：SESSION / RUN / USER / GLOBAL。",
    )

    scope_key: str = Field(
        description=(
            "维度取值：session_id / run_id / user_id / '*'。"
            "反规范化存储，便于按维度计数与排查。"
        ),
    )

    execution_type: ExecutionMatch = Field(
        description="METHOD（只看工具）/ INVOKE（工具 + 参数）。",
    )

    param_hash: str = Field(
        default="",
        description=(
            "INVOKE 时的参数 sha256；METHOD 时为空串。"
            "与 cache 拦截器同一套 canonical_json + sha256。"
        ),
    )

    match_key: str = Field(
        description=(
            "上面五段（tool_key + scope + execution + param_hash）的"
            "定长摘要。计数、加锁、索引都用它，避免多列组合比较。"
        ),
    )

    status: ToolExecutionStatus = Field(
        description=(
            "执行状态。⚠️ 「能不能重跑」的判据在守卫配置"
            "（success / error / unknown 三种独立配额）与 counted "
            "位上，不在这个字段的**名字**上。"
        ),
    )

    counted: bool = Field(
        default=True,
        description=(
            "这次尝试是否占用额度。PENDING 时先记 True（并发分支要"
            "看见它），落终态时按结果改成 False（拒绝/中断/取消/"
            "缓存命中）。"
        ),
    )

    tool_call_id: str | None = Field(
        default=None,
        description=(
            "模型给的调用 id。与 run_id 一起构成 replay/resume 的"
            "去重键：同一次工具调用在图中重跑时复用同一行，"
            "不再消耗额度。"
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
        description="创建时间（预占那一刻）。",
    )

    updated_at: datetime = Field(
        description="最后一次状态转移的时间。",
    )
