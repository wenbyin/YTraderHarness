"""
执行守卫的内部契约。

这两个结构**不出现在 OpenAPI 里**：它们不是 HTTP 请求 / 响应，
而是拦截器与服务之间、服务与仓储之间传递的判定输入与结果。
对外可配置的那一份是拦截器的 `ExecutionGuardConfig`
（见 processor/interceptor/execution_guard.py）。

为什么请求里带 `arguments` 而不是 `param_hash`
--------------------------------------------
判重键 / INVOKE 维度是"这次调用是不是同一件事"的定义，属于服务层。
让调用方自己算摘要，等于把定义复制到每一个调用点 —— 而它错了不会
报错，只会静静地每次判成"新调用"，守卫计数完全失效。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .enum import (
    ExecutionDecision,
    ExecutionMatch,
    ExecutionScope,
    GuardAction,
)

if TYPE_CHECKING:
    # 只在类型检查时导入：entities.tool_execution 反过来 import
    # schemas.enum，运行时导入会绕成一个环。
    from ..entities import ToolExecution

__all__ = [
    "ExecutionGuardRequest",
    "ExecutionReservation",
    "ExecutionStatusLimit",
]


@dataclass(frozen=True, slots=True)
class ExecutionStatusLimit:
    """
    一种执行状态的独立配额。

        limit   允许的次数；None = 不按这个状态限制；0 = 该状态一出现就拦
        action  达到上限后的动作：BLOCK / ALLOW

    「达到上限」的判定是 `count >= limit`：
        limit=1 -> 已有 1 次，下一次拦（允许 1 次）
        limit=3 -> 已有 3 次，第 4 次拦（允许 3 次）
    """

    limit: int | None
    action: GuardAction


@dataclass(frozen=True, slots=True)
class ExecutionGuardRequest:
    """一次预占请求的全部输入。"""

    tool_key: str
    tool_id: UUID | None

    session_id: UUID | None
    run_id: UUID | None
    user_id: str | None

    tool_call_id: str | None

    scope: ExecutionScope
    execution: ExecutionMatch

    #: 模型给的**原始参数**。服务层自己算 param_hash。
    arguments: Mapping[str, Any]

    #: 计数窗口（秒）。None = 该 scope 的整个生命周期。
    #: USER / GLOBAL 设为 None 时就是“永久”—— submit 工具常用。
    window_seconds: int | None

    #: 三种状态各自独立的配额。
    #:
    #: success 的计数包含 PENDING / RUNNING —— 在途尝试先占一个
    #: 成功名额，否则两个并发分支会同时看到“还没有成功”而都执行。
    success: ExecutionStatusLimit
    error: ExecutionStatusLimit
    unknown: ExecutionStatusLimit


@dataclass(frozen=True, slots=True)
class ExecutionReservation:
    """
    预占结果。

        EXECUTE  attempt 非空，调用方跑完工具后必须 settle
        BLOCK    不执行，reason 说明原因
        REPLAY   同一次工具调用已经成功过；attempt.result 是当时的结果

    BLOCK 时 `attempt` 可能为空（新调用被计数拦下），也可能非空
    （同一次 tool_call_id 撞上了 PENDING/RUNNING 等）。调用方只该看
    `reason`，不要把 attempt 当成"可以执行"。
    """

    decision: ExecutionDecision
    attempt: ToolExecution | None = None
    reason: str | None = None
