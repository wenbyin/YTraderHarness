"""
Domain entity for AgentRun.

一次执行。追加型历史：每次运行一行，状态随执行推进被改写，但行本身不被
复用（下一次运行是新的 run_id，不是把旧行改回去）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AgentRun"]


class AgentRun(BaseModel):
    """
    一次 Agent 运行。

    与观测层的 `document_pipeline_run` 是同一种东西吗
    -----------------------------------------------
    形状像（都记"一次执行"），但**不是同一张表、也不会合并**：那张表属于
    document 流水线，带拓扑快照、扇出计数这类只有它需要的列。本表是 Agent
    对话的运行，字段少得多。

    真正被两边共用的是**运行时**（`app/infra/graph/`：runner / recorder /
    dispatcher / checkpointer）。

    ⚠️ 但两条链路**不共用 recorder**
    -------------------------------
    recorder 的实现（SqlAlchemyRunRecorder）写的是 document_pipeline_run，
    且断言 subject_type 必须是 document_version。Agent 这条链路因此不用它，
    只把状态翻到本表上（见 application/agent_run_dispatch.py），
    逐节点进度不落库 —— 那由将来的 SSE 承担。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    run_id: UUID = Field(description="运行 ID。")

    session_id: UUID = Field(description="所属会话。")

    status: str = Field(
        description=(
            "queued / running / waiting_input / completed / failed / "
            "cancelled，见 schemas.enum.RunStatus。"
        ),
    )

    input: dict[str, Any] = Field(
        default_factory=dict,
        description="用户这次的输入载荷。",
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="调用方随运行携带的附加信息。",
    )

    error_type: str | None = Field(
        default=None,
        description="失败时的异常类名。未结束或成功时是 None。",
    )

    error_message: str | None = Field(
        default=None,
        description=(
            "失败时的异常文本。规格的 Run 不回它，排查只能看这里。"
        ),
    )

    user_id: str | None = Field(
        default=None,
        description=(
            "发起这次运行的用户 ID（Casdoor 的 sub）。注意是**发起人**，"
            "不是所属会话的 user_id —— 存量公共会话上仍能记录真实执行者。"
            "派发时它会被写进 ReAct 图初始 state 的 user_id。"
        ),
    )

    caller_identity: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "发起这次运行时，调用者的**身份快照**（groups / roles）。\n\n"
            "为什么要在建运行的那一刻存下来：agent 运行是**异步派发**的"
            "（请求提交后才跑），跑的时候手里没有 token，而 Casdoor 是"
            "用户的唯一真相、本地没有用户表 —— 不存这一份，工具级的"
            "鉴权（checkauth 拦截器）在派发时就没有依据。\n\n"
            "形状见 app/agent/tools/processor/context.py 的"
            " build_caller_context：{user_id, groups, roles}。\n\n"
            "⚠️ 它是**内部字段**，不进对外契约 —— `_assembly.py` 装配"
            " Run DTO 时和 input / meta / 时间戳一样丢掉它，"
            "别让它出现在 API 响应里。"
        ),
    )

    duty_key: str | None = Field(
        default=None,
        description=(
            "这次运行用哪份**职责提示词**（duty_key）。\n\n"
            "建运行那一步解析一次（见 agent_run_service_impl.create_run），"
            "此后这一轮里不再变：派发时进 ReAct 图的初始 state（builder "
            "节点据此取提示词），写消息时填进 "
            "agent_message.business_agent_id。\n\n"
            "之所以存下来而不是各处理解一次：同一轮运行里「用的哪份职责」"
            "必须只有一个答案，否则运行行、提示词、消息三处可能对不上。\n\n"
            "存量运行为 None —— 那时还没有这个概念，主循环会回落到默认职责。"
        ),
    )

    trace_id: str | None = Field(
        default=None,
        description=(
            "客户端传入的 Trace 聚合键。None = 这次运行没有开启 Trace。\n\n"
            "为什么要落库：resume 时 handler 从库里重新加载运行行，"
            "不能假设客户端会再传一次；而 trace 事件要求同一 trace 的"
            "所有事件 trace_id 完全一致。"
        ),
    )

    created_at: datetime = Field(description="创建时间。")

    updated_at: datetime = Field(description="最后更新时间。")
