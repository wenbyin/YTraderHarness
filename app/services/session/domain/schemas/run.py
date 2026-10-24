"""
运行的对外契约。

逐字段对应 session.yaml 的 components/schemas：
    CreateRunRequest / RunInput / Run / ResumeAnswer / ResumeRequest

⚠️ `input` 从自由字典收窄成 `RunInput`
------------------------------------
规格原来只写 `type: object`，于是"该传什么"没有任何一处定义 —— 而图需要
的恰恰是"谁说了什么"。现在两边一起定死成 `{"text": "..."}`：

    规格    session.yaml 的 CreateRunRequest.input（required: [text]）
    实现    本模块的 RunInput

多出来的键会被拒（`extra="forbid"`，与规格里的 additionalProperties: false
一致），而不是被静默丢掉 —— 调用方以为自己传了 user_id 却发现图根本没看见，
是最难查的一种。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "CreateRunRequest",
    "ResumeAnswer",
    "ResumeRequest",
    "Run",
    "RunInput",
]


class RunInput(APIModel):
    """
    一次运行的用户输入。

    只有 `text` 是刻意的：图的其余上下文通道（画像 / 记忆 / 检索 /
    策略覆盖）都还没有来源，先在这里留出位置等于猜一个还没设计的形状 ——
    等 ContextBuilder 真的能注入了，再按那时需要的键加。
    """

    text: str = Field(
        min_length=1,
        description=(
            "用户这一轮说的话。图把它包成一条 HumanMessage 送进主循环。"
        ),
    )


class CreateRunRequest(APIModel):
    """POST /sessions/{session_id}/runs 的请求体。"""

    input: RunInput = Field(
        description="用户输入载荷。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "调用方附加信息。原样存下，**当前没有读回的接口**"
            "（与 CreateSessionRequest.metadata 同一条，规格如此）。"
        ),
    )

    trace_id: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "客户端传入的 Trace 聚合键。**传了才开启 Trace**："
            "为空时一条 trace 事件都不写（连 Redis 键都不产生）。"
            "它可以跨 run / session 使用，因此事件里另带 run_id。"
        ),
    )


class ResumeAnswer(APIModel):
    """对**一次**中断的回答。"""

    interrupt_id: str = Field(
        min_length=1,
        description=(
            "要回答的那次中断的 id。由 `hitl.requested` 事件带出来"
            "（`data.interrupt_id`）。"
        ),
    )

    data: dict[str, Any] = Field(
        description=(
            "回答内容，原样交给 `interrupt()`。审批的形状是 "
            '`{"decision": "approve" | "reject" | "always"}`。'
        ),
    )


class ResumeRequest(APIModel):
    """
    POST /runs/{run_id}/resume 的请求体 —— 一次回答**当前所有**待审批。

    ⚠️ 为什么是"一次全部"，不是"一次一条"
    -----------------------------------
    扇出下**多个工具可以同时等审批**（每个分支各停各的，实测一次运行里
    `next=('tools', 'tools')`、两个中断并存）。所以正确的模型不是"依次
    回答"，而是"把手上所有待答的一次性回掉"。

    引擎本来就支持一次多条（`Command(resume={id: data, ...})`，实测按 id
    **精确匹配**、不会串位），所以限制只在契约这一层。

    ⚠️ 必须**不多不少**，正好覆盖当前的 pending 集合
    ---------------------------------------------
    服务端会读 checkpoint 里那条 thread 上**真实**的待处理中断集合，与
    `answers` 比对，缺一个或给多了都报 409：

        少给一个   -> 那个仍然停着，这次运行回不到 running，而调用方拿到
                      的是 200 —— 一次"成功"的请求什么都没推进
        多给/给错  -> 说明客户端手上的集合是过期的（多半漏读了后到的
                      `hitl.requested`）

    两种都属于"请求与当前状态不一致"，所以是 409 而不是 400：请求本身
    没毛病。

    ⚠️ 幂等性不做在服务端
    ------------------
    同样的请求重发第二次会拿到 409（运行已经不在 `waiting_input`），而不是
    "重复受理"。要幂等得客户端按 `interrupt_id` 自己记 —— 服务端做的话就
    得额外存一份"哪个回答用过"，那是又一份会漂移的状态。

    ⚠️ `data` 是自由字典，形状由**审批拦截器**定
    ------------------------------------------
    引擎不该知道"审批有三个选项"这件事 —— 那是工具侧的处理器的契约
    （见 processor/interceptor/approval.py 的 `ApprovalDecision`：`approve` /
    `reject` / `always`）。这里原样透传给 `interrupt()`，由那一端解析。
    """

    answers: list[ResumeAnswer] = Field(
        min_length=1,
        description=(
            "对当前**每一个**待处理中断的回答。集合必须与 "
            "`hitl.requested` 给出的待办完全一致（见类说明）。"
        ),
    )


class Run(APIModel):
    """
    一次运行。

    ⚠️ 没有 input 回显
    ----------------
    规格的 Run 只有 run_id / session_id / status。调用方提交的 input
    存进库了，但接口不回。回显在排查时很有用（"我当时到底提交了什么"），
    不过那是改契约的事，不在这里顺手加。
    """

    run_id: UUID = Field(description="运行 ID。")

    session_id: UUID = Field(description="所属会话。")

    status: str = Field(
        description=(
            "queued / running / waiting_input / completed / failed / "
            "cancelled。"
        ),
    )

    trace_id: str | None = Field(
        default=None,
        description=(
            "建运行时传入的 Trace 聚合键。为空表示这次运行没有开启 Trace。"
            "客户端自己传过的值，回显不引入新的信息。"
        ),
    )
