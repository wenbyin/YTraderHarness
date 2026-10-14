"""
Domain entity for AgentMessage.

一条**已经完成**的会话消息：不可变、整条写入、没有中间态。

⚠️ 类名带 agent_ 前缀，与另外两个实体一致
---------------------------------------
表名与类名都从 messages / Message 改成了 agent_message / AgentMessage
（见 models/sqlalchemy/agent_message.py 的说明）。**对外契约没有跟着改**：
messages.yaml 的组件仍然叫 Message —— DTO 的 title 加上
app/common/openapi.py 里的改名表把生成的组件名钉住，因此这次改名不产生
任何接口变化。


⚠️ updated_at 在，但它与 created_at **恒等**（消息仍不可变）
---------------------------------------------------------
agent_session / agent_run 都是"行被反复改写"的（状态推进、元数据更新），
它们的 updated_at 因此是真的会走的。消息相反：它只被写一次，写完就不再
变 —— "这条消息被编辑过"这件事在契约里根本不存在（messages.yaml 说
"Each message is an immutable, complete record"）。

updated_at 仍然有（两个写入点都填 created_at 的值），是为了与另外两张表
对齐、并让契约有一个统一的"最后有效时间"。它**不是**"消息会变"的暗示：
库里那一列挂着 onupdate，但没有任何代码去 UPDATE 消息 —— 唯一的例外是
手工迁移脚本，所以"想给存量行补一个字段"这件事要先想清楚会不会把这一列
推走（见 models/sqlalchemy/agent_message.py 的 reasoning_content 那段）。

⚠️ reasoning_content 与 content 是**两个不重叠的容器**
--------------------------------------------------
    content            可见内容块（text / tool_call / …）
    reasoning_content  思维链原文，一条消息最多一份

推理**不在** content 的块数组里（`message_recorder._content_blocks` 会把
推理块滤掉），因此前端不会把同一段渲染两遍。

⚠️ 但**存量行**里推理就在 content 的块数组里（实测 169 行，全是
kind=chat 的最终答复），而那批行刻意不回填 —— 所以前端的读法是

    reasoning = reasoning_content ?? content 里 type=reasoning 的块

这不是临时兼容：只读新列会让所有历史会话的推理消失，只读 content 块会
让新会话的推理消失。两个容器不重叠这条规则从**新写入**开始成立。

取不到就是 None（非推理模型、用户消息、取消公告），不是空串。

⚠️ 为什么 `meta` 而不是 `metadata`
--------------------------------
与另外两个实体同一条理由（见 agent_session.py）：一个值从契约到库要穿过
三层，中间改一次名，读代码的人就得在脑子里做一次映射。统一叫 meta，
转换只在**契约边界**（DTO）发生一次。

⚠️ content 是 list 而不是 str
---------------------------
契约里的 content 是 ContentBlock 的数组（纯文本也是 `[{"type": "text",
"text": ...}]`），因为同一条消息里可能既有文字又有工具调用块。库里那一列
带 `jsonb_typeof(content) = 'array'` 的 CHECK —— 写成字符串会当场被拒，
而这正是有那个 CHECK 的理由：形状退化不该等到前端解析时才暴露。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AgentMessage"]


class AgentMessage(BaseModel):
    """
    一条完整的会话消息。

    ⚠️ session_id 在实体上，但**不在契约里**
    -------------------------------------
    契约的 Message 没有 session_id —— 它是查询的输入（URL 里那个），
    每一条都相同，逐条回显只是噪声。因此它在实体上（写库要用），
    由 `_assembly.py` 在装配成 DTO 时丢掉，与 meta / 时间戳同一种处理。

    ⚠️ client_message_id 当前**永远是 None**
    -------------------------------------
    它是"用户提交消息的幂等 ID"，要由客户端在提交时给出。而今天的入口是
    `POST /sessions/{id}/runs`，它的请求体（CreateRunRequest）里没有这个字段
    —— 因此库里这一列建了、唯一索引也建了，但没有任何代码会写它。
    等将来出现 `POST /sessions/{id}/messages`（前端直接发消息）时才有值。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    message_id: UUID = Field(description="消息 ID。")

    session_id: UUID = Field(description="所属会话。")

    run_id: UUID | None = Field(
        default=None,
        description=(
            "产出这条消息的运行。用户消息在**建运行时**写下，因此它也有 run_id。"
        ),
    )

    seq: int = Field(
        description=(
            "会话内单调递增的序号。翻页与「先来后到」都按它，不按时间戳"
            "——同一事务里写的多条消息时间戳可能完全相同。"
        ),
    )

    role: str = Field(
        description="user / assistant，见 schemas.enum.MessageRole。",
    )

    kind: str = Field(
        description="chat / interaction / tool_activity / notice。",
    )

    business_agent_id: str | None = Field(
        default=None,
        description=(
            "产出这条消息的**职责** —— 这一轮运行用的那份职责提示词的 key"
            "（duty_key，如 `stock_research`）。\n\n"
            "来自 `agent_run.duty_key`：建运行那一刻解析一次，此后这一轮的"
            " user / assistant / 取消公告各条消息上都写同一个值。\n\n"
            "None 表示那一轮没有职责信息（这一列开始写入之前的存量行）。"
            "列名是历史命名 —— 当初预留的是业务 Agent 的位置。"
        ),
    )

    content: list[dict[str, Any]] = Field(
        default_factory=list,
        description="内容块数组。纯文本是 [{'type': 'text', 'text': ...}]。",
    )

    reasoning_content: str | None = Field(
        default=None,
        description=(
            "模型的思维链（CoT）原文。None = 这条消息没有思维链"
            "（非推理模型 / 用户消息 / 取消公告），不是空串。\n\n"
            "刻意**不放进** content 的块数组：两个容器不重叠，见类 docstring。"
        ),
    )

    status: str = Field(
        description=(
            "completed / interrupted / failed / cancelled。"
            "只表示最终结果，没有 streaming 这一档。"
        ),
    )

    client_message_id: str | None = Field(
        default=None,
        description="客户端幂等 ID。当前永远是 None，见类 docstring。",
    )

    reply_to_message_id: UUID | None = Field(
        default=None,
        description=(
            "被回复的那条消息。assistant 消息指向本轮的 user 消息；"
            "user 消息没有可指的，是 None。"
        ),
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据。对外契约里叫 metadata。",
    )

    created_at: datetime = Field(description="创建时间。")

    updated_at: datetime = Field(description="更新时间。")
