"""
实体 -> 索引行 / 契约 的纯函数组装。

纯函数、不碰数据库 —— 放在这里而不是服务里各写一份，是因为"一条消息
该以什么样的文本进索引"必须**只有一份**：关键词路与向量路共用它，
两边取到不同的文本时，两路召回就在回答不同的问题，且不报错。

同上，`notice` 那条公告（"（本次运行已取消）"）也**照常索引**：它出现在
会话历史里，把它排除会让记忆检索与消息列表对同一份历史给出不同答案。
它不是噪声 —— 搜"取消"能搜到当时那一轮，那是有用的。

⚠️ 这里**不 import langchain**
----------------------------
取消息正文走 `agent_message` 实体（纯 pydantic），不碰 langchain 的消息
类型 —— 与 session/message_recorder.py 同一条理由（那里也有同样的说明）：
知道"消息长什么样"的地方越少越好，而本模块处理的是**已经落库的**实体，
与 langchain 无关。
"""

from __future__ import annotations

from collections.abc import Sequence

from app.services.session.domain.entities import AgentMessage

from ..domain.entities import (
    FtsIndexRow,
    IndexedMessage,
    VecIndexRow,
)
from ..domain.schemas import (
    DegradedRoute,
    MemoryRecallRoute,
    MemorySearchHit,
)

__all__ = [
    "SNIPPET_CHAR_LIMIT",
    "build_degraded",
    "message_text",
    "to_fts_row",
    "to_hit",
    "to_vec_row",
]


#: 命中片段的最大字符数。
#:
#: ⚠️ 是**截断**，不是"围绕命中处的窗口"
#: ------------------------------------
#: Postgres 有 `ts_headline`，能给出带高亮、且围绕命中处的片段。这里没用，
#: 两个理由：
#:
#:   1. 它需要 tsquery，而命中可能来自**向量路** —— 那一路没有"命中词"
#:      可以居中。同一个问题同一条消息，走哪一路就得到不同的片段，
#:      这种不一致比"片段不够精致"更难解释。
#:   2. 高亮的标记（默认 `<b>` 之类）会混进 text 字段，而它是给**模型**
#:      看的。往上下文里塞标记属于噪声。
#:
#: 所以统一截断，且这 400 字符是**给模型读的量**，不是展示用的片段。
SNIPPET_CHAR_LIMIT = 400


def message_text(message: AgentMessage) -> str | None:
    """
    消息 -> 该进索引的正文。没有可索引的文本时返回 None。

    ⚠️ 只取 text 块，丢掉其余
    ----------------------
    content 是块数组，里面可能有 tool_call 块（`{type: tool_call, name,
    args}`）。工具调用的**参数**不进索引：它是机器的数据（JSON 片段、
    id、路径），把它倒进倒排只会让"搜某个参数值"命中一堆无关的调用，
    而真正说了话的消息被压下去。工具的**结果**不同 —— 它是 ToolMessage
    的 text 块，照常进来。

    ⚠️ 返回 None 而不是空串
    -------------------
    空串会写进索引、参与召回（然后因为 tokens 为空被 CHECK 拦下，
    或者更糟：分词后为空串而 CHECK 在 tokens 上、于是报一个 500）。
    None 的含义是"这条消息不该有索引行"，调用方据此**整条跳过** ——
    这正是规格里"content 为空则丢"那条规则在本仓库的落点：一轮里的
    纯工具调用消息、模型返回空串的消息，都不进索引。
    """

    blocks: Sequence[dict] = message.content or []

    parts = [
        block.get("text")
        for block in blocks
        if block.get("type") == "text"
    ]

    # `text` 可能是 None 或空串（块存在但没内容），过滤掉再判空。
    text = "\n".join(part for part in parts if part)

    return text or None


def to_fts_row(
        message: AgentMessage,
        *,
        tokens: str,
        user_id: str | None,
        org_id: str | None,
) -> FtsIndexRow:
    """
    消息 + 分好词的串 -> 关键词索引行。

    `original` 与 `tokens` 必须同源（同一个 `message_text` 的结果），
    所以这里要求调用方把 tokens 传进来而不是自己再抽一次正文 ——
    再抽一次就是第二个"正文从哪来"的答案。
    """

    return FtsIndexRow(
        message_id=message.message_id,
        session_id=message.session_id,
        run_id=message.run_id,
        seq=message.seq,
        role=message.role,
        business_agent_id=message.business_agent_id,
        user_id=user_id,
        org_id=org_id,
        created_at=message.created_at,
        tokens=tokens,
        original=message_text(message) or "",
    )


def to_vec_row(
        message: AgentMessage,
        *,
        embedding: list[float],
        user_id: str | None,
        org_id: str | None,
) -> VecIndexRow:
    """
    消息 + 向量 -> 向量索引行。

    ⚠️ 向量必须是对 `message_text(message)` 算出来的
    ---------------------------------------------
    同一个函数、同一段文本。对 `content` 整体（含工具调用块）算向量的
    话，两路索引的就是两段不同的文本了。
    """

    return VecIndexRow(
        message_id=message.message_id,
        session_id=message.session_id,
        run_id=message.run_id,
        seq=message.seq,
        role=message.role,
        business_agent_id=message.business_agent_id,
        user_id=user_id,
        org_id=org_id,
        created_at=message.created_at,
        embedding=embedding,
    )


def to_hit(
        *,
        indexed: IndexedMessage,
        score: float,
        matched_by: Sequence[MemoryRecallRoute],
) -> MemorySearchHit:
    """
    索引行 + 融合结果 -> 命中 DTO。

    `matched_by` 由调用方给（它才知道两路各自召回了什么）—— 这里只负责
    形状，不负责"哪几路命中"这个判断。
    """

    return MemorySearchHit(
        message_id=indexed.message_id,
        session_id=indexed.session_id,
        seq=indexed.seq,
        role=indexed.role,
        text=_snippet(indexed.original),
        created_at=indexed.created_at,
        score=score,
        matched_by=list(matched_by),
    )


def build_degraded(
        route: MemoryRecallRoute,
        reason: str,
) -> DegradedRoute:
    """一路没参与 -> 契约里的那一条。理由里带上路线名，便于日志里定位。"""

    return DegradedRoute(route=route, reason=reason)


def _snippet(text: str) -> str:
    """超长正文截断。见 SNIPPET_CHAR_LIMIT 的说明。"""

    if len(text) <= SNIPPET_CHAR_LIMIT:
        return text

    return text[:SNIPPET_CHAR_LIMIT] + "…"
