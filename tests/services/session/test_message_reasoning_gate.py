"""
消息接口的推理输出开关（`sinks.chat.reasoning_enabled`）。

关掉时两个容器要**一起清**：
    reasoning_content 那一列（新行）
    content 里 type=reasoning 的块（存量行）

只清一个，等于换一种方式把推理漏出去。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.services.session.application._assembly import build_message_page
from app.services.session.domain.entities import AgentMessage


def _entity() -> AgentMessage:
    now = datetime.now(timezone.utc)

    return AgentMessage(
        message_id=uuid4(),
        session_id=uuid4(),
        run_id=uuid4(),
        seq=1,
        role="assistant",
        kind="chat",
        business_agent_id=None,
        content=[
            {"type": "reasoning", "reasoning": "存量推理"},
            {"type": "text", "text": "正文"},
        ],
        reasoning_content="新推理",
        status="completed",
        client_message_id=None,
        reply_to_message_id=None,
        meta={},
        created_at=now,
        updated_at=now,
    )


def _page(*, include_reasoning: bool):
    return build_message_page(
        session_id=uuid4(),
        entities=[_entity()],
        limit=50,
        has_more_before=False,
        has_more_after=False,
        include_reasoning=include_reasoning,
    )


def test_reasoning_is_hidden_when_switched_off() -> None:
    message = _page(include_reasoning=False).messages[0]

    assert message.reasoning_content is None
    assert [block.type for block in message.content] == ["text"]
    assert message.content[0].text == "正文"


def test_reasoning_is_preserved_when_switched_on() -> None:
    message = _page(include_reasoning=True).messages[0]

    assert message.reasoning_content == "新推理"
    assert [block.type for block in message.content] == ["reasoning", "text"]
