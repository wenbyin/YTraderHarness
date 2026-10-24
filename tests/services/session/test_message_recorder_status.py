"""
HITL 暂停时消息仍是 completed。

背景
----
`waiting_input` 暂停的是 **run**，不是消息：AIMessage 已经生成完、
ToolMessage 也已经返回。把它们标成 interrupted 会让历史里所有
tool_activity 看起来都"被打断"，而它们其实都成功完成了。

"还在等人"由 `agent_run.status=waiting_input` 表达；用户的回答由
`kind=interaction` 消息承载。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from app.services.session.application.message_recorder import (
    message_status,
    to_entities,
)
from app.services.session.domain.schemas.enum import RunStatus


def test_waiting_input_messages_are_completed() -> None:
    assert message_status(RunStatus.WAITING_INPUT) == "completed"


def test_other_terminal_statuses() -> None:
    assert message_status(RunStatus.COMPLETED) == "completed"
    assert message_status(RunStatus.FAILED) == "failed"
    assert message_status(RunStatus.CANCELLED) == "cancelled"


def test_to_entities_uses_single_status() -> None:
    messages = [
        ToolMessage("工具结果", tool_call_id="call_1"),
        AIMessage("下一条"),
    ]

    entities = to_entities(
        messages,
        session_id=uuid4(),
        run_id=uuid4(),
        start_seq=1,
        status=message_status(RunStatus.WAITING_INPUT),
        business_agent_id=None,
        reply_to_message_id=None,
        created_at=datetime.now(timezone.utc),
    )

    assert [entity.status for entity in entities] == [
        "completed",
        "completed",
    ]
    assert [entity.seq for entity in entities] == [1, 2]
