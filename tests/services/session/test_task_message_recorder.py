"""任务消息落库：kind 扩展与 task_id meta。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from app.services.session.application._assembly import build_message_page
from app.services.session.application.message_recorder import (
    new_assistant_messages,
    to_entities,
)

__all__: list[str] = []


def _task_instruction() -> HumanMessage:
    return HumanMessage(
        "任务指令",
        id="task:plan-1:t1:instruction",
        additional_kwargs={
            "kind": "task_instruction",
            "task_id": "plan-1:t1",
            "plan_id": "plan-1",
            "skill_key": "research",
        },
    )


def _task_result() -> AIMessage:
    return AIMessage(
        "任务结果",
        additional_kwargs={
            "kind": "task_result",
            "task_id": "plan-1:t1",
            "plan_id": "plan-1",
            "skill_key": "research",
        },
    )


def test_task_instruction_human_is_not_filtered() -> None:
    user = HumanMessage("用户消息", id="u1")

    kept = new_assistant_messages(0, [user, _task_instruction()])

    assert [message.id for message in kept] == [
        "task:plan-1:t1:instruction",
    ]


def test_task_messages_get_new_kind_and_meta() -> None:
    entities = to_entities(
        [_task_instruction(), _task_result()],
        session_id=uuid4(),
        run_id=uuid4(),
        start_seq=1,
        status="completed",
        business_agent_id=None,
        reply_to_message_id=None,
        created_at=datetime.now(timezone.utc),
    )

    assert [entity.kind for entity in entities] == [
        "task_instruction",
        "task_result",
    ]
    assert all(entity.role == "assistant" for entity in entities)
    assert all(entity.meta["task_id"] == "plan-1:t1" for entity in entities)
    assert all(entity.meta["plan_id"] == "plan-1" for entity in entities)
    assert all(
        entity.meta["skill_key"] == "research" for entity in entities
    )


def test_message_list_returns_task_kinds_and_metadata() -> None:
    session_id = uuid4()
    entities = to_entities(
        [_task_instruction(), _task_result()],
        session_id=session_id,
        run_id=uuid4(),
        start_seq=1,
        status="completed",
        business_agent_id=None,
        reply_to_message_id=None,
        created_at=datetime.now(timezone.utc),
    )

    page = build_message_page(
        session_id=session_id,
        entities=entities,
        limit=50,
        has_more_before=False,
        has_more_after=False,
    )

    assert [message.kind for message in page.messages] == [
        "task_instruction",
        "task_result",
    ]
    assert all(
        message.metadata["task_id"] == "plan-1:t1"
        for message in page.messages
    )
