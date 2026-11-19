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
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.tools.hitl_tool import ASK_USER_QUESTION_TOOL_KEY
from app.services.session.application.message_recorder import (
    _interaction_entities,
    message_status,
    new_assistant_messages,
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


# ============================================================
# 本轮新增消息的切法：按 id 差集，不按条数
# ============================================================


def test_resumed_message_inserted_before_pending_write_is_not_lost() -> None:
    """
    真实故障：审批恢复（resume）时，被中断分支的 ToolMessage 插在暂停时
    已有 pending write **之前**，按条数切片会同时漏一条、重复一条。

    终态顺序： [H, AI, T(current), T(search), AI]
    暂停时已落库： [AI, T(search)]（T(search) 是 pending write）

        prior_ids   = {h1, a1, t_search}
        prior_count = 3

    按条数切 -> [T(search), AI]：漏掉 T(current)、重写 T(search)。
    按 id 差集 -> [T(current), AI]：正确。
    """

    human = HumanMessage("查一下", id="h1")

    first_ai = AIMessage(
        "",
        tool_calls=[{"name": "agent_current_time", "args": {}, "id": "call-0"}],
        id="a1",
    )
    search = ToolMessage(
        "search-result",
        tool_call_id="call-1",
        name="ytrader_market.search_instruments",
        id="t_search",
    )
    current = ToolMessage(
        "time-result",
        tool_call_id="call-0",
        name="agent.current_time",
        id="t_current",
    )
    final_ai = AIMessage("done", id="a2")

    produced = new_assistant_messages(
        {"h1", "a1", "t_search"},
        [human, first_ai, current, search, final_ai],
    )

    assert produced == [current, final_ai]


def test_messages_without_id_are_kept() -> None:
    """没有 id 的消息无法证明已落库 —— 保留，宁可多写不静默丢。"""

    plain = AIMessage("no-id")

    assert new_assistant_messages({"anything"}, [plain]) == [plain]


def test_human_filtered_but_task_instruction_kept() -> None:
    human = HumanMessage("用户说的", id="h1")
    instruction = HumanMessage(
        "内部任务指令",
        id="h2",
        additional_kwargs={"kind": "task_instruction"},
    )
    answer = AIMessage("回复", id="a1")

    produced = new_assistant_messages(
        set(),
        [human, instruction, answer],
    )

    assert produced == [instruction, answer]


# ============================================================
# ask_user_question：补一条 tool_activity（子图消息不进父图落库）
# ============================================================


def _question() -> dict:
    return {
        "question": "要哪种产出？",
        "header": "产出",
        "options": [
            {"label": "结构梳理", "description": "拆原文骨架"},
            {"label": "观点评价", "description": "判断论证是否站得住"},
        ],
        "multiSelect": False,
    }


def _presentation() -> SimpleNamespace:
    return SimpleNamespace(
        tool_key=ASK_USER_QUESTION_TOOL_KEY,
        renderer="inline",
        display_name="向用户提问",
        summary="向用户确认信息",
        show_input=False,
        show_output=False,
        props={},
    )


def test_interaction_gets_a_paired_tool_activity() -> None:
    question = _question()
    entities = _interaction_entities(
        session_id=uuid4(),
        run_id=uuid4(),
        start_seq=7,
        business_agent_id=None,
        questions=[question],
        answers={question["question"]: "结构梳理"},
        annotations=None,
        interrupt_id="int-1",
        tool_call_id="call-1",
        tool_args={"questions": [question], "metadata": {}},
        presentations={ASK_USER_QUESTION_TOOL_KEY: _presentation()},
        created_at=datetime.now(timezone.utc),
    )

    tool, interaction = entities

    assert (tool.seq, tool.role, tool.kind) == (
        7, "assistant", "tool_activity",
    )
    assert tool.content[0]["type"] == "tool_call"
    assert tool.content[0]["id"] == "call-1"
    assert tool.content[0]["name"] == ASK_USER_QUESTION_TOOL_KEY
    assert tool.content[0]["display"]["name"] == "向用户提问"
    assert tool.content[1]["type"] == "tool_result"
    assert tool.content[1]["tool_call_id"] == "call-1"
    assert tool.content[1]["status"] == "done"
    # show_output=false：结果正文不进块。
    assert "output" not in tool.content[1]

    assert (interaction.seq, interaction.role, interaction.kind) == (
        8, "user", "interaction",
    )
    assert interaction.meta["interrupt_id"] == "int-1"


def test_interaction_without_call_id_stays_a_single_message() -> None:
    entities = _interaction_entities(
        session_id=uuid4(),
        run_id=uuid4(),
        start_seq=1,
        business_agent_id=None,
        questions=[_question()],
        answers={},
        annotations=None,
        interrupt_id="int-1",
        tool_call_id=None,
        tool_args=None,
        presentations=None,
        created_at=datetime.now(timezone.utc),
    )

    assert len(entities) == 1
    assert entities[0].kind == "interaction"


def test_tool_activity_without_presentation_has_no_display() -> None:
    entities = _interaction_entities(
        session_id=uuid4(),
        run_id=uuid4(),
        start_seq=1,
        business_agent_id=None,
        questions=[_question()],
        answers={},
        annotations=None,
        interrupt_id="int-1",
        tool_call_id="call-1",
        tool_args={"questions": [_question()], "metadata": {}},
        presentations=None,
        created_at=datetime.now(timezone.utc),
    )

    tool = entities[0]

    assert "display" not in tool.content[0]
    assert "display" not in tool.content[1]
    assert "output" not in tool.content[1]
