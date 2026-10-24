"""agent 节点的任务结果标记：只有最终回答带 task_result。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.graph.react.agent_node import _tag_task_result

__all__: list[str] = []


def _task() -> dict:
    return {
        "id": "plan-1:t1",
        "metadata": {
            "task_id": "plan-1:t1",
            "plan_id": "plan-1",
            "skill_key": "research",
        },
    }


def test_final_answer_is_tagged_task_result() -> None:
    message = AIMessage("任务结果")

    tagged = _tag_task_result(
        message,
        {"current_task": _task()},  # type: ignore[typeddict-item]
    )

    assert tagged is not message
    assert tagged.additional_kwargs["kind"] == "task_result"
    assert tagged.additional_kwargs["task_id"] == "plan-1:t1"
    assert tagged.additional_kwargs["plan_id"] == "plan-1"
    assert tagged.additional_kwargs["skill_key"] == "research"


def test_tool_call_message_is_not_tagged() -> None:
    message = AIMessage(
        "",
        tool_calls=[
            {"name": "lookup", "args": {}, "id": "call-1"},
        ],
    )

    tagged = _tag_task_result(
        message,
        {"current_task": _task()},  # type: ignore[typeddict-item]
    )

    assert tagged is message
    assert "kind" not in tagged.additional_kwargs


def test_plain_chat_is_not_tagged() -> None:
    message = AIMessage("普通答复")

    tagged = _tag_task_result(
        message,
        {"current_task": {}},  # type: ignore[typeddict-item]
    )

    assert tagged is message
    assert "kind" not in tagged.additional_kwargs
