"""
Command 是通用能力，不是计划工具专用。

    - 控制器是**本批第一个调用**，与工具名无关；
    - 任何工具返回的 `Command.update` 都原样并入 state，不按工具类型
      过滤 `message` / `reason` / 自定义键。

（运行时注入另有一套通用测试；这里只钉图语义。）
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.agent.graph.react.routing import route_after_agent
from app.agent.graph.react.tools import _command_update


def _message(*names: str) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {"name": name, "args": {}, "id": f"call-{index}"}
            for index, name in enumerate(names)
        ],
    )


def test_first_call_is_controller_regardless_of_tool_name() -> None:
    sends = route_after_agent(
        {  # type: ignore[typeddict-item]
            "messages": [_message("some_tool", "other_tool")],
            "plan_status": "off",
        },
        None,
    )

    assert [send.arg["is_controller"] for send in sends] == [
        True, False,
    ]
    assert [send.arg["call"]["name"] for send in sends] == [
        "some_tool", "other_tool",
    ]


def test_command_update_is_passed_through_unfiltered() -> None:
    update = {
        "message": "工具结果",
        "reason": "因为",
        "custom_channel": 1,
    }

    assert _command_update(Command(update=update)) == update
