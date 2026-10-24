"""DeepSeek 思考模式：历史 assistant 的 reasoning_content 必须回传。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.common.deepseek_chat import ReasoningChatDeepSeek

__all__: list[str] = []


def _client() -> ReasoningChatDeepSeek:
    return ReasoningChatDeepSeek(
        model="deepseek-v4-flash",
        api_key="test-key",
    )


def _messages() -> list:
    return [
        HumanMessage("第一轮"),
        AIMessage(
            "第一轮回答",
            additional_kwargs={"reasoning_content": "第一轮思维链"},
        ),
        HumanMessage("第二轮"),
    ]


def test_reasoning_content_is_sent_back_in_thinking_mode() -> None:
    payload = _client()._get_request_payload(_messages())

    assistant = payload["messages"][1]

    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == "第一轮思维链"


def test_reasoning_content_is_not_added_when_thinking_is_disabled() -> None:
    payload = _client()._get_request_payload(
        _messages(),
        extra_body={"thinking": {"type": "disabled"}},
    )

    assistant = payload["messages"][1]

    assert assistant["role"] == "assistant"
    assert "reasoning_content" not in assistant


def test_reasoning_content_from_content_blocks_is_sent_back() -> None:
    messages = [
        HumanMessage("第一轮"),
        AIMessage(
            [
                {"type": "reasoning", "reasoning": "块里的思维链"},
                {"type": "text", "text": "第一轮回答"},
            ],
        ),
        HumanMessage("第二轮"),
    ]

    payload = _client()._get_request_payload(messages)

    assert payload["messages"][1]["reasoning_content"] == "块里的思维链"


def test_empty_reasoning_block_still_sends_empty_string() -> None:
    messages = [
        HumanMessage("第一轮"),
        AIMessage(
            [
                {"type": "reasoning", "reasoning": ""},
                {"type": "text", "text": "第一轮回答"},
            ],
        ),
        HumanMessage("第二轮"),
    ]

    payload = _client()._get_request_payload(messages)

    # 空串不是“没有”：DeepSeek 思考模式要求每条 assistant 都带字段。
    assert payload["messages"][1]["reasoning_content"] == ""


def test_assistant_without_any_reasoning_gets_empty_string() -> None:
    """
    分类器合成的 enter_plan_mode 这类 AIMessage 没有思维链。

    它留在历史里，下一轮思考模式如果不补空串，DeepSeek 直接 400。
    """

    messages = [
        HumanMessage("第一轮"),
        AIMessage(
            "",
            tool_calls=[{
                "name": "agent_enter_plan_mode",
                "args": {},
                "id": "classifier-plan-1",
                "type": "tool_call",
            }],
        ),
        HumanMessage("第二轮"),
    ]

    payload = _client()._get_request_payload(messages)

    assistant = payload["messages"][1]

    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == ""
