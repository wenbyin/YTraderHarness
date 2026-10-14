"""
`message_reasoning`：思维链的三个来源与优先级。

为什么值得单独一组用例
--------------------
它是**唯一**一处把"模型想过什么"从消息里取出来的地方，而取不到的失败
长这样：`agent_message.reasoning_content` 是 NULL、前端推理框永远不出现、
没有任何一处报错。键名看错、优先级排错都会得到这个症状。

装置刻意用 `SimpleNamespace`（本模块的约定：只做鸭子类型、不 import
langchain），另外用真的 `AIMessage` 验一遍本仓库实际遇到的形状。
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from app.agent.context.processor.messages import (
    is_reasoning_block,
    message_reasoning,
)


def test_reasoning_in_content_blocks() -> None:
    """provider 自己把推理写成块 —— 优先级最高。"""

    message = SimpleNamespace(
        content=[
            {"type": "reasoning", "reasoning": "想 A"},
            {"type": "text", "text": "答"},
        ]
    )

    assert message_reasoning(message) == "想 A"


def test_reasoning_blocks_are_joined_with_a_blank_line() -> None:
    """
    多个推理块用空行连接。

    直接拼会让两段思考读成一句被截断的话；空行在前端那个折叠块里也是对的。
    """

    message = SimpleNamespace(
        content=[
            {"type": "reasoning", "reasoning": "第一段"},
            {"type": "text", "text": "答"},
            {"type": "reasoning", "reasoning": "第二段"},
        ]
    )

    assert message_reasoning(message) == "第一段\n\n第二段"


def test_reasoning_falls_back_to_the_text_key() -> None:
    """块的键是 `text` 而不是 `reasoning` 时也要认（两种都存在过）。"""

    message = SimpleNamespace(content=[{"type": "reasoning", "text": "想 B"}])

    assert message_reasoning(message) == "想 B"


def test_reasoning_from_additional_kwargs() -> None:
    """DeepSeek / Ollama / XAI / Groq 的形状。"""

    message = SimpleNamespace(
        content="",
        additional_kwargs={"reasoning_content": "想 C"},
    )

    assert message_reasoning(message) == "想 C"


def test_content_blocks_win_over_additional_kwargs() -> None:
    """
    两处都有时取**声明式**的那一处。

    这个顺序不是随意的：`content` 的块是 provider 自己给的，而
    additional_kwargs 与 langchain 的归一化都可能给出同一段 —— 反过来排
    会让"上游改了翻译规则"覆盖掉 provider 原文，而两段文本不同时症状只是
    "推理内容看着不太对"，很难归因。
    """

    message = SimpleNamespace(
        content=[{"type": "reasoning", "reasoning": "声明式"}],
        additional_kwargs={"reasoning_content": "另一份"},
    )

    assert message_reasoning(message) == "声明式"


def test_no_reasoning_is_none_not_empty_string() -> None:
    """
    没有推理就是 `None`。

    ⚠️ 空串会让那一列写进 `''`，而契约/前端把它读成"有推理" —— 界面因此
    渲染出一个永远空着的推理框。NULL 才是"这次没有"。
    """

    assert message_reasoning(SimpleNamespace(content="只有正文")) is None

    # 空白块按"没有"处理（有些 provider 会回一个纯换行的推理块）。
    assert message_reasoning(
        SimpleNamespace(content=[{"type": "reasoning", "reasoning": "   \n  "}])
    ) is None


def test_unexpected_shapes_do_not_raise() -> None:
    """形状不对一律当成"没有"，不抛 —— 取思维链不该把写库那条路炸掉。"""

    assert message_reasoning(SimpleNamespace()) is None
    assert message_reasoning(SimpleNamespace(content=None)) is None
    assert message_reasoning(
        SimpleNamespace(content="", additional_kwargs=None)
    ) is None
    assert message_reasoning(
        SimpleNamespace(content="", additional_kwargs={"reasoning_content": 123})
    ) is None


def test_real_deepseek_message() -> None:
    """
    真的 `AIMessage`（本仓库实测的形状）：`content=''`、推理在
    additional_kwargs、同时带着 tool_calls。

    那一轮是"边想边调工具"—— 正文是空的，所以**只有**思维链能说明模型
    当时在想什么。
    """

    message = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "先看看有哪些工具。"},
        tool_calls=[
            {"name": "describe_platform", "args": {}, "id": "call_1", "type": "tool_call"}
        ],
    )

    assert message_reasoning(message) == "先看看有哪些工具。"


def test_is_reasoning_block() -> None:
    """块类型判定：非 Mapping、缺 type、别的 type 都不算。"""

    assert is_reasoning_block({"type": "reasoning"}) is True
    assert is_reasoning_block({"type": "text", "text": "答"}) is False
    assert is_reasoning_block({"text": "没有 type"}) is False
    assert is_reasoning_block("reasoning") is False
    assert is_reasoning_block(None) is False
