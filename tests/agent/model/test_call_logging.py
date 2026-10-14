"""
LLM 调用的正文日志（DEBUG 级，输入与输出各一条）。

背景与边界：

    observer.LoggingModelObserver 只打**结论**（谁答的、多久、几次），
    不打正文 —— 那是刻意的："日志不该是'模型说了什么'的第二份存储"。

这里补的是正文，压在 **DEBUG**：默认级别下看不见，要看的人显式调级别。
两条规则因此必须一起成立，也正是这组用例守的东西：

    默认级别下**一条都不打**（否则那个立场就白写了）
    DEBUG 下打得**全**（截断会让"模型到底看到了什么"这个唯一目的失效）
    正文先**脱敏**（redaction.py：任何要落日志的东西都先过它）
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.model import (
    LangChainProviderAdapter,
    ModelDefinition,
    ModelRequest,
)
from app.agent.model.provider import ModelClient

_LOGGER = "app.agent.model.adapters"


class _FakeChatModel:
    """鸭子类型的 langchain 客户端，响应内容可控。"""

    def __init__(self, response: str = "答复正文") -> None:
        self._response = response

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, fmt, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        return SimpleNamespace(
            content=self._response,
            tool_calls=None,
            usage_metadata=None,
        )


def _client(model, *, model_name: str = "deepseek-v4-flash") -> ModelClient:

    return ModelClient(
        definition=ModelDefinition.make("deepseek", model_name),
        client=model,
        adapter=LangChainProviderAdapter(),
        timeout=5.0,
    )


async def test_default_level_logs_no_prompt_content(caplog) -> None:
    """
    默认（INFO）下不出现正文 —— 这条是"DEBUG 才打"这个约定的守门人。

    真要有人把它提到 INFO，失败的会是这一条，而不是某天发现日志里
    躺着用户数据。
    """

    model = _FakeChatModel()
    request = ModelRequest(messages=(HumanMessage("这是一句不该出现在日志里的话"),))

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        await _client(model).adapter.ainvoke(
            _client(model), request, timeout=5.0
        )

    assert "这是一句不该出现在日志里的话" not in caplog.text


async def test_debug_logs_input_and_response(caplog) -> None:
    """DEBUG 下输入与输出都在，且能看出是哪家的哪个模型。"""

    model = _FakeChatModel("答复正文")
    request = ModelRequest(messages=(HumanMessage("用户说的话"),))

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await _client(model).adapter.ainvoke(
            _client(model), request, timeout=5.0
        )

    assert "用户说的话" in caplog.text, "输入没打出来"
    assert "答复正文" in caplog.text, "响应没打出来"
    assert "deepseek:deepseek-v4-flash" in caplog.text


async def test_request_is_logged_before_the_call(caplog) -> None:
    """
    请求打在调用**之前** —— 失败路径上没有任何"之后"。

    所以哪怕这次一定抛错，输入也必须已经记下来。
    """

    class _Boom(_FakeChatModel):
        async def ainvoke(self, messages, config=None):
            raise RuntimeError("假装 provider 挂了")

    request = ModelRequest(messages=(HumanMessage("失败前也该看得见这句话"),))

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        with pytest.raises(Exception):
            await _client(_Boom()).adapter.ainvoke(
                _client(_Boom()), request, timeout=5.0
            )

    assert "失败前也该看得见这句话" in caplog.text


async def test_prompt_is_redacted(caplog) -> None:
    """
    正文里的密钥被抹掉 —— redaction.py 的规矩：
    "任何要落日志的东西都先过这里"。

    提示词里出现密钥**不是**有人故意写进去的（用户粘贴、上游异常回显），
    所以不能靠"注意别打"。
    """

    model = _FakeChatModel()
    request = ModelRequest(
        messages=(
            HumanMessage("我的 key 是 sk-abcdef1234567890，帮我看看"),
        ),
    )

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await _client(model).adapter.ainvoke(
            _client(model), request, timeout=5.0
        )

    assert "sk-abcdef1234567890" not in caplog.text, "密钥被打进日志了"
    assert "***" in caplog.text


async def test_tool_calls_are_visible(caplog) -> None:
    """响应里的工具调用要看得见 —— 排查"模型为什么不调工具"靠它。"""

    class _ToolCalling(_FakeChatModel):
        async def ainvoke(self, messages, config=None):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_filing",
                        "args": {"symbol": "AAPL"},
                        "id": "call-1",
                    }
                ],
            )

    request = ModelRequest(messages=(HumanMessage("查一下"),))

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await _client(_ToolCalling()).adapter.ainvoke(
            _client(_ToolCalling()), request, timeout=5.0
        )

    assert "search_filing" in caplog.text
    assert "AAPL" in caplog.text
