"""
历史侧的展示过滤：messages API 与实时流用同一条「默认隐藏」。

没有 presenter 的调用块不出现在历史里；旧行（没有 display）用**当前**
配置补一份，而不是因为历史里没存就消失。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.agent.context.processor.reference import wrap_reference
from app.services.session.application._assembly import build_message
from app.services.session.domain.entities import AgentMessage
from app.services.tool.application.presentation_lookup import (
    build_presentation_lookup,
)
from app.services.tool.domain.schemas.enum import ToolRenderer


def _presentation(**overrides):
    values = {
        "renderer": ToolRenderer.TABLE,
        "display_name": "行情查询",
        "summary": "共 {output.rowCount} 条",
        "show_input": False,
        "show_output": True,
        "props": {"output_path": "/bars"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _entity(content: list[dict]) -> AgentMessage:
    now = datetime.now(timezone.utc)

    return AgentMessage(
        message_id=uuid4(),
        session_id=uuid4(),
        run_id=uuid4(),
        seq=1,
        role="assistant",
        kind="tool_activity",
        business_agent_id=None,
        content=content,
        reasoning_content=None,
        status="completed",
        client_message_id=None,
        reply_to_message_id=None,
        meta={},
        created_at=now,
        updated_at=now,
    )


def test_lookup_covers_both_name_forms() -> None:
    row = SimpleNamespace(
        tool_key="ytrader_market.search_instruments",
        renderer=ToolRenderer.TABLE,
        display_name="标的搜索",
        summary=None,
        show_input=False,
        show_output=False,
        props={},
    )

    lookup = build_presentation_lookup([row])

    assert lookup["ytrader_market.search_instruments"] is row
    assert lookup["ytrader_market_search_instruments"] is row


def test_none_means_no_filtering() -> None:
    """老调用方（不传表）行为逐字不变。"""

    message = build_message(_entity([
        {"type": "tool_call", "id": "c1", "name": "x", "args": {}},
    ]))

    block = message.content[0].model_dump()

    assert block["type"] == "tool_call"
    assert "display" not in block


def test_presented_call_gets_display() -> None:
    message = build_message(
        _entity([
            {"type": "tool_call", "id": "c1",
             "name": "ytrader_market_search_instruments",
             "args": {"keyword": "茅台"}},
        ]),
        presentations={
            "ytrader_market_search_instruments": _presentation(),
        },
    )

    block = message.content[0].model_dump()

    assert block["display"]["renderer"] == "table"
    assert block["display"]["name"] == "行情查询"


def test_unpresented_call_is_dropped_but_narration_kept() -> None:
    message = build_message(
        _entity([
            {"type": "text", "text": "我先查一下"},
            {"type": "tool_call", "id": "c1",
             "name": "ytrader_market_search_instruments", "args": {}},
        ]),
        presentations={},
    )

    assert [block.type for block in message.content] == ["text"]
    assert message.content[0].text == "我先查一下"


def test_unpresented_result_drops_block_and_raw_text() -> None:
    """结果原文也不回：前端虽不渲染，但接口不该带着被隐藏工具的数据。"""

    message = build_message(
        _entity([
            {"type": "text", "text": '{"items": []}'},
            {"type": "tool_result", "tool_call_id": "c1",
             "name": "ytrader_market.search_instruments", "status": "done"},
        ]),
        presentations={},
    )

    assert message.content == []


def test_old_result_block_heals_from_raw_text() -> None:
    """
    旧行的 `tool_result` 可能是坏的（写的时候还没解 `<ref:tool>`，
    output=[]、summary=「共  条」）。读时用文本块里的原始结果重算。
    """

    body = json.dumps({"rowCount": 1, "bars": [{"close": 12.5}]})
    wrapped = wrap_reference(
        "tool", "ytrader_market.search_instruments", body
    )

    message = build_message(
        _entity([
            {"type": "text", "text": wrapped},
            {
                "type": "tool_result",
                "tool_call_id": "c1",
                "name": "ytrader_market.search_instruments",
                "status": "done",
                "output": [],
                "display": {
                    "renderer": "table",
                    "name": "行情查询",
                    "summary": "共  条",
                },
            },
        ]),
        presentations={
            "ytrader_market.search_instruments": _presentation(),
        },
    )

    block = next(
        block for block in message.content if block.type == "tool_result"
    ).model_dump()

    assert block["output"] == [{"close": 12.5}]
    assert block["display"]["summary"] == "共 1 条"


def test_presented_result_gets_display() -> None:
    message = build_message(
        _entity([
            {"type": "tool_result", "tool_call_id": "c1",
             "name": "ytrader_market.search_instruments",
             "status": "done", "output": [{"code": "600519.SH"}]},
        ]),
        presentations={
            "ytrader_market.search_instruments": _presentation(),
        },
    )

    block = message.content[0].model_dump()

    assert block["display"]["renderer"] == "table"
    assert block["output"] == [{"code": "600519.SH"}]


def test_old_failed_result_is_healed_on_read() -> None:
    """
    旧行把错误正文当 output 写进了结果块，summary 因此是「共  条」。
    实时侧已经改成"失败只有 error"，读时按同一口径重算旧行。
    """

    message = build_message(
        _entity([
            {
                "type": "tool_result",
                "tool_call_id": "c1",
                "name": "ytrader_market.search_instruments",
                "status": "error",
                "error": "ToolError: boom",
                "output": [],
                "display": {
                    "renderer": "table",
                    "name": "行情查询",
                    "summary": "共  条",
                },
            },
        ]),
        presentations={
            "ytrader_market.search_instruments": _presentation(),
        },
    )

    block = message.content[0].model_dump()

    assert "output" not in block
    assert "summary" not in block["display"]
    assert block["display"]["name"] == "行情查询"
    assert block["error"] == "ToolError: boom"


def test_tool_call_name_is_normalized_on_read() -> None:
    """老行的 tool_call.name 是 wire name，读时按当前 presenter 归一。"""

    row = SimpleNamespace(
        tool_key="ytrader_market.search_instruments",
        renderer=ToolRenderer.TABLE,
        display_name="行情查询",
        summary=None,
        show_input=False,
        show_output=False,
        props={},
    )

    message = build_message(
        _entity([
            {"type": "tool_call", "id": "c1",
             "name": "ytrader_market_search_instruments", "args": {}},
        ]),
        presentations=build_presentation_lookup([row]),
    )

    assert message.content[0].name == "ytrader_market.search_instruments"
