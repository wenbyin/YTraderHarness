"""
工具活动的落库形状：`tool_call.display` + `tool_result`。

刷新前后渲染一致的前提是同一份数据：
    调用（AIMessage）  -> tool_call 块带 display（调用时的摘要）
    结果（ToolMessage）-> tool_result 块带 status / output / 最终 display

⚠️ 没有 presenter 就不写 `display`（缺键），不是 null / 空对象。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.context.processor.reference import wrap_reference
from app.services.session.application.message_recorder import to_entities
from app.services.tool.domain.schemas.enum import ToolRenderer


def _presentation(**overrides):
    values = {
        "renderer": ToolRenderer.TIPS,
        "display_name": "当前时间",
        "summary": "当前时间 {output.now}",
        "show_input": False,
        "show_output": True,
        "props": {"fields": ["/now"]},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _entities(messages, presentations):
    return to_entities(
        messages,
        session_id=uuid4(),
        run_id=uuid4(),
        start_seq=1,
        status="completed",
        business_agent_id=None,
        reply_to_message_id=None,
        created_at=datetime.now(timezone.utc),
        presentations=presentations,
    )


def _call_message() -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {
                "name": "agent_current_time",
                "args": {"timezone": "Asia/Shanghai"},
                "id": "call-1",
            }
        ],
    )


def _result_message(*, content: str, status: str) -> ToolMessage:
    return ToolMessage(
        content,
        tool_call_id="call-1",
        name="agent.current_time",
        status=status,
    )


def _block(content, block_type: str):
    return next(block for block in content if block["type"] == block_type)


def test_call_and_result_carry_the_presentation() -> None:
    presentation = _presentation()
    presentations = {
        # 两边名字形态都要命中：模型/工具调用用 wire name，
        # ToolMessage 的 name 是 tool_key。
        "agent_current_time": presentation,
        "agent.current_time": presentation,
    }

    call, result = _entities(
        [
            _call_message(),
            _result_message(content='{"now": "12:00"}', status="success"),
        ],
        presentations,
    )

    call_block = _block(call.content, "tool_call")
    assert call_block["display"]["renderer"] == "tips"
    assert call_block["display"]["name"] == "当前时间"
    # called 阶段没有 output：要 output 的模板整句不发（不是渲染成「共  条」），
    # 等 completed 的 display 再带上完整那句。
    assert "summary" not in call_block["display"]

    result_block = _block(result.content, "tool_result")
    assert result_block["tool_call_id"] == "call-1"
    assert result_block["status"] == "done"
    assert result_block["output"] == {"now": "12:00"}
    assert result_block["display"]["summary"] == "当前时间 12:00"


def test_without_presentation_display_key_is_absent() -> None:
    call, result = _entities(
        [
            _call_message(),
            _result_message(content='{"now": "12:00"}', status="success"),
        ],
        {},
    )

    call_block = _block(call.content, "tool_call")
    assert "display" not in call_block

    result_block = _block(result.content, "tool_result")
    assert "display" not in result_block
    # show_output 未知（没有 presenter）时也不落结果正文。
    assert "output" not in result_block
    assert result_block["status"] == "done"


def test_show_output_false_keeps_display_but_drops_output() -> None:
    presentation = _presentation(show_output=False, props={})
    presentations = {
        "agent_current_time": presentation,
        "agent.current_time": presentation,
    }

    _, result = _entities(
        [
            _call_message(),
            _result_message(content='{"now": "12:00"}', status="success"),
        ],
        presentations,
    )

    result_block = _block(result.content, "tool_result")
    assert "output" not in result_block
    assert "display" in result_block


def test_output_path_extracts_rows_for_tables() -> None:
    presentation = _presentation(
        summary="共 {output.rowCount} 条",
        props={"output_path": "/bars"},
    )
    presentations = {
        "agent_current_time": presentation,
        "agent.current_time": presentation,
    }

    _, result = _entities(
        [
            _call_message(),
            _result_message(
                content='{"rowCount": 1, "bars": [{"close": 12.5}]}',
                status="success",
            ),
        ],
        presentations,
    )

    result_block = _block(result.content, "tool_result")

    # 落库的是取段后的数组；summary 用原始包装渲染。
    assert result_block["output"] == [{"close": 12.5}]
    assert result_block["display"]["summary"] == "共 1 条"


def test_ref_wrapped_result_is_unwrapped_for_tables() -> None:
    """
    真实的 ToolMessage 正文是 `<ref:tool>` 包起来的（带引用规则尾注）。

    不先取正文就直接 json.loads，永远解析失败 —— 症状是历史里
    「共  条」+ 空表，而实时通道是好的。
    """

    presentation = _presentation(
        summary="共 {output.rowCount} 条",
        props={"output_path": "/bars"},
    )
    presentations = {
        "agent_current_time": presentation,
        "agent.current_time": presentation,
    }

    body = json.dumps({"rowCount": 1, "bars": [{"close": 12.5}]})
    wrapped = wrap_reference("tool", "agent.current_time", body)

    _, result = _entities(
        [
            _call_message(),
            ToolMessage(
                wrapped,
                tool_call_id="call-1",
                name="agent.current_time",
                status="success",
            ),
        ],
        presentations,
    )

    result_block = _block(result.content, "tool_result")

    assert result_block["output"] == [{"close": 12.5}]
    assert result_block["display"]["summary"] == "共 1 条"


def test_error_result_records_status_and_message() -> None:
    _, result = _entities(
        [_call_message(), _result_message(content="boom", status="error")],
        {},
    )

    result_block = _block(result.content, "tool_result")
    assert result_block["status"] == "error"
    assert result_block["error"] == "boom"


def test_failed_result_has_no_output_or_summary() -> None:
    """
    失败时不要把错误正文当 output：否则「共 {output.rowCount} 条」会渲染
    成「共  条」。与实时事件同一条口径 —— 失败只有 error。
    """

    presentation = _presentation(
        summary="共 {output.rowCount} 条",
        props={"output_path": "/bars"},
    )
    presentations = {
        "agent_current_time": presentation,
        "agent.current_time": presentation,
    }

    _, result = _entities(
        [
            _call_message(),
            _result_message(
                content="ToolError: boom",
                status="error",
            ),
        ],
        presentations,
    )

    result_block = _block(result.content, "tool_result")

    assert result_block["status"] == "error"
    assert "output" not in result_block
    assert "summary" not in result_block["display"]
    assert result_block["error"] == "ToolError: boom"


def test_result_block_carries_duration_from_additional_kwargs() -> None:
    """耗时随 ToolMessage.additional_kwargs 持久化，落库时写进结果块。"""

    presentation = _presentation()
    presentations = {
        "agent_current_time": presentation,
        "agent.current_time": presentation,
    }

    _, result = _entities(
        [
            _call_message(),
            ToolMessage(
                '{"now": "12:00"}',
                tool_call_id="call-1",
                name="agent.current_time",
                status="success",
                additional_kwargs={"duration_ms": 42.4},
            ),
        ],
        presentations,
    )

    assert _block(result.content, "tool_result")["duration_ms"] == 42


def test_tool_call_name_is_normalized_to_tool_key() -> None:
    """
    历史里 tool_call 块的 name 以前是 wire name，而事件与 tool_result
    块用 tool_key —— 同一条记录两种写法。落库时统一成 tool_key。
    """

    presentation = _presentation(tool_key="agent.current_time")
    presentations = {
        "agent_current_time": presentation,
        "agent.current_time": presentation,
    }

    (call,) = _entities([_call_message()], presentations)

    assert _block(call.content, "tool_call")["name"] == "agent.current_time"
