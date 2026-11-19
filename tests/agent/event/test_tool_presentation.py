"""
工具展示描述符：纯函数与投影门控。

三条要害：

    1. 没有 presenter 的工具**一条事件都不发**（默认隐藏）；
    2. show_input / show_output 关闭时，键是**不出现**，不是 null；
    3. 门控未启用（None）时保持改造前行为 —— 旧调用方不受影响。
"""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agent.event import run_event_projector as projector_module
from app.agent.event.run_event_projector import (
    RunEventProjector,
    projector_factory,
)
from app.agent.event.tool_presentation import (
    build_display,
    render_summary,
    restrict_fields,
    shape_tool_output,
)
from app.services.tool.domain.schemas import ToolPresentation


def _presentation(**overrides) -> ToolPresentation:
    return ToolPresentation(**{"renderer": "card", **overrides})


def _projector(
        presentations,
        wire_names=(),
        announced=None,
) -> RunEventProjector:
    return RunEventProjector(
        (),
        run_id="run-1",
        message_nodes={"agent"},
        tool_presentations=presentations,
        displayed_tool_wire_names=set(wire_names),
        announced_tool_calls=announced,
    )


async def _emit_called(
        projector: RunEventProjector,
        frame: dict,
) -> list[tuple]:
    events: list[tuple] = []

    async def fake_emit(event_type, *, node, data):
        events.append((event_type, dict(data)))

    projector._emit = fake_emit  # type: ignore[method-assign]

    await projector._on_tool_stream_event(frame)

    return events


@pytest.fixture
def captured_completed(monkeypatch):
    """`_emit_tool_completed` 走的是模块级 emit_event，单独拦。"""

    events: list = []

    async def fake_emit(event):
        events.append(event)

    monkeypatch.setattr(projector_module, "emit_event", fake_emit)

    return events


# ============================================================
# 纯函数
# ============================================================


def test_summary_renders_input_and_output_paths() -> None:
    summary = render_summary(
        "已查询 {input.symbol} 共 {output.count} 条",
        input_value={"symbol": "600519"},
        output_value={"count": 3},
    )

    assert summary == "已查询 600519 共 3 条"


def test_summary_renders_missing_path_as_empty() -> None:
    assert render_summary(
        "{input.nope}",
        input_value={"symbol": "600519"},
    ) == ""


def test_summary_without_template_is_none() -> None:
    assert render_summary(None) is None
    assert render_summary("") is None


def test_summary_is_omitted_until_output_exists() -> None:
    """called 阶段没有 output：不发「共  条」，completed 再发完整那句。"""

    assert render_summary("共 {output.rowCount} 条", output_value=None) is None
    assert render_summary(
        "已查询 {input.symbol}", input_value={"symbol": "600519"}
    ) == "已查询 600519"


def test_restrict_fields_keeps_only_whitelist() -> None:
    assert restrict_fields(
        {"symbol": "600519", "secret": "x"},
        ["/symbol"],
    ) == {"symbol": "600519"}


def test_restrict_fields_empty_whitelist_passes_through() -> None:
    value = {"symbol": "600519"}

    assert restrict_fields(value, None) is value
    assert restrict_fields(value, []) is value


def test_restrict_fields_nested_pointer() -> None:
    assert restrict_fields(
        {"filters": {"date_from": "2026-01-01", "x": 1}},
        ["/filters/date_from"],
    ) == {"filters": {"date_from": "2026-01-01"}}


def test_output_path_must_be_a_pointer() -> None:
    with pytest.raises(ValueError):
        _presentation(props={"output_path": "bars"})


def test_shape_tool_output_omits_output_when_not_shown() -> None:
    shaped = shape_tool_output(
        _presentation(show_output=False),
        {"x": 1},
    )

    assert shaped.has_output is False
    assert shaped.output is None
    assert shaped.display["renderer"] == "card"


def test_shape_tool_output_extracts_and_caps_rows() -> None:
    shaped = shape_tool_output(
        _presentation(
            show_output=True,
            summary="共 {output.rowCount} 条",
            props={"output_path": "/bars", "max_rows": 1},
        ),
        {"rowCount": 2, "bars": [{"a": 1}, {"a": 2}]},
    )

    assert shaped.has_output is True
    assert shaped.output == [{"a": 1}]
    # summary 用取段**之前**的包装对象渲染。
    assert shaped.display["summary"] == "共 2 条"


def test_display_carries_renderer_name_props() -> None:
    display = build_display(
        _presentation(display_name="行情查询", props={"columns": ["price"]})
    )

    assert display == {
        "renderer": "card",
        "name": "行情查询",
        "props": {"columns": ["price"]},
    }


# ============================================================
# 投影门控
# ============================================================


async def test_hidden_tool_emits_nothing(
        captured_completed,
) -> None:
    projector = _projector({})

    called = await _emit_called(projector, {
        "event": "tool.called",
        "tool_call_id": "c1",
        "tool_name": "memory_recall",
        "input": {"q": "x"},
    })

    assert called == []

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "memory_recall",
        "output": {"hits": []},
    })

    assert captured_completed == []


async def test_called_omits_input_when_not_shown() -> None:
    projector = _projector({
        "quote_get": _presentation(
            display_name="行情",
            summary="已查询 {input.symbol}",
            show_input=False,
        ),
    })

    events = await _emit_called(projector, {
        "event": "tool.called",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "input": {"symbol": "600519", "secret": "x"},
    })

    assert len(events) == 1
    _, data = events[0]

    assert "input" not in data
    assert data["display"] == {
        "renderer": "card",
        "name": "行情",
        "summary": "已查询 600519",
    }


async def test_called_restricts_input_by_fields() -> None:
    projector = _projector({
        "quote_get": _presentation(
            show_input=True,
            props={"fields": ["/symbol"]},
        ),
    })

    events = await _emit_called(projector, {
        "event": "tool.called",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "input": {"symbol": "600519", "secret": "x"},
    })

    assert events[0][1]["input"] == {"symbol": "600519"}


async def test_completed_shows_output_and_renders_summary(
        captured_completed,
) -> None:
    projector = _projector({
        "quote_get": _presentation(
            show_output=True,
            props={"fields": ["/count"]},
            summary="共 {output.count} 条",
        ),
    })

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "output": {"count": 3, "secret": "x"},
    })

    assert len(captured_completed) == 1

    data = captured_completed[0].data

    assert data["output"] == {"count": 3}
    assert data["display"]["renderer"] == "card"
    assert data["display"]["summary"] == "共 3 条"


async def test_completed_output_can_be_extracted_with_output_path(
        captured_completed,
) -> None:
    """
    `props.output_path`：从包装结果里取行数组，summary 仍读包装上的计数。

    MCP 行情工具返回 `{bars: [...], rowCount: N}`；不取段的话前端会把
    整个包装对象画成一行。
    """

    projector = _projector({
        "ytrader_market.query_market_data": _presentation(
            show_output=True,
            summary="共 {output.rowCount} 条",
            props={"output_path": "/bars"},
        ),
    })

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "ytrader_market.query_market_data",
        "output": {
            "rowCount": 2,
            "bars": [{"close": 1}, {"close": 2}],
        },
    })

    data = captured_completed[0].data

    assert data["output"] == [{"close": 1}, {"close": 2}]
    # summary 用的是**原始** output（取段之前），所以 rowCount 还在。
    assert data["display"]["summary"] == "共 2 条"


async def test_completed_output_string_is_decoded_for_display(
        captured_completed,
) -> None:
    """
    真实 custom stream 的 `output` 是 **JSON 字符串**（工具节点 `_render()`）。

    展示层按结构读，先解一层；否则 `{output.rowCount}` 与 `/bars` 全部取空，
    症状就是「共  条」+ 空表。
    """

    projector = _projector({
        "ytrader_market.query_market_data": _presentation(
            show_output=True,
            summary="共 {output.rowCount} 条",
            props={"output_path": "/bars"},
        ),
    })

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "ytrader_market.query_market_data",
        "output": '{"rowCount": 2, "bars": [{"close": 1}, {"close": 2}]}',
    })

    data = captured_completed[0].data

    assert data["output"] == [{"close": 1}, {"close": 2}]
    assert data["display"]["summary"] == "共 2 条"


async def test_max_rows_caps_the_payload(
        captured_completed,
) -> None:
    """表格只画 5 行；`max_rows` 把发送/落库的数组也压住，摘要仍报总量。"""

    projector = _projector({
        "ytrader_market.query_market_data": _presentation(
            show_output=True,
            summary="共 {output.rowCount} 条",
            props={"output_path": "/bars", "max_rows": 2},
        ),
    })

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "ytrader_market.query_market_data",
        "output": {
            "rowCount": 3,
            "bars": [{"close": 1}, {"close": 2}, {"close": 3}],
        },
    })

    data = captured_completed[0].data

    assert data["output"] == [{"close": 1}, {"close": 2}]
    assert data["display"]["summary"] == "共 3 条"


async def test_missing_output_path_yields_an_empty_array(
        captured_completed,
) -> None:
    projector = _projector({
        "ytrader_market.query_market_data": _presentation(
            show_output=True,
            props={"output_path": "/bars"},
        ),
    })

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "ytrader_market.query_market_data",
        "output": {"rowCount": 0},
    })

    # 空数组 -> 前端 buildTable 返回 null -> 退化成 card，而不是画包装对象。
    assert captured_completed[0].data["output"] == []


async def test_legacy_mode_keeps_input_and_output() -> None:
    """门控未启用（None）时行为与改造前一致。"""

    projector = _projector(None)

    events = await _emit_called(projector, {
        "event": "tool.called",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "input": {"symbol": "600519"},
    })

    assert events[0][1]["input"] == {"symbol": "600519"}
    assert "display" not in events[0][1]


def test_tool_calls_list_is_filtered_to_presented_tools() -> None:
    projector = _projector(
        {"quote_get": _presentation()},
        wire_names={"quote_get"},
    )

    assert projector._visible_tool_names(
        ["quote_get", "memory_recall"]
    ) == ["quote_get"]


def test_tool_calls_list_is_untouched_when_gating_disabled() -> None:
    projector = _projector(None)

    assert projector._visible_tool_names(
        ["quote_get", "memory_recall"]
    ) == ["quote_get", "memory_recall"]


# ============================================================
# 规范事件契约
# ============================================================


class _SendState(TypedDict, total=False):
    messages: list


async def test_send_branch_tool_events_reach_the_projector(
        monkeypatch,
) -> None:
    """
    真图 + 真 Send 分支 + 真 custom stream 的最小回归。

    工具事件不在**根**节点里写，而是在 `Send` 扇出的分支里写 ——
    这一次要钉住的正是那条路径。投影层跑的是真 `RunEventProjector`，
    只把最终 `emit_event` 换成收集器。
    """

    captured: list = []

    async def fake_emit(event):
        captured.append(event)

    monkeypatch.setattr(projector_module, "emit_event", fake_emit)

    def agent(state):
        return {}

    def route(state):
        return [Send("tools", {})]

    def tools(state):
        get_stream_writer()({
            "kind": "tool",
            "event": "tool.called",
            "tool_call_id": "c1",
            "tool_name": "agent.current_time",
            "input": {"timezone": "Asia/Shanghai"},
        })
        get_stream_writer()({
            "kind": "tool",
            "event": "tool.completed",
            "tool_call_id": "c1",
            "tool_name": "agent.current_time",
            "output": {"now": "12:00"},
        })
        return {}

    graph = StateGraph(_SendState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", END)

    presentation = _presentation(
        renderer="inline",
        display_name="当前时间",
        show_output=True,
    )

    run = await graph.compile().astream_events(
        {"messages": []},
        version="v3",
        transformers=[
            projector_factory(
                run_id="run-1",
                message_nodes={"agent"},
                tool_presentations={
                    "agent.current_time": presentation,
                    "agent_current_time": presentation,
                },
                displayed_tool_wire_names={"agent_current_time"},
            ),
        ],
    )

    async with run:
        await run.output()

    tool_events = [
        event for event in captured
        if event.type.value.startswith("tool.")
    ]

    assert [event.type.value for event in tool_events] == [
        "tool.called",
        "tool.completed",
    ]

    called, completed = tool_events

    # show_input 默认 false → 键不出现；display 仍然带文案。
    assert "input" not in called.data
    assert called.data["display"] == {
        "renderer": "inline",
        "name": "当前时间",
    }

    assert completed.data["output"] == {"now": "12:00"}
    assert completed.data["display"]["renderer"] == "inline"


def test_canonical_payload_accepts_display() -> None:
    """
    payload 模型是 extra="forbid"：新增 display 必须在 spec 里显式声明，
    否则投影层发出的每一条工具事件都会在构造 Event 时炸掉。
    """

    from app.common.event.spec import Event, EventType

    called = Event(
        type=EventType.TOOL_CALLED,
        run_id="run-1",
        data={
            "tool_call_id": "c1",
            "tool_name": "quote_get",
            "display": {"renderer": "card", "name": "行情"},
        },
    )

    assert called.data["display"] == {
        "renderer": "card",
        "name": "行情",
    }

    completed = Event(
        type=EventType.TOOL_COMPLETED,
        run_id="run-1",
        data={
            "tool_call_id": "c1",
            "tool_name": "quote_get",
            "output": {"count": 3},
            "display": {"renderer": "card"},
        },
    )

    assert completed.data["display"] == {"renderer": "card"}


# ============================================================
# 失败 / 耗时 / 重复公告
# ============================================================


async def test_failed_completion_has_no_output_or_summary(
        captured_completed,
) -> None:
    """
    失败调用只有 error：不把错误文本当 output，也不渲染「共  条」。

    custom 流失败时本来就没有 output 键（`_tool_event_result` 只回
    error）。若这里回退成 `output_of(None)`，有 output_path 的配置会
    得到 `[]`；更糟的是给 `build_display` 传一个非 None 的 output，
    模板保护失效，summary 变成空占位符。
    """

    projector = _projector({
        "quote_get": _presentation(
            show_output=True,
            summary="共 {output.rowCount} 条",
            props={"output_path": "/bars"},
        ),
    })

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "error": "ToolError: boom",
    })

    event = captured_completed[0]

    assert "output" not in event.data
    assert "summary" not in event.data["display"]
    assert event.status is not None
    assert event.error is not None


async def test_completed_prefers_execution_duration(
        captured_completed,
) -> None:
    """执行器测的耗时优先于 trace 的起止时刻（后者常常不存在）。"""

    projector = _projector({"quote_get": _presentation()})

    await projector._on_tool_stream_event({
        "event": "tool.completed",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "output": {"ok": True},
        "duration_ms": 12.4,
    })

    assert captured_completed[0].duration_ms == 12


async def test_duplicate_called_is_announced_once() -> None:
    projector = _projector({"quote_get": _presentation()})

    frame = {
        "event": "tool.called",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "input": {},
    }

    first = await _emit_called(projector, frame)
    second = await _emit_called(projector, frame)

    assert len(first) == 1
    assert second == []


async def test_announcements_are_shared_across_projector_instances() -> None:
    """
    审批 resume 会新建投影器、重跑到节点开头 —— 已公告集合必须
    跨实例共享，否则同一个 id 会再发一条 tool.called。
    """

    announced: set[str] = set()

    first = _projector({"quote_get": _presentation()}, announced=announced)
    second = _projector({"quote_get": _presentation()}, announced=announced)

    frame = {
        "event": "tool.called",
        "tool_call_id": "c1",
        "tool_name": "quote_get",
        "input": {},
    }

    assert len(await _emit_called(first, frame)) == 1
    assert await _emit_called(second, frame) == []
