"""
Trace 链路的核心断言：协议事件 + custom stream -> TraceEvent。

这里跑**真的** langgraph v3 通道（同 test_run_event_projector.py 的立场：
metadata / custom 模式都只在真运行里才成立）。只把 Redis 发布器换成
记录器，验证：

    node.*        由 tasks 投影
    model.*       由节点写入 custom stream、投影层二次校验后转发
    trace_id 为空 时一条 trace 事件都不发（custom 模式仍请求：工具事件走它）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from fakes import make_definition, make_registry

from app.agent.model import (
    ModelProfile,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
    ModelTrace,
)
from app.agent.model.adapters import LangChainProviderAdapter

from app.agent.event import run_event_projector as projector_module
from app.agent.event.run_event_projector import (
    RunEventProjector,
    projector_factory,
)
from app.common.event import (
    EventType,
    SpanKind,
    SpanStatus,
    TraceEmitter,
    TraceEvent,
)
from app.infra.graph import execution_ids


class _State(TypedDict, total=False):
    messages: Annotated[list, add_messages]


@dataclass(frozen=True, slots=True)
class _Ctx:
    pipeline_run_id: str
    session_id: str = "session-1"
    trace_id: str | None = None


def _node(state: _State, runtime: Any) -> dict[str, Any]:
    writer = get_stream_writer()

    emitter = TraceEmitter(
        trace_id=runtime.context.trace_id,
        run_id=runtime.context.pipeline_run_id,
        thread_id=runtime.context.session_id,
        writer=writer,
    )

    span = emitter.start(
        EventType.MODEL_STARTED,
        span_kind=SpanKind.MODEL,
        span_name="model.invoke",
        end_type=EventType.MODEL_COMPLETED,
        failed_type=EventType.MODEL_FAILED,
        parent_span_id=execution_ids(runtime).task_id,
        payload={
            "model_role": "reasoning",
            "provider": "fake",
            "model_name": "fake-model",
            "streaming": True,
        },
    )

    if span is not None:
        span.complete(
            payload={
                "input_tokens": 1,
                "output_tokens": 2,
                "finish_reason": "stop",
                "response_id": "resp-1",
            }
        )

    return {}


def _graph():
    builder = StateGraph(_State)
    builder.add_node("agent", _node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()


# ============================================================
# ModelRuntime -> custom stream -> 投影层
# ============================================================


@dataclass(frozen=True, slots=True)
class _ModelCtx:
    pipeline_run_id: str
    model_registry: Any
    session_id: str = "session-1"
    trace_id: str | None = None


def _model_registry() -> Any:
    adapter = LangChainProviderAdapter(
        chat_model_factory=lambda **_: GenericFakeChatModel(
            messages=iter([AIMessage(content="TOKENS")] * 4)
        )
    )

    return make_registry(
        models=[make_definition("main")],
        profiles=[ModelProfile(id="main-p", primary="fake:main")],
        bindings=[
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.REASONING,
                profile="main-p",
            ),
        ],
        adapter=adapter,
    )


def _model_graph() -> Any:
    async def _model_node(state: _State, runtime: Any) -> dict[str, Any]:
        emitter = TraceEmitter(
            trace_id=runtime.context.trace_id,
            run_id=runtime.context.pipeline_run_id,
            thread_id=runtime.context.session_id,
            writer=get_stream_writer(),
        )

        model_runtime = runtime.context.model_registry.get_runtime(
            agent_type="react",
            model_role=ModelRole.REASONING,
        )

        response = await model_runtime.ainvoke(
            ModelRequest(messages=(HumanMessage("hi"),)),
            trace=ModelTrace(
                emitter=emitter,
                parent_span_id=execution_ids(runtime).task_id,
            ),
        )

        return {"messages": [response.message]}

    builder = StateGraph(_State)
    builder.add_node("agent", _model_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()


async def _run_model_graph() -> list[TraceEvent]:
    recorded: list[TraceEvent] = []

    async def fake_publish(self: Any, event: TraceEvent) -> bool:
        recorded.append(event)
        return True

    original = projector_module.RedisTracePublisher.publish
    projector_module.RedisTracePublisher.publish = fake_publish

    try:
        run_stream = await _model_graph().astream_events(
            {"messages": []},
            {"configurable": {"thread_id": "t"}},
            version="v3",
            context=_ModelCtx(
                pipeline_run_id="run-1",
                model_registry=_model_registry(),
                trace_id="trace-1",
            ),
            transformers=[
                projector_factory(
                    run_id="run-1",
                    message_nodes={"agent"},
                    message_roles={ModelRole.REASONING},
                    trace_id="trace-1",
                    session_id="session-1",
                ),
            ],
        )

        async with run_stream:
            await run_stream.output()

    finally:
        projector_module.RedisTracePublisher.publish = original

    return recorded


@pytest.mark.asyncio
async def test_model_runtime_events_and_first_token() -> None:
    events = await _run_model_graph()
    types = [event.event_type for event in events]

    assert EventType.MODEL_STARTED in types
    assert EventType.MODEL_COMPLETED in types
    assert EventType.MODEL_FIRST_TOKEN in types

    started = next(
        event for event in events
        if event.event_type is EventType.MODEL_STARTED
    )
    completed = next(
        event for event in events
        if event.event_type is EventType.MODEL_COMPLETED
    )
    first_token = next(
        event for event in events
        if event.event_type is EventType.MODEL_FIRST_TOKEN
    )

    # 同一个 attempt：start / first_token / complete 共用一个 span_id。
    assert completed.span_id == started.span_id
    assert first_token.span_id == started.span_id
    assert first_token.parent_span_id == started.parent_span_id
    assert first_token.payload["ttft_ms"] >= 0


async def _run(trace_id: str | None) -> list[TraceEvent]:
    recorded: list[TraceEvent] = []

    async def fake_publish(self: Any, event: TraceEvent) -> bool:
        recorded.append(event)
        return True

    original = projector_module.RedisTracePublisher.publish
    projector_module.RedisTracePublisher.publish = fake_publish

    try:
        run_stream = await _graph().astream_events(
            {"messages": []},
            {"configurable": {"thread_id": "t"}},
            version="v3",
            context=_Ctx(
                pipeline_run_id="run-1",
                trace_id=trace_id,
            ),
            transformers=[
                projector_factory(
                    run_id="run-1",
                    message_nodes=set(),
                    trace_id=trace_id,
                    session_id="session-1",
                ),
            ],
        )

        async with run_stream:
            await run_stream.output()

    finally:
        projector_module.RedisTracePublisher.publish = original

    return recorded


@pytest.mark.asyncio
async def test_trace_events_from_tasks_and_custom_stream() -> None:
    events = await _run("trace-1")

    types = [event.event_type for event in events]

    assert EventType.NODE_STARTED in types
    assert EventType.NODE_COMPLETED in types
    assert EventType.MODEL_STARTED in types
    assert EventType.MODEL_COMPLETED in types

    for event in events:
        assert event.trace_id == "trace-1"
        assert event.run_id == "run-1"
        assert event.thread_id == "session-1"
        assert event.span_id

    node_start = next(
        event for event in events
        if event.event_type is EventType.NODE_STARTED
    )
    model_start = next(
        event for event in events
        if event.event_type is EventType.MODEL_STARTED
    )

    # 节点 span_id = task_id；模型 span 的父就是它。
    assert model_start.parent_span_id == node_start.span_id

    model_end = next(
        event for event in events
        if event.event_type is EventType.MODEL_COMPLETED
    )
    assert model_end.span_id == model_start.span_id
    assert model_end.payload["output_tokens"] == 2


@pytest.mark.asyncio
async def test_no_trace_id_emits_nothing() -> None:
    events = await _run(None)

    assert events == []


def test_trace_mode_is_instance_scoped() -> None:
    plain = RunEventProjector(scope=(), run_id="run-1")
    traced = RunEventProjector(
        scope=(),
        run_id="run-1",
        trace_id="trace-1",
    )

    # `custom` 始终要：工具节点把 tool.called / tool.completed 写进
    # 自定义流，由投影层统一发给前端（与 trace 的 fanout.* 同一条路）。
    assert plain.required_stream_modes == ("custom",)
    # `tools` 只给 trace 的工具 span 用，不采 trace 时不必多请求。
    assert traced.required_stream_modes == ("custom", "tools")
