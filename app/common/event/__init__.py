"""
事件的基础契约 —— **纯抽象，零实现**。

这里只放两样东西：

    event.py      事件的**词汇**：RunEventType（运行 / 消息 / 工具 / 任务）、
                  RunEvent（一条事件）、stream_key（键怎么拼），
                  以及两个集合：TERMINAL_EVENT_TYPES（运行终止）
                  与 STREAM_CLOSING_EVENT_TYPES（订阅该收尾，含暂停）
    publisher.py  发布端的**抽象接口**（EventPublisher 协议）

具体实现不在这里 —— 见 `app/agent/event`（Redis Stream 的发布端、
读取端与 SSE 帧）。这样切分的好处是 `app.common` 保持**纯叶子**：
它不认识 Redis、也不认识 agent，谁要用事件只需要依赖这一层契约。

调用方从**包**导入，不钻到叶子模块：

    from app.common.event import RunEvent, RunEventType   ✓
    from app.common.event.event import RunEvent           ✗
"""

__all__ = [
    "EventPublisher",
    "EventType",
    "RunEvent",
    "RunEventType",
    "STREAM_CLOSING_EVENT_TYPES",
    "SUBAGENT_STREAM_KIND",
    "TASK_STREAM_KIND",
    "TOOL_STREAM_KIND",
    "SpanKind",
    "SpanLink",
    "SpanStatus",
    "TERMINAL_EVENT_TYPES",
    "TraceEmitter",
    "TraceError",
    "TraceEvent",
    "TraceSpan",
    "stream_key",
    "trace_stream_key",
]

from .event import (
    STREAM_CLOSING_EVENT_TYPES,
    SUBAGENT_STREAM_KIND,
    TASK_STREAM_KIND,
    TERMINAL_EVENT_TYPES,
    TOOL_STREAM_KIND,
    RunEvent,
    RunEventType,
    stream_key,
)
from .publisher import EventPublisher
from .trace import (
    EventType,
    SpanKind,
    SpanLink,
    SpanStatus,
    TraceError,
    TraceEvent,
    trace_stream_key,
)
from .trace_emitter import TraceEmitter, TraceSpan
