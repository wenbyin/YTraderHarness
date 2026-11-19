"""
运行事件的 **Redis Stream 具体实现**。

契约（事件名、事件形状、发布端接口）在 `app/common/event`，这里放
"怎么落地"：

    config.py      读 event_stream.yaml（含 v2 的 sinks 段）
    sink.py        一条流的写入运行时：有界队列 + 分片写线程 + 丢弃策略
    router.py      放行契约：哪些事件进 chat / trace，键怎么算
    routing.py     两个 sink + EventRouter（唯一入口 emit）
    publisher.py   进程级单例 + EventSink 缝 + emit_event
    reader.py      共享词汇：StreamEntry / decode / is_terminal
    run_stream.py  读：RunStreamRegistry（**一个 run 一条连接**，
                   服务该 run 的全部订阅者）+ 探活关流
    sse.py         编码：记录 -> SSE 帧

生产者（谁在什么时候发）不在这里 —— 那是各业务自己的事：

    run.*        app/services/session/application/agent_run_dispatch.py
    node.* / message.*   app/agent/event/run_event_projector.py

调用方从**包**导入：

    from app.agent.event import emit_event                     ✓
    from app.common.event.spec import Event, EventType         ✓
"""

__all__ = [
    "HEARTBEAT",
    "EventRouter",
    "EventSink",
    "NullEventPublisher",
    "RedisStreamSink",
    "RunLiveness",
    "RunLivenessProbe",
    "RunStreamAttachment",
    "RunStreamRegistry",
    "StreamClosed",
    "StreamConfig",
    "SinkPolicy",
    "StreamEntry",
    "StreamFailed",
    "TooManyRunStreamsError",
    "build_event_publisher",
    "build_run_stream_registry",
    "chat_policy",
    "close_event_publisher",
    "emit_event",
    "get_event_publisher",
    "get_run_stream_registry",
    "is_terminal",
    "load_stream_config",
    "reset_event_publisher",
    "reset_run_stream_registry",
    "sse_frame",
    "start_event_publisher",
    "trace_policy",
]

from .config import StreamConfig, load_stream_config
from .publisher import (
    EventSink,
    NullEventPublisher,
    RedisStreamSink,
    build_event_publisher,
    close_event_publisher,
    emit_event,
    get_event_publisher,
    reset_event_publisher,
    start_event_publisher,
)
from .reader import StreamEntry, is_terminal
from .router import (
    EventRouter,
    SinkPolicy,
    chat_policy,
    trace_policy,
)
from .run_stream import (
    RunLiveness,
    RunLivenessProbe,
    RunStreamAttachment,
    RunStreamRegistry,
    StreamClosed,
    StreamFailed,
    TooManyRunStreamsError,
    build_run_stream_registry,
    get_run_stream_registry,
    reset_run_stream_registry,
)
from .sse import HEARTBEAT, sse_frame

# 事件名与事件形状是**契约**，从 app.common.event 来。这里再导出一次
# 是为了让调用方（节点、派发 handler、路由）一个 import 就够 ——
# 它们要的是"发一条 message.delta"这件事，不该关心它被分了几层。
from app.common.event import stream_key
