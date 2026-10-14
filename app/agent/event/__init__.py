"""
运行事件的 **Redis Stream 具体实现**。

契约（事件名、事件形状、发布端接口）在 `app/common/event`，这里放
"怎么落地"：

    config.py      读 event_stream.yaml
    publisher.py   写：RedisEventPublisher（写线程 + 有界队列）/
                   NullEventPublisher + 单例 + EventSink 缝
    reader.py      共享词汇：StreamEntry / decode / is_terminal
    run_stream.py  读：RunStreamRegistry（**一个 run 一条连接**，
                   服务该 run 的全部订阅者）+ 探活关流
    sse.py         编码：记录 -> SSE 帧

生产者（谁在什么时候发）不在这里 —— 那是各业务自己的事：

    run.*        app/services/session/application/agent_run_dispatch.py
    node.* / message.*   app/agent/graph/react/agent_node.py

调用方从**包**导入：

    from app.agent.event import RunEventType, publish_run_event   ✓
"""

__all__ = [
    "HEARTBEAT",
    "EventSink",
    "NullEventPublisher",
    "RedisEventPublisher",
    "RedisStreamSink",
    "RunLiveness",
    "RunLivenessProbe",
    "RunStreamAttachment",
    "RunStreamRegistry",
    "StreamClosed",
    "StreamConfig",
    "StreamEntry",
    "StreamFailed",
    "TooManyRunStreamsError",
    "build_event_publisher",
    "build_run_stream_registry",
    "close_event_publisher",
    "get_event_publisher",
    "get_run_stream_registry",
    "is_terminal",
    "load_stream_config",
    "publish_run_event",
    "reset_event_publisher",
    "reset_run_stream_registry",
    "sse_frame",
    "start_event_publisher",
]

from .config import StreamConfig, load_stream_config
from .publisher import (
    EventSink,
    NullEventPublisher,
    RedisEventPublisher,
    RedisStreamSink,
    build_event_publisher,
    close_event_publisher,
    get_event_publisher,
    publish_run_event,
    reset_event_publisher,
    start_event_publisher,
)
from .reader import StreamEntry, is_terminal
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
from app.common.event import RunEvent, RunEventType, stream_key
