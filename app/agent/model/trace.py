"""
ModelRuntime 的**每次调用** Trace 上下文。

为什么是参数
------------
`ModelRuntime` 按 `(profile, agent_type, model_role)` 缓存、被并发共享。
任何"这次调用的东西"（span 句柄、父 span）都不能挂在实例或共享
observer 上 —— 否则两次并发调用会串线。因此与 `on_delta` 一样，它是一次
调用显式传进来的参数。

谁构造它
--------
LangGraph 节点（或节点内的 Context 阶段）：
    writer = get_stream_writer()                  # 节点侧
    emitter = TraceEmitter(trace_id=..., writer=writer)
    ModelTrace(emitter=emitter, parent_span_id=<node/stage span id>)
"""

from __future__ import annotations

from dataclasses import dataclass

from app.common.event import TraceEmitter

__all__ = ["ModelTrace"]


@dataclass(frozen=True, slots=True)
class ModelTrace:
    """一次模型调用的 trace 上下文。"""

    #: 往 custom stream 写事件的发射器。trace 未开启时它是 disabled 的。
    emitter: TraceEmitter

    #: 这次调用的父 Span（发起它的节点或 Context 阶段）。
    parent_span_id: str
