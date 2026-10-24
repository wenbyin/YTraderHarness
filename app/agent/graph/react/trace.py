"""
节点侧的 Trace 构造小工具。

提示词第 5 节划的两条线，这里落实为：

    LangGraph 节点      用 `get_stream_writer()` 拿 writer
    自有 runtime        由节点把同一个 writer 显式传下去
                        （ContextBuildRequest.trace / ModelTrace）

因此本模块只做一件事：从当前节点 Runtime 里取出 trace_id / run_id /
会话 id / 节点 span_id，配一个 emitter。没有 trace_id 时 emitter 是
disabled 的，所有 emit 直接返回 None —— 不产生任何事件。
"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

from app.common.event import TraceEmitter
from app.infra.graph import execution_ids

__all__ = ["node_trace"]


def node_trace(runtime: Any) -> tuple[TraceEmitter, str]:
    """
    构造本次节点执行的 emitter 与节点 span_id。

    节点 span_id 用 langgraph 的 `task_id`：它与投影层 `tasks.data.id`
    **是同一个值**（见 infra/graph/instrumentation.py 的 execution_ids），
    于是节点侧发出的 model/context 事件能挂到投影层发出的 node.* 上。

    取不到 task_id 时回落到 run_id（仍然有父子层次，只是退化到根）。
    """

    context = getattr(runtime, "context", None)

    trace_id = getattr(context, "trace_id", None)
    run_id = getattr(context, "pipeline_run_id", None)
    session_id = getattr(context, "session_id", None)

    ids = execution_ids(runtime)

    span_id = ids.task_id or (
        str(run_id) if run_id is not None else "unknown"
    )

    emitter = TraceEmitter(
        trace_id=trace_id,
        run_id=str(run_id) if run_id is not None else "",
        thread_id=None if session_id is None else str(session_id),
        writer=get_stream_writer(),
    )

    return emitter, span_id
