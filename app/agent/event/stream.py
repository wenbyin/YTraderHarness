"""往图的 custom stream 写一条信封（工具 / 任务 / 子代理 / trace 共用）。

节点**不直接** `publish_run_event`：那会让同一次运行出现两个事件来源
（节点直发 + projector 投影），id / 顺序都会漂。统一写 custom stream，
由 `RunEventProjector` 按 `kind` 投影成规范事件。

⚠️ writer 的获取只有这一处
-------------------------
langgraph 的默认 writer 是**不抛异常的空函数**：`custom` 模式没被请求、
或节点 Runtime 没带上 writer 时，它什么都不做。只判"拿没拿到"会把 no-op
当成可用 writer —— 症状是「工具/任务明明发生了，流里一条没有」，而且
日志干净。所以这里统一做三件事：

    1. 优先用调用方显式传入的 writer（节点 Runtime 的那个）；
    2. 回落 `get_stream_writer()`；
    3. 识别 no-op 并**每进程告警一次**。

写失败也只记日志：事件是旁路，绝不把一次运行拖成失败。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer

__all__ = [
    "is_noop_stream_writer",
    "resolve_stream_writer",
    "write_custom",
]

logger = logging.getLogger(__name__)

#: no-op writer 只告警一次 —— 每条事件各打一条会把日志淹掉。
_noop_writer_warned = False


def is_noop_stream_writer(writer: Any) -> bool:
    """langgraph 的默认 writer 是个空函数：不抛，只是什么都不做。"""

    return getattr(writer, "__name__", "") == "_no_op_stream_writer"


def resolve_stream_writer(
        writer: Any = None,
) -> Callable[[Any], None] | None:
    """
    定下这次用哪个 writer；没有可用的返回 None。

    顺序：调用方给的 -> `get_stream_writer()` 的 -> 都没有。

    ⚠️ no-op 也算"没有"
    ------------------
    只判"有没有拿到"会把默认的空函数当成可用 writer，于是事件静默消失。
    """

    global _noop_writer_warned

    if writer is not None and not is_noop_stream_writer(writer):
        return writer

    try:
        fallback = get_stream_writer()
    except Exception:  # noqa: BLE001 - 图外运行，没有订阅端是正常的
        return None

    if not is_noop_stream_writer(fallback):
        return fallback

    if not _noop_writer_warned:
        _noop_writer_warned = True
        logger.warning(
            "custom 事件的 stream writer 是 no-op —— `custom` 模式没有被"
            "请求（或节点 Runtime 没带上 writer）。本进程内 custom 事件"
            "都不会进流。"
        )

    return None


def write_custom(
        envelope: dict[str, Any],
        *,
        writer: Any = None,
) -> bool:
    """
    写一条 custom 流事件；不可用 / 写失败时记日志并返回 False。

    返回 bool 只给调用方做 debug 日志用 —— 业务路径不需要处理失败。
    """

    resolved = resolve_stream_writer(writer)

    if resolved is None:
        logger.debug(
            "没有可用的 stream writer，custom 事件未发出"
            "（kind=%s, event=%s）。",
            envelope.get("kind"),
            envelope.get("event"),
        )
        return False

    try:
        resolved(envelope)

    except Exception:  # noqa: BLE001 - 观测旁路，不影响业务
        logger.warning(
            "custom 事件写入失败（kind=%s, event=%s）。",
            envelope.get("kind"),
            envelope.get("event"),
            exc_info=True,
        )
        return False

    return True
