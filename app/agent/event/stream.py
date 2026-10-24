"""往图的 custom stream 写一条信封（工具 / 任务 / 子代理共用）。

节点**不直接** `publish_run_event`：那会让同一次运行出现两个事件来源
（节点直发 + projector 投影），id / 顺序都会漂。统一写 custom stream，
由 `RunEventProjector` 按 `kind` 投影成 `RunEventType`。

图外（单测 / 直接调节点函数）没有 writer —— 静默跳过。事件是旁路：
投递失败不上抛、不影响业务。
"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

__all__ = ["write_custom"]


def write_custom(envelope: dict[str, Any]) -> None:
    """写一条 custom 流事件；writer 不可用 / 写失败时静默返回。"""

    try:
        writer = get_stream_writer()

    except Exception:  # noqa: BLE001 - 图外运行没有 writer
        return

    try:
        writer(envelope)

    except Exception:  # noqa: BLE001 - 观测旁路，不影响业务
        return
