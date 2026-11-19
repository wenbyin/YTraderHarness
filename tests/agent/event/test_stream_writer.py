"""custom stream 的 writer 解析：真 writer 优先，no-op 视为不可用。"""

from __future__ import annotations

from app.agent.event.stream import (
    is_noop_stream_writer,
    resolve_stream_writer,
)


def test_noop_writer_is_detected() -> None:
    def _no_op_stream_writer(_: object) -> None:
        return None

    assert is_noop_stream_writer(_no_op_stream_writer)
    assert not is_noop_stream_writer(lambda _: None)


def test_real_writer_wins() -> None:
    def real(_: object) -> None:
        return None

    assert resolve_stream_writer(real) is real


def test_noop_without_graph_falls_back_to_none() -> None:
    def _no_op_stream_writer(_: object) -> None:
        return None

    # 图外 get_stream_writer() 会抛 —— 没有订阅端，返回 None 是正常结果。
    assert resolve_stream_writer(_no_op_stream_writer) is None
