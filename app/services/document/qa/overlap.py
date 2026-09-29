"""
答案区间 ∩ chunk 区间 —— 定出「哪些 chunk 是这道题的目标」。

坐标系
------
`document_chunk.start_index/end_index` 与 clean 正文是**同一个字符
空间**：切分节点读的就是 `clean_ref` 的正文（见
`pipeline/nodes/chunking.py`），而本模块的答案区间也是相对同一份
正文算出来的。所以直接比大小即可，不需要任何换算。

这正是这个工具能成立的前提 —— 一旦哪天切分的输入不再是 clean 正文
（比如中间插了一步改写），这里的求交会**静默错位**：区间照样能比，
只是比的是两个不同坐标系里的位置，结果毫无意义且不报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .locate import Span

__all__ = [
    "ChunkRange",
    "ChunkOverlap",
    "chunks_overlapping",
]


@dataclass(frozen=True)
class ChunkRange:
    """一个 chunk 在正文中的区间。start/end 可为 None（未记录偏移）。"""

    chunk_id: UUID
    chunk_index: int
    start: int | None
    end: int | None


@dataclass(frozen=True)
class ChunkOverlap:
    """
    一道题与一个 chunk 的重叠情况。

    overlap_chars 让调用方看得出"这道题是整条落在这个 chunk 里，还是
    被切成了两半"。答案跨 chunk 时两个 chunk 都会返回，但各自只覆盖
    一部分 —— 评估时这两种情况的含义不同。
    """

    chunk_id: UUID
    chunk_index: int
    overlap_chars: int


def chunks_overlapping(
        span: Span,
        chunks: list[ChunkRange],
) -> list[ChunkOverlap]:
    """
    返回与 span 有重叠的 chunk，按重叠字符数降序。

    重叠判定用**左闭右开**（`chunk.start < span.end and chunk.end > span.start`）。
    因此"chunk 恰好结束在答案开始处"不算重叠 —— 那种情况下这个 chunk
    里根本没有答案的任何字符。

    `start_index`/`end_index` 为 None 的 chunk **跳过**：那是没记录偏移
    的（比如非文本模态、或切分器不产出偏移），无法判断，猜一个比跳过
    更糟。
    """

    overlaps: list[ChunkOverlap] = []

    for chunk in chunks:

        if chunk.start is None or chunk.end is None:
            continue

        left = max(span.start, chunk.start)
        right = min(span.end, chunk.end)

        if left >= right:
            continue

        overlaps.append(
            ChunkOverlap(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                overlap_chars=right - left,
            )
        )

    overlaps.sort(
        key=lambda item: (-item.overlap_chars, item.chunk_index)
    )

    return overlaps
