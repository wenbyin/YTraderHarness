"""
裁剪实现：最终返回条数与正文长度。

规格的 CutterConfig 没有 type 字段，因此这里只有一个实现。
"""

from __future__ import annotations

from ..domain.schemas import CutterConfig
from .base import Cutter, RetrieveCandidate

__all__ = ["TopKCutter"]


class TopKCutter(Cutter):
    """
    取前 top_k 条，并按 max_content_chars 截断正文。

    入参已按分数降序（融合/精排保证），这里**不重排** ——
    多一次排序会掩盖上游顺序不对的问题。

    正文截断只改 content 的副本，不改数据库里的原文：
    max_content_chars 是"这次响应别太长"的传输优化，
    不是对 Chunk 本身的重写。
    """

    async def cut(
            self,
            *,
            candidates: list[RetrieveCandidate],
            config: CutterConfig,
    ) -> list[RetrieveCandidate]:

        limit = config.top_k
        max_chars = config.max_content_chars

        selected = list(candidates[:limit])

        if max_chars is None:
            return selected

        return [
            candidate.model_copy(
                update={
                    "content": _truncate(
                        candidate.content, max_chars
                    )
                }
            )
            for candidate in selected
        ]


def _truncate(
        content: str | None,
        max_chars: int,
) -> str | None:
    """None 原样返回 —— 未回表成功的候选不该在这里变成空串。"""

    if content is None:
        return None

    if len(content) <= max_chars:
        return content

    return content[:max_chars]
