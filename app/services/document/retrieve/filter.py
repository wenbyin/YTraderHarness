"""
过滤实现：业务条件过滤（分数阈值 / 归属 / metadata）。

规格的 FilterConfig 没有 type 字段，因此这里只有一个实现。
"""

from __future__ import annotations

from typing import Any

from ..domain.schemas import FilterConfig
from .base import RetrieveCandidate, ResultFilter

__all__ = ["BasicResultFilter"]


class BasicResultFilter(ResultFilter):
    """
    四类条件的**与**关系，任一不满足即淘汰：

        score_threshold      归一化分数低于阈值
        document_ids         不属于指定 Document
        document_version_ids 不属于指定 DocumentVersion
        metadata             任一键值不精确相等

    空/None 的条件表示"不按它过滤"，而不是"全滤掉"。

    metadata 用精确相等而不是子串/包含：规格写的是
    "Exact matching semantics"，且这里面对的是
    服务端自己写进去的结构化字段（company_code 之类），
    模糊匹配会把语义问题藏到运行期。
    """

    async def apply(
            self,
            *,
            candidates: list[RetrieveCandidate],
            config: FilterConfig,
    ) -> list[RetrieveCandidate]:

        # 集合在循环外建一次：候选数按各路 top_k 算，可能几百条，
        # 在循环里 set() 是白搭的重复开销。
        document_ids = (
            frozenset(config.document_ids)
            if config.document_ids
            else None
        )

        version_ids = (
            frozenset(config.document_version_ids)
            if config.document_version_ids
            else None
        )

        kept: list[RetrieveCandidate] = []

        for candidate in candidates:

            if not _passes(
                    candidate,
                    config,
                    document_ids=document_ids,
                    version_ids=version_ids,
            ):
                continue

            kept.append(candidate)

        return kept


def _passes(
        candidate: RetrieveCandidate,
        config: FilterConfig,
        *,
        document_ids: frozenset | None,
        version_ids: frozenset | None,
) -> bool:

    threshold = config.score_threshold

    if (
            threshold is not None
            and candidate.score < threshold
    ):
        return False

    if (
            document_ids is not None
            and candidate.document_id not in document_ids
    ):
        return False

    if (
            version_ids is not None
            and candidate.document_version_id
            not in version_ids
    ):
        return False

    if config.metadata and not _matches(
            candidate.metadata, config.metadata
    ):
        return False

    return True


def _matches(
        candidate_metadata: dict[str, Any],
        conditions: dict[str, Any],
) -> bool:
    """
    conditions 里的每一项都要在候选 metadata 里精确命中。

    候选缺该键即视为不匹配 —— filter 的语义是"只保留满足条件的"，
    缺键不能当成通过。
    """

    for key, expected in conditions.items():

        if key not in candidate_metadata:
            return False

        if candidate_metadata[key] != expected:
            return False

    return True
