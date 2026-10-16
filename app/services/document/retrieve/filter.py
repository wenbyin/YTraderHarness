"""
过滤实现：业务条件过滤（分数阈值 / 归属 / metadata）。

规格的 FilterConfig 没有 type 字段，因此这里只有一个实现。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

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
            dropped: dict[UUID, str] | None = None,
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

            # ⚠️ 判定与解释是**同一次调用**。`_drop_reason` 是唯一的
            # 决策点，而"过不过"就是"有没有原因" —— 分成两个函数的话，
            # 两边迟早会对不上，而那种不一致的症状是 drop_reason 指向
            # 一个其实没生效的条件，且不报错。
            reason = _drop_reason(
                candidate,
                config,
                document_ids=document_ids,
                version_ids=version_ids,
            )

            if reason is None:
                kept.append(candidate)
                continue

            if dropped is not None:
                dropped[candidate.document_chunk_id] = reason

        return kept


def _drop_reason(
        candidate: RetrieveCandidate,
        config: FilterConfig,
        *,
        document_ids: frozenset | None,
        version_ids: frozenset | None,
) -> str | None:
    """
    这个候选为什么被淘汰；通过则返回 None。

    **四种条件里第一个不满足的那个**就是原因，顺序固定：

        score_threshold -> ownership_filter -> metadata_filter

    顺序固定是为了让"同时不满足两条"时答案确定 —— 否则同一份输入可能
    报出不同的原因，宏观统计就没法按它分组了。
    """

    threshold = config.score_threshold

    if (
            threshold is not None
            and candidate.score < threshold
    ):
        return "score_threshold"

    # 归属过滤：document_ids 与 document_version_ids 归到同一个原因。
    # 受控词表里没有单独的 document_ids / document_version_ids 两档 ——
    # 它们是同一件事（"这块不在我要的范围里"），拆开只会让宏观统计里
    # 出现两个各自样本很少的桶。
    if (
            document_ids is not None
            and candidate.document_id not in document_ids
    ):
        return "ownership_filter"

    if (
            version_ids is not None
            and candidate.document_version_id
            not in version_ids
    ):
        return "ownership_filter"

    if config.metadata and not _matches(
            candidate.metadata, config.metadata
    ):
        return "metadata_filter"

    return None


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
