"""
扇出选取规则：这个版本要跑哪些 RetrievalDefinition。

纯函数，零 LangGraph 依赖，因此可以脱离图单独测试 ——
这条规则很容易写错且后果不直观（少跑一路 = 某个索引里
静默地搜不到这份文档）。

规则
----
    版本已绑定 DocumentProcessorDefinition
        -> 只跑该 processor 经 DocumentProcessorRetrievalBinding
           关联到的 RetrievalDefinition

    版本没有绑定（走系统默认 Reader）
        -> 该 DocumentTypeDefinition 下**全部** active 的
           RetrievalDefinition 都参与

第二条来自 AnalysisMode.DEFAULT 的语义：
    "使用系统预制的中缀名 Reader。不需要配置
     DocumentProcessorDefinition，该类型下的全部
     RetrievalDefinition 都会参与。"

reprocess 的过滤
----------------
请求里给了 document_processor_definition_ids 时，先用这些
processor 过一遍绑定关系，取并集再与上面的结果求交 ——
即"只重跑这些 processor 映射到的 retrieval"。
空列表表示不过滤。
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

__all__ = [
    "RetrievalCandidate",
    "select_retrievals",
]


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """
    一个参与扇出的 RetrievalDefinition。

    只带选取规则需要的字段，不依赖 domain entity ——
    这样这条规则可以在没有任何数据库/依赖的情况下测试。
    """

    document_retrieval_definition_id: UUID

    retrieval_type: str

    chunker_name: str | None
    chunker_config: dict

    retriever_name: str | None
    retriever_config: dict

    storage_backend: str
    storage_config: dict


def select_retrievals(
        *,
        candidates: list[RetrievalCandidate],
        version_processor_definition_id: UUID | None,
        bound_retrieval_ids_by_processor: dict[
            UUID, list[UUID]
        ],
        restrict_to_processor_ids: list[UUID] | None = None,
) -> list[RetrievalCandidate]:
    """
    选出本次要跑的 retrieval 列表。

    Parameters
    ----------
    candidates
        该 DocumentTypeDefinition 下全部 active 的
        RetrievalDefinition。
    version_processor_definition_id
        版本绑定的前处理定义。None 表示走系统默认 Reader。
    bound_retrieval_ids_by_processor
        DocumentProcessorRetrievalBinding 的配置关系：
        processor definition id -> 它关联的 retrieval definition ids。
    restrict_to_processor_ids
        reprocess 请求里的过滤条件。空/None 表示不过滤。

    Returns
    -------
    去重后、按输入顺序排列的 retrieval 列表。
    """

    # --------------------------------------------------------
    # 1. 按"版本绑定的 processor"决定基础集合
    # --------------------------------------------------------

    if version_processor_definition_id is None:
        # 系统默认 Reader：该类型下全部 retrieval 都参与
        selected = list(candidates)

    else:
        bound = set(
            bound_retrieval_ids_by_processor.get(
                version_processor_definition_id, []
            )
        )

        selected = [
            c for c in candidates
            if c.document_retrieval_definition_id in bound
        ]

    # --------------------------------------------------------
    # 2. reprocess 的收窄过滤
    # --------------------------------------------------------

    if restrict_to_processor_ids:
        allowed: set[UUID] = set()

        for processor_id in restrict_to_processor_ids:
            allowed.update(
                bound_retrieval_ids_by_processor.get(
                    processor_id, []
                )
            )

        selected = [
            c for c in selected
            if c.document_retrieval_definition_id in allowed
        ]

    # --------------------------------------------------------
    # 3. 去重但保序
    # --------------------------------------------------------

    seen: set[UUID] = set()
    deduped: list[RetrievalCandidate] = []

    for candidate in selected:
        key = candidate.document_retrieval_definition_id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return deduped
