"""
痕迹采集**不得改变检索结果** —— 这是这次改动的头号约束。

为什么必须单独焊住
------------------
`RetrieveProcessor.run` 是正在工作的主链路。它现在多了一个可选的
collector，而"纯观察"这件事**没有任何类型系统兜底**：顺手
`sorted()` 之后拿返回值继续往下传、顺手 `set()` 去个重、顺手
`reversed()` —— 每一个都**不报错**，只是让结果变了。

而那种改动的影响面特别坏：评测的数字会与线上不一致，且差异看起来像
"检索质量波动"，不会有人怀疑到一段"只加了个调试开关"的代码上。

所以这里用**真跑**的方式对照：同一个输入、同一套处理器，带 trace 与
不带 trace 各跑一遍，断言 `results` **逐字段相同**。

为什么不调 HTTP
--------------
要验的是 `RetrieveProcessor.run` 这一段（融合→过滤→精排→裁剪）。
走 HTTP 要多扯进 DTO 校验、装配、鉴权，而它们与"collector 会不会改
结果"无关。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.services.document.domain.entities import DocumentTypeDefinition
from app.services.document.domain.schemas.enum import (
    AnalysisMode,
    DefinitionStatus,
    ModalityType,
    VersionMode,
)
from app.services.document.domain.schemas.retrieve import (
    DocumentRetrieveRequest,
)
from app.services.document.retrieve.base import RetrieveCandidate
from app.services.document.retrieve.processor import RetrieveProcessor
from app.services.document.retrieve.trace import RetrievalTraceCollector


def _candidate(
        *,
        score: float,
        route: str,
        doc: UUID | None = None,
) -> RetrieveCandidate:

    return RetrieveCandidate(
        document_chunk_id=uuid4(),
        score=score,
        retrieval_source=f"idx-{route}",
        pipeline_key=route,
        pipeline_keys=[route],
        document_id=doc or uuid4(),
        document_version_id=uuid4(),
        metadata={},
    )


def _definition(*, fusion_top_k: int = 3, rerank: bool = False) -> DocumentTypeDefinition:
    """
    一个最小的 DTD：只配融合的 top_k（也就是必经的那次截断）。

    精排默认**关**（`rerank_strategy=None`）—— 这样精排那一段走的就是
    "executed=False"的跳过分支，正好也把它覆盖到。
    """

    return DocumentTypeDefinition(
        document_type_definition_id=uuid4(),
        tenant_id=None,
        type_key="trace-test",
        name="痕迹对照用",
        description=None,
        modality_type=ModalityType.TEXT,
        analysis_mode=AnalysisMode.DEFAULT,
        revision=1,
        allow_update=True,
        version_mode=VersionMode.VERSIONED,
        fusion_strategy="rrf",
        fusion_config={"top_k": fusion_top_k},
        rerank_strategy="cross_encoder" if rerank else None,
        rerank_config={},
        status=DefinitionStatus.ACTIVE,
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _route_sets() -> list[list[RetrieveCandidate]]:
    """两路召回，分数刻意有交叉。"""

    dense = [
        _candidate(score=1.0, route="dense_vector"),
        _candidate(score=0.8, route="dense_vector"),
        _candidate(score=0.6, route="dense_vector"),
        _candidate(score=0.4, route="dense_vector"),
    ]

    keyword = [
        _candidate(score=0.9, route="keyword"),
        _candidate(score=0.7, route="keyword"),
        _candidate(score=0.5, route="keyword"),
    ]

    return [dense, keyword]


def _snapshot(candidates: list[RetrieveCandidate]) -> list[tuple]:
    """结果的**逐字段**快照（不只是 id 集合、也不只是顺序）。"""

    return [
        (
            candidate.document_chunk_id,
            candidate.score,
            candidate.retrieval_source,
            candidate.pipeline_key,
            tuple(candidate.pipeline_keys),
            candidate.document_id,
            candidate.document_version_id,
        )
        for candidate in candidates
    ]


async def test_trace_does_not_change_results() -> None:
    """
    带 trace 与不带 trace，结果逐字段相同。

    这条是整个改动的**底线**：它红了说明 collector 不再是纯观察者。
    """

    definition = _definition()
    result_sets = _route_sets()

    without = RetrieveProcessor.for_document_type(definition)

    plain = await without.run(query="营收是多少", result_sets=result_sets)

    with_trace = RetrieveProcessor.for_document_type(definition)

    collector = RetrievalTraceCollector(
        target_chunk_ids=[
            c.document_chunk_id
            for c in result_sets[0]
        ]
    )

    traced = await with_trace.run(
        query="营收是多少",
        result_sets=result_sets,
        trace=collector,
    )

    assert _snapshot(plain) == _snapshot(traced), (
        "带 trace 之后结果变了 —— collector 不再是纯观察者"
    )


async def test_trace_records_every_stage() -> None:
    """
    五个边界都记下来了，且 `executed` 分得清"没跑"与"跑了没丢东西"。
    """

    definition = _definition(fusion_top_k=3)
    result_sets = _route_sets()

    collector = RetrievalTraceCollector()

    await RetrieveProcessor.for_document_type(definition).run(
        query="q", result_sets=result_sets, trace=collector,
    )

    stages = {item.stage_key: item for item in collector.trace.stages}

    assert set(stages) == {"fusioner", "filter", "reranker", "cutter"}

    # 融合把 7 条截到 3 条。
    assert stages["fusioner"].input_count == 7
    assert stages["fusioner"].output_count == 3

    # 精排没配 -> executed=False，而不是"跑了没丢"。
    assert stages["reranker"].executed is False
    # 裁剪跑过了。
    assert stages["cutter"].executed is True


async def test_target_chunk_that_never_appears_is_reported_as_not_recalled() -> None:
    """
    一路都没碰到的被盯 chunk 要有 `not_recalled` 一行。

    它不满足任何 `before - after`（两边都没有它），差集算不出来 ——
    但它恰恰是最该看到的情况：目标 chunk 连召回的边都没碰到。
    """

    definition = _definition()
    result_sets = _route_sets()

    ghost = uuid4()

    collector = RetrievalTraceCollector(target_chunk_ids=[ghost])

    await RetrieveProcessor.for_document_type(definition).run(
        query="q", result_sets=result_sets, trace=collector,
    )

    collector.observe_recall(result_sets)

    missed = [
        row
        for stage in collector.trace.stages
        for row in stage.candidates
        if row.chunk_id == ghost
    ]

    assert len(missed) == 1, "只该有一条（不该在每个阶段都补一行）"
    assert missed[0].drop_reason == "not_recalled"
    assert missed[0].is_target is True
    assert missed[0].selected is False


async def test_target_chunk_survives_fusion_but_is_cut_by_cutter() -> None:
    """
    被盯的 chunk 掉在 top-N 之外时**仍然有行**。

    这是"这颗目标 chunk 在哪被丢"能回答的前提：如果只在 top-N 里
    找它，排在后面的目标 chunk 就永远查不到。
    """

    definition = _definition(fusion_top_k=50)  # 融合不截断，让裁剪来丢

    target = _candidate(score=0.05, route="dense_vector")
    # 20 条高分把它压到裁剪的 top_k(10) 之外。
    others = [
        _candidate(score=0.9, route="dense_vector") for _ in range(20)
    ]

    result_sets = [others + [target]]

    collector = RetrievalTraceCollector(
        target_chunk_ids=[target.document_chunk_id],
        top_n=10,
    )

    await RetrieveProcessor.for_document_type(definition).run(
        query="q", result_sets=result_sets, trace=collector,
    )

    cutter = next(
        stage for stage in collector.trace.stages
        if stage.stage_key == "cutter"
    )

    rows = [
        row for row in cutter.candidates
        if row.chunk_id == target.document_chunk_id
    ]

    assert len(rows) == 1, "被盯的 chunk 必须无条件有一行"
    assert rows[0].selected is False
    assert rows[0].drop_reason == "final_top_k_cutoff"
    assert rows[0].rank == 21, "记的是它死时的名次"
    assert (
        len(cutter.candidates) <= 10 + 10 + 1
    ), "top-N ∪ 丢弃 top-N ∪ 被盯的，不该把 21 条全记下来"


async def test_filter_reason_is_per_candidate_not_per_stage() -> None:
    """
    过滤那一段的原因**逐条不同**（四种条件），不是按阶段写死一种。

    这条焊住的是"原因必须由阶段自己回答"：如果改成调用方按 config 猜，
    下面这三个 chunk 会拿到同一个原因。
    """

    definition = _definition(fusion_top_k=50)

    keeper_doc = uuid4()

    low = _candidate(score=0.1, route="dense_vector", doc=keeper_doc)
    other_doc = _candidate(score=0.9, route="dense_vector")
    bad_meta = _candidate(score=0.9, route="dense_vector", doc=keeper_doc)

    result_sets = [[low, other_doc, bad_meta]]

    # 阈值 0.5 + 只保留 keeper_doc + metadata 要求 k=v
    processor = RetrieveProcessor.for_document_type(definition)
    processor.filter.score_threshold = 0.5
    processor.filter.document_ids = [keeper_doc]
    processor.filter.metadata = {"k": "v"}

    collector = RetrievalTraceCollector(
        target_chunk_ids=[
            low.document_chunk_id,
            other_doc.document_chunk_id,
            bad_meta.document_chunk_id,
        ],
    )

    await processor.run(query="q", result_sets=result_sets, trace=collector)

    filter_stage = next(
        stage for stage in collector.trace.stages
        if stage.stage_key == "filter"
    )

    reasons = {
        row.chunk_id: row.drop_reason
        for row in filter_stage.candidates
        if not row.selected
    }

    assert reasons[low.document_chunk_id] == "score_threshold"
    assert reasons[other_doc.document_chunk_id] == "ownership_filter"
    assert reasons[bad_meta.document_chunk_id] == "metadata_filter"


@pytest.mark.parametrize("top_n", [1, 5, 10])
async def test_top_n_does_not_change_results(top_n: int) -> None:
    """
    换 top-N 只影响记多少行，不影响结果。

    （`top_n` 是"记几条"，不是"留几条" —— 这条把两者焊开。）
    """

    definition = _definition()
    result_sets = _route_sets()

    plain = await RetrieveProcessor.for_document_type(definition).run(
        query="q", result_sets=result_sets,
    )

    collector = RetrievalTraceCollector(top_n=top_n)

    traced = await RetrieveProcessor.for_document_type(definition).run(
        query="q", result_sets=result_sets, trace=collector,
    )

    assert _snapshot(plain) == _snapshot(traced)


# ============================================================
# 请求层的两条约束
# ============================================================
#
# 都是"拒绝而不是静默降级"。放在这里而不是路由用例里：它们是**契约**上
# 的约束（pydantic 校验），与 HTTP 那一层无关。


def test_target_ids_without_the_switch_are_rejected() -> None:
    """
    给了 `target_chunk_ids` 却没开 `include_stage_trace` -> 拒绝。

    静默打开的话，调用方会以为自己显式控制着采集开关，而其实有一个
    参数根本没被读过。
    """

    with pytest.raises(ValueError, match="include_stage_trace"):

        DocumentRetrieveRequest(
            document_type_definition_ids=[uuid4()],
            input_text="q",
            target_chunk_ids=[uuid4()],
        )


def test_multi_dtd_with_trace_is_rejected() -> None:
    """
    多 DTD + 开 trace -> 拒绝。

    多 DTD 会把 `RetrieveProcessor` 跑 **N+1 次**（每个 DTD 各一次，
    再过公共链），于是 fusioner / filter / reranker / cutter 各出现
    N+1 遍、`stage_key` 撞在一起，而"这一行的融合是哪一次融合"没有答案。

    拒绝而不是产出一堆分不清来源的行 —— 评测侧也不需要它（一条 case
    只打一个 DTD）。
    """

    with pytest.raises(ValueError, match="不支持同时检索多个"):

        DocumentRetrieveRequest(
            document_type_definition_ids=[uuid4(), uuid4()],
            input_text="q",
            include_stage_trace=True,
        )


def test_multi_dtd_without_trace_still_works() -> None:
    """
    多 DTD **不开** trace 照旧 —— 这条约束不该影响既有调用方。
    """

    request = DocumentRetrieveRequest(
        document_type_definition_ids=[uuid4(), uuid4()],
        input_text="q",
    )

    assert request.include_stage_trace is False
    assert len(request.document_type_definition_ids) == 2
