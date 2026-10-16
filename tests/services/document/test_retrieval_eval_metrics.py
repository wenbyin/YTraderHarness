"""
评测执行的**纯函数**：命中判定、漏斗归因、指标分母。

为什么这些必须是纯函数用例
--------------------------
它们不碰库、不碰索引后端，而它们错了的后果**全都不可见**：

    命中判定取错了名次      → 召回率整体偏移，而偏移量与检索无关
    归因规则错了            → 宏观统计指向错误的阶段，于是改错地方
    failed 混进 miss 分母   → "检索挂了"看起来像"召回变差"

三条都不是"崩了"，而是"数字看着合理但指向别的东西"。所以这里逐条焊死，
用构造的输入（毫秒级，不需要 ES/Milvus）。
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.services.document.application.retrieval_eval_evaluation import (
    build_retrieve_request,
    build_stage_rows,
    compute_metrics,
    judge_hit,
    percentile,
)
from app.services.document.domain.entities import (
    RetrievalEvalCaseResult,
)
from app.services.document.domain.schemas.enum import (
    EvalCaseResultStatus,
)
from app.services.document.domain.schemas.retrieve import (
    DocumentRetrieveResponse,
    RetrieveStageTrace,
    RetrieveTrace,
    TracedChunkCandidate,
)


def _response(chunk_ids: list[UUID]) -> DocumentRetrieveResponse:
    """一列结果，名次就是顺序。"""

    from app.services.document.domain.schemas.retrieve import (
        DocumentRetrieveResult,
    )

    return DocumentRetrieveResponse(
        retrieve_id=uuid4(),
        retrieve_scope="single",
        retrieve_processor_mode="dtd",
        document_type_definition_ids=[uuid4()],
        results=[
            DocumentRetrieveResult(
                rank=index + 1,
                score=1.0 - index * 0.1,
                document_chunk_id=chunk_id,
                document_version_id=uuid4(),
                document_id=uuid4(),
                document_type_definition_id=uuid4(),
                content="x",
            )
            for index, chunk_id in enumerate(chunk_ids)
        ],
    )


def _result(
        *,
        status: EvalCaseResultStatus = EvalCaseResultStatus.COMPLETED,
        hit: bool | None = True,
        rank: int | None = 1,
        latency: int | None = 10,
        attempt_no: int = 1,
) -> RetrievalEvalCaseResult:

    return RetrievalEvalCaseResult(
        eval_case_result_id=uuid4(),
        eval_run_id=uuid4(),
        eval_case_id=uuid4(),
        attempt_no=attempt_no,
        gold_hit=hit,
        gold_rank=rank,
        status=status,
        latency_ms=latency,
    )


# ============================================================
# 命中判定
# ============================================================


def test_hit_uses_the_best_gold_rank_not_the_first() -> None:
    """
    多个金标时取**最好**的名次。

    这条不只是"更准"：取第一个碰到的会让 MRR 系统性偏低，而偏低与检索
    质量无关 —— 它只是"返回顺序里哪个金标先出现"的巧合。
    """

    good = uuid4()
    also_good = uuid4()

    # 第 3 名是另一个金标，第 5 名才是"第一个被碰到的"
    response = _response(
        [uuid4(), uuid4(), also_good, uuid4(), good]
    )

    hit, rank = judge_hit(
        response=response,
        gold_chunk_ids={good, also_good},
        max_k=10,
    )

    assert hit is True
    assert rank == 3, "应当取最好的那个名次"


def test_hit_respects_max_k_but_still_reports_the_true_rank() -> None:
    """
    金标排第 11、max_k=10 -> **不算命中，但名次要报出来（11）**。

    ⚠️ 名次**不是** None，这一点是刻意的：`max_k` 只决定"判不判命中"，
    不决定"我们能看多远"。旧写法在 `results[:max_k]` 里找，于是 max_k=10
    时第 11 名直接变 None —— 事后就无法回答"如果看前 20 条会怎样"，只能
    重跑一轮别的 max_k（还要重付全部检索成本）。

    有了真实名次，同一轮数据能读出任意 k 的召回率曲线
    （见 compute_metrics 的 hit_rate_curve）。
    """

    gold = uuid4()

    response = _response([uuid4() for _ in range(10)] + [gold])

    hit, rank = judge_hit(
        response=response, gold_chunk_ids={gold}, max_k=10,
    )

    assert hit is False, "第 11 名在 max_k=10 之外，不算命中"
    assert rank == 11, "但真实名次要报出来，否则曲线读不出来"


def test_rank_is_none_when_gold_never_returned() -> None:
    """金标**根本不在返回集合里** -> 名次才是 None（这是真正的没召回到）。"""

    gold = uuid4()

    hit, rank = judge_hit(
        response=_response([uuid4() for _ in range(10)]),
        gold_chunk_ids={gold},
        max_k=10,
    )

    assert hit is False
    assert rank is None


# ============================================================
# 指标分母
# ============================================================


def test_collapse_by_case_keeps_the_best_attempt() -> None:
    """
    逐轮结果折叠成"每条 case 一行"，取最好的一次。

    ⚠️ 这是**表头指标正确的唯一前提**。不折叠的话，一条"第 1 轮没中、
    第 2 轮命中"的 case 会同时进分母两次、分子一次 —— 逐轮放宽的收益被
    稀释掉，看起来像"扩了两轮没用"。

    实测口径（同一批 40 条反复跑出来的 240 行）：按行 43%、按 case 70%。
    """

    from app.services.document.application.retrieval_eval_evaluation import (
        collapse_by_case,
    )

    case_id = uuid4()

    miss_then_hit = [
        _result(hit=False, rank=None),   # 第 1 轮没中
        _result(hit=True, rank=3),       # 第 2 轮命中
    ]
    for item in miss_then_hit:
        item.eval_case_id = case_id

    # 另一条：三轮都没中
    never_id = uuid4()
    never = [_result(hit=False, rank=None) for _ in range(3)]
    for item in never:
        item.eval_case_id = never_id

    collapsed = collapse_by_case(miss_then_hit + never)

    assert len(collapsed) == 2, "两条 case -> 两行"

    by_case = {item.eval_case_id: item for item in collapsed}

    assert by_case[case_id].gold_hit is True, "取命中那一次"
    assert by_case[case_id].gold_rank == 3

    # 都命中时取名次最小的（**同一个** case 的两行，才会被折叠）
    shared = uuid4()
    both_hit = [_result(hit=True, rank=5), _result(hit=True, rank=2)]
    for item in both_hit:
        item.eval_case_id = shared
    picked = collapse_by_case(both_hit)
    assert len(picked) == 1, "同一个 case 的两行要折成一行"
    assert picked[0].gold_rank == 2, "都命中取名次最小的"

    # 折叠之后算出来的才是"最终找到没"
    metrics = compute_metrics(collapsed, max_k=10)
    assert metrics["total_cases"] == 2
    assert metrics["hit_rate_at_k"] == 0.5


def test_collapse_breaks_ties_by_the_later_round() -> None:
    """
    完全同级（三轮都没中）时必须**定序**，不能看库的返回顺序。

    不定的话，同一条 case 的两行在两次查询里可能给出不同的痕迹页 ——
    乃至不同的 latency 分位（取哪一行的 `latency_ms` 跟着变）。取更晚的
    那轮：逐轮放宽里它就是参数最宽、最接近"最终找不到"的那次。
    """

    from app.services.document.application.retrieval_eval_evaluation import (
        collapse_by_case,
    )

    case_id = uuid4()

    rows = [
        _result(hit=False, rank=None, attempt_no=no) for no in (3, 1, 2)
    ]
    for item in rows:
        item.eval_case_id = case_id

    collapsed = collapse_by_case(rows)

    assert len(collapsed) == 1
    assert collapsed[0].attempt_no == 3, "同级取更晚的一轮"


def test_representative_result_is_the_collapsed_row() -> None:
    """
    单条 case 的代表行与 `collapse_by_case` 必须是**同一行**。

    痕迹页用它、列表与表头用 `collapse_by_case`（每条 case 一行）——
    两处规则若各写一遍，同一次评测在三个页面上会给出三种"这条 case 的
    结果"，而三处单看都说得通。
    """

    from app.services.document.application.retrieval_eval_evaluation import (
        collapse_by_case,
        representative_result,
    )

    case_id = uuid4()

    rows = [
        _result(hit=False, rank=None, attempt_no=1),
        _result(hit=True, rank=7, attempt_no=2),
    ]
    for item in rows:
        item.eval_case_id = case_id

    picked = representative_result(rows)

    assert picked is not None
    assert picked.attempt_no == 2, "命中优先"

    # 逐行比对，而不是只比 attempt_no：哪天两处挑出的是同一轮但不同的
    # 对象（复制了一份行），只有 `is` 看得出来。
    assert picked is collapse_by_case(rows)[0]

    assert representative_result([]) is None, "没有结果行时不该硬造一条"


def test_failed_cases_are_excluded_from_the_denominator() -> None:
    """
    **执行失败的不进分母。**

    它们混进去的话，"检索挂了"会表现为"召回率下降" —— 而那是最容易被
    误判成检索质量问题的一种假象。失败数另有 `failed_cases` 一列。
    """

    results = [
        _result(hit=True, rank=1),                       # 命中
        _result(hit=False, rank=None),                   # 没命中
        _result(
            status=EvalCaseResultStatus.FAILED,
            hit=None, rank=None,
        ),                                               # 挂了 -> 不进分母
    ]

    metrics = compute_metrics(results, max_k=10)

    assert metrics["total_cases"] == 3
    assert metrics["failed_cases"] == 1
    # 分母是 2（判过的），不是 3。
    assert metrics["hit_rate_at_k"] == 0.5
    assert metrics["hit_count"] == 1
    assert metrics["miss_count"] == 1


def test_hit_rate_at_5_is_null_when_max_k_is_smaller() -> None:
    """
    `max_k < 5` 时 `hit_rate_at_5` 是 **NULL，不是 0**。

    0 会被读成"top5 一个都没中"，而 NULL 说的是"这个指标在本次配置下
    没有意义"。两者是完全不同的结论。
    """

    metrics = compute_metrics([_result(hit=True, rank=1)], max_k=3)

    assert metrics["hit_rate_at_5"] is None
    assert metrics["hit_rate_at_1"] == 1.0


def test_metrics_with_no_judged_case_are_null_not_zero() -> None:
    """一条都没判过 -> 比率是 NULL。除以 0 得到 0 会把"没数据"说成"全错"。"""

    metrics = compute_metrics(
        [_result(status=EvalCaseResultStatus.FAILED, hit=None)],
        max_k=10,
    )

    assert metrics["hit_rate_at_k"] is None
    assert metrics["mrr_at_k"] is None
    assert metrics["hit_rate_at_1"] is None


def test_mrr_uses_best_rank_per_case() -> None:
    """MRR = 各自最好名次的倒数的平均。"""

    metrics = compute_metrics(
        [
            _result(hit=True, rank=1),    # 1.0
            _result(hit=True, rank=2),    # 0.5
            _result(hit=False, rank=None),  # 0.0
        ],
        max_k=10,
    )

    assert metrics["mrr_at_k"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_percentile_returns_a_real_observed_value() -> None:
    """
    分位取**观测到的真实值**，不插值。

    插值会给出一个库里从未发生过的延迟，而它的用途只是"看一眼慢到什么
    程度" —— 一个没发生过的数比略保守的真实值更容易误导。
    """

    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0.5) in values
    assert percentile(values, 0.95) in values
    assert percentile([], 0.5) is None


# ============================================================
# 前端覆盖的往返
# ============================================================


def test_override_round_trip_keeps_fields_set() -> None:
    """
    前端覆盖存进库、再读回来执行时，**必须只覆盖它显式传的那些字段**。

    这条守的是 `create_eval_run` 里那一行 `model_dump` 的参数选择：

        exclude_none   ✗ 保留所有**非 None** 的字段 —— 而 pydantic 已经把
                        没传的填成了默认值，于是默认值变成"显式传入"。
                        实测前端只传 {"fusioner": {"rrf_k": 30}} 时，
                        fields_set 会变成 {enabled, rrf_k, top_k, type}
                        —— **DTD 配的 top_k 会被悄悄重置成 50**。
        exclude_unset  ✓ 只存显式传过的

    `_apply_override` 恰恰是按 `model_fields_set` 合并的，所以这个往返
    一旦丢信息，"只改一个字段"就变成"把同段其它字段一起钉死"，而且
    不报错 —— 只是某个 DTD 的配置失效了，没人看得出来。
    """

    from app.services.document.domain.schemas.retrieve import (
        RetrieveProcessorOverride,
    )

    # 前端只调融合的 rrf_k
    given = RetrieveProcessorOverride.model_validate(
        {"fusioner": {"rrf_k": 30}}
    )

    stored = given.model_dump(mode="json", exclude_unset=True)
    replayed = RetrieveProcessorOverride.model_validate(stored)

    assert replayed.fusioner is not None
    assert replayed.fusioner.model_fields_set == {"rrf_k"}, (
        "往返之后 fields_set 变了 —— 覆盖范围被放大了"
    )
    assert replayed.fusioner.rrf_k == 30
    # 没传的那些仍在"继承"状态（值看着是默认值，但不在 fields_set 里，
    # 所以合并时**不会**盖掉 DTD 的配置）。
    assert "top_k" not in replayed.fusioner.model_fields_set

    # 前端完全没提的段保持 None —— `_apply_override(base, None)` 会直接
    # 返回 base，即整段继承。
    assert replayed.reranker is None


def test_override_can_enable_rerank() -> None:
    """
    前端可以用覆盖**把 DTD 没配的精排打开**（这正是那个参数的用途）。

    注意 `enabled` 不传也成立：`_apply_override` 只认显式字段，而未传的
    `enabled` 停在默认 True，所以只需要给出 `type`。
    """

    from app.services.document.domain.schemas.retrieve import (
        RetrieveProcessorOverride,
    )

    given = RetrieveProcessorOverride.model_validate(
        {"reranker": {"type": "cross_encoder", "top_n": 50}}
    )

    stored = given.model_dump(mode="json", exclude_unset=True)

    assert stored == {
        "reranker": {"type": "cross_encoder", "top_n": 50}
    }, "只该存显式传的那两个字段"


# ============================================================
# 逐轮：请求**真的**用了那一轮的参数吗
# ============================================================
#
# 这一组串起「阶梯 -> 请求」那一段，而它此前**从没被任何测试走过**。
# 后果实测过四次（同一类）：DTO 加了、openapi 同步了、测试全绿，而前端
# 传的 `attempts` 被接受、被校验、然后**丢掉** —— 跑的永远是内置档
# （第 3 档带精排），报的表现是「我没传 rerank，第三轮却在跑精排」。
#
# 所以断言必须落在**构造出来的那个请求**上，而不是「没抛异常」。


def _case():
    from app.services.document.domain.entities import (
        RetrievalEvalCase,
    )

    return RetrievalEvalCase(
        eval_case_id=uuid4(),
        bundle_id=uuid4(),
        document_type_definition_id=uuid4(),
        source_chunk_id=uuid4(),
        source_content_hash="h",
        question="q",
        answer_original="a",
        question_order=0,
        question_sha256="s" * 64,
    )


def _run(**overrides):
    from app.services.document.domain.entities import RetrievalEvalRun

    payload = {
        "eval_run_id": uuid4(),
        "bundle_id": uuid4(),
        "max_k": 10,
    }
    payload.update(overrides)

    return RetrievalEvalRun(**payload)


def _spec(**overrides):
    from app.services.document.domain.schemas.retrieval_eval import (
        EvalAttemptSpec,
    )

    return EvalAttemptSpec(**overrides)


def test_attempt_without_reranker_must_not_enable_it() -> None:
    """
    ⚠️ **阶梯里没给精排，构造出的请求里就不能有精排。**

    这条直接对应线上报的那个现象：「我没传 rerank，第三轮却在跑精排」。
    原因是执行侧压根没读前端传的阶梯（`getattr` 恒为 None -> 走内置档）。
    """

    from app.services.document.domain.schemas.retrieve import (
        CutterConfig,
        FusionerConfig,
        RetrieveProcessorOverride,
    )

    request = build_retrieve_request(
        case=_case(),
        run=_run(),
        gold_chunk_ids=[],
        attempt=_spec(
            recall_top_k=500,
            retrieve_processor=RetrieveProcessorOverride(
                fusioner=FusionerConfig(top_k=200),
                cutter=CutterConfig(top_k=50),
            ),
        ),
    )

    assert request.retrieve_processor is not None
    assert request.retrieve_processor.reranker is None, (
        "阶梯里没给精排，请求里就不该有 —— 有的话就是又跑了别处的参数"
    )
    assert request.retrieve_processor.fusioner.top_k == 200
    assert request.retrieve_processor.cutter.top_k == 50


def test_attempt_recall_top_k_reaches_every_route() -> None:
    """这一轮的 `recall_top_k` 要铺到**每一条**召回路上。"""

    request = build_retrieve_request(
        case=_case(), run=_run(), gold_chunk_ids=[],
        attempt=_spec(recall_top_k=400),
    )

    assert request.retrieval_config, "逐路配置不能是空的"
    assert {
        config.top_k for config in request.retrieval_config.values()
    } == {400}
    # 逐路超时仍在（它保证某一路卡住时只降级那一路）。
    assert all(
        config.timeout_ms for config in request.retrieval_config.values()
    )


def test_without_attempt_the_run_override_still_applies() -> None:
    """没有阶梯在跑时，运行级的 override 仍然生效（向后兼容那条路）。"""

    from app.services.document.domain.schemas.retrieve import (
        CutterConfig,
        RetrieveProcessorOverride,
    )

    run = _run(
        query_processor_override=RetrieveProcessorOverride(
            cutter=CutterConfig(top_k=50),
        ).model_dump(mode="json", exclude_unset=True),
    )

    request = build_retrieve_request(
        case=_case(), run=run, gold_chunk_ids=[],
    )

    assert request.retrieve_processor is not None
    assert request.retrieve_processor.cutter.top_k == 50


def test_resolve_ladder_prefers_the_request() -> None:
    """
    `_resolve_ladder` 的两条分支，以及**来源**要能区分。

    来源会被写进快照 —— 「这几个数是按哪套阶梯跑出来的」必须事后可查。
    """

    from app.services.document.application.retrieval_eval_service_impl import (
        _resolve_ladder,
    )
    from app.services.document.domain.schemas.retrieval_eval import (
        CreateEvalRunRequest,
    )

    given = [_spec(name="mine", recall_top_k=123)]
    request = CreateEvalRunRequest(
        bundle_id=uuid4(), attempts=given,
    )

    ladder, source = _resolve_ladder(request, calibrated_override=None)

    assert source == "request"
    assert [item.name for item in ladder] == ["mine"]

    # 没给 attempts -> 内置三档
    ladder, source = _resolve_ladder(
        CreateEvalRunRequest(bundle_id=uuid4()),
        calibrated_override=None,
    )

    assert source == "default_ladder"
    assert len(ladder) == 3


# ============================================================
# 漏斗归因
# ============================================================


def _trace(stages: list[RetrieveStageTrace]) -> RetrieveTrace:
    return RetrieveTrace(target_chunk_ids=[], stages=stages, notes=[])


def _candidate(
        chunk_id: UUID,
        *,
        rank: int,
        selected: bool,
        reason: str | None = None,
) -> TracedChunkCandidate:

    return TracedChunkCandidate(
        chunk_id=chunk_id,
        rank=rank,
        score=0.5,
        selected=selected,
        drop_reason=reason,
        is_target=True,
    )


def test_drop_is_attributed_only_to_the_stage_that_lost_it() -> None:
    """
    **只有"进来时还在、出去时没了"的那一段才算丢了它。**

    这是这张表的语义核心。不这么做的话：金标在召回那一路根本没出现，
    后面的 filter / rerank / cutter 每一段都"出阶段后不在" —— 三段各记
    一次丢弃，宏观统计里它们会背上一大堆不是自己造成的锅。

    构造：金标在 fusioner 就被丢了，后面的段它压根没进去过。
    """

    gold = uuid4()

    trace = _trace([
        RetrieveStageTrace(
            stage_key="fusioner", source_key="", executed=True,
            input_count=10, output_count=3,
            candidates=[
                _candidate(uuid4(), rank=1, selected=True),
                _candidate(gold, rank=5, selected=False,
                           reason="top_k_cutoff"),
            ],
        ),
        # 金标没进这一段 —— 痕迹里没有它的行。
        RetrieveStageTrace(
            stage_key="filter", source_key="", executed=True,
            input_count=3, output_count=3,
            candidates=[_candidate(uuid4(), rank=1, selected=True)],
        ),
    ])

    observations, _ = build_stage_rows(
        eval_case_result_id=uuid4(),
        trace=trace,
        gold_chunk_ids={gold},
    )

    by_stage = {item.stage_key: item for item in observations}

    fusion = by_stage["fusioner"]

    assert fusion.gold_present_before is True
    assert fusion.gold_present_after is False
    assert fusion.drop_reason == "top_k_cutoff"
    assert fusion.gold_rank_before == 5

    # ⚠️ 后面那一段**不能**再记一次丢弃。
    later = by_stage["filter"]

    assert later.gold_present_before is False
    assert later.gold_present_after is False
    assert later.drop_reason == "not_applicable", (
        "更早的阶段就丢了，这里不是它的锅"
    )


def test_not_recalled_is_reported_not_hidden_as_not_applicable() -> None:
    """
    一路都没碰到的金标记 `not_recalled`，**不是** not_applicable。

    它对任何一段都是"进来前就不在"，但恰恰是最该被看见的那种失败 ——
    记成"不适用"会让"金标连召回的边都没碰到"从统计里消失。
    """

    gold = uuid4()

    trace = _trace([
        RetrieveStageTrace(
            stage_key="recall", source_key="", executed=True,
            input_count=0, output_count=0,
            candidates=[
                _candidate(gold, rank=1, selected=False,
                           reason="not_recalled"),
            ],
        ),
    ])

    observations, _ = build_stage_rows(
        eval_case_result_id=uuid4(),
        trace=trace,
        gold_chunk_ids={gold},
    )

    observation = observations[0]

    assert observation.drop_reason == "not_recalled"
    assert observation.gold_present_before is False
    assert observation.gold_present_after is False


def test_surviving_gold_has_no_drop_reason() -> None:
    """金标活着穿过一段 -> `drop_reason` 是 None（不是 not_applicable）。"""

    gold = uuid4()

    trace = _trace([
        RetrieveStageTrace(
            stage_key="cutter", source_key="", executed=True,
            input_count=3, output_count=1,
            candidates=[
                _candidate(gold, rank=1, selected=True),
            ],
        ),
    ])

    observations, candidates = build_stage_rows(
        eval_case_result_id=uuid4(),
        trace=trace,
        gold_chunk_ids={gold},
    )

    assert observations[0].drop_reason is None
    assert observations[0].gold_present_before is True
    assert observations[0].gold_present_after is True

    # 候选行：活下来的写 after 那一套（库上的 CHECK 要求至少一个非空）。
    assert candidates[0].rank_after == 1
    assert candidates[0].rank_before is None


def test_dropped_candidate_records_the_rank_it_died_at() -> None:
    """
    被丢的候选记的是**死时**的名次。

    那正是"卡在门槛上"的证据：看到 rank=11 / score=0.29 而阈值是 0.3，
    一眼看出是分数砍的还是排序本来就错。
    """

    gold = uuid4()

    trace = _trace([
        RetrieveStageTrace(
            stage_key="cutter", source_key="", executed=True,
            input_count=11, output_count=10,
            candidates=[
                _candidate(gold, rank=11, selected=False,
                           reason="final_top_k_cutoff"),
            ],
        ),
    ])

    _, candidates = build_stage_rows(
        eval_case_result_id=uuid4(),
        trace=trace,
        gold_chunk_ids={gold},
    )

    assert candidates[0].rank_before == 11
    assert candidates[0].rank_after is None
    assert candidates[0].drop_reason == "final_top_k_cutoff"
