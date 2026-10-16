"""
评测执行的**纯函数**部分：请求构造、命中判定、痕迹映射、指标计算。

为什么单独一个模块
------------------
它们没有一个碰数据库、碰模型 —— 全是"输入什么、输出什么"。而它们是这
一层**唯一需要反复验证**的东西：

    命中判定错了      → 召回率整体偏移，且偏移量与检索无关
    归因规则错了      → 宏观统计指向错误的阶段，改错地方
    指标分母错了      → failed 混进 miss，"检索挂了"看起来像"召回变差"

把它们从服务实现里拿出来，就能脱离库、脱离索引后端反复跑。
`retrieval_eval_service_impl.py` 只负责"按顺序调它们、把结果落库"。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from ..domain.entities import (
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    RetrievalEvalRun,
    RetrievalEvalStageCandidate,
    RetrievalEvalStageObservation,
)
from ..domain.schemas.enum import EvalCaseResultStatus
from ..domain.schemas.retrieve import (
    DocumentRetrieveRequest,
    DocumentRetrieveResponse,
    QueryRewriteInput,
    RetrieveProcessorOverride,
)

__all__ = [
    "EVAL_ROUTE_TIMEOUT_MS",
    "collapse_by_case",
    "representative_result",
    "HIT_RATE_CURVE_KS",
    "build_retrieve_request",
    "build_stage_rows",
    "compute_metrics",
    "judge_hit",
    "override_of",
    "pass_rate",
    "percentile",
]


#: 每一路召回的默认超时（毫秒）。
#:
#: ⚠️ 这是**逐路**的超时，与 `CASE_TIMEOUT_SECONDS`（整条 case 的兜底）
#: 分工不同：
#:
#:     本常量           某一路卡住 -> 那一路进 degraded_routes，其余路照常，
#:                      这条 case 照常判命中/未命中 —— **真正的降级**
#:     CASE_TIMEOUT     兜底：整条 case 超过 60 秒就判 failed，别让一轮挂死
#:
#: 不设它的话，`retrieval_config` 为空 -> 每路都用后端自己的超时（也就是
#: 不设）-> **任何一路卡住都会拖死整条 case**。而正确的语义是"其余路召回
#: 的内容是真实可检索的，没道理让整个查询连带失败"（见 README 的
#: degraded_routes 一节）。
#:
#: 取 10 秒：实测两路都在 0.2 秒量级，所以 10 秒是 p95 的 **50 倍** —— 正常
#: 路径碰不到，而真卡住时 10 秒就降级，不必等到 60 秒的兜底。
#:
#: ⚠️ 这个数**没有**做成请求参数：`CreateEvalRunRequest` 的
#: `query_processor_override` 只覆盖召回**之后**的四段，召回参数按公开契约
#: 不是调用方该管的（recaller 是内部实现细节）。要按部署调，改这里。
EVAL_ROUTE_TIMEOUT_MS = 10_000

#: 召回率曲线读哪几个 k。
HIT_RATE_CURVE_KS = (1, 5, 10, 20, 50)


# ============================================================
# 请求构造
# ============================================================


def override_of(run: RetrievalEvalRun) -> RetrieveProcessorOverride | None:
    """
    把运行上存的 override 还原成 DTO。

    `create_eval_run` 已经按 `max_k` 调过 `cutter.top_k`（压到不超过它）
    并存下来了，这里只是读回去 —— **判定口径必须与建轮次时一致**。
    在执行时再算一遍会出现"建的时候按 10 存、跑的时候按 50 算"，而两者
    都能自圆其说。
    """

    if not run.query_processor_override:
        return None

    return RetrieveProcessorOverride.model_validate(
        run.query_processor_override
    )


def build_retrieve_request(
        *,
        case: RetrievalEvalCase,
        run: RetrievalEvalRun,
        gold_chunk_ids: list[UUID],
        attempt: Any | None = None,
        query_rewrite: QueryRewriteInput | None = None,
) -> DocumentRetrieveRequest:
    """
    一条 case -> 一次检索请求。

    ⚠️ `document_type_definition_ids` **只有一个元素**，这是整个评测的
    口径所在。把 bundle 的整个 DTD 范围传进去的话，`retrieve_scope` 会
    变成 multi，装配路径切到 `COMMON_RETRIEVE_PROCESSOR_CHAIN` ——
    生效配置就变了（不再走该 DTD 自己的融合/精排配置）。

    ⚠️ `target_chunk_ids` 传的是**金标集合**，不是 `source_chunk_id` 单个。
    一条 case 可能有多个金标（跨 chunk 裁定加进来的），而"目标在哪被丢"
    对每一个都成立 —— 只盯出题那一块，会漏掉"另一个金标其实被召回到了"。

    ⚠️ `only_active=True` 与抽样口径一致：数据集是从 active 版本的 chunk
    抽的。设 false 会把 superseded 版本的 chunk 也召回，而那些 chunk 的
    id 与数据集里的**不同**（reprocess 只增不删）—— 金标永远命不中，
    召回率凭空变低。
    """

    return DocumentRetrieveRequest(
        document_type_definition_ids=[
            case.document_type_definition_id
        ],
        input_text=case.question,
        only_active=True,
        # ⚠️ **阶梯在跑时，这一轮用什么就是什么。**
        #
        # 此前这里是"这一轮没给 retrieve_processor 就回落到运行级
        # override" —— 而契约里写的是"给了 attempts 就忽略
        # query_processor_override"。两者矛盾，且回落会让某一轮**悄悄**
        # 拿到另一轮的参数。现在：这一轮没给 = 整段继承 DTD。
        retrieve_processor=(
            attempt.retrieve_processor
            if attempt is not None
            else override_of(run)
        ),
        # 改写**不回落**到任何运行级默认：阶梯没给这一轮配改写器，就是
        # "这一轮不改写"。给个兜底会让"某轮悄悄用上了别的轮次的改写"
        # 变成可能，而那正是 `query_processor_override` 那处已经踩过的
        # 坑（见上面那段注释）。
        #
        # ⚠️ 与 `query_rewrite` **二选一**：调用方（批量化那条路）已经把
        # 这一轮的题一次改完了，这时就不再让服务去调改写器 —— 两者同时发
        # 会有一个静默失效（服务只用其中之一），而那正是契约要避免的。
        query_rewriter=(
            None if query_rewrite is not None
            else (attempt.query_rewriter if attempt is not None else None)
        ),
        query_rewrite=query_rewrite,
        include_stage_trace=run.include_stage_trace,
        # 空列表与 None 在契约上是两回事：`None` = 不盯任何东西（只采漏斗），
        # 而空列表会被 `trace_chunk_ids` 的校验当成"给了值但没开开关"。
        # 所以没有金标时给 None。
        target_chunk_ids=sorted(gold_chunk_ids, key=str) or None,
        # ⚠️ 逐路超时 —— 见 `EVAL_ROUTE_TIMEOUT_MS` 的说明。
        retrieval_config=_route_timeouts(
            recall_top_k=(
                attempt.recall_top_k if attempt is not None else None
            ),
            explicit=(
                attempt.retrieval_config if attempt is not None else None
            ),
        ),
    )


def _route_timeouts(
        *,
        recall_top_k: int | None = None,
        explicit: dict | None = None,
) -> dict:
    """
    给**每一种** retrieval_type 都配上超时。

    ⚠️ 要枚举全部类型，不能只写 `dense_vector` / `keyword`。
    `retrieval_config` 的语义是"未出现的**路**用该路的默认"，而默认就是
    **不设超时** —— 只列两路的话，某个 DTD 上第三条路（比如 graph）卡住
    照样拖死整条 case，而"我只配了我认识的那两路"看起来毫无问题。

    （实测踩过一次偶发卡顿：同一查询反复打都是 0.2 秒，说明不是查询形状；
    但当时任何一路都没有超时，一卡就是整条 case 挂住。）
    """

    # 逐路细化优先：调用方点名的那些路用它给的配置，其余仍拿默认超时。
    # 不合并而是整体替代 —— 合并要处理"这些路给了、那些路没给"，而
    # `retrieval_config` 的语义本来就是"未出现的路用该路默认"。
    if explicit:
        return {
            key: (
                value
                if value.timeout_ms is not None
                else value.model_copy(
                    update={"timeout_ms": EVAL_ROUTE_TIMEOUT_MS}
                )
            )
            for key, value in explicit.items()
        }

    from ..domain.schemas.enum import RetrievalType
    from ..domain.schemas.retrieve import RetrievalRecallConfig

    return {
        retrieval_type: RetrievalRecallConfig(
            top_k=recall_top_k,
            timeout_ms=EVAL_ROUTE_TIMEOUT_MS,
        )
        for retrieval_type in RetrievalType
    }


# ============================================================
# 命中判定
# ============================================================


def judge_hit(
        *,
        response: DocumentRetrieveResponse,
        gold_chunk_ids: set[UUID],
        max_k: int,
) -> tuple[bool, int | None]:
    """
    金标命中判定。返回 `(是否命中, 最好的名次)`。

    **纯 ID 比对** —— 不调 LLM、不看文本相似度。这是第一层的定义：它测的是
    召回，所以判据只能是"金标在不在返回里"。

    多个金标时取**最好**的那个名次（MRR 用它），不是第一个碰到的 —— 后者
    会让排序看起来比实际差。

    ⚠️ **名次在"全部返回结果"里找，而不是在 `results[:max_k]` 里找。**
    -----------------------------------------------------------------
    这两个口径必须解耦，否则 `max_k` 会把信息毁掉：max_k=10 时金标排第
    15，旧写法直接返回 `rank=None` —— 于是**事后无法回答"如果看前 20 条
    会怎样"**，只能重跑一轮别的 max_k（还要重付全部检索成本）。

    现在的语义是：

        gold_rank   金标在**返回集合**里的真实名次（没返回就是 None）
        gold_hit    名次 <= max_k

    这样 `max_k` 只决定"判不判命中"，不再决定"我们能看得多远"，于是
    同一轮数据可以读出**任意 k 的召回率曲线**（见 compute_metrics 的
    `hit_rate_curve`）。
    """

    for index, item in enumerate(response.results):

        if item.document_chunk_id in gold_chunk_ids:

            rank = index + 1

            return rank <= max_k, rank

    return False, None


# ============================================================
# 痕迹 -> 两张表的行
# ============================================================


def build_stage_rows(
        *,
        eval_case_result_id: UUID,
        trace: Any,
        gold_chunk_ids: set[UUID],
) -> tuple[
    list[RetrievalEvalStageObservation],
    list[RetrievalEvalStageCandidate],
]:
    """
    `response.trace` -> 逐阶段观察与候选行。

    归因规则（**这张表的语义核心**）：

        一个金标在某段"出阶段后不在"了，只有它**进这一段前还在**时才
        算这一段丢的。痕迹里出现一条金标行就表示"它在这一段的输入里"
        （`RetrievalTraceCollector` 对从没见过的 chunk **不补行**），
        所以这条直接可读。

        否则：召回那一路没碰到的金标，会让后面 filter / rerank / cutter
        每一段都"出阶段后不在" —— 三段各记一次丢弃，宏观统计里它们会
        背上一大堆不是自己造成的锅。

    例外是召回那条**合成行**（`drop_reason=not_recalled`）：它是"所有路
    都没碰到"，进来之前就不在，所以前/后都记 False，而**不是**
    not_applicable —— 它恰恰是最该被看见的那种失败。
    """

    observations: list[RetrievalEvalStageObservation] = []
    candidates: list[RetrievalEvalStageCandidate] = []

    for stage in trace.stages:

        gold_rows = [
            row for row in stage.candidates
            if row.chunk_id in gold_chunk_ids
        ]

        survivors = [row for row in gold_rows if row.selected]
        dropped = [row for row in gold_rows if not row.selected]

        synthetic = (
            bool(dropped)
            and not survivors
            and all(row.drop_reason == "not_recalled" for row in dropped)
        )

        if synthetic:

            present_before = False
            present_after = False
            rank_before = None
            rank_after = None
            score_before = None
            score_after = None
            drop_reason: str | None = "not_recalled"

        else:

            present_before = bool(gold_rows)
            present_after = bool(survivors)

            # 名次用 `min(..., default=None)` 而不是 0：0 会被读成"排第 0"。
            rank_before = min(
                (row.rank for row in gold_rows), default=None
            )
            rank_after = min(
                (row.rank for row in survivors), default=None
            )
            score_before = _best_score(gold_rows)
            score_after = _best_score(survivors)

            if not present_before:
                # 更早的阶段就丢了。记 not_applicable 而不是再记一次丢弃。
                drop_reason = "not_applicable"
            elif present_after:
                drop_reason = None
            else:
                drop_reason = dropped[0].drop_reason or "unknown"

        observations.append(
            RetrievalEvalStageObservation(
                stage_observation_id=uuid4(),
                eval_case_result_id=eval_case_result_id,
                stage_key=stage.stage_key,
                source_key=stage.source_key,
                executed=stage.executed,
                input_count=stage.input_count,
                output_count=stage.output_count,
                gold_present_before=present_before,
                gold_present_after=present_after,
                gold_rank_before=rank_before,
                gold_rank_after=rank_after,
                gold_score_before=score_before,
                gold_score_after=score_after,
                # 分数名与方向**留空**：本管线暴露的分数始终是归一化到
                # [0,1] 的（见 stage_candidate 的说明），编一个名字比留空更糟。
                gold_score_name=None,
                gold_score_direction=None,
                drop_reason=drop_reason,
                details={},
            )
        )

        for row in stage.candidates:

            # 用哪一套排名/分数取决于它在**这一段**是活是死：
            #
            #     活下来   -> rank_after / score_after
            #     死在这里 -> rank_before / score_before
            #
            # （库上的 CHECK 要求至少一个非空，这个映射天然满足。）
            candidates.append(
                RetrievalEvalStageCandidate(
                    stage_candidate_id=uuid4(),
                    eval_case_result_id=eval_case_result_id,
                    stage_key=stage.stage_key,
                    source_key=stage.source_key,
                    chunk_id=row.chunk_id,
                    rank_after=row.rank if row.selected else None,
                    rank_before=None if row.selected else row.rank,
                    score_after=row.score if row.selected else None,
                    score_before=None if row.selected else row.score,
                    score_name_after=None,
                    score_name_before=None,
                    score_direction_after=None,
                    score_direction_before=None,
                    selected=row.selected,
                    drop_reason=row.drop_reason,
                    metadata={"is_target": row.is_target},
                )
            )

    return observations, candidates


def _best_score(rows: list[Any]) -> float | None:
    """名次最靠前那一行的分数；都没有分数就 None。"""

    ranked = [row for row in rows if row.score is not None]

    if not ranked:
        return None

    return min(ranked, key=lambda row: row.rank).score


# ============================================================
# 指标
# ============================================================


def collapse_by_case(
        results: list[RetrievalEvalCaseResult],
) -> list[RetrievalEvalCaseResult]:
    """
    每条 case 只留**最好**的那一次，供表头指标使用。

    ⚠️ 这一步不做，逐轮放宽的收益会被**表头稀释掉**
    --------------------------------------------
    `compute_metrics` 是按**行**算的。而逐轮评测里一条 case 会有多行
    （一轮一行），于是：

        第 1 轮没中、第 2 轮命中  ->  一行 miss + 一行 hit
        => 它同时进了分母两次，分子一次

    实测口径（同一批 40 条反复跑出来的 240 行）：

        按行数命中  103/240 = 43%
        按 case 命中  28/40  = 70%     <- 这才是「最终找到没」

    于是「3 轮把 37% 提到 60%」会被算成 40% —— 看起来像"扩了两轮几乎
    没用"，而实际上它们救回了 9 条。

    「哪一轮救回来的」由 `by_attempt` 回答；表头回答的是**最终**结果。
    两个问题，两个口径。

    取哪一行：命中优先；都命中取名次最小的；都没命中时**非执行失败**
    优先（一条 case 在第一轮检索挂了、第二轮正常跑但没中，它的状态
    应当是「没中」而不是「挂了」）。完全同级时取更晚的一轮。
    """

    best: dict[Any, RetrievalEvalCaseResult] = {}

    for result in results:

        current = best.get(result.eval_case_id)

        if current is None or _better(result, current):
            best[result.eval_case_id] = result

    return list(best.values())


def representative_result(
        results: list[RetrievalEvalCaseResult],
) -> RetrievalEvalCaseResult | None:
    """
    一条 case 的代表行 —— `collapse_by_case` 的单条版本。

    单条 case 的痕迹页只需要一行，而**取哪一行必须与列表、表头同一条**：
    三处各写一遍规则的话，同一次评测在三个页面上会给出三种「这条 case
    的结果」，而三处单看都说得通。

    传空列表回 None（没有结果行可挑）。
    """

    best: RetrievalEvalCaseResult | None = None

    for result in results:

        if best is None or _better(result, best):
            best = result

    return best


def _better(
        candidate: RetrievalEvalCaseResult,
        current: RetrievalEvalCaseResult,
) -> bool:
    """`candidate` 是否比 `current` 更能代表这条 case 的最终结果。"""

    def _key(item: RetrievalEvalCaseResult) -> tuple:
        # 命中 > 没命中 > 执行失败
        tier = 2 if item.gold_hit else (1 if item.gold_hit is False else 0)
        # 名次越小越好；None 排在最后。
        #
        # ⚠️ 末位的轮次是**定序用**的，不是业务口径：三轮都没命中的 case
        # 两两同级，不加这一位取哪一行就取决于数据库返回顺序 —— 于是同样
        # 的两行，两次查询给出不同的痕迹页（乃至不同的 latency 分位）。
        # 取更晚的一轮：逐轮放宽里它就是参数最宽、最接近"最终找不到"的那次。
        return (tier, -(item.gold_rank or 0), item.attempt_no)

    return _key(candidate) > _key(current)


def compute_metrics(
        results: list[RetrievalEvalCaseResult],
        max_k: int,
) -> dict[str, Any]:
    """
    确定性指标。**没有一条来自 LLM 判分。**

    ⚠️ 分母是"判过"的那些，**不是** `total_cases`
    ---------------------------------------------
    检索执行失败的那些（后端挂了、超时）**不进分母**：它们是**故障**，
    不是**质量**。混进去的话，"检索挂了"看起来就像"召回率变差了"
    —— 而那是最容易误判成检索问题的一种假象。

    （失败数另有一列 `failed_cases`，跑通与全挂都能一眼看出来。）

    ⚠️ `hit_rate_at_5` 在 `max_k < 5` 时是 NULL 而不是 0
    ------------------------------------------------
    它没有意义。而"没有意义"必须与"命中率是 0"分开 —— 后者会被读成
    "top5 一个都没中"，是完全不同的结论。
    """

    completed = [
        result for result in results
        if result.status is EvalCaseResultStatus.COMPLETED
    ]
    failed = [
        result for result in results
        if result.status is not EvalCaseResultStatus.COMPLETED
    ]

    judged = [
        result for result in completed if result.gold_hit is not None
    ]

    def _rank_of(
            result: RetrievalEvalCaseResult,
            k: int | None = None,
    ) -> int | None:
        """真实名次；超过 k 就是 None（k 默认取本轮的 max_k）。"""

        rank = result.gold_rank
        limit = max_k if k is None else k
        return rank if rank is not None and rank <= limit else None

    hit = sum(1 for result in judged if _rank_of(result) is not None)

    def _rate(numerator: int, denominator: int) -> float | None:
        if not denominator:
            return None
        return round(numerator / denominator, 6)

    latencies = sorted(
        float(result.latency_ms)
        for result in judged
        if result.latency_ms is not None
    )

    mrr: float | None = None

    if judged:
        mrr = round(
            sum(
                1.0 / _rank_of(result)
                for result in judged
                if _rank_of(result) is not None
            )
            / len(judged),
            6,
        )

    return {
        "total_cases": len(results),
        "completed_cases": len(completed),
        "failed_cases": len(failed),
        "hit_count": hit,
        "miss_count": len(judged) - hit,
        "hit_rate_at_1": _rate(
            sum(1 for item in judged if _rank_of(item) == 1),
            len(judged),
        ),
        "hit_rate_at_5": (
            None
            if max_k < 5
            else _rate(
                sum(
                    1 for item in judged
                    if _rank_of(item) is not None
                    and _rank_of(item) <= 5
                ),
                len(judged),
            )
        ),
        "hit_rate_at_k": _rate(hit, len(judged)),
        # 召回率曲线：同一轮数据读出任意 k 的命中率。
        #
        # 存在的理由就是 `max_k` 那件事：线上到底喂几条给模型是个**部署
        # 事实**，而它一变，单个 hit@k 的数就要重跑一轮才知道。有了曲线，
        # "口径对齐"从"重跑"变成"读另一个点"。
        #
        # k 超过实际返回条数时那个点**没有意义**（分母还是判过的那些，
        # 分子不可能再涨）—— 调用方看曲线时要拿 `retrieved_max` 对照。
        "hit_rate_curve": {
            str(k): _rate(
                sum(
                    1 for item in judged
                    if _rank_of(item, k) is not None
                ),
                len(judged),
            )
            for k in HIT_RATE_CURVE_KS
        },
        "mrr_at_k": mrr,
        # 实际返回了几条（读曲线时要对照 —— 超过它的 k 没有意义）。
        "retrieved_max": max(
            (
                len(item.retrieved_chunk_ids or [])
                for item in judged
            ),
            default=0,
        ),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "max_k": max_k,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """
    最近秩法取分位。`values` 必须**已排序**。

    不用插值：样本少时插值会给出一个**库里不存在的**延迟值，而它的用途
    只是"看一眼慢到什么程度" —— 一个从未发生过的数比略保守的真实值更
    容易误导。
    """

    if not values:
        return None

    index = max(
        0, min(len(values) - 1, int(round(fraction * len(values))) - 1)
    )

    return round(values[index], 1)


def pass_rate(before: int, after: int) -> float | None:
    """一个阶段的金标通过率。没有"进来时还在"的样本时为 None。"""

    if not before:
        return None

    return round(after / before, 6)
