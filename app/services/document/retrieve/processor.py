"""
RetrieveProcessor：四段配置的解析 + 一次装配的执行。

配置从哪来（重要）
------------------
系统里**只有两段查询配置有存放处** ——
DocumentTypeDefinition 的扁平字段：

    fusion_strategy / fusion_config   -> FusionerConfig
    rerank_strategy / rerank_config   -> RerankerConfig

Filter 与 Cutter 在 DTD 表里**没有对应列**，也没有公共链配置文件。
因此这两段（以及公共链自身的基线）的默认值直接**编码为本类的
类属性** —— 见下面的 default_* 常量。改默认值就改这里，
不要去 DTD 表加列。

覆盖优先级（对应规格的 runtime override precedence）：

    请求的 retrieve_processor 字段
        > DTD 配置
        > 本类默认值

"未给" 与 "给了默认值" 是两回事
--------------------------------
规格写明 "Omitted fields inherit"。而 pydantic 会把没传的字段
填上默认值，直接 model_copy 会把继承变成重置 —— 例如 DTD 配了
top_k=100，调用方只想改 fusioner.type，结果 top_k 被悄悄还原成 50。

因此合并时只看 `model_fields_set`（本次**显式传入**过的字段名），
见 _apply_override。这是本模块最容易写错的地方。

Recaller 不在这里
-----------------
规格：Recaller 是内部实现细节，不通过公开 API 暴露，也不参与
override。候选集由服务层内联调用 IndexBackend.search 得到，
在调用 run() 之前就已就绪。
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import ValidationError

from ..domain.entities import DocumentTypeDefinition
from ..domain.schemas import (
    CutterConfig,
    EffectiveRetrieveProcessor,
    FilterConfig,
    FusionerConfig,
    RetrieveProcessorOverride,
    RerankerConfig,
)
from ..domain.schemas.enum import (
    FusionerType,
    RetrieveProcessorSource,
    RerankerType,
)
from .base import RetrieveCandidate, normalize_scores
from .registry import RetrieveProcessorRegistry, get_registry
from .trace import RetrievalTraceCollector

__all__ = ["RetrieveProcessor"]


def _stage_log(stage_key: str, note: str = "") -> float:
    """
    一个阶段**开始**了 —— 返回起始时刻。

    ⚠️ 只打"开始"、不打"结束"，是刻意的：这样**没有出现的那一行**
    就是卡住的那一段。打"结束"的话，卡住时你看到的是最后一条"结束"
    日志，还得自己推下一个是谁；只打"开始"则直接指向它。
    """
    import time

    logger.info("[retrieve] 阶段 %-16s 开始%s", stage_key, note)
    return time.monotonic()


def _stage_done(stage_key: str, started: float, count: int) -> None:
    """阶段结束了。与上面的"开始"配对 —— 有开始没结束 = 它卡住了。"""
    import time

    logger.info(
        "[retrieve] 阶段 %-16s 结束 %.0fms  -> %d 条",
        stage_key,
        (time.monotonic() - started) * 1000,
        count,
    )


def _flatten_by_score(
        result_sets: list[list[RetrieveCandidate]],
) -> list[RetrieveCandidate]:
    """
    多路候选拼成一列，按分数降序。

    **只给痕迹用**（融合那一阶段的"进"长什么样），不参与任何计算 ——
    所以它不需要与 RRF 的排序口径一致，见 `run()` 里的说明。
    """

    merged = [
        candidate
        for candidates in result_sets
        for candidate in candidates
    ]

    merged.sort(key=lambda item: item.score, reverse=True)

    return merged

logger = logging.getLogger(__name__)


class RetrieveProcessor:
    """
    一组已解析的四段配置 + 把它们串起来跑一遍的能力。

    实例是**不可变的配置载体**：解析完就不再改。
    一次查询里会被构造多次（每个 DTD 一个，加公共链一个）。
    """

    # ========================================================
    # 系统默认值
    # ========================================================
    #
    # 用户可见的默认行为就写在这里。DTD 没有 filter / cutter 列，
    # 公共链也没有配置文件，所以这两段的基线只存在于本处。
    #
    # 注意这些是**不可变**的 Config 实例：所有派生都走
    # model_copy，绝不原地改，否则会污染进程内共享的默认值。

    default_fusioner: ClassVar[FusionerConfig] = (
        FusionerConfig()
    )

    default_filter: ClassVar[FilterConfig] = FilterConfig()

    default_reranker: ClassVar[RerankerConfig] = (
        RerankerConfig()
    )

    default_cutter: ClassVar[CutterConfig] = CutterConfig()

    def __init__(
            self,
            *,
            source: RetrieveProcessorSource,
            fusioner: FusionerConfig,
            filter: FilterConfig,
            reranker: RerankerConfig,
            cutter: CutterConfig,
            registry: RetrieveProcessorRegistry | None = None,
    ) -> None:

        self.source = source
        self.fusioner = fusioner
        self.filter = filter
        self.reranker = reranker
        self.cutter = cutter

        self._registry = registry or get_registry()

    # ========================================================
    # 构造
    # ========================================================

    @classmethod
    def for_document_type(
            cls,
            definition: DocumentTypeDefinition,
            *,
            override: RetrieveProcessorOverride | None = None,
            registry: RetrieveProcessorRegistry | None = None,
    ) -> "RetrieveProcessor":
        """
        按 DTD 的扁平字段解析，再叠加请求级 override。

        source 的判定见 _resolve_source：**只有 override 真的改变了
        生效配置**才报 runtime_override，否则报
        document_type_definition —— 传了一模一样的值不该改口径。
        """

        base = cls._from_definition(definition)

        final = cls(
            source=RetrieveProcessorSource
            .DOCUMENT_TYPE_DEFINITION,
            fusioner=base.fusioner,
            filter=base.filter,
            reranker=base.reranker,
            cutter=base.cutter,
            registry=registry,
        )._with_override(override)

        final.source = cls._resolve_source(
            base=base, final=final
        )

        return final

    @classmethod
    def common_chain(
            cls,
            *,
            override: RetrieveProcessorOverride | None = None,
            registry: RetrieveProcessorRegistry | None = None,
    ) -> "RetrieveProcessor":
        """
        COMMON_RETRIEVE_PROCESSOR_CHAIN：多 DTD 场景下，各 DTD 各自
        跑完之后再过的那条链。

        它没有归属的 DTD，因此基线就是本类的系统默认值。
        """

        base = cls(
            source=RetrieveProcessorSource
            .COMMON_RETRIEVE_PROCESSOR_CHAIN,
            fusioner=cls.default_fusioner,
            filter=cls.default_filter,
            reranker=cls.default_reranker,
            cutter=cls.default_cutter,
            registry=registry,
        )

        final = base._with_override(override)

        final.source = cls._resolve_source(
            base=base, final=final
        )

        return final

    # ========================================================
    # 执行
    # ========================================================

    async def run(
            self,
            *,
            query: str,
            result_sets: list[list[RetrieveCandidate]],
            trace: RetrievalTraceCollector | None = None,
    ) -> list[RetrieveCandidate]:
        """
        跑一遍 融合 -> 过滤 -> 精排 -> 裁剪。

        result_sets 是"每路一个列表"（多 DTD 场景下是"每个 DTD
        一个列表"）。返回已按分数降序、已裁剪的最终结果。

        `trace` —— 逐阶段痕迹的收集器，**可选**，默认 None
        --------------------------------------------------
        它是**纯观察者**：只读每一步已经算完的列表，不改、不去重、
        不排序，返回值一律不用。传 None 时这条链路与从前逐字节相同
        （一个 collector 都不建、一次集合运算都不做）。

        ⚠️ 调用点全部在"这一步的结果已经产生之后"。**不要**为了少存一次
        中间变量而把 observe 插到阶段中间，更不要在 collect 里顺手
        `sorted()`/`set()` 之后拿返回值继续往下传 —— `run()` 是正在
        工作的主链路，那样做会**静默改变检索结果**。
        """

        # 融合的"进"：多路拼起来、按分数降序。
        #
        # ⚠️ 这个名次只是**报告口径**，不是 RRF 看到的东西 —— RRF 用的是
        # **每一路内部**的排名（`Σ 1/(rrf_k + rank)`），跨路的分数本就
        # 不可比（各自按最大值归一化，头名都是 1.0）。所以它回答的是
        # "这一条在所有候选里大致处于什么位置"，而不是"融合按什么排的"。
        fused_input = (
            _flatten_by_score(result_sets)
            if trace is not None
            else None
        )

        _t = _stage_log("fusioner")
        candidates = await self._fuse(query, result_sets)
        _stage_done("fusioner", _t, len(candidates))

        if trace is not None:
            trace.observe_stage(
                stage_key="fusioner",
                before=fused_input,
                after=candidates,
                reason="top_k_cutoff",
            )

        if self.filter.enabled:

            # 只在开着 trace 时才收集原因 —— 关了就不产生这个 dict，
            # 过滤器的热循环里少一次哈希写入。
            dropped: dict | None = {} if trace is not None else None

            before_filter = candidates
            _t = _stage_log("filter", f"（enabled={self.filter.enabled}）")

            candidates = await self._registry.filter.apply(
                candidates=candidates,
                config=self.filter,
                dropped=dropped,
            )
            _stage_done("filter", _t, len(candidates))

            if trace is not None:
                trace.observe_stage(
                    stage_key="filter",
                    before=before_filter,
                    after=candidates,
                    # 逐个原因（过滤有四种），而不是按阶段写死一种。
                    reasons=dropped,
                )

        elif trace is not None:
            # 过滤被关掉了。仍然报一段 —— 但 `executed=False`。
            # 不报的话，"候选怎么突然少了"会在痕迹里断档；而报成
            # executed=True 又会让"没配过滤"看起来像"过滤没丢东西"。
            trace.observe_stage(
                stage_key="filter",
                before=candidates,
                after=candidates,
                executed=False,
            )

        before_rerank = candidates
        _t = _stage_log("reranker", f"（type={self.reranker.type}）")

        # 实现往这里写"我没能真跑"的原因（第三方不可用、超时…）。
        # 形制照 `ResultFilter.apply(dropped=...)` —— 原因必须由实现
        # 回答，调用方分不清"本地模型挂了"与"远端 502"。
        rerank_degraded: dict[str, str] = {}

        candidates = await self._rerank(
            query, candidates, degraded=rerank_degraded,
        )
        _stage_done("reranker", _t, len(candidates))

        if rerank_degraded:
            logger.warning(
                "精排降级（type=%s）：%s —— 本轮按融合顺序返回。",
                self.reranker.type,
                rerank_degraded.get("reason", "未记录原因"),
            )

        if trace is not None:
            trace.observe_stage(
                stage_key="reranker",
                before=before_rerank,
                after=candidates,
                reason="rerank_cutoff",
                # 三种"没真跑"：enabled=false、type 为空（DTD 没配精排、
                # 调用方也没指定）、以及**实现自己报告降级**。前两种不是
                # "精排跑了但没改动结果"，第三种更不是 —— 它连模型都没
                # 碰到。混进"跑了"会让宏观统计读成"精排没能改善排序"，
                # 而真实原因是那次调用根本没成功。
                executed=(
                    self.reranker.enabled
                    and self.reranker.type is not None
                    and not rerank_degraded
                ),
            )

        if rerank_degraded and trace is not None:
            trace.note(
                "精排未能执行（type="
                f"{self.reranker.type}）："
                f"{rerank_degraded.get('reason', '未记录原因')}"
                " —— 本轮结果按融合顺序，与「精排跑了但没改善排序」"
                "不是一回事。"
            )

        before_cut = candidates
        _t = _stage_log("cutter", f"（top_k={self.cutter.top_k}）")

        candidates = await self._registry.cutter.cut(
            candidates=candidates,
            config=self.cutter,
        )
        _stage_done("cutter", _t, len(candidates))

        if trace is not None:
            trace.observe_stage(
                stage_key="cutter",
                before=before_cut,
                after=candidates,
                reason="final_top_k_cutoff",
            )

        return candidates

    async def _fuse(
            self,
            query: str,
            result_sets: list[list[RetrieveCandidate]],
    ) -> list[RetrieveCandidate]:

        sets = [s for s in result_sets if s]

        if not sets:
            return []

        # 只有一路时不做融合：融合是"把多路并成一路"的操作，
        # 对单路跑 RRF 会把原始分换成 1/(k+rank)，白白丢掉
        # "这条到底有多相关"的信息。此时各路的分数在召回后
        # 已按路归一化，直接沿用。
        if len(sets) == 1:
            return normalize_scores(sets[0])

        if not self.fusioner.enabled:

            merged = [item for s in sets for item in s]

            merged.sort(
                key=lambda item: item.score,
                reverse=True,
            )

            return normalize_scores(merged)

        fusioner = self._registry.fusioner(
            self.fusioner.type
        )

        return await fusioner.fuse(
            query=query,
            result_sets=sets,
            top_k=self.fusioner.top_k,
            config=self.fusioner,
        )

    async def _rerank(
            self,
            query: str,
            candidates: list[RetrieveCandidate],
            degraded: dict[str, str] | None = None,
    ) -> list[RetrieveCandidate]:

        if not candidates:
            return []

        if not self.reranker.enabled:
            return candidates

        # type 为空 = 这条链没有配置精排（DTD 的 rerank_strategy
        # 为 NULL 且调用方也没指定）。与 enabled=False 一样跳过，
        # 而不是去找一个"默认精排"—— 规格里 type 没有默认值。
        if self.reranker.type is None:
            return candidates

        # 送进精排的候选数默认沿用融合的 top_k：融合已经决定了
        # "候选池有多大"，精排只该在这个池子里重排。
        top_n = self.reranker.top_n or self.fusioner.top_k

        reranker = self._registry.reranker(
            self.reranker.type
        )

        return await reranker.rerank(
            query=query,
            candidates=candidates,
            top_n=top_n,
            config=self.reranker,
            degraded=degraded,
        )

    # ========================================================
    # 回显
    # ========================================================

    def effective(self) -> EffectiveRetrieveProcessor:
        """
        本次实际生效的配置，原样回显给调用方。

        调用方（外部 LLM）靠它知道这一轮到底用了什么参数，
        从而决定下一轮要不要带 override 再查。
        """

        return EffectiveRetrieveProcessor(
            source=self.source,
            fusioner=self.fusioner,
            filter=self.filter,
            reranker=self.reranker,
            cutter=self.cutter,
        )

    # ========================================================
    # 解析
    # ========================================================

    @classmethod
    def _from_definition(
            cls,
            definition: DocumentTypeDefinition,
    ) -> "RetrieveProcessor":
        """
        DTD 的扁平字段 -> 四段配置。

        DTD 只提供 fusion / rerank 两段；filter 与 cutter 用
        系统默认（DTD 表里没有它们的列）。
        """

        return cls(
            source=RetrieveProcessorSource
            .DOCUMENT_TYPE_DEFINITION,
            fusioner=_merge(
                cls.default_fusioner,
                definition.fusion_config,
                strategy=definition.fusion_strategy,
                strategy_field="type",
                kind="fusion",
            ),
            filter=cls.default_filter,
            reranker=_merge(
                cls.default_reranker,
                definition.rerank_config,
                strategy=definition.rerank_strategy,
                strategy_field="type",
                kind="rerank",
            ),
            cutter=cls.default_cutter,
        )

    def _with_override(
            self,
            override: RetrieveProcessorOverride | None,
    ) -> "RetrieveProcessor":
        """
        叠加请求级 override，返回新实例。

        只作用于显式传入的字段 —— 见模块 docstring。
        """

        if override is None:
            return self

        return RetrieveProcessor(
            source=self.source,
            fusioner=_apply_override(
                self.fusioner, override.fusioner
            ),
            filter=_apply_override(
                self.filter, override.filter
            ),
            reranker=_apply_override(
                self.reranker, override.reranker
            ),
            cutter=_apply_override(
                self.cutter, override.cutter
            ),
            registry=self._registry,
        )

    @staticmethod
    def _resolve_source(
            *,
            base: "RetrieveProcessor",
            final: "RetrieveProcessor",
    ) -> RetrieveProcessorSource:
        """
        规格：runtime_override 表示"请求带了至少一项**真正改变了
        生效配置**的 override"。
        """

        changed = (
            final.fusioner != base.fusioner
            or final.filter != base.filter
            or final.reranker != base.reranker
            or final.cutter != base.cutter
        )

        if changed:
            return RetrieveProcessorSource.RUNTIME_OVERRIDE

        return base.source


# ============================================================
# 合并工具
# ============================================================


def _apply_override(
        base: Any,
        patch: Any,
):
    """
    把 patch 里**显式传入**的字段盖到 base 上。

    不能用 patch 的当前值整体替换：pydantic 已经把未传字段填成了
    默认值，全量替换会把"继承"变成"重置"。
    """

    if patch is None:
        return base

    explicit = patch.model_fields_set

    if not explicit:
        return base

    return base.model_copy(
        update={
            name: getattr(patch, name)
            for name in explicit
        }
    )


def _merge(
        base: Any,
        raw: dict | None,
        *,
        strategy: str | None,
        strategy_field: str,
        kind: str,
):
    """
    把 DTD 的 (strategy, config) 扁平字段并到 base 配置上。

    strategy 是字符串列（例如 "rrf"），config 是自由 JSONB。
    两处都可能为空，也可能过时（枚举成员被删过），因此：

        - 认不出的 strategy / 未知的 config 键 -> 记 warning 后忽略，
          不让整条查询 500。历史 DTD 行不该因为枚举收敛而不可查。
        - 认得出但**取值非法**（例如 top_k 超界）-> 抛 ValueError，
          这是配置写错了，静默降级会让错误一直藏着。
    """

    updates: dict[str, Any] = {}

    if strategy:
        try:
            updates[strategy_field] = _STRATEGY_ENUMS[
                kind
            ](strategy)

        except ValueError:
            logger.warning(
                "DTD 的 %s_strategy=%r 不是已知取值，已忽略。"
                "已知取值：%s",
                kind,
                strategy,
                [member.value for member in _STRATEGY_ENUMS[kind]],
            )

    known = set(type(base).model_fields)

    for key, value in (raw or {}).items():

        if key not in known:
            logger.warning(
                "DTD 的 %s_config 含未知键 %r，已忽略。"
                "可用键：%s",
                kind,
                key,
                sorted(known),
            )
            continue

        updates[key] = value

    if not updates:
        return base

    try:
        # 走 model_validate 而不是 model_copy：model_copy 不做校验，
        # 越界的 top_k / 非法的 score_threshold 会被原样带进管线。
        return type(base).model_validate(
            {**base.model_dump(), **updates}
        )

    except ValidationError as exc:
        raise ValueError(
            f"DocumentTypeDefinition 的 {kind} 配置非法：{exc}"
        ) from exc


_STRATEGY_ENUMS = {
    "fusion": FusionerType,
    "rerank": RerankerType,
}
