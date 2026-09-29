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

__all__ = ["RetrieveProcessor"]

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
    ) -> list[RetrieveCandidate]:
        """
        跑一遍 融合 -> 过滤 -> 精排 -> 裁剪。

        result_sets 是"每路一个列表"（多 DTD 场景下是"每个 DTD
        一个列表"）。返回已按分数降序、已裁剪的最终结果。
        """

        candidates = await self._fuse(query, result_sets)

        if self.filter.enabled:

            candidates = await self._registry.filter.apply(
                candidates=candidates,
                config=self.filter,
            )

        candidates = await self._rerank(query, candidates)

        return await self._registry.cutter.cut(
            candidates=candidates,
            config=self.cutter,
        )

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
