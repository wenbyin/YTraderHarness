"""
查询期四阶段契约（Fusioner / Filter / Reranker / Cutter）。

角色划分对应 document_retrieve.yaml 的 RetrieveProcessorOverride：
四段都可以被运行时覆盖，**Recaller 不在其中** —— 它是内部实现
细节，不通过公开 API 暴露，也不在这里。

与 policy/base.py 的关系（待收敛）
----------------------------------
policy/base.py 里还有两个并列的 FusionStrategy / RerankStrategy：
**未实现、全项目无引用**，且接收的 IndexSearchResult 只有
chunk_id / score / retrieval_source / metadata，**没有正文** ——
而精排恰恰需要正文。

查询期因此改用本模块的 RetrieveCandidate 与 Fusioner / Reranker。
两者目前并存，属于历史遗留：policy/base.py 的那两个应当删除，
否则"融合实现在哪"有两个答案（policy/embedder.py 的注释也早就
提醒过不要重复定义它们）。本次未动它，是因为它不在本次改动
授权范围内 —— 建议后续单独清理。

分数的不变量（重要）
--------------------
管线内流动的 RetrieveCandidate.score **始终是归一化到 [0, 1] 的分数**。

为什么必须归一化：规格里的 filter.score_threshold 取值域是
[0, 1] 且描述为 "Minimum normalized score"，响应示例的分数
也是 0.9421 / 0.9317 这种量级。而底层召回给的是 ES BM25 原始分
（可能到几十）和 RRF 的 1/(k+rank)（约 0.016）—— 两者都落不进
[0, 1]，不归一化的话 score_threshold=0.3 会把结果全部滤掉。

因此每一段的产出一律归一化（见 normalize_scores）：
召回后按路归一化，融合后归一化，精排后归一化。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..domain.schemas import (
    CutterConfig,
    FilterConfig,
    FusionerConfig,
    RerankerConfig,
)
from ..domain.schemas.enum import FusionerType, RerankerType

__all__ = [
    "Cutter",
    "Fusioner",
    "RetrieveCandidate",
    "Reranker",
    "ResultFilter",
    "merge_highlights",
    "merge_metadata",
    "merge_routes",
    "normalize_scores",
]


class RetrieveCandidate(BaseModel):
    """
    管线内流动的一条候选结果。

    与 IndexSearchResult 的分工：
        IndexSearchResult 是**单路召回**的产物，只有 chunk_id /
        score / retrieval_source / metadata。
        RetrieveCandidate 是多路融合后的统一形态，额外带上回表补齐的
        正文与归属（content / document_* ），因为过滤要看归属、
        精排要看正文。

    可变的：融合会重写 score，回表会回填 content 与归属字段。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ---- 召回阶段就有 ----

    document_chunk_id: UUID

    score: float = Field(
        description="归一化到 [0, 1] 的分数（见模块 docstring）。",
    )

    retrieval_source: str = Field(
        description=(
            "产出该候选的召回来源，取自 index_identifier。"
            "多路融合时靠它区分结果来自哪，"
            "也是 weighted_sum 的 weights 的键。"
        ),
    )

    pipeline_key: str | None = Field(
        default=None,
        description=(
            "业务检索方向，取自 "
            "DocumentRetrievalDefinition.retrieval_type。"
            "\n\n"
            "融合之后它只是**代表值**（首次命中的那一路）—— 要看这条"
            "总共被哪几路命中，读 pipeline_keys。"
        ),
    )

    pipeline_keys: list[str] = Field(
        default_factory=list,
        description=(
            "**命中过这条**的全部业务检索方向（去重，保持首次出现顺序）。"
            "\n\n"
            "融合之前，每条候选只来自一路，这里就是它自己那一路；"
            "融合之后同一条被多路命中时会被合并成一条，这里列出全部 ——"
            "「被几路同时命中」本身就是最强的相关性信号，丢掉它等于把"
            "融合白做了（各路的分数尺度不可比，但'都被选中'是可比的）。"
            "\n\n"
            "融合产出以它为准；pipeline_key 只作代表值保留。"
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    highlights: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "**按路分组**的索引侧高亮片段：`{pipeline_key: [片段, ...]}`。"
            "\n\n"
            "刻意不做成扁平列表：多路命中时各路的片段回答的是不同问题"
            "（'关键词命中在哪' vs '稠密召回借词面标出了什么'），"
            "混在一起就再也分不出哪段是谁标的。前端要按路展示，"
            "拿到扁平列表只能猜。"
            "\n\n"
            "只收录**产生了片段**的路 —— 没产生的不出现在这里。"
            "要知道一条 chunk 总共被哪几路命中，读 pipeline_keys。"
        ),
    )


    # ---- 回表补齐 ----
    #
    # PostgreSQL 是正文与归属的唯一事实来源：召回结果里没有它们
    # （见 index/es.py 的 search —— 只回 chunk_id / score /
    # retrieval_source / metadata）。

    content: str | None = None

    document_version_id: UUID | None = None

    document_id: UUID | None = None

    document_type_definition_id: UUID | None = None

    def normalized(
            self,
            score: float,
    ) -> "RetrieveCandidate":
        """产出一份改了 score 的副本，不改原对象。"""

        return self.model_copy(update={"score": score})

    def fused(
            self,
            score: float,
            pipeline_keys: list[str],
            highlights: dict[str, list[str]],
            metadata: dict[str, Any] | None = None,
    ) -> "RetrieveCandidate":
        """融合产出的副本：新分数 + **命中过它的全部路** + 各路的高亮。"""

        return self.model_copy(update={
            "score": score,
            "pipeline_keys": list(pipeline_keys),
            # 深拷一层：accumulator 是逐轮累加出来的，直接挂上去会让
            # 后续融合的写入漏进已经产出的候选里。
            "highlights": {
                route: list(fragments)
                for route, fragments in highlights.items()
            },
            # 合并后的 metadata：留首次那一路的会让 dense 先到时丢掉
            # ES 那份（Milvus 根本不回 metadata）。见 merge_metadata。
            "metadata": dict(
                self.metadata if metadata is None else metadata
            ),
        })

    def routes(self) -> list[str]:
        """
        本候选来自哪几路（去重、保序）。

        融合过的候选身上已经有整份列表（多 DTD 时公共链拿到的是各 DTD
        融合后的结果，一条可能已带多路），所以优先读它；没融合过的候选
        只有单数的 pipeline_key。
        """

        if self.pipeline_keys:
            return list(self.pipeline_keys)

        return [self.pipeline_key] if self.pipeline_key else []


def merge_routes(
        accumulator: dict,
        candidate: "RetrieveCandidate",
) -> None:
    """
    把候选的来源路并进 `{chunk_id: [路, ...]}`，去重且保持首次出现顺序。

    两个 Fusioner 共用这一份 —— 各写一遍迟早会在"多路命中同一块"
    这件事上给出不同答案，而那正是融合唯一的产出。
    """

    routes = accumulator.setdefault(
        candidate.document_chunk_id, []
    )

    for route in candidate.routes():
        if route not in routes:
            routes.append(route)


def merge_metadata(
        accumulator: dict,
        candidate: "RetrieveCandidate",
) -> None:
    """
    把候选的 metadata 并进 `{chunk_id: {...}}`。

    为什么必须合并而不是"留首次命中那一路的"
    ----------------------------------------
    各路返回的 metadata **内容不同**，而且有的路根本没有：

        ES      存并回 `metadata`（阶段 8 继承的 title / section /
                company_code 都在这里）
        Milvus  只回 _OUTPUT_FIELDS 那几个标量 ——
                collection schema 里**没有 metadata 这一列**

    于是"首次命中哪一路"会决定 metadata 有没有内容。dense 先被融合到，
    ES 那份带 title / section 的 metadata 就被悄悄丢掉了 ——
    表现是响应里查不到继承下来的字段，而索引里明明有。

    后来的路**填补**先到的没覆盖的键，不反向覆盖已有的：
    先到那一路先占住键，后面的只补空缺，避免两路给出同一键时结果
    取决于遍历顺序。
    """

    merged = accumulator.setdefault(
        candidate.document_chunk_id, {}
    )

    for key, value in (candidate.metadata or {}).items():
        if key not in merged:
            merged[key] = value


def merge_highlights(
        accumulator: dict,
        candidate: "RetrieveCandidate",
) -> None:
    """
    把候选的高亮片段并进 `{chunk_id: {路: [片段, ...]}}`。

    与 merge_highlights 吃的是同一批数据，区别只有一个：**留着路**。
    扁平那份是对外的主字段（合成一个列表就够用来在结果页上标命中），
    但「这一段是哪一路标出来的」在扁平化的那一刻就没了 —— 想按路
    切换高亮（点某一枚路径标签、只描那一路的命中）就无从下手。

    一条候选可能身兼多路（多 DTD 的公共链会拿到**已被融合过**的候选，
    见 RetrieveCandidate.routes）。这时它身上的 highlights 本身已经是
    合并结果、按字面拆不回各路，于是退回「整份记给它的每一路」——
    宁可多标，不可丢标。新构造的候选永远只来自一路，走不到这条分支。
    """

    if not candidate.highlights:
        return

    by_route = accumulator.setdefault(
        candidate.document_chunk_id, {}
    )

    for route, fragments in candidate.highlights.items():
        bucket = by_route.setdefault(route, [])
        for fragment in fragments:
            if fragment not in bucket:
                bucket.append(fragment)


def normalize_scores(
        candidates: list[RetrieveCandidate],
) -> list[RetrieveCandidate]:
    """
    把一组候选的分数线性缩放到 [0, 1]（按最大值）。

    按最大值而不是 min-max：min-max 会把每组的最低分压成 0，
    丢掉"整体都很相关"与"整体都不相关"的区别；除以最大值则
    保留了相对形状。

    最大值 <= 0 时原样返回 —— 全是 0 或出现负分（不该发生，
    但别在这里制造 NaN）。
    """

    if not candidates:
        return []

    top = max(candidate.score for candidate in candidates)

    if top <= 0:
        return list(candidates)

    return [
        candidate.normalized(candidate.score / top)
        for candidate in candidates
    ]


# ============================================================
# 融合
# ============================================================


class Fusioner(ABC):
    """
    多路召回结果的融合契约。

    实现按 FusionerType 注册（见 registry.py）。
    """

    #: 注册表按它索引；不声明的实现不会被注册。
    fusioner_type: ClassVar[FusionerType | None] = None

    @abstractmethod
    async def fuse(
            self,
            *,
            query: str,
            result_sets: list[list[RetrieveCandidate]],
            top_k: int,
            config: FusionerConfig,
    ) -> list[RetrieveCandidate]:
        """
        合并多路候选，产出**已按分数降序**的单一列表。

        返回的分数必须归一化到 [0, 1]（见模块 docstring）。

        result_sets 是"每路一个列表"，路的顺序与身份由
        RetrieveCandidate.retrieval_source 标识，不靠下标。
        空列表要能正常处理 —— 某一路召回为空是常态。
        """
        raise NotImplementedError


# ============================================================
# 过滤
# ============================================================


class ResultFilter(ABC):
    """
    候选过滤契约。

    规格里 FilterConfig **没有 type 字段** —— 过滤只有一种，
    因此不按名字注册，实现唯一（见 filter.py）。
    """

    @abstractmethod
    async def apply(
            self,
            *,
            candidates: list[RetrieveCandidate],
            config: FilterConfig,
    ) -> list[RetrieveCandidate]:
        """
        按配置裁剪候选，保持原有顺序。

        注意这里做的是**业务条件**过滤（分数阈值、归属、
        metadata）。召回阶段的 document_version_id 过滤在
        IndexBackend 里（走索引过滤），两者不冲突。
        """
        raise NotImplementedError


# ============================================================
# 精排
# ============================================================


class Reranker(ABC):
    """
    融合之后的精排契约。

    实现按 RerankerType 注册（见 registry.py）。

    这里用**复数** reranker_types，与 Fusioner 的单数
    fusioner_type 不同：一个融合算法只服务一个类型，而一个
    精排实现可能覆盖多个类型（当前的占位实现就同时服务
    cross_encoder / llm / custom —— 见 rerank.py 的说明）。
    """

    #: 本实现服务哪些 RerankerType。空元组表示不注册。
    reranker_types: ClassVar[tuple[RerankerType, ...]] = ()

    @abstractmethod
    async def rerank(
            self,
            *,
            query: str,
            candidates: list[RetrieveCandidate],
            top_n: int,
            config: RerankerConfig,
    ) -> list[RetrieveCandidate]:
        """
        重新打分并产出**已按分数降序**的列表，最多 top_n 条。

        返回的分数必须归一化到 [0, 1]。

        实现可以假设候选已回表补齐（content 非空）——
        精排要看正文，因此管线把回表排在精排之前。
        """
        raise NotImplementedError


# ============================================================
# 裁剪
# ============================================================


class Cutter(ABC):
    """
    最终结果裁剪契约。

    与 ResultFilter 同理，规格的 CutterConfig 没有 type 字段，
    实现唯一（见 cut.py）。
    """

    @abstractmethod
    async def cut(
            self,
            *,
            candidates: list[RetrieveCandidate],
            config: CutterConfig,
    ) -> list[RetrieveCandidate]:
        """
        截断到 config.top_k 条（并可选地截断正文长度）。

        入参已按分数降序，实现只需取前 N 条 —— 不要在这里重排。
        """
        raise NotImplementedError
