"""
查询期契约（QueryRewriter / Fusioner / Filter / Reranker / Cutter）。

角色划分对应 document_retrieve.yaml 的 RetrieveProcessorOverride：
四段都可以被运行时覆盖，**Recaller 不在其中** —— 它是内部实现
细节，不通过公开 API 暴露，也不在这里。

QueryRewriter 是后加的第一个**召回之前**的环节，所以它不走
RetrieveProcessorOverride（那四段全在召回之后），而是
`DocumentRetrieveRequest.query_rewriter` 上的并列字段。

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
from dataclasses import dataclass, field
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..domain.schemas import (
    CutterConfig,
    FilterConfig,
    FusionerConfig,
    QueryRewriterConfig,
    RerankerConfig,
)
from ..domain.schemas.enum import (
    FusionerType,
    QueryRewriterKind,
    RerankerType,
)

__all__ = [
    "Cutter",
    "Fusioner",
    "QueryRewriter",
    "Reranker",
    "ResultFilter",
    "RetrieveCandidate",
    "RewriteResult",
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
# 查询改写（召回之前）
# ============================================================


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """
    一次改写的产物。

    刻意用 dataclass 而不是 APIModel：它是**流水线内部**的产物，像
    RetrieveCandidate 一样只在管线里流动；对外的形状由
    `QueryRewriteSummary` 那个 DTO 负责，两者不共用类型（共用会让
    "内部多一个字段"变成"公开契约多一个字段"）。
    """

    #: 改写后的文本。**没改动时就是原文**。
    text: str

    #: 文本是否真的变了。`False` 与"没开改写"是两件事，所以要分开报。
    changed: bool = False

    #: 改写失败、已退回原文。检索照常进行，但调用方要看得出这件事。
    degraded: bool = False

    #: 应当被**短语加权**的片段（关键词那一路）。
    #:
    #: 与 `text` 是两条不同的路：
    #:     text            换个说法**喂给两路**（dense 的向量、BM25 的查询串）
    #:     boost_phrases   只对**关键词**那一路，把这几段按"位置相邻"加权
    #:
    #: 为什么需要它：查询串里多几个词只是让 BM25 多几个词项，而它们会与
    #: 其余词项一起被 `minimum_should_match` 摊薄 —— 日期这种**唯一的
    #: 判别词**因此推不动排序。`match_phrase` 是位置相邻匹配，不受那条
    #: 比例约束。实测（同一份语料、同一批 case）：把日期当短语加权，
    #: keyword 名次 140→4 / 123→11 / 48→6 / 18→8。
    #:
    #: ⚠️ 只对**能表达短语查询**的后端有意义（当前是 ES）。向量那一路
    #: 无视它 —— 所以它治不了"dense 没召回"，那是另一回事。
    #:
    #: 由改写器给（它才知道哪一段是判别词），索引后端只负责应用。
    boost_phrases: tuple[str, ...] = ()

    #: 应当参与**精确筛选/加权**的日期键（`2007年4月12日` / `2007-04-12`
    #: 都是 `2007412`，形态见 `services/document/dates.py`）。
    #:
    #: 与 `boost_phrases` 的分工：
    #:     boost_phrases   在**正文**里按位置相邻找这些串（拼给人的写法）
    #:     date_keys       与索引期算好的 `content_dates` 字段**同一形态**，
    #:                     所以"查询写中文、正文写短横线"也能对上 ——
    #:                     短语匹配做不到这一点（它比的是字符串本身）
    #:
    #: 由改写器给，索引后端按自己的能力应用（ES 打 `content_dates` 字段、
    #: Milvus 做标量筛选）；认不了的后端忽略。
    date_keys: tuple[str, ...] = ()

    #: 改写器自己报的细节（命中的日期、错误原因…）。形状随 kind 变。
    detail: dict[str, Any] = field(default_factory=dict)


class QueryRewriter(ABC):
    """
    召回之前的查询改写契约。

    实现按 QueryRewriterKind 注册（见 registry.py）。

    ⚠️ **契约：本方法永不抛异常。**
    改写是"锦上添花"，它失败不该让一次检索失败 —— 任何异常、超时、
    解析不了的回包，都要**自己吞掉**并返回原文 + `degraded=True`。
    调用方（`DocumentRetrieveServiceImpl`）另有一层兜底，但那层只是
    保险，不是让实现偷懒的理由：抛出去的代价是整次检索 500，而正确
    的表现是"这次没改写，检索照常"。
    """

    #: 本实现服务哪些 kind。空元组表示不注册。
    rewriter_kinds: ClassVar[tuple[QueryRewriterKind, ...]] = ()

    @abstractmethod
    async def rewrite(
            self,
            *,
            query: str,
            config: QueryRewriterConfig,
    ) -> RewriteResult:
        """
        改写 `query`，返回新文本。

        实现要保证**幂等**：把改写后的文本再喂进来，结果应当与上一次
        相同（`changed=False`）。否则重试/续跑会不断往 query 上叠东西，
        而那种劣化在结果里看不出来。
        """
        raise NotImplementedError

    async def rewrite_many(
            self,
            *,
            queries: list[str],
            config: QueryRewriterConfig,
    ) -> list[RewriteResult]:
        """
        **一批**查询一起改写。返回与入参**等长、同序**的结果。

        默认实现就是逐条 `rewrite()` —— 对 `common` 这种不调模型的实现，
        批量没有意义（它就是几个正则）。调模型的实现应当覆盖它：一次
        调用改一批，省掉每条都重付的固定开销（系统提示词就是 500 多
        token）。

        ⚠️ 覆盖它的实现**必须自己保证"条数与顺序"**：批量最危险的失败
        不是出错，是**串号** —— 把甲的改写安到乙头上时，每条读起来都
        合理。所以调用方（评测那一层）只按位置取结果，而"位置对不对"
        只能由实现自己校验（它知道模型回了什么）。校验不过就**整批
        退化成"不改"**，或回落到逐条 —— 两者都比串号好。
        """
        return [
            await self.rewrite(query=query, config=config)
            for query in queries
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
            dropped: dict[UUID, str] | None = None,
    ) -> list[RetrieveCandidate]:
        """
        按配置裁剪候选，保持原有顺序。

        注意这里做的是**业务条件**过滤（分数阈值、归属、
        metadata）。召回阶段的 document_version_id 过滤在
        IndexBackend 里（走索引过滤），两者不冲突。

        `dropped` —— 为什么淘汰，由**实现自己**回答
        ----------------------------------------
        可选出口。传了的话，实现把 `{chunk_id: 淘汰原因}` 填进去；不传
        就与从前逐字节相同。

        ⚠️ **必须由实现填，不能让调用方按 config 反猜。** 过滤有四种
        条件（分数阈值 / document_ids / document_version_ids /
        metadata），只看「它不在了」分不出是哪一个；而按 config 反猜等于
        把过滤逻辑抄了第二份 —— 两份一旦不一致，`drop_reason` 会指向一个
        其实没生效的阈值，且不报错。

        原因取值的受控词表见 `domain/schemas/enum.py` 的
        `EvalDropReason`（`score_threshold` / `metadata_filter` /
        `ownership_filter`）。
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
            degraded: dict[str, str] | None = None,
    ) -> list[RetrieveCandidate]:
        """
        重新打分并产出**已按分数降序**的列表。

        返回的分数必须归一化到 [0, 1]。

        ⚠️ **不是"最多 top_n 条"，而是"原样长度"**：精排只改变顺序，
        最终条数由 `cutter.top_k` 决定。`top_n` 是**送进**精排的条数
        （见 RerankerConfig 的说明）—— 拿它截断输出会让精排变成第二次
        裁剪，排在 top_n 之后的候选再也回不来。

        实现可以假设候选已回表补齐（content 非空）——
        精排要看正文，因此管线把回表排在精排之前。

        `degraded` —— 这一段到底跑没跑，由**实现自己**回答
        --------------------------------------------------
        可选出口，形制照 `ResultFilter.apply(dropped=...)`：传了的话，
        实现在"没能真跑"（第三方服务不可用、超时、回包解析不了）时写入
        `{"reason": ...}`；不传就与从前逐字节相同。

        ⚠️ **必须由实现填，不能让调用方按异常反猜。** 降级有两个完全
        不同的来源（本地模型不可用 / 远端不可用），只有实现分得清；而
        调用方拿到的是一个已经排好序的列表 —— 它看起来与成功一模一样。
        调用方据它把这一段标成 `executed=False`（"能力缺口"），而不是
        "跑了但没改动结果"（"检索质量"）。
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
