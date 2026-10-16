"""
业务级文档检索的对外契约（POST /documents/retrieve）。

规格见 app/services/document/document_retrieve.yaml。

与 processing.py 的内部契约不同
-------------------------------
processing.py 里的 IndexSearchRequest / FusionRequest / RerankRequest
是**服务内部**的 DTO：调用方不构造它们，也不知道有几个 Index 参与。

本模块是**对外**契约：调用方只说"查哪几个 DocumentTypeDefinition、
查什么文本"，不指定 Elasticsearch / Milvus / retrieval_type /
hybrid=True/False —— 那些由 DTD 的配置决定。

一次调用 = 一轮检索
-------------------
本契约刻意**没有**分页、没有游标、没有"继续检索"的入参。
一次调用就是一轮，服务不会自动扩召回、重试检索或启动
agent loop —— 什么时候再查一次由调用方（外部 LLM）决定。

查询改写是唯一一个"发生在检索内部"的动作，所以它必须由调用方
**显式**指定（`query_rewriter`，不传就不改），并且改了什么要看得出来
（响应里的 `query_text` / `query_rewrite`）。

Recaller 不在这里
-----------------
规格明确：Recaller 是内部实现细节，不通过公开 API 暴露。
因此 RetrieveProcessorOverride 只有 fusioner / filter / reranker /
cutter 四段，没有 recaller —— 这不是遗漏。

字段命名与规格一一对应（snake_case 直接上线，本项目不做别名转换）。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from .common import APIModel
from .enum import (
    FusionerType,
    QueryRewriterKind,
    RetrievalType,
    RetrieveProcessorMode,
    RetrieveProcessorSource,
    RetrieveScope,
    RerankerType,
)


# ============================================================
# RetrieveProcessor 四段配置
# ============================================================


class FusionerConfig(APIModel):
    """
    多路召回结果的融合配置。

    调用方显式给定的字段会覆盖生效配置里的同名字段；
    未给定的字段继承（见 document_retrieve.yaml 的
    "runtime override precedence"）。
    """

    enabled: bool = Field(
        default=True,
        description="关闭后跳过融合，各路结果原样进入下一阶段。",
    )

    type: FusionerType = Field(
        default=FusionerType.RRF,
        description="融合算法。",
    )

    top_k: int = Field(
        default=50,
        ge=1,
        le=5000,
        description="融合后保留的候选数。",
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
        description=(
            "RRF 的排名平滑常数。"
            "只有 type=rrf 时有意义。"
        ),
    )

    weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "weighted_sum 的各路权重。"
            "键是服务端定义的召回来源标识 —— 即结果里的 "
            "retrieval_source（index_identifier）。"
            "type=weighted_sum 且未给权重时各路等权。"
        ),
    )


class FilterConfig(APIModel):
    """
    召回结果的过滤条件。

    过滤能力本身没有 DTD 级配置 —— DTD 表里只有 fusion / rerank
    两段（见 DocumentTypeDefinition.fusion_strategy 等），
    filter 的默认值由 RetrieveProcessor 的类默认值给出。
    """

    enabled: bool = Field(
        default=True,
        description="关闭后跳过过滤。",
    )

    score_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "保留的最低归一化分数。"
            "None 表示不按分数过滤。"
        ),
    )

    document_ids: list[UUID] | None = Field(
        default=None,
        description="只保留这些 Document 的结果。",
    )

    document_version_ids: list[UUID] | None = Field(
        default=None,
        description="只保留这些 DocumentVersion 的结果。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "服务端支持的 metadata 精确匹配条件。"
            "空 dict 表示不按 metadata 过滤。"
        ),
    )


class RerankerConfig(APIModel):
    """
    融合之后的精排配置。

    注意 top_n 与 cutter.top_k 的区别：
        top_n 是**送进**精排的候选数；
        top_k 是精排**之后**最终返回的数（在 CutterConfig 上）。
    """

    enabled: bool = Field(
        default=True,
        description="关闭后跳过精排。",
    )

    type: RerankerType | None = Field(
        default=None,
        description=(
            "精排算法。None 表示沿用 DTD 的 rerank_strategy；"
            "两者都为空时不做精排。"
        ),
    )

    model: str | None = Field(
        default=None,
        description=(
            "精排模型标识。"
            "None 表示沿用生效配置里的 model。"
        ),
    )

    top_n: int | None = Field(
        default=None,
        ge=1,
        le=5000,
        description=(
            "送进精排的候选数。"
            "None 表示沿用生效配置里的 top_n。"
        ),
    )


class QueryRewriterConfig(APIModel):
    """
    召回**之前**的查询改写配置。

    与 `RerankerConfig` 那几个不同，这一层发生在召回之前：改写后的文本
    同时喂给两路召回（dense 的向量化、keyword 的 BM25）与之后那四段。

    **不传（或 kind 为 None）= 不改写** —— 这里没有"沿用 DTD 配置"的
    回落，因为改写会改变 query 本身，不该由文档类型悄悄决定。评测侧把
    它做成阶梯的逐轮参数。
    """

    enabled: bool = Field(
        default=True,
        description="关闭后跳过改写（保留字段是为了让调用方能显式停用）。",
    )

    kind: QueryRewriterKind | None = Field(
        default=None,
        description=(
            "改写策略。None = 不改写。"
            "\n\n"
            "`common` 不调模型（日期归一化 + 两种写法展开），"
            "`llm` 调一次模型按提示词改写。"
        ),
    )

    model: str | None = Field(
        default=None,
        description=(
            "改写模型标识，只对 `llm` 有意义。"
            "None 表示用该 model_role 绑定里的模型。"
        ),
    )

    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=60_000,
        description=(
            "改写这一步的墙钟上限（毫秒）。"
            "\n\n"
            "**None = 不额外设限**，交给该 `model_role` 绑定 Profile 自己的"
            "超时/重试策略（见 model.yaml 的 profiles）。显式给值才会套一层"
            "更紧的上限 —— 那一层会**盖掉** profile 里的值（含它的重试预算）。"
            "\n\n"
            "超时**不算失败**：退回原文并标 `degraded`，检索照常进行。"
        ),
    )


class QueryRewriteInput(APIModel):
    """
    调用方**已经算好**的改写结果。

    为什么需要它：改写要调模型，而一次评测的某一轮里有几十条 case ——
    逐条调就是几十次，而每次都要重付一遍那 500 多 token 的系统提示词。
    **评测那一层知道"这一轮有哪些题"**，所以由它一次改完，再把每条的结果
    交回给那次检索（检索本身只看得到一条 query）。

    给了它就不再调改写器（`query_rewriter` 会被忽略）—— 两者同时生效会有
    一个静默失效，与 `attempts` / `query_processor_override` 是同一条口径。

    ⚠️ 三个字段必须**一起**给：只给 `text` 的话短语加权与日期键就没了
    （见 `RewriteResult` 那两个字段的说明），而少了它们只是"排序变差"，
    不会报错。
    """

    text: str = Field(
        description="改写后的查询文本（实际发给检索的那句）。",
    )

    boost_phrases: list[str] = Field(
        default_factory=list,
        description="位置相邻加权的片段（`RewriteResult.boost_phrases`）。",
    )

    date_keys: list[str] = Field(
        default_factory=list,
        description=(
            "日期键（补零 `YYYYMMDD`，`RewriteResult.date_keys`）。"
        ),
    )


class CutterConfig(APIModel):
    """
    最终结果的裁剪配置。

    与 FilterConfig 同理，裁剪没有 DTD 级配置，
    默认值由 RetrieveProcessor 的类默认值给出。
    """

    top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="最终返回的结果条数上限。",
    )

    max_content_chars: int | None = Field(
        default=None,
        ge=1,
        le=200000,
        description=(
            "每条结果返回的最大正文字符数。"
            "None 表示不截断。"
        ),
    )


# ============================================================
# 召回参数（融合之前，每路一份）
# ============================================================


class RetrievalRecallConfig(APIModel):
    """
    单路**召回**的参数。融合之前的阶段。

    为什么是请求级、而不是写进 DTD
    ------------------------------
    document_retrieve.yaml 里写明 "Recaller is an internal implementation
    detail and is not exposed through the public API"，RetrieveProcessorOverride
    也刻意只有 fusioner / filter / reranker / cutter 四段 —— 那四段全部发生在
    召回**之后**。

    这四段参数决定的是"在已经召回的池子里怎么排"，本类决定的恰恰是
    "池子本身有多大、有多干净"。两件事，所以不塞进那个 override。

    未出现的字段一律 None = 用默认；未出现的**路**用该路的默认。
    """

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description=(
            "这一路召回多少条。None = 用默认（200）。"
            "\n\n"
            "这是**候选池**大小，不是最终返回条数（那是 cutter.top_k）。"
            "池子太窄时语义高度相关但排序靠后的 chunk 根本进不来，"
            "后面的融合 / 精排 / 裁剪再准也救不回来。"
            "\n\n"
            "上限 500 与 IndexSearchRequest.top_k 的 le=500 对齐。"
        ),
    )

    min_score: float | None = Field(
        default=None,
        description=(
            "这一路的分数下限，**按原始分**（不做归一化）。"
            "None = 不过滤。"
            "\n\n"
            "**语义随 retrieval_type 而变**，因为各路原始分的量纲不同：\n"
            "    keyword        BM25，无上界（几十是常态）\n"
            "    sparse_vector  稀疏点积，无界\n"
            "    dense_vector   余弦，[-1, 1] 有绝对语义\n"
            "所以同一个数值在不同路上的效果完全不一样，必须分路给。"
            "\n\n"
            "**必须下推到索引层**（ES 的 min_score / Milvus 的 radius），"
            "不能在召回后过滤：每路召回后会立刻做**按最大值**归一化"
            "（normalize_scores 是 score / max(score)），归一化之后这一路"
            "的头名恒为 1.0 —— 拿归一化分做阈值永远滤不掉任何东西。"
        ),
    )

    search_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "该路索引后端的检索参数，原样透传。"
            "\n\n"
            "**当前只有 dense_vector（Milvus）消费**，例如 HNSW 的 ef / "
            "nprobe。其余后端没有对应概念，传了会被忽略。"
            "\n\n"
            "注意它和 top_k 是两回事：ef 太小会**真的丢邻居** —— "
            "top_k 设 200 而 ef 只有 10，拿到的也不是真 top 200。"
            "\n\n"
            "metric_type 不接受覆盖：它是建表时定死的物理事实，由后端"
            "自己填（见 index/milvus.py 的说明）。"
        ),
    )

    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=60_000,
        description=(
            "这一路召回的独立超时（毫秒）。None = 不设，交给后端自己的"
            "超时。"
            "\n\n"
            "超时只让**这一路**进 degraded_routes，其余路照常返回 —— "
            "与后端报错时的处理一致：其余路召回的内容是真实可检索的，"
            "没道理让整个查询连带失败。"
        ),
    )


class RetrieveProcessorOverride(APIModel):
    """
    本次请求对 RetrieveProcessor 的临时覆盖。

    只对当前这一次调用生效，不写回 DTD。
    未出现的段整体继承生效配置。
    """

    fusioner: FusionerConfig | None = None

    filter: FilterConfig | None = None

    reranker: RerankerConfig | None = None

    cutter: CutterConfig | None = None


# ============================================================
# 请求
# ============================================================


class DocumentRetrieveRequest(APIModel):
    """
    一次检索轮次的请求。

    document_type_definition_ids 的个数决定装配路径：
        1 个   -> 该 DTD 的 RetrieveProcessor
        >1 个  -> 各 DTD 各自处理，再过 COMMON_RETRIEVE_PROCESSOR_CHAIN
    """

    document_type_definition_ids: list[UUID] = Field(
        min_length=1,
        max_length=50,
        description=(
            "参与本次检索的业务文档类型定义。"
            "至少一个，最多 50 个，且不可重复。"
        ),
    )

    input_text: str = Field(
        min_length=1,
        max_length=10_000,
        description="自然语言查询文本。",
    )

    only_active: bool = Field(
        default=True,
        description=(
            "是否只检索**当前生效版本**。默认 true。"
            "\n\n"
            "为 false 时会把历史版本（superseded / pending 等）的 chunk "
            "也一并召回 —— 排查'某个旧版本里到底写了什么'时才需要。"
            "\n\n"
            "过滤在**索引层**做，不是召回后回表再剔：索引里同时存着"
            "旧版本与新版本的 chunk（reprocess 只增不删），回表过滤会"
            "让 top_k 被陈旧版本吃掉，返回条数少得莫名其妙。"
        ),
    )

    retrieval_types: list[RetrievalType] | None = Field(
        default=None,
        min_length=1,
        description=(
            "只召回这几路。None（默认）= 该 DTD 配置的全部路。"
            "\n\n"
            "**取交集**：传进来的集合只用来**筛**该 DTD 实际配了的 "
            "retrieval，不会凭空多出几路。所以传了文档里没有的类型不是\n"
            "错误，只是那一项被筛掉 —— 想要的路全都没有时结果是空，\n"
            "而不是 400。"
            "\n\n"
            "**空数组会被拒绝**（422）而不是当成'全部'：None 已经表达了\n"
            "'全部'，再让 [] 也表达同一个意思就有两种写法；而按交集语义\n"
            "[] 本来就该是'一路都不要'，那是调用方几乎不会想要的结果。"
            "拒绝它比猜它的意图安全。"
            "\n\n"
            "定向召回之后，融合看到的就是**变少的那几路** —— 被筛掉的\n"
            "路不会贡献分数，也不会出现在 pipeline_keys 里。"
        ),
    )

    retrieval_config: dict[
        RetrievalType,
        RetrievalRecallConfig,
    ] | None = Field(
        default=None,
        description=(
            "按检索路给召回参数。键是 retrieval_type，值是那一路的参数。"
            "没给的路、没给的字段都用默认。"
            "\n\n"
            "与 retrieval_types 的分工：那个决定**跑哪几路**，这个决定"
            "**跑起来多大**。"
            "\n\n"
            "**键必须是合法的 retrieval_type**，拼错（如 dense_vectorr）"
            "直接 422 而不是被忽略 —— 配置写错却静默不生效是最难查的"
            "那类问题。这与 retrieval_types 的'取交集'宽松语义不同："
            "那里传了文档没配的类型只是被筛掉，因为那是**筛选**语义；"
            "这里是**配置**语义，键对不上就是写错了。"
            "\n\n"
            "合法但本次 retrieval_types 没选中的路，其配置只是闲置，"
            "不算错。"
        ),
    )

    retrieve_processor: RetrieveProcessorOverride | None = Field(
        default=None,
        description=(
            "可选的运行时覆盖。"
            "只影响本次调用，不修改 DTD。"
        ),
    )

    # 召回**之前**的那一步，与上面四段不同层 —— 所以是并列字段，
    # 不塞进 `RetrieveProcessorOverride`（那四段全发生在召回之后）。
    query_rewriter: QueryRewriterConfig | None = Field(
        default=None,
        description=(
            "可选的查询改写。**不传 = 不改写**。"
            "\n\n"
            "改写发生在召回之前，改后的文本同时喂给两路召回与之后四段；"
            "这一轮实际发出的文本会随响应回带（`query_text`），"
            "评测侧把它落库 —— 否则事后答不上「这条命中的是哪句话」。"
            "\n\n"
            "与 `query_rewrite` 二选一：给了那个（调用方已经算好的）就用"
            "那个，本字段被忽略。"
        ),
    )

    query_rewrite: QueryRewriteInput | None = Field(
        default=None,
        description=(
            "调用方**已经算好**的改写结果 —— 用于批量改写（见 "
            "`QueryRewriteInput` 的说明）。给了就不再调改写器。"
        ),
    )

    # ========================================================
    # 逐阶段痕迹（诊断用）
    # ========================================================

    include_stage_trace: bool = Field(
        default=False,
        description=(
            "是否采集逐阶段痕迹。**默认关**。"
            "\n\n"
            "关着的时候这条链路与从前**逐字节相同** —— 不建 collector、"
            "不做任何集合运算。开着才付那份开销。"
            "\n\n"
            "开启后每个阶段（各路召回、融合、过滤、精排、裁剪）会记下："
            "进出条数、**进出各 top-N 候选及其分数**，以及被盯住的 chunk "
            "在该阶段的排名/分数/是否被丢/丢的原因。"
            "\n\n"
            "⚠️ 它**不影响** `results` 一个字节。collector 是纯观察者。"
        ),
    )

    target_chunk_ids: list[UUID] | None = Field(
        default=None,
        max_length=50,
        description=(
            "额外**盯住**的 chunk。它们会被并进每个阶段的候选行，"
            "即使掉到第 200 名也保证有记录（带当时的排名与分数）。"
            "\n\n"
            "用途是回答「**这颗目标 chunk 在哪一段被丢的**」—— 评测侧会"
            "把一条 case 的金标集合填在这里。"
            "\n\n"
            "⚠️ 与 `include_stage_trace` 的关系：给了这个就必须同时打开"
            "那个开关（否则 422）。**不做静默打开** —— 与"
            "`document_type_definition_ids` 拒绝重复、`retrieval_types` "
            "拒绝空数组同一口径：静默补默认值会让调用方以为自己在调参，"
            "其实有一个没生效。"
            "\n\n"
            "⚠️ 上限 50：它是被逐阶段扫描的，不设上限时一个调用方塞一万个 "
            "id 就能把每次检索变成一次全表比对。"
        ),
    )

    @model_validator(mode="after")
    def _trace_requires_the_switch(self) -> "DocumentRetrieveRequest":
        """
        采集开关的两条前置约束。

        拒绝而不是静默降级：静默补默认值会让调用方以为自己在调参，而其实
        有一个没生效（仓库既有口径，见本类 `_unique_ids` 与
        `retrieval_types` 的空数组处理）。
        """

        if self.target_chunk_ids and not self.include_stage_trace:

            raise ValueError(
                "target_chunk_ids 给了值，但 include_stage_trace 是 false ——"
                "没有采集开关就没有地方承载这些 id。请同时置 "
                "include_stage_trace=true（不静默打开是刻意的）。"
            )

        # ⚠️ 多 DTD 不支持采集
        # ------------------
        # 多 DTD 时装配路径是"每个 DTD 各跑一遍 RetrieveProcessor，再过
        # 公共链"，也就是 `RetrieveProcessor.run` 会被调 **N+1 次** ——
        # 每次都产出 fusioner / filter / reranker / cutter 四段，于是
        # `stage_key` 撞成 N+1 组同名阶段，而"这一行的融合是哪一次融合"
        # 没有答案。
        #
        # 要做对得给每个阶段再带一个 DTD 维度（表上也要加列），那是另一件
        # 事。而**评测侧永远不需要**它 —— 设计里写死一条 case 只打一个
        # DTD。所以这里拒绝，而不是产出一堆分不清来源的行。
        if self.include_stage_trace and len(
                self.document_type_definition_ids
        ) > 1:

            raise ValueError(
                "include_stage_trace 目前不支持同时检索多个 "
                "DocumentTypeDefinition：多 DTD 会把 RetrieveProcessor "
                "跑 N+1 次，各段痕迹的 stage_key 会撞且无法区分来源。"
                "请一次只传一个 DTD（评测侧本来就是一条 case 一个 DTD）。"
            )

        return self

    @field_validator("document_type_definition_ids")
    @classmethod
    def _unique_ids(
            cls,
            value: list[UUID],
    ) -> list[UUID]:
        """
        规格要求 uniqueItems。

        这里选择**拒绝**而不是静默去重：重复 ID 几乎总是调用方
        拼装参数时的 bug，静默去重会把它藏起来。
        """

        if len(set(value)) != len(value):
            raise ValueError(
                "document_type_definition_ids 不能包含重复项。"
            )

        return value


# ============================================================
# 响应
# ============================================================


class DocumentRetrieveResult(APIModel):
    """
    一条最终结果。

    content 与 document_id / document_type_definition_id 不在
    召回结果里 —— 外层 IndexSearchResult 只回 chunk_id / score /
    retrieval_source / metadata，因此这些字段由服务在裁剪后
    回表补齐（见 DocumentRetrieveServiceImpl._hydrate）。
    """

    rank: int = Field(
        ge=1,
        description="最终排名，从 1 开始。",
    )

    score: float = Field(
        description="该结果在当前阶段下的分数。",
    )

    document_chunk_id: UUID

    document_version_id: UUID

    document_id: UUID

    document_type_definition_id: UUID

    pipeline_keys: list[str] = Field(
        default_factory=list,
        description=(
            "**命中过这条结果**的全部业务检索方向，取自 "
            "DocumentRetrievalDefinition.retrieval_type"
            "（keyword / dense_vector / sparse_vector / graph …）。"
            "\n\n"
            "是数组而不是单值：同一条 chunk 被多路同时命中是融合最有价值"
            "的产出 —— 各路的分数尺度不可比（BM25 是几十、余弦是 0..1），"
            "但「都被选中」是可比的。只报一路等于把这个信号丢掉。"
            "\n\n"
            "仅供调用方理解结果来源，不是可回传的检索参数。"
        ),
    )

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    highlights: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "**按路分组**的索引侧高亮片段：`{pipeline_key: [片段, ...]}`，"
            "键与 pipeline_keys 同一个取值域（retrieval_type）。"
            "\n\n"
            "刻意不做成扁平列表：多路命中时各路的片段回答的是不同问题"
            "（'关键词命中在哪' vs '稠密召回借词面标出了什么'），"
            "合并成一个列表就再也分不出哪段是谁标的 —— 前端要按路切换"
            "展示时就无从下手。"
            "\n\n"
            "只收录**产生了片段**的路。一条 chunk 被哪几路命中看 "
            "pipeline_keys；这里回答的是'那几路各自标出了什么'。"
            "\n\n"
            "标不出来是正常的，不代表没命中：ES 的 sparse_vector 查询"
            "不经过高亮器；Milvus 走的是**词面**高亮，未必对应"
            "'向量为什么命中'。"
        ),
    )



class DegradedRoute(APIModel):
    """
    本轮**没能参与**的一路召回。

    多路召回里一路的后端挂了（Milvus 在恢复、ES 超时……）不该让整个
    查询失败 —— 其余路召回的内容是**真实可检索**的。但静默跳过会让
    调用方分不清"没召回到"和"后端挂了"，所以这里如实列出。

    degraded_routes 为空 = 所有路都正常参与了。
    """

    document_retrieval_definition_id: UUID

    retrieval_type: RetrievalType = Field(
        description="该路原本要走的检索方向，便于定位是哪条后端。",
    )

    reason: str = Field(
        description=(
            "该路没参与的原因（后端异常信息，或'尚未接通'）。"
            "会同时打一条 warning 日志 —— 响应里给出是为了**不静默**。"
        ),
    )


class EffectiveRetrieveProcessor(APIModel):
    """
    本次调用**实际生效**的 RetrieveProcessor，原样回显。

    回显是刻意的：调用方（外部 LLM）由此知道这一轮到底用了什么
    融合/过滤/精排参数，从而决定下一轮是否要带 override 再查。
    """

    source: RetrieveProcessorSource = Field(
        description="生效配置的来源。",
    )

    fusioner: FusionerConfig | None = None

    filter: FilterConfig | None = None

    reranker: RerankerConfig | None = None

    cutter: CutterConfig | None = None


# ============================================================
# 逐阶段痕迹
# ============================================================
#
# 这三个模型**住在 retrieve 这一侧**（而不是从 eval 的 schemas 里 import）：
# 痕迹是检索管线自己的产物，让 retrieve 去依赖 eval 的 DTO 是反向依赖 ——
# 与"不让检索链路去查 bundle 拿金标"是同一条理由。
#
# eval 侧另有一套**同名同义**的 DTO（StageTraceItem / ScoredChunkCandidate），
# 那是**落库并聚合之后**的视图（多了 gold_present_before/after 这类汇总）。
# 两者的 `stages[].candidates[]` 字段名与含义一致，前端一套渲染能吃两种来源。


class TracedChunkCandidate(APIModel):
    """一个候选在某个阶段边界上的状态。"""

    chunk_id: UUID

    rank: int = Field(
        ge=1,
        description="1-based 排名。**顺序即排名**（候选列表本身没有 rank 字段）。",
    )

    score: float | None = Field(
        default=None,
        description=(
            "该阶段的分数。**归一化到 [0, 1]、higher-is-better** —— "
            "管线里流动的分数始终如此（召回后按路归一化、融合后归一化、"
            "精排后归一化）。"
            "\n\n"
            "⚠️ 即便如此也不要跨阶段比：归一化是'除以本阶段最大值'，"
            "两段的 0.9 基数不同。"
        ),
    )

    source_key: str | None = Field(
        default=None,
        description="这个名字属于哪一路召回（融合之后为空）。",
    )

    selected: bool = Field(
        default=True,
        description="这一阶段是否保留了它。false 即「死在这一段」。",
    )

    drop_reason: str | None = Field(
        default=None,
        description=(
            "被丢的原因（受控词表，见 domain/schemas/enum.py 的 "
            "EvalDropReason）。selected=true 时为空。"
        ),
    )

    is_target: bool = Field(
        default=False,
        description=(
            "是否在请求的 `target_chunk_ids` 里。"
            "被盯住的候选**无条件保留一行**（即使它的排名远在 top-N 之外）"
            "—— 这正是「这颗目标 chunk 在哪被丢」能回答的原因。"
        ),
    )


class RetrieveStageTrace(APIModel):
    """一个阶段边界上的进出。"""

    stage_key: str = Field(
        description=(
            "阶段标识。真实阶段名，不另造一套："
            "`recall.<retrieval_type>`（每路一个）、`fusioner`、"
            "`filter`、`reranker`、`cutter`。"
        ),
    )

    source_key: str = Field(
        default="",
        description=(
            "并行分支标识（哪一路召回）。非分支阶段用空串 —— "
            "空串能直接进唯一约束，NULL 在唯一约束里不相等。"
        ),
    )

    executed: bool = Field(
        default=True,
        description=(
            "这一阶段本次是否真的跑了。"
            "\n\n"
            "`false` 指「配置关掉了 / 没接通」（比如 DTD 没配精排、filter "
            "被 enabled=false 跳过）。它与「跑了但没召回」是两回事："
            "后者是**检索质量**，前者是**能力缺口** —— 混在一起时，"
            "一个没配精排的 DTD 看起来会像「精排把好结果排没了」。"
        ),
    )

    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)

    candidates: list[TracedChunkCandidate] = Field(
        default_factory=list,
        description=(
            "**top-N ∪ 被盯住的**候选，按该阶段结束时的顺序。"
            "\n\n"
            "这是「被阈值砍掉」与「排序本来就错」唯一能分清的地方："
            "看到 rank=11 / score=0.29 而阈值 0.3，一眼看得出。"
        ),
    )


class RetrieveTrace(APIModel):
    """
    一次检索的逐阶段痕迹。

    它回答两件事：

        **这颗目标 chunk 在哪一段被丢的** —— 看 `is_target=true` 的行，
        找到 `selected=false` 的那一段
        **回来的都是什么、各多少分** —— 看 top-N 的行。逆向提问时
        "别的 chunk 里也有正确答案"就靠它识别
    """

    target_chunk_ids: list[UUID] = Field(
        default_factory=list,
        description="本次盯住的 chunk。空表示只采了阶段漏斗、没有指定目标。",
    )

    stages: list[RetrieveStageTrace] = Field(default_factory=list)

    notes: list[str] = Field(
        default_factory=list,
        description=(
            "采集的**已知边界**。目前只有一条：被 `min_score`（下推到"
            "索引层的过滤）砍掉的候选我们**从没见过**，因此不会出现在"
            "任何一行里 —— 它们在痕迹看来就是「没召回到」。"
            "\n\n"
            "放在响应里而不是只写在注释里：不写的话，使用者会把"
            "「没出现在痕迹里」读成「这轮没有被过滤掉的东西」。"
        ),
    )


class QueryRewriteSummary(APIModel):
    """
    这一次改写做了什么。

    存在的理由：改写发生在**服务内部**，而需要"这次到底发了什么"的是
    **服务之外**的调用方 —— 评测侧要把那一句话落进结果行（见
    `retrieval_eval_case_result.query_text`），不回声它就永远填不上。
    """

    kind: QueryRewriterKind | None = Field(
        default=None,
        description="实际用的改写策略。None 表示没改写（没传或显式关掉）。",
    )

    changed: bool = Field(
        default=False,
        description="改后的文本是否与原文不同。",
    )

    degraded: bool = Field(
        default=False,
        description=(
            "改写失败、已退回原文。**不算检索失败** —— 所以调用方要"
            "分得清「没改写」与「想改写但没改成」。"
        ),
    )

    detail: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "改写器自己报的细节（命中的日期、错误原因等）。"
            "形状随 kind 变，调用方不要依赖它做判断。"
        ),
    )


class DocumentRetrieveResponse(APIModel):
    """
    一次检索轮次的结果。

    刻意没有 total / 分页字段：这是"一轮检索"而不是"一个可分页
    的资源集合"，返回条数由 cutter.top_k 决定。
    """

    retrieve_id: UUID = Field(
        description="本次调用的标识，用于日志与链路追踪对齐。",
    )

    query_text: str | None = Field(
        default=None,
        description=(
            "本次**实际发给检索**的文本（改写之后）。"
            "\n\n"
            "没改写时等于请求里的 `input_text`。它必须回带：评测侧"
            "拿它落库，否则「这条命中的是哪句话」事后无从回答 ——"
            "而 LLM 改写是不可复现的。"
        ),
    )

    query_rewrite: QueryRewriteSummary | None = Field(
        default=None,
        description="这次改写的摘要。None = 这轮没开改写。",
    )

    retrieve_scope: RetrieveScope

    retrieve_processor_mode: RetrieveProcessorMode

    document_type_definition_ids: list[UUID]

    results: list[DocumentRetrieveResult]

    degraded_routes: list[DegradedRoute] = Field(
        default_factory=list,
        description=(
            "本轮没能参与的路。为空表示全部正常。"
            "\n\n"
            "**调用方应当检查它**：非空意味着 results 只来自剩余那几路，"
            "可能不全 —— 而不是'库里就这些'。"
        ),
    )

    retrieve_processor: EffectiveRetrieveProcessor | None = None

    trace: RetrieveTrace | None = Field(
        default=None,
        description=(
            "逐阶段痕迹。**只有请求里 `include_stage_trace=true` 时才有值**"
            "（否则是 null）。"
            "\n\n"
            "它与 `results` 完全独立：`results` 不受开关影响，"
            "一个字节都不会变。"
        ),
    )


__all__ = [
    "CutterConfig",
    "DegradedRoute",
    "DocumentRetrieveRequest",
    "DocumentRetrieveResponse",
    "DocumentRetrieveResult",
    "EffectiveRetrieveProcessor",
    "FilterConfig",
    "FusionerConfig",
    "QueryRewriteInput",
    "QueryRewriteSummary",
    "QueryRewriterConfig",
    "RerankerConfig",
    "RetrieveProcessorOverride",
    "RetrieveStageTrace",
    "RetrieveTrace",
    "TracedChunkCandidate",
]
