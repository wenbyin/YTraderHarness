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
一次调用就是一轮，服务不会自动扩召回、改写查询、重试检索或启动
agent loop —— 什么时候再查一次由调用方（外部 LLM）决定。

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

from pydantic import Field, field_validator

from .common import APIModel
from .enum import (
    FusionerType,
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


class DocumentRetrieveResponse(APIModel):
    """
    一次检索轮次的结果。

    刻意没有 total / 分页字段：这是"一轮检索"而不是"一个可分页
    的资源集合"，返回条数由 cutter.top_k 决定。
    """

    retrieve_id: UUID = Field(
        description="本次调用的标识，用于日志与链路追踪对齐。",
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


__all__ = [
    "CutterConfig",
    "DegradedRoute",
    "DocumentRetrieveRequest",
    "DocumentRetrieveResponse",
    "DocumentRetrieveResult",
    "EffectiveRetrieveProcessor",
    "FilterConfig",
    "FusionerConfig",
    "RetrieveProcessorOverride",
    "RerankerConfig",
]
