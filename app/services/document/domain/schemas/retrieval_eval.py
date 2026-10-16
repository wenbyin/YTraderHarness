"""
第一层检索评测（Retrieval Eval）的请求与响应。

契约权威在 `rag_eval_design/retrieval_eval_openapi.yaml` —— 本文件是它的
Pydantic 表达，字段名与约束一一对应。

这一层的两句话
--------------
**造题用 LLM，跑评测不用。** 所以这里既有「让模型出题」的请求
（`CreateEvalBundleRequest` 带着 questions_per_chunk / chunks_per_window /
max_input_tokens），也有「纯 ID 比对」的响应（`EvalMetrics` 里没有任何
生成质量指标）。

**一条 case 只打一个 DTD。** 不指定 DTD 的场景（让 LLM 自己路由走哪个
DTD）属于 agentic_rag，不在这一层 —— 所以 `document_type_definition_ids`
可以有多个（抽样范围），而 `EvalCase` 上的那个是单值。

几个刻意的取舍
--------------
`document_type_definition_ids` **拒绝重复而不是静默去重**：重复 ID 几乎
总是调用方 bug（复制粘贴时漏改），静默去重会让这个 bug 永远不暴露。这与
`DocumentRetrieveRequest` 同一口径。

`EvalCase.answer_original` 是**源 chunk 的逐字片段**，不是模型生成的答案
—— 理由见 models/sqlalchemy/retrieval_eval.py 里 case 表的说明。

`EvalMetrics` 里 `hit_rate_at_5` 可以是 NULL：max_k < 5 时它没有意义，
而「没有意义」必须与「是 0」分开（0 会被读成「top5 一个都没中」）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from .common import APIModel
from .enum import (
    EvalBundleStatus,
    EvalCaseResultStatus,
    EvalCaseValidationStatus,
    EvalJobStatus,
    EvalSamplingStrategy,
    EvalScoreDirection,
)

# 运行时覆盖四段配置 —— **直接复用检索那边的定义**。
#
# 刻意不在这里另立一份：那份 override 的语义（未显式传的字段继承，
# 只认 model_fields_set）与检索侧的实施方式是绑死的，抄一份出来必然
# 漂移 —— 而漂移的表现是「评测时改了参数但没生效」，极难归因。
from .retrieve import (
    CutterConfig,
    FusionerConfig,
    QueryRewriterConfig,
    RerankerConfig,
    RetrievalRecallConfig,
    RetrieveProcessorOverride,
)
from .enum import QueryRewriterKind, RetrievalType, RerankerType

__all__ = [
    "DEFAULT_ATTEMPT_LADDER",
    "DEFAULT_CHUNKS_PER_WINDOW",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_K",
    "DEFAULT_QUESTIONS_PER_CHUNK",
    "DEFAULT_TOP_CANDIDATES_PER_STAGE",
    "QWEN_RERANK_MODEL",
    "CreateEvalBundleAcceptedResponse",
    "CreateEvalBundleRequest",
    "CreateEvalRunAcceptedResponse",
    "CreateEvalRunRequest",
    "EvalAttemptSpec",
    "EvalBundleDetail",
    "EvalBundleExport",
    "EvalBundleExportItem",
    "EvalBundleListResponse",
    "EvalBundleSummary",
    "EvalCase",
    "EvalCaseListResponse",
    "EvalCaseTraceResponse",
    "EvalMetrics",
    "EvalRunCase",
    "EvalRunCaseListResponse",
    "EvalRunDetail",
    "GenerationRetryAcceptedResponse",
    "GenerationRunDetail",
    "GenerationWindowSummary",
    "ScoredChunkCandidate",
    "StageStatItem",
    "StageStatsResponse",
    "StageTraceItem",
]


# ============================================================
# 默认值
# ============================================================
#
# 与 retrieval_eval_openapi.yaml 的 default 一致。集中在这里而不是散在
# 各 Field 上：服务层要把它们展开进 config_snapshot（数据集必须记下
# 「当时用的到底是几」），散着写就会出现「接口说默认 3、快照里记了
# 别的」。

#: 每个源 chunk 出几道题。
DEFAULT_QUESTIONS_PER_CHUNK = 3

#: 一次 LLM 调用最多放几个源 chunk。
DEFAULT_CHUNKS_PER_WINDOW = 5

#: 一次 LLM 调用的输入 token 预算上限。
DEFAULT_MAX_INPUT_TOKENS = 12_000

#: 评测默认取前几条判定命中。
DEFAULT_MAX_K = 10

#: 每个阶段保留几条候选（诊断用）。
#:
#: ⚠️ 这个值与 `retrieve/trace.py` 的 `DEFAULT_TOP_N` **是同一个量**，而
#: 它们曾经不一致（这里 5、那边 10）—— 后果是 trace 产 21 条、DTO 卡在 5，
#: 真跑一轮才炸 `List should have at most 5 items`。
#: 改一处必须改另一处（或者干脆只留一处）。
DEFAULT_TOP_CANDIDATES_PER_STAGE = 10


# ============================================================
# 造 bundle
# ============================================================


class CreateEvalBundleRequest(APIModel):
    """
    建一份 QA 数据集并启动造题。

    三个数量词的关系（最容易混的地方）：

        sampled_chunk_count (N)   总共抽多少个源 chunk（跨 DTD 合计）
        questions_per_chunk (M)   每个 chunk 出几道题
        chunks_per_window   (L)   一次 LLM 调用最多放几个 chunk

    所以 LLM 调用次数约等于 N / L，而产出上限约等于 N × M。
    **L 是受上下文窗口所限的那个数** —— 它同时受 token 预算约束，
    两者取更紧的那个（见服务层的窗口切分）。
    """

    name: str = Field(
        min_length=1,
        max_length=255,
        description="数据集名称。",
    )

    description: str | None = Field(
        default=None,
        max_length=10_000,
        description="数据集说明。",
    )

    document_type_definition_ids: list[UUID] = Field(
        min_length=1,
        max_length=50,
        description=(
            "抽样范围：从这些 DTD 的文档里抽 chunk。"
            "**至少一个，最多 50 个，不可重复**（重复会 422 而不是被"
            "静默去重 —— 重复 ID 几乎总是调用方 bug）。"
            "\n\n"
            "注意这与评测时的调用是两件事：评测时每条 case 只打"
            "**它自己那一个** DTD，不会把这几个 DTD 一起传进去。"
        ),
    )

    sampled_chunk_count: int = Field(
        ge=1,
        le=100_000,
        description=(
            "总共抽多少个源 chunk。**是跨全部选中 DTD 的合计，不是"
            "每个 DTD 抽这么多**。"
        ),
    )

    questions_per_chunk: int = Field(
        default=DEFAULT_QUESTIONS_PER_CHUNK,
        ge=1,
        le=20,
        description=(
            "每个源 chunk 期望出几道题。"
            "\n\n"
            "是**期望值不是保证值**：chunk 里凑不出这么多道互不相同的"
            "明确事实时，模型应当少出并标注 —— 硬凑出来的题是数据集里"
            "的噪音。所以最终题数可能少于 N × M，那个差额会体现在"
            "bundle 状态与窗口统计上，而不是被抹平。"
        ),
    )

    chunks_per_window: int = Field(
        default=DEFAULT_CHUNKS_PER_WINDOW,
        ge=1,
        le=50,
        description=(
            "一次 LLM 调用最多放几个源 chunk（L）。"
            "\n\n"
            "**这是拆分请求的那个数**：受限于模型的上下文窗口，"
            "N 个 chunk 不可能一次丢进去，所以按 L 个一组分批。"
            "实际每组的数量还会被 max_input_tokens 进一步压小 ——"
            "两个约束取更紧的那个。"
        ),
    )

    max_input_tokens: int = Field(
        default=DEFAULT_MAX_INPUT_TOKENS,
        ge=1_000,
        le=1_000_000,
        description=(
            "单次调用的输入 token 预算。"
            "\n\n"
            "**必须真的有这个约束**：只按 chunk 个数分组是不够的 ——"
            "不同 chunk 的字符数差一个数量级（表格段几千字、事件段"
            "几十字），按个数分组会让某一批直接把上下文撑爆，"
            "而症状是「这个窗口每次都失败」，看起来像模型的问题。"
            "\n\n"
            "预算里要给指令与期望输出**留余量**，服务层会再打一个折扣。"
        ),
    )

    enable_cross_chunk_judge: bool = Field(
        default=False,
        description=(
            "是否启用**跨 chunk 判金标**（调第二个模型，判定「别的块是否也"
            "答得上这道题」）。**默认关闭。**"
            "\n\n"
            "关着的时候连模型角色都不解析 —— 也就是说 judge 的绑定没配好"
            "**不会**影响造题，它根本不会被碰到。"
            "\n\n"
            "关着会失去什么：每道题的金标集合就只剩**出题那一块**"
            "（`gold_chunk_ids == {source_chunk_id}`）。于是「检索返回了"
            "另一块也能答这个问题的 chunk」会被判成 miss —— 那是**冤枉**，"
            "它确实答得上。这类假 miss 让召回率偏低，而偏低的部分从数据上"
            "看不出来。"
            "\n\n"
            "⚠️ 代价与收益取决于语料形状：候选只取**同一份文档**里的其它块"
            "（另一家公司的文档答不了「关于这家公司」的问题）。随机跨公司"
            "抽样时同文档候选通常为空 —— 那种情况下开着它也是零调用、"
            "零收益；真正点亮它的是「一份文档贡献多块」。"
            "\n\n"
            "这个值会写进数据集的 `generation_config` 快照 —— 它**改变数据集"
            "内容**，所以必须能事后分辨某份数据集是从哪种模式产出的。"
        ),
    )

    sampling_strategy: EvalSamplingStrategy = Field(
        default=EvalSamplingStrategy.BALANCED_BY_DTD,
        description=(
            "多个 DTD 时怎么分配这 N 个名额。"
            "样本数悬殊时 uniform 会让大的那个 DTD 吃掉绝大部分题目，"
            "而它的检索表现未必代表其余 DTD。"
        ),
    )

    seed: int | None = Field(
        default=None,
        ge=0,
        le=9_223_372_036_854_775_807,
        description=(
            "抽样种子。**不传时由服务端生成并存下来** ——"
            "存下来才有「同种子 + 同语料 = 同一批 chunk」这个可验证的"
            "性质，否则调参时无法分辨「这次结果不同」是配置变了还是"
            "抽到的样本变了。"
        ),
    )

    @field_validator("document_type_definition_ids")
    @classmethod
    def _reject_duplicate_dtds(
            cls,
            value: list[UUID],
    ) -> list[UUID]:
        """
        重复直接拒绝，不静默去重。

        静默去重的代价：调用方以为抽了 3 个 DTD 的范围，实际只有 2
        个，而数据集本身不会体现这个差别 —— 等到「怎么老是没有 A 类的
        题」被发现时，已经过去很久了。
        """

        if len(set(value)) != len(value):
            raise ValueError(
                "document_type_definition_ids 里存在重复的 DTD ——"
                "重复 ID 几乎总是调用方 bug，请修正后重试"
                "（不静默去重是刻意的）。"
            )

        return value


class CreateEvalBundleAcceptedResponse(APIModel):
    """bundle 已建、造题已入队（202）。"""

    bundle_id: UUID
    generation_run_id: UUID
    status: EvalBundleStatus
    message: str | None = None


# ============================================================
# bundle 查询
# ============================================================


class EvalBundleSummary(APIModel):
    """选择器里的一行。"""

    bundle_id: UUID
    name: str
    description: str | None = None
    status: EvalBundleStatus

    document_type_definition_ids: list[UUID] = Field(
        default_factory=list,
        description="这份数据集覆盖了哪些 DTD（抽样范围）。",
    )

    source_chunk_count: int = Field(
        ge=0,
        description="实际抽中的源 chunk 数（**分母**，不是期望值）。",
    )

    case_count: int = Field(
        ge=0,
        description="校验通过的 QA 条数。",
    )

    latest_generation_status: EvalJobStatus | None = Field(
        default=None,
        description="最近一次造题运行的状态。",
    )

    dataset_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
        description="导出内容指纹。拿它比对两次评测是不是同一份数据。",
    )

    created_at: datetime
    completed_at: datetime | None = None


class EvalBundleListResponse(APIModel):
    items: list[EvalBundleSummary] = Field(default_factory=list)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)


class EvalBundleDetail(EvalBundleSummary):
    """
    bundle 明细。

    比 summary 多出的都是**可复现性**所需的东西：抽样/造题配置与
    schema 版本。少了它们，这份数据集就只能被使用、不能被解释 ——
    而「这批题是怎么来的」恰恰是调参时最常被问到的。
    """

    schema_version: str
    selection_config: dict[str, Any] = Field(default_factory=dict)
    generation_config: dict[str, Any] = Field(default_factory=dict)
    latest_generation_run_id: UUID | None = None
    updated_at: datetime


class EvalCase(APIModel):
    """
    一条 QA（预览用）。

    `chunk_id` 与 `gold_chunk_ids` 的差别是这一层最重要的一个区分：

        chunk_id        出题用的**源** chunk
        gold_chunk_ids  **全部**能回答这个问题的 chunk（含源 chunk）

    后者可能不止一个 —— 逆向造的题（从 chunk 出题）在别的 chunk 里
    也可能有正确答案，而造题时那个 chunk 未必在同一个窗口里（上下文
    窗口所限）。判命中用后者：返回了另一个也能答的 chunk 不该算 miss。
    """

    eval_case_id: UUID
    document_type_definition_id: UUID
    document_version_id: UUID | None = None
    document_retrieval_definition_id: UUID | None = None

    # ---- 源文档身份 ----
    #
    # 出题时把主体（公司名 / 证券代码）从**文档**取来写进问题，因为正文
    # 里常常写的是「本公司」。反过来，问题一旦写了公司名，答案的正确性
    # 就依赖"这块 chunk 属于那家公司" —— 所以这三列是这类题**能否复核**
    # 的前提，必须在数据集里看得到。
    document_id: UUID | None = Field(
        default=None,
        description="源 chunk 所属 Document。",
    )

    document_title: str | None = Field(
        default=None,
        description=(
            "源文档标题，即**主体名称**（本语料里=公司名，如「特锐德」）。"
            "\n\n"
            "复核对得上：问题里写的主体应当就是它。"
        ),
    )

    document_key: str | None = Field(
        default=None,
        description=(
            "源文档的业务标识（本语料里=证券代码，如「300001.SZ」）。"
        ),
    )

    chunk_id: UUID = Field(
        description="出题用的源 chunk（**不可变的金标**）。",
    )

    gold_chunk_ids: list[UUID] = Field(
        min_length=1,
        description=(
            "全部经验证能回答这个问题的 chunk，**必然包含 chunk_id**。"
            "\n\n"
            "⚠️ 这个集合只覆盖**本次抽样到的 chunk 池**，不是全语料 ——"
            "所以它是下界：没命中不等于语料里没有别的答案。"
        ),
    )

    question: str

    answer_original: str = Field(
        description=(
            "源 chunk 里的**逐字片段**，不是模型生成的摘要。"
            "比对位置而非措辞，才能把「没召回到」与「措辞对不上」分开。"
        ),
    )

    answer_start_offset: int | None = Field(default=None, ge=0)
    answer_end_offset: int | None = Field(default=None, ge=0)

    question_order: int = Field(ge=0)
    validation_status: EvalCaseValidationStatus
    validation_error: str | None = None


class EvalCaseListResponse(APIModel):
    items: list[EvalCase] = Field(default_factory=list)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)


class EvalBundleExportItem(APIModel):
    """
    导出物里的一条。外部工具（ragas 等）直接读这个形状。

    ⚠️ 带**源文档身份**（标题 + 业务标识）：本语料里正文常写「本公司」，
    问题里的主体只能来自文档 —— 所以"这道题是哪家公司的"必须随导出物
    一起走，否则外部工具无法复核这条 ground truth 该不该命中。
    """

    chunk_id: UUID = Field(description="源 chunk。")
    gold_chunk_ids: list[UUID] = Field(min_length=1)
    document_type_definition_id: UUID
    document_version_id: UUID | None = None
    document_id: UUID | None = Field(
        default=None, description="源 chunk 所属 Document。",
    )
    document_title: str | None = Field(
        default=None,
        description="源文档标题（本语料里=公司名）。问题里的主体应当是它。",
    )
    document_key: str | None = Field(
        default=None,
        description="源文档业务标识（本语料里=证券代码）。",
    )
    question: str
    answer_original: str


class EvalBundleExport(APIModel):
    """
    确定性导出。

    **确定性**指的是同样的数据导出两次得到字节相同的结果 —— 所以
    条目顺序有固定口径（按 DTD、源 chunk、题序），不能依赖数据库的
    返回顺序。`dataset_sha256` 就是在这个确定性上算出来的：它让
    「两次评测用的是不是同一份数据」变成一个可比较的值。
    """

    schema_version: str
    bundle_id: UUID
    bundle_name: str
    dataset_sha256: str | None = None
    generation_config: dict[str, Any] = Field(default_factory=dict)
    document_type_definition_ids: list[UUID] = Field(default_factory=list)
    items: list[EvalBundleExportItem] = Field(default_factory=list)


# ============================================================
# 造题进度
# ============================================================


class GenerationWindowSummary(APIModel):
    """
    一个窗口的进度。

    窗口是**重试的单位**，所以这里必须能看出「哪个窗口该重试」：
    status + attempt_count + error_code 三者合起来才够判断。
    """

    generation_window_id: UUID | None = None
    window_index: int = Field(ge=0)
    status: EvalJobStatus
    input_chunk_count: int = Field(ge=0)
    input_token_estimate: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(ge=0)
    valid_case_count: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None


class GenerationRunDetail(APIModel):
    generation_run_id: UUID
    bundle_id: UUID
    status: EvalJobStatus
    model_role: str

    model_name: str | None = Field(
        default=None,
        description=(
            "实际解析到的模型名。"
            "与 model_role 分开记：换模型后要重建数据集，而「这批是哪"
            "次模型产出的」只能从这里读。"
        ),
    )

    sample_seed: int
    total_chunks: int = Field(ge=0)
    total_windows: int = Field(ge=0)
    completed_windows: int = Field(ge=0)
    failed_windows: int = Field(ge=0)
    generated_cases: int = Field(ge=0)
    error_message: str | None = None
    windows: list[GenerationWindowSummary] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class GenerationRetryAcceptedResponse(APIModel):
    generation_run_id: UUID
    status: EvalJobStatus
    retried_window_count: int = Field(ge=0)


# ============================================================
# 跑评测
# ============================================================


class EvalAttemptSpec(APIModel):
    """
    **一轮**扩展的参数。

    前端可以一次把三轮都传进来；不传就用内置的三档阶梯（见
    `DEFAULT_ATTEMPT_LADDER`）。

    ⚠️ 三轮是**逐轮放宽**，不是三个独立的实验
    ----------------------------------------
    第 N 轮只跑第 N-1 轮**没命中**的 case，所以每轮的样本量不同、
    分母也不同。报「第 2 轮命中率」时必须带上分母，否则会被读成
    "全体的命中率"而错一个量级。

    ⚠️ 这确实是用 gold 驱动重试 —— 但**每轮分开统计**之后它是正当的：
    得到的是「1 次找到 37% / 3 次找到 52%」，两个数都真实。只有把三轮
    混成一个数才算作弊。
    """

    recall_top_k: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description=(
            "**每一路**召回多少条。None = 用该路的默认（200）。"
            "\n\n"
            "上限 500 与 `IndexSearchRequest.top_k` 的 le=500 对齐。"
        ),
    )

    retrieval_config: dict[RetrievalType, RetrievalRecallConfig] | None = Field(
        default=None,
        description=(
            "逐路细化覆盖（键是 retrieval_type）。给了就盖过 "
            "`recall_top_k` —— 后者是「所有路都设成这个数」的简写。"
        ),
    )

    retrieve_processor: RetrieveProcessorOverride | None = Field(
        default=None,
        description=(
            "这一轮的四段覆盖（融合 / 过滤 / 精排 / 裁剪）。"
            "\n\n"
            "**第 3 轮加精排就是在这里给 `reranker`** —— 精排贵（实测 "
            "`top_n=50` 要 45 秒，`top_n=10` 约 9 秒），所以只让前两轮都"
            "没找到的那批去付。附带好处：第 3 轮的**增量**就是精排的净"
            "收益，不必另开一轮对照实验。"
        ),
    )

    query_rewriter: QueryRewriterConfig | None = Field(
        default=None,
        description=(
            "这一轮的查询改写。**不传 = 这一轮不改写**。"
            "\n\n"
            "与另外两个字段的分工：`recall_top_k` / `retrieval_config` 管"
            "**召回多少**，`retrieve_processor` 管**召回之后怎么处理**，"
            "而它管**查什么** —— 改写发生在召回之前，改后的文本这一轮"
            "两路召回都吃。"
            "\n\n"
            "⚠️ 逐轮给不同的改写器时，轮的**增量**就不再只是那一个变量的"
            "贡献了（例如第 2 轮换 llm 改写、第 3 轮又换回来）。要读"
            "「纯精排收益」就给相邻两轮配同一个改写器。"
        ),
    )

    name: str | None = Field(
        default=None,
        max_length=64,
        description="给这一轮起个名（报表里显示用）。None = 用序号。",
    )


#: **前端默认按这个规约来传** —— 不传 `attempts` 时服务端就用它。
#:
#: 三档是按**实测的失败分布**定标的，不是拍脑袋：
#:
#:     21 条死于 fusioner top_k=50    进融合时金标平均排第 65.5
#:     39 条死于 cutter   top_k=10    进 cutter 时金标平均排第 14.3
#:     15 条两路都没碰到
#:
#: 所以每一档都瞄准上一档暴露出的那个缺口：
#:
#:     round 1  基线：召回 200 / 融合 50 / 裁剪 10，无精排
#:               + common 改写（日期两种写法都展开 —— 不调模型，免费）
#:     round 2  治"死在融合 top_k"那批：融合提到 100、裁剪提到 20
#:               + llm 改写（换一种说法再查一次，这是"换说法"最便宜的实现）
#:     round 3  治"排序不够好"那批：**这时才开精排** —— 它贵（第三方
#:               API，按次计费），只让前两轮都没找到的那批去付。
#:               改写**不传**（= 不改写）：第三轮要读的是精排的增量。
#:
#: ⚠️ 第 3 轮的增量不是"干净的精排 delta"
#: --------------------------------------
#: 第 2 轮用 llm 改写、第 3 轮不用 —— 所以 `round3 - round2` 里既有精排
#: 的收益、也有"少了一次改写"的损失。要读净精排收益，就给第 3 轮也配上
#: 同一个改写器（一行的事），或另跑一轮对照。这里选"不传"是因为第三轮
#: 的输入已经是前两轮都没找到的最难那批，凭改写救回来的希望最小。
#:
#: ⚠️ 改这里 = 改"一轮评测的默认含义"。历史结果的可比性依赖它稳定 ——
#: 整套阶梯会进 `effective_config_snapshot`，所以事后查得出某批数字是
#: 哪套阶梯跑出来的。

#: 默认阶梯第三轮用的精排模型（阿里云 DashScope 的文本精排）。
#:
#: 与 `app/config/rerank.yaml` 里的默认值是同一个量 —— 这里给只是为了让
#: 阶梯快照能自证"这批数字是用哪个模型跑出来的"；真正发请求时以配置里的
#: 为准（配置能改，快照不能）。
QWEN_RERANK_MODEL = "qwen3.7-text-rerank"


DEFAULT_ATTEMPT_LADDER: tuple[EvalAttemptSpec, ...] = (
    EvalAttemptSpec(
        name="round1",
        recall_top_k=200,
        retrieve_processor=RetrieveProcessorOverride(
            # 融合窗口 50 -> 100（实测 +1 条：金标在融合结果里排第 53 名的
            # 那条，50 时被切、100 时留下）。**不取 200**：另外两条更深的
            # 排在 151/155，250 以内都捞不回来，那是第 2/3 轮窗口（100/200）
            # 该管的事 —— 第 1 轮就开到 200 会把阶梯的"逐轮放宽"抹平。
            #
            # 代价：下游（过滤/精排/裁剪）看到的候选从 50 条变 100 条。
            fusioner=FusionerConfig(top_k=100),
            cutter=CutterConfig(top_k=10),
        ),
        query_rewriter=QueryRewriterConfig(
            kind=QueryRewriterKind.COMMON,
        ),
    ),
    EvalAttemptSpec(
        name="round2",
        recall_top_k=400,
        retrieve_processor=RetrieveProcessorOverride(
            fusioner=FusionerConfig(top_k=100),
            cutter=CutterConfig(top_k=20),
        ),
        query_rewriter=QueryRewriterConfig(
            kind=QueryRewriterKind.LLM,
        ),
    ),
    EvalAttemptSpec(
        name="round3",
        recall_top_k=500,
        retrieve_processor=RetrieveProcessorOverride(
            fusioner=FusionerConfig(top_k=200),
            cutter=CutterConfig(top_k=30),
            # 第三方 API 精排。`top_n=50` 与 `_MAX_CANDIDATES` 同值、
            # 高于本轮的 cutter.top_k=30 —— 精排只改顺序，最终条数归裁剪
            # 管（见 rerank.py 的说明）。模型标识写在这里而不是硬编码在
            # 实现里：换模型不改代码。
            reranker=RerankerConfig(
                type=RerankerType.API,
                model=QWEN_RERANK_MODEL,
                top_n=50,
            ),
        ),
    ),
)


class CreateEvalRunRequest(APIModel):
    """
    在固定 bundle 上跑一轮检索。

    这里**没有任何「用不用 LLM」的开关** —— 这一层不用 LLM 判分，
    所以没有可关的东西。
    """

    bundle_id: UUID

    max_k: int = Field(
        default=DEFAULT_MAX_K,
        ge=1,
        le=1000,
        description=(
            "命中判定取前几条。"
            "\n\n"
            "**必须与最终返回条数的口径一致**（cutter.top_k）："
            "max_k 大于实际返回条数时，多出来的部分恒为空 ——"
            "算出来的 hit_rate_at_k 会偏低，而那偏低是配置不一致"
            "造成的，不是检索的问题。"
            "\n\n"
            "服务端的做法是**把返回条数压到不超过 max_k**（`min`）。"
            "⚠️ 所以它**不保证**两者相等：逐轮阶梯的某些轮返回得比 max_k "
            "少时（默认阶梯第 1/2 轮是 10/20 对 50），标签仍写着 at_50、"
            "实际量的是 at_10 —— 不报错。要么把那些轮调宽，要么把这个值"
            "调到不超过最窄那一轮。"
        ),
    )

    include_stage_trace: bool = Field(
        default=True,
        description=(
            "是否采集逐阶段痕迹。"
            "\n\n"
            "关掉之后只剩最终命中结果，「为什么丢了」完全无从回答 ——"
            "宏观丢因统计会退化成整体命中率。只有确实不需要归因、"
            "且要压 trace 开销时才关。"
        ),
    )

    top_candidates_per_stage: int = Field(
        default=DEFAULT_TOP_CANDIDATES_PER_STAGE,
        ge=1,
        le=20,
        description=(
            "每个阶段/分支保留几条候选的分数。"
            "\n\n"
            "**被分数阈值砍掉的金标候选无条件保留**，即使它不在前 N ——"
            "否则「是阈值砍的还是排序错了」这个最常见的问题查不出来。"
        ),
    )

    query_processor_override: RetrieveProcessorOverride | None = Field(
        default=None,
        description=(
            "本次评测对四段处理器的临时覆盖（融合/过滤/精排/裁剪）。"
            "\n\n"
            "⚠️ 只在本轮生效、不写回 DTD。合并时**只认显式传过的字段**："
            "pydantic 会给没传的字段填默认值，直接合并会把「继承 DTD "
            "配置」悄悄变成「重置成类默认值」。"
            "\n\n"
            "**召回本身不可覆盖**（recaller 是内部实现细节），"
            "这与公开检索契约一致。"
            "\n\n"
            "⚠️ 传了 `attempts` 时**以 `attempts` 为准**，这个字段被忽略 —— "
            "两者都表达「本轮用什么参数」，同时给会有一个静默不生效。"
            "（没传 attempts 时它是唯一那份参数。）"
        ),
    )

    attempts: list[EvalAttemptSpec] | None = Field(
        default=None,
        max_length=3,
        description=(
            "**逐轮的扩展参数**，一次可以全传进来。"
            "\n\n"
            "不传 = 用内置的三档阶梯（`DEFAULT_ATTEMPT_LADDER`），"
            "也就是前端**默认按这个规约来传**："
            "\n\n"
            "```\n"
            "round 1   召回 200 / 融合  50 / 裁剪 10   无精排   ← 基线\n"
            "round 2   召回 400 / 融合 100 / 裁剪 20   无精排\n"
            "round 3   召回 500 / 融合 200 / 裁剪 30   精排 top_n=10\n"
            "```\n\n"
            "语义：**第 N 轮只跑第 N-1 轮没命中的 case**。所以每轮的"
            "样本量不同、分母也不同 —— 报「第 2 轮命中率」必须带分母，"
            "否则会被读成「全体的命中率」。"
            "\n\n"
            "⚠️ 这确实是用 gold 驱动重试。**每轮分开统计之后它是正当的**"
            "（得到的是「1 次找到 37% / 3 次找到 52%」，两个数都真实）；"
            "把三轮混成一个数才是作弊。"
            "\n\n"
            "⚠️ 整个阶梯会进 `effective_config_snapshot` —— 它**改变"
            "结果的含义**，「这几个数是几轮跑出来的」必须事后可查。"
        ),
    )


class CreateEvalRunAcceptedResponse(APIModel):
    eval_run_id: UUID
    bundle_id: UUID
    status: EvalJobStatus
    message: str | None = None


class EvalMetrics(APIModel):
    """
    确定性指标。**没有一条来自 LLM 判分。**

    为什么 hit_rate_at_5 可以是 NULL
    -------------------------------
    max_k < 5 时它没有意义。而「没有意义」必须与「命中率是 0」分开 ——
    后者会被读成「top5 一个都没中」，是完全不同的结论。

    为什么 failed 与 miss 分开
    --------------------------
    检索执行失败（后端挂了、超时）会让命中率下降，而那是**故障**不是
    **质量问题**。混在一起时，「检索挂了」看起来就像「召回率变差了」。
    """

    total_cases: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    hit_count: int = Field(ge=0)
    miss_count: int = Field(ge=0)

    hit_rate_at_1: float | None = Field(default=None, ge=0, le=1)
    hit_rate_at_5: float | None = Field(default=None, ge=0, le=1)
    hit_rate_at_k: float | None = Field(default=None, ge=0, le=1)

    mrr_at_k: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "平均倒数排名。多个金标时取**最好的**那个金标的排名 ——"
            "不是第一个碰到的。"
        ),
    )

    by_attempt: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "逐轮指标，键是轮次号（「1」「2」「3」）。"
            "\n\n"
            "⚠️ **每轮的分母不同** —— 第 N 轮只跑第 N-1 轮未命中的那些。"
            "所以每一项都带 `cases`（这一轮跑了几条）与 `input_share`"
            "（占全部 case 的比例）。读比率时必须连分母一起看，"
            "否则会被读成「全体命中率」而错一个量级。"
            "\n\n"
            "这个分离正是「用 gold 驱动重试」能成立的原因："
            "「1 轮找到」「3 轮找到」两个数都是真的，而混成一个就不是。"
        ),
    )

    hit_rate_curve: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "召回率曲线，形如 {「1」: 0.05, 「10」: 0.375, 「50」: 0.7}。"
            "\n\n"
            "存在的理由就是 `max_k` 那件事：**线上到底喂几条给模型是个"
            "部署事实**，而它一变，单个 hit@k 的数就要重跑一轮才知道。"
            "有了曲线，「口径对齐」从**重跑**变成**读另一个点**。"
            "\n\n"
            "⚠️ k 超过 `retrieved_max` 的点没有意义（分子不可能再涨）—— "
            "读到接近 0 时先确认检索到底返回了几条。"
        ),
    )

    retrieved_max: int | None = Field(
        default=None,
        ge=0,
        description=(
            "本轮单条最多返回了几条。**读曲线时必须对照它** —— "
            "`hit_rate_curve` 里超过这个数的 k，其 0 值只说明"
            "「没取回来」，不说明「召回差」。"
        ),
    )

    p50_latency_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    max_k: int = Field(ge=1)


class EvalRunDetail(APIModel):
    eval_run_id: UUID
    bundle_id: UUID
    status: EvalJobStatus
    max_k: int = Field(ge=1)
    include_stage_trace: bool
    query_processor_override: dict[str, Any] = Field(default_factory=dict)
    effective_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    total_cases: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    metrics: EvalMetrics
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ============================================================
# 宏观丢因统计
# ============================================================


class StageStatItem(APIModel):
    """
    一个 (阶段, 分支) 的汇总。

    `gold_drop_count` 的语义被「只归因于真的丢了的阶段」限制住：
    进阶段前金标就不在了的，在这个阶段记 not_applicable，不计入丢弃。
    不这么做的话，dense 一路没召回到会让后面每个阶段各记一次丢弃，
    宏观统计里 filter / rerank 会背上一大堆不是它们造成的锅。
    """

    stage_key: str
    source_key: str = ""
    observed_cases: int = Field(ge=0)
    executed_cases: int = Field(ge=0)
    average_input_count: float | None = Field(default=None, ge=0)
    average_output_count: float | None = Field(default=None, ge=0)
    gold_present_before_count: int = Field(ge=0)
    gold_present_after_count: int = Field(ge=0)
    gold_drop_count: int = Field(ge=0)
    gold_pass_rate: float | None = Field(default=None, ge=0, le=1)

    drop_reason_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "按受控词表（EvalDropReason）分组的丢弃计数。"
            "这是「chunk 为什么被丢」的直接答案。"
        ),
    )


class StageStatsResponse(APIModel):
    eval_run_id: UUID
    stages: list[StageStatItem] = Field(default_factory=list)


# ============================================================
# 单条 case 的痕迹
# ============================================================


class ScoredChunkCandidate(APIModel):
    """
    一个候选 chunk 在某个时刻的排名与分数。

    ⚠️ 实测口径：管线里流动的分数**始终是归一化到 [0,1] 的**
    -------------------------------------------------------
    召回后按路归一化、融合后归一化、精排后归一化（三处都调
    `normalize_scores`，见 retrieve/base.py）。方向一律 higher-is-better。

    所以 `score_name` / `score_direction` 目前**几乎不携带信息** ——
    它们保留着是为了将来记录"下推到索引层的原始分"（`min_score` 那一类
    过滤用的是原始量纲：BM25 可到几十、余弦 [-1,1]），那是唯一需要它们
    的场景。列可空，别为了填满而编一个名字。

    ⚠️ 即便如此也**不要跨阶段比分数**：现在同为 [0,1] 只是因为都被
    "除以本阶段最大值"归一化了，两个阶段的 0.9 基数不同。
    """

    chunk_id: UUID

    rank: int = Field(
        ge=1,
        description="1-based 排名。",
    )

    score: float | None = Field(
        default=None,
        description="该阶段的原始分。**跨阶段不可比。**",
    )

    score_name: str | None = Field(
        default=None,
        description=(
            "分数名，例如 cosine_similarity / bm25 / rrf_score / "
            "reranker_score。没有它，数值无法解释。"
        ),
    )

    score_direction: EvalScoreDirection | None = None

    source_key: str | None = Field(
        default=None,
        description="召回分支 / 阶段来源。",
    )

    selected: bool = True

    drop_reason: str | None = Field(
        default=None,
        description="被丢的原因（受控词表）。",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class StageTraceItem(APIModel):
    """
    一个阶段在这一条 case 上的进出。

    `gold_ranks_before` / `gold_scores_before` 是**按 chunk_id 键的
    map**（而不是单值）：一道题可能有多个金标，各在何处、各得几分
    都要能看见 —— 只报最好的那个会掩盖「另一个金标掉到了第 40 名」
    这类信息。
    """

    stage_key: str
    source_key: str = ""
    input_count: int | None = Field(default=None, ge=0)
    output_count: int | None = Field(default=None, ge=0)
    gold_present_before: bool | None = None
    gold_present_after: bool | None = None
    gold_ranks_before: dict[str, int | None] = Field(default_factory=dict)
    gold_scores_before: dict[str, float | None] = Field(default_factory=dict)
    gold_ranks_after: dict[str, int | None] = Field(default_factory=dict)
    gold_scores_after: dict[str, float | None] = Field(default_factory=dict)
    drop_reason: str | None = None

    top_candidates: list[ScoredChunkCandidate] = Field(
        default_factory=list,
        # ⚠️ 上界不是"每段几条"，而是**一行最多能出现几条**：
        #
        #     存活 top-N ∪ 丢弃 top-N ∪ 被盯住的
        #
        # N 默认 10，"被盯住的"由请求给（上限 50）—— 一段最多 2N+50=70。
        # 取 128 是留余量而不是精确值：**精确值写在这里会在 N 或盯住上限
        # 变动时静默错**，而这个字段只用于防止响应无限膨胀，不表达业务语义。
        #
        # （实测踩过：原本写 5，而 trace 产 21 条 —— 真跑一轮才炸
        # `List should have at most 5 items after validation, not 21`。）
        max_length=128,
        description=(
            "该阶段进出的候选（含分数）：存活 top-N ∪ 丢弃 top-N ∪ "
            "被盯住的那些。"
            "\n\n"
            "这是「分被阈值砍掉」与「排序本来就错」这两件事唯一能分清"
            "的地方：看到 rank=3 / score=0.29 而阈值 0.3，一眼看得出。"
        ),
    )


class EvalCaseTraceResponse(APIModel):
    eval_run_id: UUID
    eval_case_id: UUID

    attempt_no: int = Field(
        ge=1,
        description=(
            "**这份痕迹是哪一轮的**（逐轮放宽，一轮一行）。"
            "\n\n"
            "不传 `attempt_no` 时给的是**代表行**那一轮 —— 与列表、表头"
            "指标同一条（命中优先，再比名次），所以「这条 case 的结果」在"
            "三个页面上是同一个。"
            "\n\n"
            "⚠️ 必须跟着响应走：第 1 轮与第 3 轮的参数不同（第 3 轮才开"
            "精排、召回与裁剪都更宽），不标出来的话这份痕迹会被读成"
            "「这条 case 的检索过程」，而它是「这一轮的检索过程」。"
        ),
    )

    available_attempts: list[int] = Field(
        default_factory=list,
        description=(
            "这一轮评测里这条 case **实际跑过的轮次**，升序。"
            "\n\n"
            "逐轮放宽的语义是「第 N 轮只跑第 N-1 轮没命中的 case」——"
            "所以第 1 轮就命中的 case 只有 `[1]`，而三轮都没命中的是"
            "`[1, 2, 3]`。想逐轮看，就拿这里的值去传 `attempt_no`。"
        ),
    )

    query_text: str | None = Field(
        default=None,
        description=(
            "这一轮**实际发给检索**的文本（改写之后）。"
            "\n\n"
            "与 case 的 `question` 可能不同：逐轮阶梯里每一轮的改写器不同"
            "（第 1 轮日期归一化、第 2 轮大模型改写、第 3 轮不改），而大"
            "模型改写不可复现 —— 想复核「这段痕迹是哪句话跑出来的」只能"
            "看它。"
        ),
    )

    gold_chunk_ids: list[UUID] = Field(default_factory=list)
    stages: list[StageTraceItem] = Field(default_factory=list)


# ============================================================
# 单条 case 的结果
# ============================================================


class EvalRunCase(APIModel):
    """
    一条 case 的命中结果 —— **一条 case 一行**（不是一轮一行）。

    逐轮放宽让同一条 case 在一轮评测里有多行，列表给的是**代表行**：
    命中优先，再比名次（`collapse_by_case`）。所以这里的 `gold_hit` 是
    「这条 case 最终找到没有」，与表头指标、与轮次的 `total_cases` 同一
    口径；「哪一轮救回来的」在 `EvalRunDetail.metrics["by_attempt"]` 里，
    要看某一轮的逐段痕迹就 `attempt_no` 指给 `get_case_trace`。
    """

    eval_case_id: UUID
    question: str
    chunk_id: UUID
    document_type_definition_id: UUID
    status: EvalCaseResultStatus

    attempt_no: int = Field(
        ge=1,
        description=(
            "这一行来自**哪一轮**（第 1 轮就命中的 case 是 1，三轮都没"
            "命中的是 3 —— 逐轮放宽里取更晚的那轮）。"
            "\n\n"
            "它同时是「这条 case 花了检索几次才找到 / 没找到」的读数，"
            "也是拿去查逐段痕迹的 handle。"
        ),
    )

    gold_hit: bool | None = Field(
        default=None,
        description=(
            "**任意**一个金标被返回即为 true。"
            "None = 这一条没跑成（与「没命中」是两回事）。"
        ),
    )

    query_text: str | None = Field(
        default=None,
        description=(
            "代表行那一轮**实际发给检索**的文本（改写之后）。"
            "没改写时等于 case 的 `question`。"
        ),
    )

    gold_rank: int | None = Field(default=None, ge=1)

    gold_chunk_ids: list[UUID] = Field(default_factory=list)

    retrieved_chunk_ids: list[UUID] = Field(
        default_factory=list,
        description="最终返回的 chunk id（有序）。",
    )

    top_candidates: list[ScoredChunkCandidate] = Field(
        default_factory=list,
        # 最终结果本来就由 `cutter.top_k` 决定条数（默认 10），这里取
        # 20 只为了防止客户端把 max_k 调到很大时响应跟着膨胀。
        max_length=20,
        description=(
            "最终 top-N 候选，**带分数**。默认前 10 —— 与"
            "「top 10 chunk 也要保留结果和 score」那条要求一致。"
            "\n\n"
            "逆向提问时别的 chunk 也可能答得上 —— 只看 id 集合会丢掉"
            "判断「它算不算也该命中」的依据。"
        ),
    )

    latency_ms: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    error_message: str | None = None


class EvalRunCaseListResponse(APIModel):
    items: list[EvalRunCase] = Field(default_factory=list)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)

    total: int = Field(
        ge=0,
        description=(
            "**case 数**（不是结果行数）—— 与轮次的 `total_cases` 同口径。"
            "\n\n"
            "⚠️ 逐轮放宽让一条 case 有多行（一轮一行），按行数报的话这个"
            "数会大一倍以上（实测 40 条 case 报 84），而分页是按它翻的："
            "翻到底也看不到最后几条 case。"
        ),
    )
