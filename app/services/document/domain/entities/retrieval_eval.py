"""
第一层检索评测的领域实体。

为什么这些实体挤在一个模块里（与其它实体一文件一个不同）
------------------------------------------------------
它们分两组，各自是**一个聚合**：

    数据集侧  bundle 及其抽样尝试、LLM 调用窗口、产出的 QA、每题的金标
    评测侧    一次评测轮次、单条结果、逐阶段金标进出、逐阶段候选

分开八个文件只会让人在八个地方来回跳，而它们没有各自的行为 —— 都是记录。

已有的先例是 `pipeline.py`（document_pipeline_run 与它的节点行同处一个
模块），理由一样。

（评测侧原先说"等做到那一层时再决定放哪"，现在做到了：放在同一个模块
里，因为两组的**字段风格与服务层用法完全一致**，而分模块反而会让人以为
它们属于不同的服务。）

实体存在的意义
--------------
仓储层返回实体而不是 ORM 行：ORM 行带着 session 生命周期（离开事务就
过期）、带着懒加载关系、而且能被随手改。服务层拿到实体之后就是一个
**快照**，读写都必须显式回到仓储 —— 这是本仓库各模块一致的纪律。

⚠️ 与实体不同，`generation_window.input_chunks` 是**输入快照**（含正文）。
它在实体上原样保留：重试一个窗口时靠它取回当时的正文，不再回源查
document_chunk（源可能已经被重新切分）。见模型层的说明。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas.enum import (
    EvalBundleStatus,
    EvalCaseResultStatus,
    EvalCaseValidationStatus,
    EvalGoldValidationMethod,
    EvalJobStatus,
)

__all__ = [
    "RetrievalEvalBundle",
    "RetrievalEvalCase",
    "RetrievalEvalCaseGoldChunk",
    "RetrievalEvalCaseResult",
    "RetrievalEvalGenerationRun",
    "RetrievalEvalGenerationWindow",
    "RetrievalEvalRun",
    "RetrievalEvalStageCandidate",
    "RetrievalEvalStageObservation",
]


class _EvalEntity(BaseModel):
    """三个实体共用的配置：从 ORM 行构造、拒绝未知字段。"""

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )


# ============================================================
# Bundle
# ============================================================


class RetrievalEvalBundle(_EvalEntity):
    """
    一份不可变的数据集。

    `status` 的流转由服务层拥有，实体只承载值 —— 但有一条不变量值得
    在这里说清楚：**`ready` 之后内容不可变**。改抽样量 / DTD 范围 / M /
    L / 出题模型都只能新建一份，因为历史评测结果要能对上一份固定的分母。
    """

    bundle_id: UUID
    tenant_id: UUID | None = None

    name: str
    description: str | None = None

    status: EvalBundleStatus = EvalBundleStatus.DRAFT
    schema_version: str = "1.0"

    selection_config: dict[str, Any] = Field(default_factory=dict)
    generation_config: dict[str, Any] = Field(default_factory=dict)

    dataset_sha256: str | None = None

    source_chunk_count: int = Field(default=0, ge=0)
    case_count: int = Field(default=0, ge=0)

    created_by: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_immutable(self) -> bool:
        """
        内容是否已冻结。

        `ready` 是唯一的不可变状态 —— `partial_failed` / `failed` 都还
        允许重试补窗口（那正是这个状态存在的意义）。
        """

        return self.status is EvalBundleStatus.READY


# ============================================================
# Generation run
# ============================================================


class RetrievalEvalGenerationRun(_EvalEntity):
    """
    一次可复现的抽样 + 造题尝试。

    `sample_seed` 是一等字段而不是塞在 `config_snapshot` 里：它是**复现
    的钥匙**，"同种子 + 同语料 = 同一批 chunk" 是一条可断言的性质，
    值得能被单独读到。
    """

    generation_run_id: UUID
    bundle_id: UUID

    status: EvalJobStatus = EvalJobStatus.QUEUED
    sample_seed: int

    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    model_role: str = "eval.qa_generation"
    model_name: str | None = None

    total_chunks: int = Field(default=0, ge=0)
    total_windows: int = Field(default=0, ge=0)
    completed_windows: int = Field(default=0, ge=0)
    failed_windows: int = Field(default=0, ge=0)
    generated_cases: int = Field(default=0, ge=0)

    error_message: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        """是否已经跑完（成功或失败都算）。"""

        return self.status in (
            EvalJobStatus.COMPLETED,
            EvalJobStatus.PARTIAL_FAILED,
            EvalJobStatus.FAILED,
            EvalJobStatus.CANCELLED,
        )


# ============================================================
# Generation window
# ============================================================


class RetrievalEvalGenerationWindow(_EvalEntity):
    """
    一次 LLM 调用 = 一个窗口。**重试的单位**。

    `input_chunks` 是含正文的快照。`attempt_count` 与 case 侧的唯一约束
    （`bundle_id + source_chunk_id + question_order`）共同保证重试幂等：
    重试会重新写 case，而唯一约束让"同一个 chunk 的同序号问题"插不进
    第二条，于是重复产出被库拒绝而不是靠应用层判断。
    """

    generation_window_id: UUID
    generation_run_id: UUID

    window_index: int = Field(ge=0)
    status: EvalJobStatus = EvalJobStatus.QUEUED

    input_chunks: list[dict[str, Any]] = Field(default_factory=list)
    input_token_estimate: int | None = Field(default=None, ge=0)

    request_sha256: str | None = None
    raw_response: dict[str, Any] | None = None

    attempt_count: int = Field(default=0, ge=0)
    valid_case_count: int = Field(default=0, ge=0)

    error_code: str | None = None
    error_message: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def chunk_ids(self) -> list[Any]:
        """本窗口的源 chunk id（按快照顺序）。"""

        return [
            chunk.get("chunk_id") for chunk in self.input_chunks
        ]

    @property
    def is_retryable(self) -> bool:
        """
        这个窗口值不值得重试。

        `failed` 与 `partial_failed` 都算 —— 后者指"跑了、但产出的 case
        数少于期望"，那通常意味着模型这次没出够题，重试有意义。

        ⚠️ `queued` **不算**可重试：它只是还没跑，重复提交它会造成两个
        worker 同时处理同一个窗口。
        """

        return self.status in (
            EvalJobStatus.FAILED,
            EvalJobStatus.PARTIAL_FAILED,
        )


# ============================================================
# Case
# ============================================================


class RetrievalEvalCase(_EvalEntity):
    """
    一条 QA。

    `answer_start_offset` / `answer_end_offset` 与 `answer_original` 是
    一致的：`answer_original == 源 chunk 正文[start:end]`。这不是巧合 ——
    服务端在写入前会把模型给的串换成**原文切出来的那一段**（见
    eval/generator.py 的 validate_items），正是为了让这条等式恒成立。
    校验（或人工）随时可以拿 offset 回原 chunk 复核。
    """

    eval_case_id: UUID
    bundle_id: UUID
    generation_window_id: UUID | None = None

    document_type_definition_id: UUID
    document_version_id: UUID | None = None
    document_retrieval_definition_id: UUID | None = None

    # ---- 源文档身份（快照） ----
    #
    # ⚠️ 见模型层的长说明：这不是"顺手多存点"。本语料的正文常写
    # 「本公司」，出题时必须把主体从**文档**取来写进问题；而问题一旦
    # 写了公司名，**答案的正确性就只能靠"这块 chunk 属于那家公司"来
    # 保证**（正文里的数字不带主体）。所以不记这三列，这类题无法复核。
    document_id: UUID | None = None
    document_title: str | None = None
    document_key: str | None = None

    source_chunk_id: UUID
    source_content_hash: str

    question: str
    answer_original: str
    answer_start_offset: int | None = Field(default=None, ge=0)
    answer_end_offset: int | None = Field(default=None, ge=0)

    question_order: int = Field(ge=0)
    question_sha256: str

    validation_status: EvalCaseValidationStatus = (
        EvalCaseValidationStatus.VALIDATED
    )
    validation_error: str | None = None

    created_at: datetime | None = None

    @model_validator(mode="after")
    def _check_offsets(self) -> "RetrievalEvalCase":
        """
        偏移的先后关系。

        与库上那条 CHECK 是**同一件事的两道防线**：库拦的是任何写入路径
        （包括运维手工 SQL），这里拦的是"错误在进入事务之前就被发现"。
        只留库那道的话，能撞出 IntegrityError 的地方会离原因很远。
        """

        start = self.answer_start_offset
        end = self.answer_end_offset

        if start is not None and end is not None and end < start:
            raise ValueError(
                f"answer_end_offset ({end}) 小于 answer_start_offset "
                f"({start})。"
            )

        return self


class RetrievalEvalCaseGoldChunk(_EvalEntity):
    """
    一个能回答某道题的 chunk。

    `is_source_chunk` 让统计能分开"命中来自出题的那一块"与"命中来自
    别的块" —— 后者才是跨 chunk 裁定真正救回来的那些。两者的比例是个
    有意义的信号：如果后者占比很高，说明语料里同一事实反复出现，那本身
    就值得知道。
    """

    eval_case_gold_chunk_id: UUID
    eval_case_id: UUID

    chunk_id: UUID
    is_source_chunk: bool = False

    validation_method: EvalGoldValidationMethod = (
        EvalGoldValidationMethod.SOURCE_CHUNK
    )

    evidence_excerpt: str | None = None
    source_content_hash: str | None = None
    judgment: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime | None = None


# ============================================================
# 评测侧
# ============================================================
#
# 与数据集侧（bundle / generation_* / case）分开的另一半：这里的东西
# **全部由一次评测轮次产生**，不需要 LLM，也没有"重试某个窗口"那种粒度。
#
# 四张表是一只漏斗的四层：
#
#     run              一次评测轮次（配置快照 + 聚合指标缓存）
#     case_result      单条 QA 的命中结果（最终 top-N 与分数在 retrieved_chunk_ids 里）
#     stage_observation 单条 QA 的**逐阶段金标进出** —— 宏观统计的来源
#     stage_candidate   单条 QA 的**逐阶段候选** —— 只给单个 case 诊断用
#
# 后两张的分工是刻意的：observation 少而可聚合（宏观），candidate 多而
# 只用于个案（诊断）。把两者合成一张表的结果，要么聚合时扫一堆用不上的
# 行，要么诊断时发现细节被丢光了。


class RetrievalEvalRun(_EvalEntity):
    """
    一轮评测。

    `effective_config_snapshot` 不是冗余：DTD 的融合/精排配置**是可改的**，
    而结果是"在当时那套配置下"得出的。不存快照的话，改完 DTD 再去读历史
    结果，会以为那些数字是当前配置的表现。

    `query_processor_override` 与它分开存：前者是**调用方要求了什么**，
    后者是**实际生效了什么**。两者不等时，差异就是答案所在。
    """

    eval_run_id: UUID
    bundle_id: UUID

    status: EvalJobStatus = EvalJobStatus.QUEUED
    max_k: int = Field(default=10, ge=1)

    query_processor_override: dict[str, Any] = Field(default_factory=dict)
    effective_config_snapshot: dict[str, Any] = Field(default_factory=dict)

    include_stage_trace: bool = True

    total_cases: int = Field(default=0, ge=0)
    completed_cases: int = Field(default=0, ge=0)
    failed_cases: int = Field(default=0, ge=0)

    metrics: dict[str, Any] = Field(default_factory=dict)

    created_by: UUID | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            EvalJobStatus.COMPLETED,
            EvalJobStatus.PARTIAL_FAILED,
            EvalJobStatus.FAILED,
            EvalJobStatus.CANCELLED,
        )


class RetrievalEvalCaseResult(_EvalEntity):
    """
    一条 case 在一轮评测里的结果。

    `gold_hit` 用 NULL 表示**没跑到**（不是"没命中"）—— 三值而不是两值，
    因为 failed 与 miss 必须分得开。把它们并成 false 的话，"检索挂了"
    会表现为"召回率下降"，而那是最容易误判成检索质量问题的一种假象。
    """

    eval_case_result_id: UUID
    eval_run_id: UUID
    eval_case_id: UUID

    #: 第几轮（从 1 起）。**同一条 case 每轮一行** —— 逐轮放宽时，
    #: 「第几轮找到的」是核心指标之一。
    attempt_no: int = Field(default=1, ge=1)

    query_id: UUID | None = Field(
        default=None,
        description=(
            "检索侧本次调用的 `retrieve_id`。"
            "**这是把评测结果与检索日志对上的唯一凭据** —— 按它去翻日志，"
            "而不是按问题文本（同一个问题文本可能挂在不同的 chunk 上）。"
        ),
    )

    query_text: str | None = Field(
        default=None,
        description=(
            "本轮**实际发给检索**的文本（改写之后）。"
            "\n\n"
            "与 `query_id` 是一对：那条指向日志（会被轮转），这条自己"
            "就是内容。逐轮评测里每轮可以用不同改写器，而 LLM 改写不可"
            "复现 —— 所以它是「这条命中的是哪句话」的唯一可靠答案。"
        ),
    )

    gold_hit: bool | None = None

    gold_rank: int | None = Field(
        default=None,
        ge=1,
        description="**最好**的那个金标的 1-based 排名（多个金标时取最好的）。",
    )

    retrieved_chunk_ids: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "最终结果的有序候选，元素形如 {chunk_id, rank, score, "
            "score_name, source_key, metadata}。**带分数** —— 逆向提问时"
            "别的 chunk 也可能答得上，只看 id 集合会丢掉判断依据。"
        ),
    )

    latency_ms: int | None = Field(default=None, ge=0)

    #: 本轮**没能参与**的召回路线（超时/后端故障/未实现）。
    #:
    #: 记它是为了让「降级跑的 miss」与「检索真的差」分得开 —— 没有它，
    #: 一次后端超时会伪装成召回质量下降。见模型层的长说明。
    degraded_routes: list[dict[str, Any]] = Field(default_factory=list)

    status: EvalCaseResultStatus = EvalCaseResultStatus.COMPLETED
    error_code: str | None = None
    error_message: str | None = None

    created_at: datetime | None = None


class RetrievalEvalStageObservation(_EvalEntity):
    """
    金标在**一个阶段**前后的位置。宏观丢因统计就建在它上面。

    归因规则（这张表的语义核心）：一个阶段只有在"进阶段前金标**还在**、
    出阶段后**没了**"时才算丢了它。不这么判的话，金标在召回那一路根本
    没出现，后面的 filter / rerank / cutter 每一段都会"出阶段后不在" ——
    于是三段各记一次丢弃，宏观统计里它们会背上一大堆不是自己造成的锅。
    """

    stage_observation_id: UUID
    eval_case_result_id: UUID

    stage_key: str
    source_key: str = ""

    executed: bool = True

    input_count: int | None = Field(default=None, ge=0)
    output_count: int | None = Field(default=None, ge=0)

    gold_present_before: bool | None = None
    gold_present_after: bool | None = None

    gold_rank_before: int | None = Field(default=None, ge=1)
    gold_rank_after: int | None = Field(default=None, ge=1)

    gold_score_before: float | None = None
    gold_score_after: float | None = None
    gold_score_name: str | None = None
    gold_score_direction: str | None = None

    drop_reason: str | None = None

    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class RetrievalEvalStageCandidate(_EvalEntity):
    """
    一个阶段进出的候选（top-N ∪ 被盯住的）。

    ⚠️ 分数**只在同一阶段/分支内可比**。本管线暴露出来的分数始终是归一化
    到 [0,1] 的（召回后按路归一化、融合后归一化、精排后归一化），所以
    `score_name` 与 `score_direction` 目前几乎不携带信息 —— 保留它们是为了
    将来记录**下推到索引层的原始分**（`min_score` 那类过滤用的是原始量纲：
    BM25 可到几十、余弦 [-1,1]）。别为了填满而编一个名字。
    """

    stage_candidate_id: UUID
    eval_case_result_id: UUID

    stage_key: str
    source_key: str = ""

    chunk_id: UUID

    #: 进/出两套排名与分数。
    #:
    #: 用哪一套取决于这个候选在**这一段**是活是死：
    #:     活下来 -> rank_after / score_after
    #:     死在这里 -> rank_before / score_before
    #: 所以两条里恰好有一条非空（库上有 CHECK 兜着）。
    rank_before: int | None = Field(default=None, ge=1)
    rank_after: int | None = Field(default=None, ge=1)
    score_before: float | None = None
    score_after: float | None = None
    score_name_before: str | None = None
    score_name_after: str | None = None
    score_direction_before: str | None = None
    score_direction_after: str | None = None

    selected: bool = True
    drop_reason: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
