"""
检索评测数据集服务的实现。

一次造题执行的形状
------------------
    create_bundle（同步，几十毫秒）
        校验 DTD -> 抽 chunk -> 取正文 -> 切窗口 -> 落库 -> 提交任务

    generate（异步，分钟级）
        逐个窗口：认领 -> 调模型 -> 校验 -> 写 case 与金标 -> 记状态
        最后：从窗口行重算运行计数、定 bundle 的状态

几处刻意的决定
--------------
**一、窗口数有上限，且不发子任务。**
一次执行最多跑 `MAX_WINDOWS_PER_TASK_RUN` 个窗口，剩下的留在 `queued`，
由再次提交同一条任务继续（`retry_failed_windows` 或重新触发）。理由见
那个常量的说明 —— 是 lease 的上限，不是工程上的偷懒。

**二、计数从窗口行重算，不累加。**
任务是至少一次投递，累加在重跑时会算重。见
`retrieval_eval_bundle.count_windows_by_status`。

**三、单个窗口失败**不抛异常**。**
抛出去会让任务重试，而重试会把已经成功的窗口的模型调用再付一遍钱。
失败记进窗口行，bundle 落 `partial_failed` / `failed` —— 两个状态都说得
清楚"这批数据集不完整"，比一个 500 诚实。
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.agent.model import ModelRole, get_model_registry
from app.common.exceptions import ConflictError, NotFoundError
from app.common.factories import get_service_factory
from app.common.service.base import BaseService, ServiceContext
from app.common.task import TaskPermanentError
from app.common.task.models import Task
from app.common.utils import canonical_json, sha256_text
from app.services.task.application import TaskService

from ..chunker.text import TextDocumentChunker
from ..domain.entities import (
    RetrievalEvalBundle,
    RetrievalEvalCase,
    RetrievalEvalCaseGoldChunk,
    RetrievalEvalCaseResult,
    RetrievalEvalGenerationRun,
    RetrievalEvalGenerationWindow,
    RetrievalEvalRun,
    RetrievalEvalStageCandidate,
    RetrievalEvalStageObservation,
)
from ..domain.schemas.enum import (
    DefinitionStatus,
    EvalBundleStatus,
    EvalCaseResultStatus,
    EvalCaseValidationStatus,
    EvalGoldValidationMethod,
    EvalJobStatus,
    EvalSamplingStrategy,
    QueryRewriterKind,
)
from ..domain.schemas.retrieval_eval import (
    DEFAULT_ATTEMPT_LADDER,
    DEFAULT_TOP_CANDIDATES_PER_STAGE,
    CreateEvalBundleAcceptedResponse,
    CreateEvalBundleRequest,
    CreateEvalRunAcceptedResponse,
    CreateEvalRunRequest,
    EvalBundleDetail,
    EvalBundleExport,
    EvalBundleExportItem,
    EvalBundleListResponse,
    EvalBundleSummary,
    EvalAttemptSpec,
    EvalCase,
    EvalCaseListResponse,
    EvalCaseTraceResponse,
    EvalMetrics,
    EvalRunCase,
    EvalRunCaseListResponse,
    EvalRunDetail,
    GenerationRetryAcceptedResponse,
    GenerationRunDetail,
    GenerationWindowSummary,
    ScoredChunkCandidate,
    StageStatItem,
    StageStatsResponse,
    StageTraceItem,
)
from ..domain.schemas.retrieve import (
    CutterConfig,
    DocumentRetrieveRequest,
    DocumentRetrieveResponse,
    QueryRewriteInput,
    QueryRewriterConfig,
    RetrieveProcessorOverride,
)
from ..repository.retrieval_eval_run import RetrievalEvalRunRepository
from . import retrieval_eval_evaluation as evaluation
from .document_retrieve_service import DocumentRetrieveService
from ..domain.tasks import (
    RETRIEVAL_EVAL_EVALUATE_HANDLER,
    RETRIEVAL_EVAL_GENERATE_HANDLER,
    RetrievalEvalEvaluateParams,
    RetrievalEvalGenerateParams,
)
from ..eval import adjudication
from ..eval import generator as eval_generator
from ..eval import sample_chunks
from ..eval.windowing import plan_windows
from ..repository.document_chunk import DocumentChunkRepository
from ..repository.retrieval_eval_bundle import (
    RetrievalEvalBundleRepository,
)
from ..repository.retrieval_eval_case import (
    RetrievalEvalCaseRepository,
)
from ..repository.type_definition import (
    DocumentTypeDefinitionRepository,
)
from .retrieval_eval_service import RetrievalEvalService

__all__ = [
    "MAX_WINDOWS_PER_TASK_RUN",
    "WINDOW_CONCURRENCY",
    "RetrievalEvalServiceImpl",
]

logger = logging.getLogger(__name__)


#: 一次造题执行最多处理的窗口数。
#:
#: **这是 lease 的上限，不是性能取舍。** Worker 的 lease 默认 900 秒且
#: **不续约**（见 app/infra/task/worker.py）—— 跑过头的任务会被 Sweeper
#: 判定为失联、回收、重跑，于是同一条运行的窗口可能被两个执行同时处理
#: （重复的模型调用，而模型是按次计费的）。
#:
#: 算式：`WINDOW_CONCURRENCY` 个并行 × 单窗口几十秒。取 20 是保守值：
#: 单窗口按 30 秒算，串行 20 个是 10 分钟，仍在 15 分钟以内。
#: 单窗口更慢（长 chunk、思考型模型）时要么调小它，要么调大 worker 的
#: lease —— 但**不要**只是把它调大而不看 lease。
MAX_WINDOWS_PER_TASK_RUN = 20

#: 同时跑几个窗口。
#:
#: ⚠️ **必须 ≤ 你账号的并发上限，而不只是 RPM。** 两者是不同的限制，
#: 而且并发限制往往小得多：实测某个账号 RPM=3、**并发上限只有 1** ——
#: 那时并发设成 4 会让每个窗口都直接 429，整批变成 `failed`，而
#: 窗口行上写的是"模型返回 429"，看起来像服务端的问题。
#:
#: 取 1 是安全的默认。**没有把它做成请求参数**是刻意的：公开的
#: OpenAPI 契约里没有这个字段，而"多一个未在契约里声明的入参"会让
#: 前端以为它可以调。要按部署调，改这里。
WINDOW_CONCURRENCY = 1

#: 单条 case 的检索超时（秒）。
#:
#: ⚠️ **必须自己设。** 不设的话整轮就交给索引后端自己的超时了，而
#: `retrieval_config.timeout_ms` 默认**不设** —— 一个不返回的后端能让
#: 一轮评测无限等下去：没有 error、没有进度日志、库里也什么都看不到
#: （一个事务）。
#:
#: 取 60 秒是从实测反推的：p50 约 190ms、p95 约 350ms，所以 60 秒是
#: p95 的 **170 倍** —— 正常路径不可能碰到它，而真的卡住时 60 秒就能
#: 把这一条判成 failed 继续往下走。既不要宽到等于没设，也不要紧到把
#: 偶尔一次的慢查询误判成故障。
CASE_TIMEOUT_SECONDS = 60.0

#: 每多少条 case **提交一次**（也是打一行进度的粒度）。
#:
#: ⚠️ 这是整个评测执行里最重要的一个数。
#:
#: 整轮一个事务有三个后果，实测都撞过：
#:
#:     运行中**不可观测**      库里的 `status` 还是 `queued`、结果行 0 条，
#:                             与"从没启动"完全一样。有一个任务跑了
#:                             527 秒而没有任何可查的痕迹，只能靠猜。
#:     中途失败**全丢**        跑到第 39 条炸了，前 38 次的检索一起回滚，
#:                             然后重试再从第 1 条开始。
#:     被租约回收**从零重来**  超过 900 秒的轮次永远跑不完：回收 -> 重跑
#:                             -> 再超 -> 直到 3 次尝试耗尽。
#:
#: 取 10 的取舍：批太大则上面三条的粒度变粗（一批 100 条，中间炸了丢 100
#: 条）；批太小则事务开销与日志噪音上来（每批一次提交 + 一次检索会话）。
#: 按实测 0.3–0.5 秒/条，一批约 3–5 秒 —— 进度粒度够细，开销可忽略。
BATCH_COMMIT_EVERY = 10


#: 收尾时"把这一轮的全部结果读回来算指标"的上界。
#:
#: 一轮评测的 case 数由数据集决定（抽样量 / M）。取 10 万是给"抽样一万块"
#: 这种极端配置留的余量；达到它说明这一轮大到该分批统计了 —— 而那时
#: **指标会算少**，所以超界要记 warning，不能静默截断。
MAX_CASES_PER_RUN = 100_000


#: 抽样池的硬上界。
#:
#: 语料可能几万块（实测某 DTD 下 39697 块），而抽样只需要"有哪些候选"。
#: 取 5 万是为了让常见语料一页取完。**池子被这个值截断时会记一条
#: warning** —— 静默截断会让"数据集只覆盖了语料的一小部分"没有任何痕迹。
MAX_SAMPLE_POOL = 50_000

#: 一条题最多送几个候选去判金标。
#:
#: 候选已经被收窄到"同一份文档"（见 `generate` 里的说明），所以正常
#: 情况下是 0 到几个 —— 一个窗口的 5 块来自同一份文档本来就少见。
#: 只有"一份文档贡献了很多块"时才会撞到这个上界，而那时**截断会记
#: 一条 warning**：它直接缩小了金标搜索范围。
MAX_ADJUDICATION_CANDIDATES = 20


class RetrievalEvalServiceImpl(RetrievalEvalService):

    def __init__(
            self,
            *,
            bundle_repository: RetrievalEvalBundleRepository,
            case_repository: RetrievalEvalCaseRepository,
            run_repository: RetrievalEvalRunRepository,
            chunk_repository: DocumentChunkRepository,
            type_definition_repository: DocumentTypeDefinitionRepository,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._bundles = bundle_repository
        self._cases = case_repository
        self._runs = run_repository
        self._chunks = chunk_repository
        self._definitions = type_definition_repository

    # ========================================================
    # 建 bundle
    # ========================================================

    async def create_bundle(
            self,
            request: CreateEvalBundleRequest,
            *,
            tenant_id: UUID | None = None,
            created_by: UUID | None = None,
    ) -> CreateEvalBundleAcceptedResponse:

        # ---- 1. 校验 DTD ----
        #
        # 先校验再抽样，而不是等切窗口时才发现 —— 后者会在抽完、写完
        # bundle 之后才失败，留下一个永远跑不起来的 bundle。
        definitions = await self._definitions.get_many(
            list(request.document_type_definition_ids)
        )

        found = {
            definition.document_type_definition_id
            for definition in definitions
        }
        missing = [
            str(dtd_id)
            for dtd_id in request.document_type_definition_ids
            if dtd_id not in found
        ]

        if missing:
            raise NotFoundError(
                "这些 DocumentTypeDefinition 不存在或已逻辑删除："
                + "、".join(missing)
            )

        inactive = [
            str(definition.document_type_definition_id)
            for definition in definitions
            if definition.status is not DefinitionStatus.ACTIVE
        ]

        if inactive:
            raise ConflictError(
                "这些 DocumentTypeDefinition 不是 ACTIVE 状态，"
                "不能用它们出题：" + "、".join(inactive)
            )

        # ---- 2. 抽样 ----
        seed = (
            request.seed
            if request.seed is not None
            else _new_seed()
        )

        pool = await self._chunks.list_samplable_chunk_ids(
            document_type_definition_ids=list(
                request.document_type_definition_ids
            ),
            # 只抽规范正文块。图那一路的 chunk 不是"一段正文"，
            # 从它出题会得到一道语料里没有答案的题。
            chunker_keys=[TextDocumentChunker.name],
            max_pool=MAX_SAMPLE_POOL,
        )

        if len(pool) >= MAX_SAMPLE_POOL:
            logger.warning(
                "抽样池达到上界 %d，本次抽样只在其中进行 ——"
                "数据集覆盖的语料范围小于全部候选。"
                "要覆盖更大范围请调大 MAX_SAMPLE_POOL。",
                MAX_SAMPLE_POOL,
            )

        if not pool:
            raise ValueError(
                "选中的 DTD 下没有任何可出题的 chunk（正文非空、"
                "chunker 为 %s 且版本未删除）。"
                "这通常意味着这些 DTD 还没有跑过写入管线。" % (
                    TextDocumentChunker.name,
                )
            )

        picked = sample_chunks(
            pool,
            count=min(request.sampled_chunk_count, len(pool)),
            seed=seed,
            strategy=request.sampling_strategy,
        )

        # ---- 3. 取正文 + 切窗口 ----
        contents = await self._load_contents(picked)

        # ⚠️ 快照里的 id **一律转成字符串**，这一步不是风格问题。
        #
        # 这个 dict 会整体进 `retrieval_eval_generation_window.input_chunks`
        # （JSONB），而 **JSON 没有 UUID 类型** —— 直接放 UUID 对象会在
        # INSERT 那一刻炸在 json 序列化里，报的是
        # `Object of type UUID is not JSON serializable`，离"我在建快照"
        # 这件事很远。
        #
        # 转成字符串之后还有第二个好处：模型回填的 chunk_id 也是字符串，
        # 于是"模型给的 id"与"快照里的 id"是同一个类型，比较时不必在
        # 两边各做一次转换（那种不对称最容易漏掉一边，症状是静默少写
        # 几条 case）。
        window_chunks = [
            {
                "chunk_id": str(item["chunk_id"]),
                "document_type_definition_id": str(
                    item["document_type_definition_id"]
                ),
                "document_version_id": str(item["document_version_id"]),
                # 文档身份随快照一起走。出题时按 chunk 喂给模型 ——
                # 正文里常写「本公司」，不给主体它就写不出可检索的问题。
                # 见 eval_qa_generation 提示词的「主体从哪来」那一节。
                "document_id": str(item["document_id"]),
                "document_title": item.get("document_title"),
                "document_key": item.get("document_key"),
                "content": contents[item["chunk_id"]].content,
                "content_hash": contents[item["chunk_id"]].content_hash,
                "token_count": contents[item["chunk_id"]].token_count,
            }
            for item in picked
            if item["chunk_id"] in contents
        ]

        if not window_chunks:
            raise ValueError(
                "抽中的 chunk 一块正文都没读到 —— 抽样池与正文表不一致，"
                "不该发生。"
            )

        windows, skipped = plan_windows(
            window_chunks,
            chunks_per_window=request.chunks_per_window,
            max_input_tokens=request.max_input_tokens,
            questions_per_chunk=request.questions_per_chunk,
        )

        for item in skipped:
            # 跳过是**要说出来**的：它直接减少数据集的覆盖面，而"少了
            # 几个 chunk"在最终产物里没有任何别的痕迹。
            logger.warning(
                "跳过 chunk %s：%s", item.chunk_id, item.reason,
            )

        if not windows:
            raise ValueError(
                f"抽中的 {len(window_chunks)} 个 chunk 切不出任何窗口 ——"
                "全部超过了 max_input_tokens。调大它，或选内容更短的 DTD。"
            )

        # ---- 4. 落库 ----
        now = _now()

        bundle = RetrievalEvalBundle(
            bundle_id=uuid4(),
            tenant_id=tenant_id,
            name=request.name,
            description=request.description,
            status=EvalBundleStatus.GENERATING,
            selection_config={
                "document_type_definition_ids": [
                    str(x)
                    for x in request.document_type_definition_ids
                ],
                "sampled_chunk_count": request.sampled_chunk_count,
                "sampling_strategy": request.sampling_strategy.value,
                "seed": seed,
                "pool_size": len(pool),
                "skipped_chunk_count": len(skipped),
            },
            generation_config={
                "questions_per_chunk": request.questions_per_chunk,
                "chunks_per_window": request.chunks_per_window,
                "max_input_tokens": request.max_input_tokens,
                "model_role": ModelRole.EVAL_QA_GENERATION,
                # ⚠️ 这个开关**改变数据集内容**（有没有跨 chunk 金标），
                # 所以必须进快照 —— 否则事后无法分辨某份数据集是从哪种
                # 模式产出的，而"金标集合为什么这么小"就只能靠猜。
                "enable_cross_chunk_judge": (
                    request.enable_cross_chunk_judge
                ),
            },
            source_chunk_count=len(window_chunks),
            case_count=0,
            # ⚠️ 显式填 `created_at` —— 不能只靠列的 server_default。
            # PostgreSQL 的 now() 是**事务开始时间**，同一事务插的多行会
            # 拿到相同的时间戳，于是"最近一次"掉到随机 UUID 的 tiebreaker
            # 上。详见仓储接口的说明。
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )

        await self._bundles.add_bundle(bundle)

        run = RetrievalEvalGenerationRun(
            generation_run_id=uuid4(),
            bundle_id=bundle.bundle_id,
            status=EvalJobStatus.QUEUED,
            sample_seed=seed,
            config_snapshot=dict(bundle.generation_config),
            model_role=ModelRole.EVAL_QA_GENERATION,
            total_chunks=len(window_chunks),
            total_windows=len(windows),
            created_at=now,
        )

        await self._bundles.add_generation_run(run)

        await self._bundles.add_windows(
            [
                RetrievalEvalGenerationWindow(
                    generation_window_id=uuid4(),
                    generation_run_id=run.generation_run_id,
                    window_index=plan.window_index,
                    status=EvalJobStatus.QUEUED,
                    # 输入快照**含正文**：重试时不再回源查 chunk
                    # （源可能已被重新切分，那样"重试"会悄悄换掉输入）。
                    input_chunks=plan.chunks,
                    input_token_estimate=plan.token_estimate,
                    created_at=now,
                )
                for plan in windows
            ]
        )

        # ---- 5. 提交任务 ----
        #
        # ⚠️ 提交必须发生在**本事务提交之后**才能保证任务读到数据。
        # 这里直接提交，靠的是 ServiceFactory.scope 的 post_commit 机制：
        # `TaskService.submit` 内部会把投递挂到 post_commit 上，所以这里
        # await 返回时任务已落库、但投递还没发生。
        await _submit_generation(run.generation_run_id)

        logger.info(
            "数据集已建：bundle=%s run=%s 抽中 %d 块 -> %d 个窗口"
            "（跳过 %d 块）",
            bundle.bundle_id,
            run.generation_run_id,
            len(window_chunks),
            len(windows),
            len(skipped),
        )

        return CreateEvalBundleAcceptedResponse(
            bundle_id=bundle.bundle_id,
            generation_run_id=run.generation_run_id,
            status=EvalBundleStatus.GENERATING,
            message=(
                f"造题已入队：{len(window_chunks)} 个源 chunk，"
                f"{len(windows)} 个窗口。"
            ),
        )

    # ========================================================
    # 读
    # ========================================================

    async def list_bundles(
            self,
            *,
            status: EvalBundleStatus | None = None,
            document_type_definition_id: UUID | None = None,
            page: int = 1,
            page_size: int = 20,
            tenant_id: UUID | None = None,
    ) -> EvalBundleListResponse:

        # ⚠️ 按 DTD 筛是在**内存里**做的（筛 selection_config），不是在
        # SQL 里。所以它筛的是"已经在当页的那些" —— 而这是错的。
        #
        # 所以这里**明确拒绝**这个组合而不是返回一个悄悄不对的结果：
        # 分页 + 后置过滤 = 少返回、且用户不知道。
        if document_type_definition_id is not None:
            raise ValueError(
                "按 DTD 筛列表暂不支持与分页组合（DTD 范围记在 "
                "selection_config 这个 JSONB 里，无法在 SQL 层可靠分页）。"
                "要看某个 DTD 的题，请用 GET /bundles/{id}/cases 的 "
                "document_type_definition_id 过滤。"
            )

        items, total = await self._bundles.list_bundles(
            tenant_id=tenant_id,
            status=status,
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        latest = await self._bundles.list_latest_generation_runs(
            [item.bundle_id for item in items]
        )

        return EvalBundleListResponse(
            items=[
                _to_summary(item, latest.get(item.bundle_id))
                for item in items
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def get_bundle(
            self,
            bundle_id: UUID,
    ) -> EvalBundleDetail:

        bundle = await self._bundles.get_bundle(bundle_id)

        if bundle is None:
            raise NotFoundError(f"数据集不存在：{bundle_id}")

        latest = await self._bundles.get_latest_generation_run(bundle_id)

        summary = _to_summary(bundle, latest)

        return EvalBundleDetail(
            **summary.model_dump(),
            schema_version=bundle.schema_version,
            selection_config=dict(bundle.selection_config),
            generation_config=dict(bundle.generation_config),
            latest_generation_run_id=(
                None if latest is None else latest.generation_run_id
            ),
            updated_at=bundle.updated_at or bundle.created_at,
        )

    async def list_cases(
            self,
            bundle_id: UUID,
            *,
            document_type_definition_id: UUID | None = None,
            validation_status: EvalCaseValidationStatus | None = None,
            page: int = 1,
            page_size: int = 50,
    ) -> EvalCaseListResponse:

        if await self._bundles.get_bundle(bundle_id) is None:
            raise NotFoundError(f"数据集不存在：{bundle_id}")

        cases, total = await self._cases.list_cases(
            bundle_id=bundle_id,
            document_type_definition_id=document_type_definition_id,
            validation_status=validation_status,
            offset=(page - 1) * page_size,
            limit=page_size,
        )

        gold = await self._cases.list_gold_chunks(
            [case.eval_case_id for case in cases]
        )

        return EvalCaseListResponse(
            items=[
                _to_case_dto(
                    case,
                    [
                        item.chunk_id
                        for item in gold.get(case.eval_case_id, [])
                    ],
                )
                for case in cases
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def export_bundle(
            self,
            bundle_id: UUID,
    ) -> EvalBundleExport:

        bundle = await self._bundles.get_bundle(bundle_id)

        if bundle is None:
            raise NotFoundError(f"数据集不存在：{bundle_id}")

        if bundle.status is not EvalBundleStatus.READY:
            raise ConflictError(
                f"数据集当前状态是 {bundle.status.value}，不能导出。"
                "只有 ready 的数据集才是完整的 —— 用不完整的数据集跑评测，"
                "得到的偏低召回率来自数据集本身，而它在结果里与"
                "「检索真的差」长得一模一样。"
            )

        cases = await self._cases.list_all_validated_cases(bundle_id)
        gold = await self._cases.list_gold_chunks(
            [case.eval_case_id for case in cases]
        )

        items = [
            EvalBundleExportItem(
                chunk_id=case.source_chunk_id,
                gold_chunk_ids=[
                    item.chunk_id
                    for item in gold.get(case.eval_case_id, [])
                ] or [case.source_chunk_id],
                document_type_definition_id=(
                    case.document_type_definition_id
                ),
                document_version_id=case.document_version_id,
                # 文档身份随导出物走：问题里的主体来自它，不带上就复核不了。
                document_id=case.document_id,
                document_title=case.document_title,
                document_key=case.document_key,
                question=case.question,
                answer_original=case.answer_original,
            )
            for case in cases
        ]

        payload = {
            "schema_version": bundle.schema_version,
            "bundle_id": str(bundle.bundle_id),
            "bundle_name": bundle.name,
            "generation_config": bundle.generation_config,
            "document_type_definition_ids": bundle.selection_config.get(
                "document_type_definition_ids", []
            ),
            "items": [
                {**item.model_dump(mode="json")} for item in items
            ],
        }

        # 指纹建在**确定性序列化**上（键序固定、条目顺序固定）。
        # 顺序一旦依赖库的返回顺序，同一份数据导出两次会得到不同的
        # 指纹 —— 而它的用途恰恰是回答"这两次评测用的是不是同一份数据"。
        return EvalBundleExport(
            schema_version=bundle.schema_version,
            bundle_id=bundle.bundle_id,
            bundle_name=bundle.name,
            dataset_sha256=sha256_text(canonical_json(payload)),
            generation_config=dict(bundle.generation_config),
            document_type_definition_ids=list(
                bundle.selection_config.get(
                    "document_type_definition_ids", []
                )
            ),
            items=items,
        )

    # ========================================================
    # 造题进度
    # ========================================================

    async def get_generation_run(
            self,
            generation_run_id: UUID,
    ) -> GenerationRunDetail:

        run = await self._bundles.get_generation_run(generation_run_id)

        if run is None:
            raise NotFoundError(f"造题运行不存在：{generation_run_id}")

        windows = await self._bundles.list_windows(generation_run_id)

        return GenerationRunDetail(
            generation_run_id=run.generation_run_id,
            bundle_id=run.bundle_id,
            status=run.status,
            model_role=run.model_role,
            model_name=run.model_name,
            sample_seed=run.sample_seed,
            total_chunks=run.total_chunks,
            total_windows=run.total_windows,
            completed_windows=run.completed_windows,
            failed_windows=run.failed_windows,
            generated_cases=run.generated_cases,
            error_message=run.error_message,
            windows=[
                GenerationWindowSummary(
                    generation_window_id=window.generation_window_id,
                    window_index=window.window_index,
                    status=window.status,
                    input_chunk_count=len(window.input_chunks),
                    input_token_estimate=window.input_token_estimate,
                    attempt_count=window.attempt_count,
                    valid_case_count=window.valid_case_count,
                    error_code=window.error_code,
                    error_message=window.error_message,
                )
                for window in windows
            ],
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

    async def retry_failed_windows(
            self,
            generation_run_id: UUID,
    ) -> GenerationRetryAcceptedResponse:

        run = await self._bundles.get_generation_run(generation_run_id)

        if run is None:
            raise NotFoundError(f"造题运行不存在：{generation_run_id}")

        unfinished = await self._bundles.list_unfinished_windows(
            generation_run_id
        )

        if not unfinished:
            raise ConflictError(
                "这次造题没有可再跑的窗口（全部已完成）。"
            )

        await _submit_generation(generation_run_id)

        return GenerationRetryAcceptedResponse(
            generation_run_id=generation_run_id,
            status=run.status,
            retried_window_count=len(unfinished),
        )

    # ========================================================
    # 后台任务主体
    # ========================================================

    async def generate(
            self,
            params: RetrievalEvalGenerateParams,
            ctx: Any,
    ) -> None:

        run = await self._bundles.get_generation_run(
            params.generation_run_id
        )

        if run is None:
            # ⚠️ 必须是 `TaskPermanentError`，不能是 `NotFoundError`。
            #
            # 重试策略的默认是**重试**，而它的 `_FATAL` 清单里只有
            # UnknownHandlerError / TaskParamsError / TaskStateError /
            # pydantic 的 ValidationError / CancelledError —— 我们的
            # NotFoundError 不在其中，于是会被当成"暂时性故障"重试满
            # `max_attempts`（默认 3）次。
            #
            # 而"这次运行不存在"是**永久性**的：它不会自己出现，重试只是
            # 白烧三次。`TaskPermanentError` 是 Handler 声明这件事的唯一
            # 方式（见 app/common/task/exceptions.py 的说明：默认重试，
            # 由 Handler 明确说"别试了"）。
            #
            # 不静默返回的理由：那会留下一条**成功**的任务而什么都没做，
            # 在任务面板上看起来像跑过了。
            raise TaskPermanentError(
                f"造题运行不存在：{params.generation_run_id}。"
                "这通常意味着任务与它的运行行不同步（运行被删了、或库被"
                "回滚过）—— 重试不会让它出现。"
            )

        bundle = await self._bundles.get_bundle(run.bundle_id)

        if bundle is None:
            raise TaskPermanentError(
                f"数据集不存在：{run.bundle_id}（运行 "
                f"{run.generation_run_id} 指向它）。与上一条同理，"
                "这是永久性的。"
            )

        now = _now()

        run.status = EvalJobStatus.RUNNING
        run.started_at = run.started_at or now
        await self._bundles.update_generation_run(run)

        runtime = get_model_registry().get_runtime(
            agent_type=eval_generator.EVAL_AGENT_TYPE,
            model_role=ModelRole.EVAL_QA_GENERATION,
        )

        # 判金标是**另一个 role**（判据要保守，与出题的取舍不同）。
        #
        # ⚠️ **开关关着时连解析都不做。**
        #
        # 这不是省一次查表 —— `get_runtime` 在 role 没配绑定时会直接抛
        # `ModelResolutionError`。无条件解析的话，"judge 这一路还没接通"
        # 会让**整批造题失败**，而这恰恰是关闭开关本该避免的事。
        # 换句话说：关着的时候，judge 的配置对不对**与我们无关**。
        judge_enabled = bool(
            bundle.generation_config.get("enable_cross_chunk_judge")
        )

        runtime_judge = (
            get_model_registry().get_runtime(
                agent_type=adjudication.ADJUDICATE_AGENT_TYPE,
                model_role=ModelRole.EVAL_CROSS_CHUNK_JUDGE,
            )
            if judge_enabled
            else None
        )

        if not judge_enabled:
            logger.info(
                "跨 chunk 判金标**未启用**（默认关闭）—— 每道题的金标集合"
                "只含出题那一块，即 gold_chunk_ids 是**下界**。"
            )

        # DTD 的**名称与描述**（不是 id）。要在 `generate` 里再取一次：
        # create_bundle 那次取到的 definitions 不在这个作用域里。
        definitions = await self._definitions.get_many(
            [
                UUID(str(x))
                for x in (
                    bundle.selection_config.get(
                        "document_type_definition_ids", []
                    )
                    or []
                )
            ]
        )

        document_context = _document_context(definitions)

        # 判金标的候选池：本 bundle **全部窗口**的输入快照，按 document_id 分组。
        #
        # 为什么按文档分组（而不是拿整个池子两两比对）
        # ----------------------------------------
        # 出题时主体已经写进了问题（公司名/证券代码），所以**另一家公司的
        # 文档答不了这道题** —— 「特锐德 2007 年经审计净资产」在万科的档案
        # 里没有答案。这条过滤：
        #
        #     省掉绝大多数无意义的模型调用（跨公司抽样时通常一个候选都没有）
        #     同时避免把「相关」当成「能答」—— 那是判金标最容易犯的错，
        #     而它会让召回率虚高且无法察觉
        #
        # ⚠️ 代价要说清楚：随机跨公司抽样（20 块抽自 100 家）时，同一份
        # 文档极少贡献两块以上 —— 于是**候选池通常是空的、一次模型都不调**。
        # 这不是缺陷，是这类语料的本来形状：同一事实在另一家公司文档里
        # 出现，本来就答不了「关于这家公司」的问题。
        #
        # 真正会点亮它的是「一份文档贡献多块」的场景（同一事件在摘要段与
        # 详情段各出现一次）。按公司抽样、或把 N 调大，命中率就上来了。
        pool_by_document = await self._load_pool_by_document(
            params.generation_run_id
        )

        unfinished = await self._bundles.list_unfinished_windows(
            params.generation_run_id
        )

        batch = unfinished[:MAX_WINDOWS_PER_TASK_RUN]

        if len(unfinished) > len(batch):
            logger.info(
                "本次执行处理 %d/%d 个窗口（上限 %d）—— 其余留在 queued，"
                "再次提交同一任务即可继续。",
                len(batch),
                len(unfinished),
                MAX_WINDOWS_PER_TASK_RUN,
            )

        gate = asyncio.Semaphore(max(1, WINDOW_CONCURRENCY))

        async def run_one(window) -> None:

            # ⚠️ **先认领再跑。** 认领是一条带条件的 UPDATE，两个人同时
            # 提交任务时只有一个能拿到这个窗口 —— 而模型调用是按次计费的，
            # 重复那一遍的钱已经花掉了。见仓储的 `claim_window`。
            async with gate:

                if not await self._bundles.claim_window(
                    window.generation_window_id
                ):
                    logger.info(
                        "窗口 %d 已被别的执行认领，跳过。",
                        window.window_index,
                    )
                    return

                await self._run_window(
                    window=window,
                    bundle=bundle,
                    runtime=runtime,
                    runtime_judge=runtime_judge,
                    document_context=document_context,
                    pool_by_document=pool_by_document,
                )

        await asyncio.gather(*(run_one(window) for window in batch))

        await self._finalize(run=run, bundle=bundle)

    # ========================================================
    # 评测执行
    # ========================================================

    async def create_eval_run(
            self,
            request: CreateEvalRunRequest,
            *,
            created_by: UUID | None = None,
    ) -> CreateEvalRunAcceptedResponse:

        bundle = await self._bundles.get_bundle(request.bundle_id)

        if bundle is None:
            raise NotFoundError(f"数据集不存在：{request.bundle_id}")

        if bundle.status is not EvalBundleStatus.READY:
            raise ConflictError(
                f"数据集当前状态是 {bundle.status.value}，不能拿来评测。"
                "只有 ready 的数据集才是完整的 —— 用不完整的数据集跑出来的"
                "偏低召回率来自数据集本身，而它在结果里与"
                "「检索真的差」长得一模一样。"
            )

        # `max_k` 与"最终返回条数"的关系：**返回条数不超过判定口径**。
        #
        # `max_k` 是命中判定的截断位置，实际返回条数由 cutter.top_k 决定
        # （类默认 10）。一条排在 max_k 之外的结果既不会改变
        # `hit_rate_at_k` 的分子，也照常占用过滤/精排/裁剪的算力 ——
        # 所以这里把返回条数**压到不超过 max_k**。
        #
        # ⚠️ 它**不负责**让 `at_k` 名实相符
        # --------------------------------
        # 当某轮的实际返回条数 < max_k 时，分子只能数到"返回的那几条"。
        # 默认阶梯的第 1/2 轮就是这种情况（cutter 10/20 对着 max_k=50）：
        # **标签写着 50、量的还是 10**，而且不报错 —— 看起来只是"这个数
        # 偏低"。要让标识与事实一致，只能二选一：
        #
        #     a) 把每轮的返回条数抬到 ≥ max_k（方向相反）；
        #     b) 把 max_k 设成 ≤ 最窄那一轮的返回条数，即承认这批数字
        #        的口径就是 at_10。
        #
        # 曲线点同理：`HIT_RATE_CURVE_KS` 里的 20/50 要读得到，max_k 至少
        # 得是 50 —— "压到不超过 max_k"不会替调用方补齐这件事（此前那句
        # `max(..., max(HIT_RATE_CURVE_KS))` 会，但它与"不超过"互相矛盾）。
        #
        # 另外三段原样不动 —— 只认 model_fields_set，不能把"继承 DTD 配置"
        # 悄悄变成"重置成类默认值"。
        override = request.query_processor_override

        cutter = (
            override.cutter.model_copy(deep=True)
            if override is not None and override.cutter is not None
            else CutterConfig()
        )

        cutter.top_k = min(cutter.top_k, request.max_k)

        if override is None:
            override = RetrieveProcessorOverride(cutter=cutter)
        else:
            override = RetrieveProcessorOverride(
                fusioner=override.fusioner,
                filter=override.filter,
                reranker=override.reranker,
                cutter=cutter,
            )

        # ---- 把这一轮评测的**阶梯**定下来 ----
        #
        # ⚠️ 这一步此前**根本不存在**，而 `attempts` 在 DTO 里已经能被
        # 接受校验 —— 于是前端传的参数被接受、被校验、然后**丢掉**，
        # 跑的永远是内置三档（第 3 档带精排）。
        #
        # 存到 run 上而不是塞进任务参数，有两个实际好处：
        #   重跑/重试（reusable key、被租约回收后重投）读的是同一份阶梯，
        #   不依赖提交那一刻的参数；
        #   它同时是**可查的快照** —— "这几个数是按哪套阶梯跑出来的"
        #   事后必须答得上来（与 created_at 必须显式盖章同一类理由）。
        ladder, ladder_source = _resolve_ladder(request, override)

        now = _now()

        run = RetrievalEvalRun(
            eval_run_id=uuid4(),
            bundle_id=bundle.bundle_id,
            status=EvalJobStatus.QUEUED,
            max_k=request.max_k,
            # ⚠️ **必须是 `exclude_unset`，不能是 `exclude_none`。**
            #
            # 这一行决定"前端传的覆盖"到底覆盖了什么。`_apply_override`
            # 是按 `patch.model_fields_set` 合并的，而这里把 override 存成
            # JSON、执行时再 validate 回来 —— 这个往返必须**保住
            # fields_set**，否则前端只改一个字段会顺手把同段里的其它字段
            # 一起钉死。
            #
            # `exclude_none` 保不住：它保留所有**非 None** 的字段，而
            # pydantic 已经把没传的填成了默认值 —— 于是默认值变成"显式
            # 传入"。实测：
            #
            #     前端传 {"fusioner": {"rrf_k": 30}}
            #     exclude_none  ->  存成 {enabled, type, top_k, rrf_k}
            #                       fields_set = {enabled, rrf_k, top_k, type}
            #                       ↑ DTD 配的 top_k=100 会被重置成 50
            #     exclude_unset ->  存成 {rrf_k}；fields_set = {rrf_k}
            #
            # 这就是 `document_retrieve.yaml` 里"合并时只认 model_fields_set"
            # 那条警告说的坑，落在这里。
            query_processor_override=override.model_dump(
                mode="json", exclude_unset=True
            ),
            include_stage_trace=request.include_stage_trace,
            # 阶梯进快照 —— 它**改变结果的含义**（三轮 vs 一轮）。
            effective_config_snapshot={
                "attempts": [
                    spec.model_dump(mode="json", exclude_unset=True)
                    for spec in ladder
                ],
                "attempts_source": ladder_source,
            },
            total_cases=bundle.case_count,
            created_by=created_by,
            created_at=now,
        )

        await self._runs.add_run(run)

        await _submit_evaluation(run.eval_run_id)

        logger.info(
            "评测已启动：run=%s bundle=%s max_k=%d（cutter.top_k=%d）",
            run.eval_run_id,
            bundle.bundle_id,
            request.max_k,
            cutter.top_k,
        )

        return CreateEvalRunAcceptedResponse(
            eval_run_id=run.eval_run_id,
            bundle_id=bundle.bundle_id,
            status=EvalJobStatus.QUEUED,
            message=f"将对 {bundle.case_count} 条 case 各执行一次检索。",
        )

    async def get_eval_run(self, eval_run_id: UUID) -> EvalRunDetail:

        run = await self._runs.get_run(eval_run_id)

        if run is None:
            raise NotFoundError(f"评测轮次不存在：{eval_run_id}")

        return EvalRunDetail(
            eval_run_id=run.eval_run_id,
            bundle_id=run.bundle_id,
            status=run.status,
            max_k=run.max_k,
            include_stage_trace=run.include_stage_trace,
            query_processor_override=dict(run.query_processor_override),
            effective_config_snapshot=dict(
                run.effective_config_snapshot
            ),
            total_cases=run.total_cases,
            completed_cases=run.completed_cases,
            failed_cases=run.failed_cases,
            metrics=EvalMetrics(**run.metrics) if run.metrics else EvalMetrics(
                total_cases=run.total_cases,
                completed_cases=0,
                failed_cases=0,
                hit_count=0,
                miss_count=0,
                max_k=run.max_k,
            ),
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

    async def get_stage_stats(
            self,
            eval_run_id: UUID,
    ) -> StageStatsResponse:

        if await self._runs.get_run(eval_run_id) is None:
            raise NotFoundError(f"评测轮次不存在：{eval_run_id}")

        rows = await self._runs.aggregate_stage_stats(eval_run_id)

        return StageStatsResponse(
            eval_run_id=eval_run_id,
            stages=[
                StageStatItem(
                    stage_key=row["stage_key"],
                    source_key=row["source_key"],
                    observed_cases=row["observed_cases"],
                    executed_cases=row["executed_cases"],
                    average_input_count=row["average_input_count"],
                    average_output_count=row["average_output_count"],
                    gold_present_before_count=(
                        row["gold_present_before_count"]
                    ),
                    gold_present_after_count=(
                        row["gold_present_after_count"]
                    ),
                    gold_drop_count=row["gold_drop_count"],
                    gold_pass_rate=evaluation.pass_rate(
                        row["gold_present_before_count"],
                        row["gold_present_after_count"],
                    ),
                    drop_reason_counts=row["drop_reason_counts"],
                )
                for row in rows
            ],
        )

    async def list_run_cases(
            self,
            eval_run_id: UUID,
            *,
            hit: bool | None = None,
            document_type_definition_id: UUID | None = None,
            page: int = 1,
            page_size: int = 50,
    ) -> EvalRunCaseListResponse:

        if await self._runs.get_run(eval_run_id) is None:
            raise NotFoundError(f"评测轮次不存在：{eval_run_id}")

        # ⚠️ 先折叠成「一条 case 一行」，再筛、再分页。
        #
        # 逐轮放宽让同一条 case 有多行（一轮一行），按行分页的话：同一条
        # case 在列表里出现 2~3 次，而响应里没有轮次字段可以把它们分开
        # ——前端看到的是两条一模一样的行；`total` 也是行数（实测一轮 40
        # 条 case 报 84），于是「这条 case 到底中找到没有」在列表里答不
        # 上来。表头指标早就是这么折的，两条读路径没跟上。
        #
        # 筛选也必须在折叠**之后**：一条 case 第 1 轮没中、第 2 轮命中，
        # 它的结论是**命中**。按行筛 `hit=false` 会把它捞出来，而列表里
        # 那一行显示的 `gold_hit` 是 true。
        #
        # 先读**标量**（`*_briefs`）折叠，再按挑出来的一页取整行。
        #
        # 不一次读整行：那样每页都要把整个轮次的 `retrieved_chunk_ids`
        # （带分数的 top-N，实测均值 7.8 KB/行）读出来，而 1000 条 case
        # 三轮是二十多 MB —— 只为了丢掉其中的绝大部分。也**不做**
        # SQL 侧的 DISTINCT ON：那要在 SQL 里再抄一遍代表行规则，而规则
        # 只能有一份（`collapse_by_case`，另有单元用例钉着）。
        briefs = await self._runs.list_case_result_briefs(eval_run_id)

        results = evaluation.collapse_by_case(briefs)

        cases = await self._load_cases_by_id(
            [result.eval_case_id for result in results]
        )

        pairs: list[tuple[RetrievalEvalCase, RetrievalEvalCaseResult]] = []

        for result in results:

            case = cases.get(result.eval_case_id)

            if case is None:
                # 结果行指向一条读不出来的 case（不该发生）。跳过而不是
                # 编一条：一条字段全空的 case 比少一条更容易误导。
                logger.warning(
                    "结果 %s 指向的 case %s 读不出来，跳过。",
                    result.eval_case_result_id,
                    result.eval_case_id,
                )
                continue

            if (
                document_type_definition_id is not None
                and case.document_type_definition_id
                != document_type_definition_id
            ):
                continue

            # `is not` 而不是 `!=`：三值语义（NULL = 没跑到，既不属于
            # true 也不属于 false），见仓储接口那段说明。
            if hit is not None and result.gold_hit is not hit:
                continue

            pairs.append((case, result))

        # 排序键取自 case —— 就是旧 SQL 那三列（DTD、源 chunk、题号）。
        # 结果行的自然顺序是**按轮次分组**的，直接按它分页会让同一页里
        # 混着不同 DTD 的 case，而这件事前端靠着排序才读得懂。
        pairs.sort(
            key=lambda pair: (
                str(pair[0].document_type_definition_id),
                str(pair[0].source_chunk_id),
                pair[0].question_order,
            )
        )

        offset = (page - 1) * page_size
        page_pairs = pairs[offset:offset + page_size]

        # 这一页的**整行** —— 折叠时读的 briefs 没有候选与分数，而响应要。
        # 按 `(case, 轮次)` 取：折叠已经定了要哪一轮。
        full = await self._runs.list_results_by_identity(
            [
                (result.eval_case_id, result.attempt_no)
                for _, result in page_pairs
            ]
        )

        by_identity = {
            (item.eval_case_id, item.attempt_no): item for item in full
        }

        gold = await self._cases.list_gold_chunks(
            [case.eval_case_id for case, _ in page_pairs]
        )

        items: list[EvalRunCase] = []

        for case, brief in page_pairs:

            result = by_identity.get((brief.eval_case_id, brief.attempt_no))

            if result is None:
                # 折叠与取整行之间那一行没了（并发删了轮次）。与上面
                # 「case 读不出来」同一种处理：跳过并说出来，不编一行。
                logger.warning(
                    "结果行 %s 的第 %d 轮在读取途中消失，跳过。",
                    brief.eval_case_id,
                    brief.attempt_no,
                )
                continue

            items.append(
                _to_run_case(
                    case=case,
                    result=result,
                    gold_chunk_ids=[
                        item.chunk_id
                        for item in gold.get(case.eval_case_id, [])
                    ],
                )
            )

        return EvalRunCaseListResponse(
            items=items,
            page=page,
            page_size=page_size,
            # **case 数**，不是结果行数 —— 与轮次的 `total_cases` 同口径。
            total=len(pairs),
        )

    async def get_case_trace(
            self,
            eval_run_id: UUID,
            eval_case_id: UUID,
            *,
            attempt_no: int | None = None,
    ) -> EvalCaseTraceResponse:

        # ⚠️ 一条 case 在这一轮里可能有**多行**（逐轮放宽，一轮一行），
        # 所以这里拿到的是列表而不是一行。
        results = await self._runs.list_results_for_trace(
            eval_run_id, eval_case_id
        )

        if not results:
            raise NotFoundError(
                f"这一轮里没有这条 case 的结果：run={eval_run_id} "
                f"case={eval_case_id}"
            )

        if attempt_no is None:

            # 不指定轮次 = 与列表、表头同一条代表行。各写各的规则会让同一
            # 次评测在三个页面上给出三种「这条 case 的结果」。
            result = evaluation.representative_result(results)

        else:

            result = next(
                (
                    item for item in results
                    if item.attempt_no == attempt_no
                ),
                None,
            )

        if result is None:
            # 只可能是「指定了没有的那一轮」：上面刚确认过 `results` 非空，
            # 而非空就有代表行。
            raise NotFoundError(
                f"这条 case 在本轮没有第 {attempt_no} 轮的结果："
                f"run={eval_run_id} case={eval_case_id} "
                f"（实际有 {[item.attempt_no for item in results]} 轮）"
            )

        observations = await self._runs.list_stage_observations(
            result.eval_case_result_id
        )
        candidates = await self._runs.list_stage_candidates(
            result.eval_case_result_id
        )

        gold = await self._cases.list_gold_chunks([eval_case_id])

        gold_ids = {
            item.chunk_id for item in gold.get(eval_case_id, [])
        }

        # 按 (stage_key, source_key) 把候选挂回它所属的那一段。
        #
        # `stage_observation` 与 `stage_candidate` 是两张表，而响应里每段
        # 要同时带"金标进出"与"候选" —— 在这里拼起来，而不是让前端自己
        # 按 stage_key 对齐（那会把"两张表怎么拼"的知识泄漏到前端）。
        by_stage: dict[
            tuple[str, str], list[RetrievalEvalStageCandidate]
        ] = {}

        for candidate in candidates:

            by_stage.setdefault(
                (candidate.stage_key, candidate.source_key), []
            ).append(candidate)

        stages: list[StageTraceItem] = []

        for observation in observations:

            key = (observation.stage_key, observation.source_key)
            stage_candidates = by_stage.get(key, [])

            stages.append(
                StageTraceItem(
                    stage_key=observation.stage_key,
                    source_key=observation.source_key,
                    input_count=observation.input_count,
                    output_count=observation.output_count,
                    gold_present_before=observation.gold_present_before,
                    gold_present_after=observation.gold_present_after,
                    # ⚠️ 这两张 map 从**候选行**重建，不是从 observation。
                    #
                    # observation 上只存了"最好的那个金标"的名次与分数
                    # （一个数），而一道题可能有多个金标，各在何处都要能
                    # 看见 —— 只报最好的那个会掩盖"另一个金标掉到了第 40
                    # 名"。逐 chunk 的那份只在候选行里。
                    gold_ranks_before=_gold_rank_map(
                        stage_candidates, gold_ids, before=True,
                    ),
                    gold_scores_before=_gold_score_map(
                        stage_candidates, gold_ids, before=True,
                    ),
                    gold_ranks_after=_gold_rank_map(
                        stage_candidates, gold_ids, before=False,
                    ),
                    gold_scores_after=_gold_score_map(
                        stage_candidates, gold_ids, before=False,
                    ),
                    drop_reason=observation.drop_reason,
                    top_candidates=[
                        _to_scored_candidate(item)
                        for item in stage_candidates
                    ],
                )
            )

        return EvalCaseTraceResponse(
            eval_run_id=eval_run_id,
            eval_case_id=eval_case_id,
            # 这一份痕迹是**哪一轮**的，必须跟着响应走：逐轮放宽里第 1 轮
            # 与第 3 轮的参数不同（第 3 轮才开精排），不写出来的话这份痕迹
            # 会被当成「这条 case 的检索过程」，而它是「这一轮的检索过程」。
            attempt_no=result.attempt_no,
            available_attempts=[item.attempt_no for item in results],
            # 这一轮的 query 原文：复核痕迹时"它当时问的是什么"与
            # "读的是哪一轮"一样重要。
            query_text=result.query_text,
            gold_chunk_ids=sorted(gold_ids, key=str),
            stages=stages,
        )

    async def evaluate(
            self,
            params: RetrievalEvalEvaluateParams,
            ctx: Any,
    ) -> None:

        run = await self._runs.get_run(params.eval_run_id)

        if run is None:
            raise TaskPermanentError(
                f"评测轮次不存在：{params.eval_run_id}。重试不会让它出现。"
            )

        bundle = await self._bundles.get_bundle(run.bundle_id)

        if bundle is None:
            raise TaskPermanentError(
                f"数据集不存在：{run.bundle_id}（轮次 "
                f"{run.eval_run_id} 指向它）。"
            )

        cases = await self._cases.list_all_validated_cases(
            bundle.bundle_id
        )

        if not cases:
            raise TaskPermanentError(
                f"数据集 {bundle.bundle_id} 里没有校验通过的 case，"
                "没有可评测的东西。"
            )

        gold = await self._cases.list_gold_chunks(
            [case.eval_case_id for case in cases]
        )

        factory = get_service_factory()

        run.status = EvalJobStatus.RUNNING
        run.started_at = run.started_at or _now()
        run.total_cases = len(cases)
        # ⚠️ **合并**，不是覆盖 —— create_eval_run 已经把阶梯放在这里了，
        # 直接赋值会把它冲掉，而"这一轮按哪套阶梯跑的"就此丢失。
        run.effective_config_snapshot = {
            **run.effective_config_snapshot,
            **await self._snapshot_effective(bundle, cases),
        }

        # ⚠️ **这几行必须在自己的 scope 里提交一次。**
        #
        # 此前它写在**外层事务**里，而外层事务只有整轮跑完才提交 —— 于是
        # 运行期间前端读到的 `status` 还是 `queued`、`total_cases` 还是旧值，
        # 与"从没启动"完全分辨不出来。
        #
        # 分批提交只让**结果行**可见（那部分是对的），**运行状态没有** ——
        # 所以"进度可见"这件事只做成了一半。开跑前先把这几行提交掉，
        # 前端第一秒就能看到 `running` + 正确的 `total_cases`。
        async with factory.scope(RetrievalEvalService) as marker:
            await marker._runs.update_run(run)

        factory = get_service_factory()

        # ---- 已有结果的跳过：**这就是"被租约回收后接着跑"** ----
        #
        # 没有这一步，回收后的重跑会从第 1 条开始，把已经花掉的检索成本
        # 再付一遍 —— 而超过租约的轮次正是靠"分批提交 + 跳过已完成"
        # 才能分几次跑完的。
        #
        # `add_case_results` 本身有 ON CONFLICT DO NOTHING，所以**正确性**
        # 不依赖这一步；它省的是**检索的钱**。
        existing, _ = await self._runs.list_case_results(
            eval_run_id=run.eval_run_id, offset=0, limit=MAX_CASES_PER_RUN,
        )
        done_ids = {
            item.eval_case_id for item in existing
        }

        pending = [
            case for case in cases if case.eval_case_id not in done_ids
        ]

        # ⚠️ 这里**不能**再写一次 `run.status = RUNNING` / 快照 / update_run。
        #
        # 曾经有过——上面那块是它的更新版（合并快照 + 单独提交），而这一块
        # 是原版残留。它在上面那块**之后**执行，于是把刚合并好的
        # `effective_config_snapshot` 又用"不带 attempts 的覆盖版"写了一遍，
        # 把阶梯抹掉了。
        #
        # 症状：前端传了不带精排的阶梯、库里也存对了，而第 3 轮照样跑精排
        # ——因为下面读 `.get("attempts")` 读到的是 None，回落内置档。
        # **同一份状态被写两次、后一次赢**，而两次的内容不同。

        started = time.monotonic()

        logger.info(
            "评测开始：run=%s 共 %d 条 case（本轮待跑 %d 条，已有结果 %d 条，"
            "逐条超时 %.0f 秒，每 %d 条提交一次）",
            run.eval_run_id,
            len(cases),
            len(pending),
            len(done_ids),
            CASE_TIMEOUT_SECONDS,
            BATCH_COMMIT_EVERY,
        )

        # 阶梯从 **run 的快照**读，不从任务参数读。
        #
        # ⚠️ 这里此前写的是 `getattr(params, "attempts", None)` —— 而
        # `RetrievalEvalEvaluateParams` 上**根本没有** attempts 字段，于是
        # 它**恒为 None**，`or DEFAULT_ATTEMPT_LADDER` 永远走默认档。
        # 用 `getattr(..., None)` 兜一个"本该存在"的字段，正是让这个 bug
        # 一直沉默的原因：它把"字段不存在"伪装成了"这次没传"。
        
        request_attempts = [
            EvalAttemptSpec.model_validate(item)
            for item in (
                run.effective_config_snapshot.get("attempts") or []
            )
        ] or None


        # ---- 分批跑 + 分批提交 ----
        #
        # "提交"不是仓储层的动作（仓储禁止 commit），而是**每批开一个自己
        # 的 scope** —— scope 退出即提交。顺带让每批各开各的会话，一批里
        # 的故障不会污染下一批。
        # ---- 逐轮：第 N 轮只跑第 N-1 轮**没命中**的 ----
        #
        # ⚠️ 每轮的分母都不同（第 2 轮的输入是第 1 轮失败的那批），
        # 所以逐轮的命中率**必须连分母一起报**，否则会被读成「全体命中率」
        # 而错一个量级。
        attempt_specs = list(request_attempts or DEFAULT_ATTEMPT_LADDER)

        round_source = pending

        for attempt_no, spec in enumerate(attempt_specs, start=1):

            if not round_source:
                logger.info("第 %d 轮：无可跑 case，跳过。", attempt_no)
                break

            # 这一轮里已经跑过的跳过 —— 与整体的「接着跑」同一条理由：
            # 被租约回收后重跑时，前几轮的结果是上一次执行提交的。
            this_round, _ = await self._runs.list_case_results(
                eval_run_id=run.eval_run_id,
                attempt_no=attempt_no,
                offset=0,
                limit=MAX_CASES_PER_RUN,
            )
            already = {item.eval_case_id for item in this_round}
            batch_source = [
                case for case in round_source
                if case.eval_case_id not in already
            ]

            # ⚠️ **把这一轮实际用的参数打出来。**
            #
            # 只打"第 N 轮开始"是不够的：内置三档与前端传的阶梯在
            # 融合/裁剪上可能恰好相同（都是 200/30），而**精排**是唯一的
            # 判据。实测踩过：明明传了不带精排的阶梯，日志里却在跑精排，
            # 而因为没有这一行，无法从日志判断用的是哪一套。
            _rp = spec.retrieve_processor

            logger.info(
                "第 %d 轮开始（%s）：待跑 %d 条  召回=%s 融合=%s 裁剪=%s "
                "精排=%s  来源=%s",
                attempt_no,
                spec.name or "round%d" % attempt_no,
                len(batch_source),
                spec.recall_top_k,
                _rp.fusioner.top_k if _rp and _rp.fusioner else None,
                _rp.cutter.top_k if _rp and _rp.cutter else None,
                _rp.reranker.type if _rp and _rp.reranker else None,
                run.effective_config_snapshot.get("attempts_source"),
            )

            # 这一轮的题**一次性**改完，再逐条跑检索。
            #
            # 为什么在**这一层**批量：改写要调模型，而"这一轮有哪些题"只有
            # 这里知道。逐条改就是每条一次调用（第 2 轮几十条 = 几十次），
            # 而每次都要重付一遍 500 多 token 的系统提示词。批量之后是一
            # 次调用改一批 —— 而检索那边只看得到一条 query，所以结果要
            # 顺着 `query_rewrite` 传回去（见 `QueryRewriteInput`）。
            #
            # 失败姿态：`rewrite_many` 自己保证"永不抛"（批处理不成形时
            # 回落到逐条），最坏情况是每条都退回原文 —— 与逐条调用时一样。
            precomputed = await _precompute_rewrites(
                batch_source,
                spec.query_rewriter,
            )

            for start in range(0, len(batch_source), BATCH_COMMIT_EVERY):

                batch = batch_source[start:start + BATCH_COMMIT_EVERY]

                async with factory.scope(RetrievalEvalService) as writer:

                    for case in batch:

                        await writer._evaluate_one(
                            run=run,
                            bundle=bundle,
                            case=case,
                            gold_chunk_ids=[
                                item.chunk_id
                                for item in gold.get(case.eval_case_id, [])
                            ],
                            factory=factory,
                            attempt_no=attempt_no,
                            attempt=spec,
                            query_rewrite=precomputed.get(case.eval_case_id),
                        )

                done = min(start + len(batch), len(batch_source))
                elapsed = time.monotonic() - started
                rate = elapsed / max(1, done)

                logger.info(
                    "第 %d 轮进度 %d/%d 已提交（已用 %.0fs，%.2fs/条，"
                    "预计本轮剩余 %.0fs）",
                    attempt_no,
                    done,
                    len(batch_source),
                    elapsed,
                    rate,
                    rate * (len(batch_source) - done),
                )

            # 这一轮的未命中 = 下一轮的输入。
            round_rows, _ = await self._runs.list_case_results(
                eval_run_id=run.eval_run_id,
                attempt_no=attempt_no,
                offset=0,
                limit=MAX_CASES_PER_RUN,
            )
            misses = {
                item.eval_case_id for item in round_rows
                if not item.gold_hit
            }

            logger.info(
                "第 %d 轮结束：跑 %d 条，命中 %d 条，未命中 %d 条"
                "（分母是这一轮的输入，不是全部 case）",
                attempt_no,
                len(round_rows),
                len(round_rows) - len(misses),
                len(misses),
            )

            round_source = [
                case for case in round_source
                if case.eval_case_id in misses
            ]

        # ---- 收尾 ----
        #
        # 计数与指标**整体重算并写回**，不在循环里累加 —— 任务是至少一次
        # 投递，被 reclaimed 重跑时累加会算重（与生成侧同一条纪律）。
        # 重算还有个好处：它把**别人的批次**也算进去（回收后重跑时，前一半
        # 的结果是上一次执行提交的）。
        results, total_results = await self._runs.list_case_results(
            eval_run_id=run.eval_run_id, offset=0, limit=MAX_CASES_PER_RUN,
        )

        if total_results > len(results):
            # 静默截断会让指标基于一部分结果算出来 —— 那是个看起来正常
            # 但偏掉的数。说出来。
            logger.warning(
                "结果行 %d 条超过读取上界 %d，指标只基于前 %d 条 ——"
                "请调大 MAX_CASES_PER_RUN。",
                total_results,
                MAX_CASES_PER_RUN,
                len(results),
            )

        completed = sum(
            1 for item in results
            if item.status is EvalCaseResultStatus.COMPLETED
        )
        failed = len(results) - completed

        run.completed_cases = completed
        run.failed_cases = failed

        # ⚠️ 表头指标用**折叠后**的结果（每条 case 只留最好的一次）。
        #
        # 逐轮评测里一条 case 有多行，直接按行算会让"第 1 轮没中、第 2 轮
        # 命中"的那条同时进分母两次、分子一次 —— 于是逐轮放宽的收益被
        # 稀释掉（实测同一批数据：按行 43%、按 case 70%）。
        # 「哪一轮救回来的」由下面的 by_attempt 回答。
        run.metrics = evaluation.compute_metrics(
            evaluation.collapse_by_case(results), run.max_k,
        )

        # 逐轮分开统计 —— 这是「分三次 call」这套设计**能不能读**的关键。
        #
        # 第 N 轮的输入是第 N-1 轮未命中的那些，所以每轮的分母不同。
        # 混成一个数会让人以为「3 次找到 52%」是「一次 52%」。
        run.metrics["by_attempt"] = await self._metrics_by_attempt(run)
        run.completed_at = _now()

        if failed and completed:
            run.status = EvalJobStatus.PARTIAL_FAILED
        elif completed:
            run.status = EvalJobStatus.COMPLETED
        else:
            run.status = EvalJobStatus.FAILED

        await self._runs.update_run(run)

        logger.info(
            "评测收尾：run=%s 状态=%s %d/%d 条跑通，命中率@%d=%s",
            run.eval_run_id,
            run.status.value,
            completed,
            len(cases),
            run.max_k,
            run.metrics.get("hit_rate_at_k"),
        )

    # ========================================================
    # internals
    # ========================================================

    async def _run_window(
            self,
            *,
            window: RetrievalEvalGenerationWindow,
            bundle: RetrievalEvalBundle,
            runtime: Any,
            runtime_judge: Any,
            document_context: str,
            pool_by_document: dict[str, list[dict]],
    ) -> None:
        """
        跑一个窗口。**不抛异常** —— 任何失败都记进窗口行。

        抛出去会让任务重试，把已经成功的窗口的模型调用再付一遍钱。
        """

        questions_per_chunk = int(
            bundle.generation_config.get("questions_per_chunk", 3)
        )

        outcome = await eval_generator.generate_for_window(
            window_chunks=list(window.input_chunks),
            runtime=runtime,
            questions_per_chunk=questions_per_chunk,
            document_context=document_context,
        )

        inserted = 0

        if outcome.cases:

            # 产出里的 chunk_id 是**字符串**（模型原样回填它），而快照里
            # 是真 UUID —— 用同一个键化函数对齐，免得一边 str 一边 UUID
            # 查不到、然后静默少写几条 case。
            chunks = _chunk_by_id(window.input_chunks)

            cases: list[RetrievalEvalCase] = []

            for generated in outcome.cases:

                chunk = chunks.get(str(generated.chunk_id))

                if chunk is None:
                    # validate_items 已经保证 id 在本窗口内，走到这里说明
                    # 快照与产出对不上 —— 不该发生，但少写一条比写一条
                    # 归属错误的 case 好（后者会让金标指向别的块）。
                    logger.warning(
                        "窗口 %d 的产出引用了快照里没有的 chunk %s，丢弃。",
                        window.window_index,
                        generated.chunk_id,
                    )
                    continue

                cases.append(
                    RetrievalEvalCase(
                        eval_case_id=uuid4(),
                        bundle_id=bundle.bundle_id,
                        generation_window_id=(
                            window.generation_window_id
                        ),
                        document_type_definition_id=(
                            chunk["document_type_definition_id"]
                        ),
                        document_version_id=chunk.get(
                            "document_version_id"
                        ),
                        document_id=chunk.get("document_id"),
                        document_title=chunk.get("document_title"),
                        document_key=chunk.get("document_key"),
                        source_chunk_id=chunk["chunk_id"],
                        source_content_hash=(
                            chunk.get("content_hash") or ""
                        ),
                        question=generated.question,
                        answer_original=generated.answer_original,
                        answer_start_offset=(
                            generated.answer_start_offset
                        ),
                        answer_end_offset=(
                            generated.answer_end_offset
                        ),
                        question_order=generated.question_order,
                        question_sha256=generated.question_sha256,
                        validation_status=(
                            EvalCaseValidationStatus.VALIDATED
                        ),
                        created_at=_now(),
                    )
                )

            # ⚠️ 幂等插入：返回的是**真的写了几条**。重跑同一个窗口时，
            # 已经写过的会被唯一约束挡掉，于是这个数小于 len(cases) ——
            # 那正是"重试不产生重复"的落点，也是 valid_case_count 的口径。
            inserted = await self._cases.add_cases(cases)

            if inserted:
                # ---- ① 源 chunk：无条件就是金标 ----
                #
                # 它是**构造出来的事实**，不需要判 —— 题目就是从它出的。
                # 先落它，后面裁定失败也不会让这道题没有金标。
                await self._cases.add_gold_chunks(
                    [
                        RetrievalEvalCaseGoldChunk(
                            eval_case_gold_chunk_id=uuid4(),
                            eval_case_id=case.eval_case_id,
                            chunk_id=case.source_chunk_id,
                            is_source_chunk=True,
                            validation_method=(
                                EvalGoldValidationMethod.SOURCE_CHUNK
                            ),
                            source_content_hash=(
                                case.source_content_hash
                            ),
                            created_at=_now(),
                        )
                        for case in cases
                    ]
                )

                # ---- ② 跨 chunk 裁定：别的块是不是也答得上 ----
                #
                # 三重条件，缺一不可：
                #
                #   开关打开（API 参数，**默认 false**）
                #   真的插进去了 —— 重跑一个已完成的窗口时 inserted == 0，
                #                    金标也早就在库里了，再裁一遍是白付钱
                #   runtime_judge 非 None（开关关着时它是 None，见 generate）
                #
                # （部分重复时（inserted < len(cases)）会对整批再裁一次，
                # 少数几条重复付费。重试本来就少，不值得为它加一层
                # "哪几条是新的"的查询。）
                if runtime_judge is not None:
                    await self._adjudicate_cases(
                        cases=cases,
                        runtime=runtime_judge,
                        pool_by_document=pool_by_document,
                    )

        window.status = _window_status(outcome, inserted)
        window.valid_case_count = inserted
        window.raw_response = outcome.raw_response
        window.error_code = outcome.error_code
        window.error_message = outcome.error_message
        window.completed_at = _now()
        # `request_sha256` 不动：重试若改了输入（换了提示词、换了模型），
        # 首轮那个指纹是"这个窗口原本跑的是什么"的唯一记录。

        await self._bundles.update_window(window)

    async def _finalize(
            self,
            *,
            run: RetrievalEvalGenerationRun,
            bundle: RetrievalEvalBundle,
    ) -> None:
        """
        收尾：**从窗口行重算**计数与状态，然后定 bundle 的状态。

        重算而不是累加 —— 任务是至少一次投递，累加在重跑时算重。
        """

        counts = await self._bundles.count_windows_by_status(
            run.generation_run_id
        )

        total = sum(counts.values())
        completed = counts.get(EvalJobStatus.COMPLETED.value, 0)
        failed = (
            counts.get(EvalJobStatus.FAILED.value, 0)
            + counts.get(EvalJobStatus.PARTIAL_FAILED.value, 0)
        )
        queued = (
            counts.get(EvalJobStatus.QUEUED.value, 0)
            + counts.get(EvalJobStatus.RUNNING.value, 0)
        )

        case_count = await self._cases.count_cases(
            bundle.bundle_id,
            validation_status=EvalCaseValidationStatus.VALIDATED,
        )

        run.total_windows = total
        run.completed_windows = completed
        run.failed_windows = failed
        run.generated_cases = case_count

        if queued:
            # 本次执行到窗口数上限为止。**不推进 bundle 的状态**：还有
            # 窗口没跑，数据集就不完整，说它 ready 会让一次不完整的评测
            # 看起来像完整的。运行本身也留在 running（它确实还没跑完）。
            run.error_message = (
                f"本次执行处理到窗口数上限（{MAX_WINDOWS_PER_TASK_RUN}），"
                f"仍有 {queued} 个窗口未处理。再次触发该任务即可继续"
                "（已完成的窗口会被跳过）。"
            )
            await self._bundles.update_generation_run(run)
            return

        bundle.source_chunk_count = (
            await self._cases.count_distinct_source_chunks(bundle.bundle_id)
        )
        bundle.case_count = case_count

        if completed and failed:
            bundle.status = EvalBundleStatus.PARTIAL_FAILED
            run.status = EvalJobStatus.PARTIAL_FAILED
        elif completed:
            bundle.status = EvalBundleStatus.READY
            run.status = EvalJobStatus.COMPLETED
        else:
            bundle.status = EvalBundleStatus.FAILED
            run.status = EvalJobStatus.FAILED

        run.error_message = None
        run.completed_at = _now()

        if bundle.status in (
            EvalBundleStatus.READY,
            EvalBundleStatus.PARTIAL_FAILED,
            EvalBundleStatus.FAILED,
        ):
            bundle.completed_at = bundle.completed_at or _now()

        bundle.updated_at = _now()

        await self._bundles.update_generation_run(run)
        await self._bundles.update_bundle(bundle)

        logger.info(
            "造题收尾：bundle=%s 状态=%s 窗口 %d/%d 成功，case %d 条",
            bundle.bundle_id,
            bundle.status.value,
            completed,
            total,
            case_count,
        )

    async def _evaluate_one(
            self,
            *,
            run: RetrievalEvalRun,
            bundle: RetrievalEvalBundle,
            case: RetrievalEvalCase,
            gold_chunk_ids: list[UUID],
            factory: Any,
            attempt_no: int = 1,
            attempt: Any | None = None,
            query_rewrite: QueryRewriteInput | None = None,
    ) -> RetrievalEvalCaseResult:
        """
        跑一条 case，落它的结果、逐阶段观察与候选。**不抛异常**。

        抛出去会让另外 999 条白跑。而这一层不用 LLM，重跑便宜 —— 但没有
        理由让一次网络抖动把整轮结果作废。失败记进 `status=failed` +
        `gold_hit=NULL`，那是**故障**不是**质量**（见 compute_metrics）。

        ⚠️ 每条 case 各开一个 short scope 取检索服务，而不是整轮共用一个。
        ------------------------------------------------------------------
        共用的话，一次检索失败会让那个 session 的事务进入 aborted 状态，
        后面每一条都会跟着失败 —— 于是一次抖动变成整轮全挂。逐条开
        scope 的代价是多几次事务，而每次都很小。
        """

        request = evaluation.build_retrieve_request(
            case=case, run=run, gold_chunk_ids=gold_chunk_ids,
            attempt=attempt,
            # 这一轮的题在跑之前已经**批量改**过了（见 _precompute_rewrites）；
            # None = 这一轮没开改写，或批处理那一步整体放弃了。
            query_rewrite=query_rewrite,
        )

        started = time.monotonic()

        try:

            async with factory.scope(DocumentRetrieveService) as retriever:
                # ⚠️ **必须自己设超时。**
                #
                # 不设的话整轮就交给索引后端自己的超时了，而
                # `retrieval_config.timeout_ms` 默认是**不设** —— 一个不
                # 返回的后端能让这一轮无限等下去：没有 error、没有进度、
                # 库里也什么都看不到（一个事务）。
                #
                # 实测踩过一次：40 条的一轮，邻居两轮各 10–20 秒跑完，
                # 而某一轮卡了 527 秒还没结束。
                #
                # 超时算 `failed` + `gold_hit=NULL` —— 那正是"故障"与
                # "没命中"要分开的用处（见 compute_metrics）。
                response = await asyncio.wait_for(
                    retriever.retrieve_documents(request),
                    timeout=CASE_TIMEOUT_SECONDS,
                )

        except TimeoutError as exc:  # noqa: BLE001 —— 见 docstring

            logger.warning(
                "检索超时（case=%s，超过 %.0f 秒）：question=%r",
                case.eval_case_id,
                CASE_TIMEOUT_SECONDS,
                case.question[:60],
            )

            result = RetrievalEvalCaseResult(
                eval_case_result_id=uuid4(),
                eval_run_id=run.eval_run_id,
                eval_case_id=case.eval_case_id,
                # ⚠️ **必须带 attempt_no。** 漏掉它时实体默认是 1，于是
                # 第 2/3 轮的超时行会撞上第 1 轮那行 —— `add_case_results`
                # 是 ON CONFLICT DO NOTHING，结果是**这次超时一行都写不
                # 进去且不报错**：看起来像"这一轮全都没超时"。
                # （下面那个 except 分支一直带着它，只有这里漏了。）
                attempt_no=attempt_no,
                query_text=case.question,
                # **NULL，不是 False** —— 超时是故障，不是没命中。
                gold_hit=None,
                gold_rank=None,
                retrieved_chunk_ids=[],
                status=EvalCaseResultStatus.FAILED,
                error_code="retrieval_timeout",
                error_message=(
                    f"检索超过 {CASE_TIMEOUT_SECONDS:.0f} 秒未返回"
                ),
            )

            await self._runs.add_case_results([result])

            return result

        except Exception as exc:  # noqa: BLE001 —— 见 docstring

            logger.warning(
                "检索失败（case=%s question=%r）：%s: %s",
                case.eval_case_id,
                case.question[:60],
                type(exc).__name__,
                exc,
            )

            result = RetrievalEvalCaseResult(
                eval_case_result_id=uuid4(),
                eval_run_id=run.eval_run_id,
                eval_case_id=case.eval_case_id,
                attempt_no=attempt_no,
                # 失败行也带上"本来要查什么"：改写发生在服务内部，失败时
                # 拿不到响应回带的文本，退回落库的 `question` —— 至少不是
                # 空值（空值会让这一行连"问的是什么"都答不上来）。
                query_text=case.question,
                # **NULL，不是 False** —— 没跑到与没命中是两回事。
                gold_hit=None,
                gold_rank=None,
                retrieved_chunk_ids=[],
                status=EvalCaseResultStatus.FAILED,
                error_code=type(exc).__name__,
                error_message=str(exc)[:2000],
            )

            await self._runs.add_case_results([result])

            return result

        gold_ids = set(gold_chunk_ids)

        hit, rank = evaluation.judge_hit(
            response=response, gold_chunk_ids=gold_ids, max_k=run.max_k,
        )

        result = RetrievalEvalCaseResult(
            eval_case_result_id=uuid4(),
            eval_run_id=run.eval_run_id,
            eval_case_id=case.eval_case_id,
            attempt_no=attempt_no,
            # 与检索日志对齐的唯一凭据（按问题文本对是不行的 —— 同一个
            # 问题文本可能挂在不同的 chunk 上）。
            query_id=response.retrieve_id,
            # 这一轮实际发出去的文本。用**响应回带的**而不是
            # `case.question`：阶梯里每一轮的改写器不同，而 LLM 改写
            # 不可复现 —— 存 question 的话，"这轮到底查的是哪句话"就丢了。
            #
            # 回落 `case.question` 只兜"检索服务没回带"这一种情况（旧版
            # 服务、或没传改写器时它等于原文）。
            query_text=response.query_text or case.question,
            gold_hit=hit,
            gold_rank=rank,
            # **带分数**：逆向提问时别的 chunk 也可能答得上，只看 id 集合
            # 会丢掉判断"它算不算也该命中"的依据。
            retrieved_chunk_ids=[
                {
                    "chunk_id": str(item.document_chunk_id),
                    "rank": item.rank,
                    "score": item.score,
                    "score_name": None,
                    # ⚠️ 是 `pipeline_keys`（**列表**），不是 `retrieval_source`。
                    #
                    # `retrieval_source` 只存在于**融合之前**的候选
                    # （`RetrieveCandidate`）上 —— 那时一条候选只来自一路。
                    # 融合之后同一条 chunk 可能被多路同时命中，所以这里
                    # 是一个列表。
                    #
                    # 这个字段值得留全：README 明写"同一条 chunk 被多路
                    # 同时命中是融合最有价值的产出"—— 只留第一路等于把
                    # 那个信号丢掉。（`source_key` 只是给 DTO 用的单值。）
                    #
                    # （实测踩过：这里原本写的是 `item.retrieval_source`，
                    # 真跑一轮才炸 —— AttributeError，而构造的输入里
                    # 永远看不到。）
                    "pipeline_keys": list(item.pipeline_keys or []),
                    "source_key": (
                        (item.pipeline_keys or [None])[0]
                    ),
                    "metadata": dict(item.metadata or {}),
                }
                for item in response.results
            ],
            latency_ms=int((time.monotonic() - started) * 1000),
            # ⚠️ **必须记它。** 一次降级跑的 miss 与一次正常跑的 miss 在
            # 数据上原本完全一样，而前者是「那一路当时不可用」、后者才是
            # 「检索质量差」。检索响应里本来就有这个字段（README 明写
            # 「调用方应当检查它」），此前是评测这一层把它丢掉了。
            degraded_routes=[
                {
                    "retrieval_type": str(
                        item.retrieval_type.value
                        if hasattr(item.retrieval_type, "value")
                        else item.retrieval_type
                    ),
                    "document_retrieval_definition_id": str(
                        item.document_retrieval_definition_id
                    ),
                    "reason": item.reason,
                }
                for item in (response.degraded_routes or [])
            ],
            status=EvalCaseResultStatus.COMPLETED,
        )

        await self._runs.add_case_results([result])

        if response.trace is not None:

            observations, candidates = evaluation.build_stage_rows(
                eval_case_result_id=result.eval_case_result_id,
                trace=response.trace,
                gold_chunk_ids=gold_ids,
            )

            await self._runs.add_stage_observations(observations)
            await self._runs.add_stage_candidates(candidates)

        return result

    async def _snapshot_effective(
            self,
            bundle: RetrievalEvalBundle,
            cases: list[RetrievalEvalCase],
    ) -> dict[str, Any]:
        """
        记录本轮**实际生效**的检索配置（按 DTD）。

        不是冗余：DTD 的融合/精排配置**可改**，而结果是"在当时那套配置下"
        得出的。不存快照的话，改完 DTD 再去读历史结果，会以为那些数字是
        当前配置的表现。

        这里存的是**建库时的 DTD 配置 + 运行上那份 override 的合并结果**，
        与执行时逐条 case 实际用的东西同源（都走 `override_of`）。
        """

        definitions = await self._definitions.get_many(
            [
                UUID(str(x))
                for x in (
                    bundle.selection_config.get(
                        "document_type_definition_ids", []
                    ) or []
                )
            ]
        )

        return {
            "mode": "single_dtd_per_case",
            "max_k": None,
            "dtds": [
                {
                    "document_type_definition_id": str(
                        definition.document_type_definition_id
                    ),
                    "type_key": definition.type_key,
                    "name": definition.name,
                    "fusion_strategy": definition.fusion_strategy,
                    "fusion_config": dict(
                        definition.fusion_config or {}
                    ),
                    "rerank_strategy": definition.rerank_strategy,
                    "rerank_config": dict(
                        definition.rerank_config or {}
                    ),
                }
                for definition in definitions
            ],
            "case_count": len(cases),
            # 每案的 DTD 只有一个 —— 这是第一层的口径，写进快照免得事后
            # 有人以为测的是多 DTD 路由。
            "note": (
                "每条 case 只打它自己那一个 DTD（scope=single）。"
                "线上 agentic_rag 可能一次问多个 DTD，本层不测那个。"
            ),
        }

    async def _metrics_by_attempt(self, run: RetrievalEvalRun) -> dict:
        """
        逐轮的指标。

        ⚠️ 每轮的**分母不一样**（第 N 轮只跑第 N-1 轮未命中的那些），
        所以每一项都同时给出 `cases`（这一轮跑了几条）与 `input_share`
        （占全部 case 的多少）。只报比率会被读成「全体命中率」。
        这正是不把它并进总指标、而是单列一份的理由。
        """

        total = max(1, run.total_cases)
        by_attempt: dict[str, dict] = {}

        for attempt_no in range(1, 4):

            rows, _ = await self._runs.list_case_results(
                eval_run_id=run.eval_run_id,
                attempt_no=attempt_no,
                offset=0,
                limit=MAX_CASES_PER_RUN,
            )

            if not rows:
                continue

            metrics = evaluation.compute_metrics(rows, run.max_k)

            by_attempt[str(attempt_no)] = {
                **metrics,
                # 分母是这一轮实际跑了几条 —— 读比率时必须连它一起看。
                "cases": len(rows),
                "input_share": round(len(rows) / total, 6),
            }

        return by_attempt

    async def _load_cases_by_id(
            self,
            eval_case_ids: list[UUID],
    ) -> dict[UUID, RetrievalEvalCase]:
        """
        批量取 case，按 id 建索引。

        ⚠️ 不能拿 `list_cases(bundle_id=...)` 凑合：那个**按 bundle 分页**，
        而这里手上只有一批 case id。走 bundle 去捞要先把整个数据集的 case
        读一遍、再挑出当页那 50 条 —— 在 1000 条的数据集上是 20 倍的浪费，
        而且从结果上完全看不出来。
        """

        if not eval_case_ids:
            return {}

        found = await self._cases.get_many(eval_case_ids)

        return {case.eval_case_id: case for case in found}

    async def _load_pool_by_document(
            self,
            generation_run_id: UUID,
    ) -> dict[str, list[dict]]:
        """
        本 bundle 全部窗口的输入快照，按 `document_id` 分组。

        用**快照**而不是回源查 chunk 表：快照是重试时复现当时输入的权威
        来源，回源查会拿到"现在"的内容（源可能已被重新切分），于是裁定
        结果与出题时的正文对不上。

        没有 `document_id` 的块直接跳过（旧数据或文档被删）—— 它们进不了
        任何一组，也就不会被当成候选。**不静默兜底成"全池候选"**：那会
        把跨公司比对重新引回来，而那是判金标最容易犯的错。
        """

        windows = await self._bundles.list_windows(generation_run_id)

        pool: dict[str, list[dict]] = {}

        for window in windows:

            for chunk in window.input_chunks:

                document_id = str(chunk.get("document_id") or "")

                if not document_id:
                    continue

                pool.setdefault(document_id, []).append(chunk)

        return pool

    async def _adjudicate_cases(
            self,
            *,
            cases: list[RetrievalEvalCase],
            runtime: Any,
            pool_by_document: dict[str, list[dict]],
    ) -> int:
        """
        给这批 case 补**跨 chunk 金标**。返回补写了几行。

        为什么需要它
        ----------
        题目是**逆向**造的（从 chunk 出题），所以同一份档案里另一段把同一
        个事实又写一遍是很自然的。只把出题那块算金标的话，检索返回了另
        一块会被判 miss —— 那是**冤枉**：它确实答得上。这类假 miss 让召回
        率偏低，而偏低的部分从数据上完全看不出来。

        ⚠️ 候选只取**同一份文档**里的其它块。见 `generate` 里的长说明：
        出题时主体已写进问题，另一家公司的文档答不了它。

        裁定失败只记 warning：金标退化成"只有源 chunk"是**安全的默认值**
        （集合是下界，不是错的），而抛出去会让整个窗口陪葬。
        """

        if not cases:
            return 0

        gold: list[RetrievalEvalCaseGoldChunk] = []

        for case in cases:

            candidates = [
                chunk
                for chunk in pool_by_document.get(
                    str(case.document_id or ""), []
                )
                # 排除它自己那块 —— 它是金标这件事不需要判。
                if str(chunk.get("chunk_id")) != str(case.source_chunk_id)
            ]

            if not candidates:
                continue

            if len(candidates) > MAX_ADJUDICATION_CANDIDATES:

                # 截断**要说出来**：它直接缩小了金标搜索范围，而"金标集合
                # 比预期小"在结果里没有任何别的痕迹。
                logger.info(
                    "判金标候选被截断：case=%s 同文档候选 %d 个，"
                    "只判前 %d 个（调大 MAX_ADJUDICATION_CANDIDATES "
                    "可覆盖更多）。",
                    case.eval_case_id,
                    len(candidates),
                    MAX_ADJUDICATION_CANDIDATES,
                )

                candidates = candidates[:MAX_ADJUDICATION_CANDIDATES]

            outcome = await adjudication.adjudicate_question(
                question=case.question,
                answer_original=case.answer_original,
                candidates=candidates,
                runtime=runtime,
            )

            if outcome.error_message:
                logger.warning(
                    "判金标失败（case=%s，金标退化为只有源 chunk）：%s",
                    case.eval_case_id,
                    outcome.error_message,
                )
                continue

            by_id = {
                str(chunk.get("chunk_id")): chunk
                for chunk in candidates
            }

            for accepted in outcome.accepted:

                source_chunk = by_id.get(str(accepted.chunk_id))

                gold.append(
                    RetrievalEvalCaseGoldChunk(
                        eval_case_gold_chunk_id=uuid4(),
                        eval_case_id=case.eval_case_id,
                        chunk_id=accepted.chunk_id,
                        # **不是**源 chunk：这一行才是裁定真正救回来的那些。
                        # 统计里两者要分得开。
                        is_source_chunk=False,
                        validation_method=(
                            EvalGoldValidationMethod.LLM_CROSS_CHUNK_JUDGE
                        ),
                        evidence_excerpt=accepted.evidence_excerpt,
                        source_content_hash=(
                            None
                            if source_chunk is None
                            else source_chunk.get("content_hash")
                        ),
                        judgment=dict(accepted.judgment),
                        created_at=_now(),
                    )
                )

        if not gold:
            return 0

        # 幂等：重跑裁定不会在同一个 (case, chunk) 上留两行。
        return await self._cases.add_gold_chunks(gold)

    async def _load_contents(
            self,
            picked: list[dict],
    ) -> dict:
        """
        按抽中的 id 取正文。

        分页取而不是一次 `in_` 全部：抽中的可能有几万条，而一条 SQL 的
        参数上限（PostgreSQL 是 65535）与单次往返的体积都会成为问题。
        """

        by_id: dict = {}

        chunk_ids = [item["chunk_id"] for item in picked]

        for start in range(0, len(chunk_ids), _CONTENT_PAGE):

            for chunk in await self._chunks.list_by_ids(
                chunk_ids[start:start + _CONTENT_PAGE]
            ):
                by_id[chunk.document_chunk_id] = chunk

        return by_id


# ============================================================
# module helpers
# ============================================================

#: 取正文时的分页大小。
_CONTENT_PAGE = 500


def _resolve_ladder(
        request: CreateEvalRunRequest,
        calibrated_override: RetrieveProcessorOverride | None,
) -> tuple[list[EvalAttemptSpec], str]:
    """
    定出这次评测用哪套阶梯。返回 `(阶梯, 来源)`。

        请求给了 attempts    -> 用它，**忽略 query_processor_override**
                                （两者都表达"本轮用什么参数"，同时生效会有
                                 一个静默失效）
        没给                 -> 内置三档；若请求给了 query_processor_override，
                                它作为**第 1 轮**的四段（校准过 cutter.top_k
                                的那份），第 2/3 轮仍是内置档

    来源会被写进 `effective_config_snapshot.attempts_source` —— "这几个数
    是按哪套阶梯跑出来的"必须事后答得上来。
    """

    if request.attempts:
        return list(request.attempts), "request"

    ladder = list(DEFAULT_ATTEMPT_LADDER)

    if calibrated_override is not None:

        ladder[0] = ladder[0].model_copy(
            update={"retrieve_processor": calibrated_override}
        )

        return ladder, "default_ladder_with_request_round1"

    return ladder, "default_ladder"


async def _precompute_rewrites(
        cases: list[RetrievalEvalCase],
        config: QueryRewriterConfig | None,
) -> dict[UUID, QueryRewriteInput]:
    """
    按轮批量改写：`{case_id: 改写结果}`。没开改写就返回空 dict（= 不改）。

    ⚠️ **只改模型的那些 kind**：`common` 不调模型（几个正则），批量化对它
    没有收益 —— 交给检索那边逐条跑（`query_rewriter`）还更简单，也免得
    "预处理过的 common"与"服务里现算的 common"两处各算一遍。

    ⚠️ 改不动或失败时**照样回一条**（原文），而不是从 dict 里漏掉：漏掉会
    让那条 case 回落到"服务自己再改写一次"（如果这一轮配了改写器的话），
    于是一轮里出现两种行为。这里明确"要么都预处理过，要么都不"。
    """

    if (
        config is None
        or not config.enabled
        or config.kind is None
        or not cases
    ):
        return {}

    if config.kind is not QueryRewriterKind.LLM:
        return {}

    from ..retrieve.registry import get_registry

    rewriter = get_registry().rewriter(config.kind)

    questions = [case.question for case in cases]

    started = time.monotonic()

    results = await rewriter.rewrite_many(
        queries=questions, config=config,
    )

    # 条数必须一一对应 —— 批量最危险的失败是**串号**（把甲的改写安到乙
    # 头上），而调用方只按位置取，所以这里核对一次长度；长度不对就整体
    # 放弃预处理（让服务走它自己的逐条路径），绝不按错位的位置去配。
    if len(results) != len(cases):

        logger.warning(
            "批量改写返回 %d 条、输入 %d 条 —— 位置对不上，本轮不走预处理"
            "（改由检索侧逐条处理）。",
            len(results),
            len(cases),
        )

        return {}

    logger.info(
        "第 %d 条的题已批量改写完成：%d 条，耗时 %.1fs",
        len(cases),
        len(results),
        time.monotonic() - started,
    )

    return {
        case.eval_case_id: QueryRewriteInput(
            text=result.text,
            boost_phrases=list(result.boost_phrases),
            date_keys=list(result.date_keys),
        )
        for case, result in zip(cases, results)
    }


def _now() -> datetime:
    """
    带时区的当前时间。

    ⚠️ 数据库侧的 `server_default=now()` **不够用** —— 它是**事务开始
    时间**，同一事务里插的多行会拿到完全相同的时间戳，于是任何按
    `created_at` 排序的地方都会掉到 UUID 的 tiebreaker 上（随机）。
    见 retrieval_eval_bundle 仓储接口里的说明。
    """

    return datetime.now(timezone.utc)


def _new_seed() -> int:
    """
    服务端生成的抽样种子。

    用 `uuid4` 的位数而不是 `random`：种子要**持久化**，而"两个并发的
    建 bundle 请求拿到同一个种子"会让两份数据集抽到同一批 chunk ——
    那看起来像抽样没生效。
    """

    return uuid4().int % (2 ** 63)


def _chunk_by_id(input_chunks: list[dict]) -> dict:
    """窗口的输入快照按 chunk_id 建索引。"""

    return {
        str(chunk.get("chunk_id")): chunk for chunk in input_chunks
    }


def _window_status(outcome: Any, inserted: int) -> EvalJobStatus:
    """
    一个窗口最终算什么状态。

    三种情形，三种落法：

        有产出                  completed
        没产出但**没报错**      partial_failed
        报错了                  failed

    第二档（提示词明确允许的 `insufficient_fact`：模型说"这块凑不出
    那么多题"）**不能**算 `completed` —— 那会让 bundle 变 ready，而实际
    题数少于配置，评测出来的分母是错的却看不出来。

    它也不算 `failed`：那是提示词允许的结果，重试多半还是同一个答案，
    把它标成失败等于诱导人去重试一件不会变的事。

    第三档（`no_valid_case`：模型给了条目但全部没通过校验）算 `failed`
    —— 那说明它在编，值得重试，而 `failed` 正是重试端点会捡起来的。
    """

    if outcome.error_code is None:
        return (
            EvalJobStatus.COMPLETED
            if inserted
            else EvalJobStatus.PARTIAL_FAILED
        )

    return EvalJobStatus.FAILED


def _document_context(definitions: list[Any]) -> str:
    """
    给模型的**语料级背景**（不是主体 —— 主体是逐块给的）。

    ⚠️ 这里曾经给的是**一串裸 UUID**（`document_type_definition_ids`
    直接拼出来）。那是纯噪音：模型拿到 `28f35eaa-e276-...` 这类东西学不到
    任何东西，还占着提示词的注意力。改成 DTD 的**名称与描述**，那才是
    人能读懂、模型也用得上的信息（「上市公司大事提醒档案」）。

    主体的来源在别处：每一段正文上面标着「所属文档：特锐德（300001.SZ）」，
    由提示词模块的 `render_user_message` 逐块渲染 —— 因为一个窗口里的
    chunk 可能来自不同文档，全局给一次会把 A 公司的块写成 B 公司的题。
    """

    lines = [
        f"- {definition.name}"
        + (f"（{definition.type_key}）" if definition.type_key else "")
        + (f"：{definition.description}" if definition.description else "")
        for definition in definitions
    ]

    if not lines:
        return ""

    return "本次出题覆盖的文档类型：\n" + "\n".join(lines)


async def _submit_generation(generation_run_id: UUID) -> None:
    """
    提交（或再次提交）造题任务。

    ⚠️ 用 `scope(TaskService)` 而不是 `ScopedTaskSubmitter`
    ---------------------------------------------------
    与 memory 的 trigger.py 同一取舍：后者刻意不挂 dispatcher，靠 outbox
    relay 投递，那是给 Handler 派生子任务用的（子任务不着急）。这里希望
    尽快投递 —— 用户刚建完数据集就在看进度。

    ⚠️ `reusable_key=True` 是**必需**的
    --------------------------------
    默认的幂等键是**一次性**的：任务进终态之后键仍归它，再提交拿回的是
    那条已终结的任务（不会再跑）。而本层的"继续跑剩下的窗口"正是靠
    **再次提交同一个键**实现的（一次执行有窗口数上限，且不发子任务）。
    置 True 让键在任务进终态时被释放，同一个键于是可以反复用。

    失败只记 warning 并**不抛** —— 抛出去会让 create_bundle 在数据已经
    落库之后报 500，而调用方需要知道的是"数据集建好了但没跑起来"
    （bundle 停在 generating、没有任何窗口被处理，那是个可诊断的状态）。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        # 没跑过 lifespan（冒烟脚本 / 工具脚本）。不是故障。
        logger.debug(
            "[RetrievalEval] 任务基建不可用，跳过造题投递（run=%s）：%s",
            generation_run_id, exc,
        )
        return

    try:
        async with factory.scope(TaskService) as service:

            await service.submit(
                Task(
                    handler=RETRIEVAL_EVAL_GENERATE_HANDLER,

                    params=RetrievalEvalGenerateParams(
                        generation_run_id=generation_run_id,
                    ),
                    causation_id=generation_run_id,
                    idempotency_key=(
                        f"{RETRIEVAL_EVAL_GENERATE_HANDLER}:"
                        f"{generation_run_id}"
                    ),
                    # 见上面「reusable_key 是必需的」。
                    reusable_key=True,
                )
            )

    except Exception as exc:  # noqa: BLE001 —— 旁路，见 docstring
        logger.warning(
            "[RetrievalEval] 提交造题任务失败（run=%s，数据集已落库，"
            "可用重试端点补跑）：%s",
            generation_run_id, exc,
        )


async def _submit_evaluation(eval_run_id: UUID) -> None:
    """
    提交评测执行任务。

    与 `_submit_generation` 同一形状，包括 `reusable_key=True`（重跑一轮
    评测靠再次提交同一个键）与"提交失败只记 warning 不抛"（抛出去会让
    `create_eval_run` 在轮次已经落库之后报 500，而调用方需要知道的是
    "轮次建好了但没跑起来" —— 那是个可诊断的状态）。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        logger.debug(
            "[RetrievalEval] 任务基建不可用，跳过评测投递（run=%s）：%s",
            eval_run_id, exc,
        )
        return

    try:
        async with factory.scope(TaskService) as service:

            await service.submit(
                Task(
                    handler=RETRIEVAL_EVAL_EVALUATE_HANDLER,
                    params=RetrievalEvalEvaluateParams(
                        eval_run_id=eval_run_id,
                    ),
                    causation_id=eval_run_id,
                    idempotency_key=(
                        f"{RETRIEVAL_EVAL_EVALUATE_HANDLER}:{eval_run_id}"
                    ),
                    reusable_key=True,
                )
            )

    except Exception as exc:  # noqa: BLE001 —— 旁路，见 docstring
        logger.warning(
            "[RetrievalEval] 提交评测任务失败（run=%s，轮次已落库，"
            "可再次提交同一 id 补跑）：%s",
            eval_run_id, exc,
        )


def _to_run_case(
        *,
        case: RetrievalEvalCase,
        result: RetrievalEvalCaseResult,
        gold_chunk_ids: list[UUID],
) -> EvalRunCase:

    top = [
        ScoredChunkCandidate(
            chunk_id=UUID(str(item["chunk_id"])),
            rank=int(item["rank"]),
            score=item.get("score"),
            score_name=item.get("score_name"),
            score_direction=None,
            source_key=item.get("source_key"),
            selected=True,
            drop_reason=None,
            metadata=dict(item.get("metadata") or {}),
        )
        for item in (result.retrieved_chunk_ids or [])
        if item.get("chunk_id") and item.get("rank")
    ][:DEFAULT_TOP_CANDIDATES_PER_STAGE]

    return EvalRunCase(
        eval_case_id=case.eval_case_id,
        question=case.question,
        chunk_id=case.source_chunk_id,
        document_type_definition_id=case.document_type_definition_id,
        status=result.status,
        attempt_no=result.attempt_no,
        gold_hit=result.gold_hit,
        query_text=result.query_text,
        gold_rank=result.gold_rank,
        gold_chunk_ids=gold_chunk_ids or [case.source_chunk_id],
        retrieved_chunk_ids=[
            UUID(str(item["chunk_id"]))
            for item in (result.retrieved_chunk_ids or [])
            if item.get("chunk_id")
        ],
        top_candidates=top,
        latency_ms=result.latency_ms,
        error_code=result.error_code,
        error_message=result.error_message,
    )


def _to_scored_candidate(
        item: RetrievalEvalStageCandidate,
) -> ScoredChunkCandidate:
    """
    逐阶段候选行 -> 响应 DTO。

    ⚠️ 用哪一套排名/分数取决于它是活是死（与写入时同一个映射，反向用）：
    活下来的读 after，死在这一段的读 before。**不要**用
    `rank_after or rank_before` 这种写法 —— 它在 rank 为 0 或分数为 0.0 时
    会取错（`or` 把 0 当假值）。
    """

    return ScoredChunkCandidate(
        chunk_id=item.chunk_id,
        rank=(
            item.rank_after
            if item.rank_after is not None
            else item.rank_before
        ) or 1,
        score=(
            item.score_after
            if item.rank_after is not None
            else item.score_before
        ),
        score_name=(
            item.score_name_after
            if item.rank_after is not None
            else item.score_name_before
        ),
        score_direction=None,
        source_key=None,
        selected=item.selected,
        drop_reason=item.drop_reason,
        metadata=dict(item.metadata or {}),
    )


def _gold_rank_map(
        candidates: list[RetrievalEvalStageCandidate],
        gold_chunk_ids: set[UUID],
        *,
        before: bool,
) -> dict[str, int | None]:
    """
    `{chunk_id: 名次}` —— 只含金标，且**逐 chunk**。

    这是 `stage_observation` 上那个"最好的名次"给不出的东西：一道题可能
    有多个金标，只报最好的那个会掩盖"另一个金标掉到了第 40 名"。
    """

    result: dict[str, int | None] = {}

    for item in candidates:

        if item.chunk_id not in gold_chunk_ids:
            continue

        result[str(item.chunk_id)] = (
            item.rank_before if before else item.rank_after
        )

    return result


def _gold_score_map(
        candidates: list[RetrievalEvalStageCandidate],
        gold_chunk_ids: set[UUID],
        *,
        before: bool,
) -> dict[str, float | None]:
    """同 `_gold_rank_map`，取分数。"""

    result: dict[str, float | None] = {}

    for item in candidates:

        if item.chunk_id not in gold_chunk_ids:
            continue

        result[str(item.chunk_id)] = (
            item.score_before if before else item.score_after
        )

    return result


def _to_summary(
        bundle: RetrievalEvalBundle,
        latest: RetrievalEvalGenerationRun | None,
) -> EvalBundleSummary:

    return EvalBundleSummary(
        bundle_id=bundle.bundle_id,
        name=bundle.name,
        description=bundle.description,
        status=bundle.status,
        document_type_definition_ids=list(
            bundle.selection_config.get(
                "document_type_definition_ids", []
            )
        ),
        source_chunk_count=bundle.source_chunk_count,
        case_count=bundle.case_count,
        latest_generation_status=(
            None if latest is None else latest.status
        ),
        dataset_sha256=bundle.dataset_sha256,
        created_at=bundle.created_at,
        completed_at=bundle.completed_at,
    )


def _to_case_dto(
        case: RetrievalEvalCase,
        gold_chunk_ids: list[UUID],
) -> EvalCase:

    return EvalCase(
        eval_case_id=case.eval_case_id,
        document_type_definition_id=case.document_type_definition_id,
        document_version_id=case.document_version_id,
        document_retrieval_definition_id=(
            case.document_retrieval_definition_id
        ),
        document_id=case.document_id,
        document_title=case.document_title,
        document_key=case.document_key,
        chunk_id=case.source_chunk_id,
        # 源 chunk **必然**在金标里（它是构造出来的事实）。空列表只可能
        # 出现在迁移途中，兜一下比返回一个违反契约的空数组好。
        gold_chunk_ids=gold_chunk_ids or [case.source_chunk_id],
        question=case.question,
        answer_original=case.answer_original,
        answer_start_offset=case.answer_start_offset,
        answer_end_offset=case.answer_end_offset,
        question_order=case.question_order,
        validation_status=case.validation_status,
        validation_error=case.validation_error,
    )
