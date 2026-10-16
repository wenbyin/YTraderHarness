"""
第一层检索评测的端点。

契约权威是 `rag_eval_design/retrieval_eval_openapi.yaml` —— 本文件是它的
路由实现，路径、状态码、参数名与那份 yaml 一一对应。

这一组端点的**主语**看路径第二段就能读出来：

    /retrieval-eval/bundles            数据集本身（建、列、看、导出）
    /retrieval-eval/generation-runs     造题执行（进度、重试）
    /retrieval-eval/runs                评测执行（进度、指标、宏观丢因、逐条痕迹）

两个 POST 都是**异步**的（202），因为真正的活在后台任务里：造题要调几百
次模型（分钟级），评测要逐条调检索。同步做会让一个 HTTP 请求挂几十分钟。

**只有造题用 LLM。** 评测执行是纯 ID 比对 —— 判"金标在不在返回里"，
没有生成、没有 LLM 判分。这是第一层的定义（它测召回，不测生成）。

⚠️ 逐阶段痕迹有**两条**路径，分工不同
-----------------------------------
    POST /documents/retrieve 带上 include_stage_trace
        **即时**：当场返回漏斗，不落库。适合"随手输一句、看这次为什么没召回"
    GET  /runs/{id}/cases/{case}/trace
        **读库**：读的是跑过的评测轮次里存下来的痕迹，含金标进出与
        top-N 候选分数。适合"这道基准题为什么没命中"

⚠️ 身份字段（created_by / tenant_id）目前**留空**
-----------------------------------------------
`AuthenticatedUser` 的身份是**不透明的字符串**（Casdoor 的 `sub` /
`org_id`，实测形如 `built-in/admin` 这种也可能出现），而数据集的这两列
是 UUID。硬转会为那些非 UUID 的 subject 抛异常 —— 变成"建数据集时 500"，
而原因与建数据集无关。

所以先留 NULL，等"身份怎么映射到 UUID 列"这件事有明确答案再接。列可空，
不影响任何现有行为；`created_by` 半填（能转的填、不能转的留空）比全留空
更糟 —— 那会让"这一列没值"有两种互相矛盾的读法。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, status

from app.common.utils import to_http_error

from ...services.dependencies import Service
from ...services.document.application.retrieval_eval_service import (
    RetrievalEvalService,
)
from ...services.document.domain.schemas.enum import (
    EvalBundleStatus,
    EvalCaseValidationStatus,
)
from ...services.document.domain.schemas.retrieval_eval import (
    CreateEvalBundleAcceptedResponse,
    CreateEvalBundleRequest,
    CreateEvalRunAcceptedResponse,
    CreateEvalRunRequest,
    EvalBundleDetail,
    EvalBundleExport,
    EvalBundleListResponse,
    EvalCaseListResponse,
    EvalCaseTraceResponse,
    EvalRunCaseListResponse,
    EvalRunDetail,
    GenerationRetryAcceptedResponse,
    GenerationRunDetail,
    StageStatsResponse,
)

router = APIRouter(
    tags=["Retrieval Eval"],
)


# ============================================================
# 数据集
# ============================================================


@router.post(
    "/retrieval-eval/bundles",
    response_model=CreateEvalBundleAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_retrieval_eval_bundle(
        request: CreateEvalBundleRequest,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> CreateEvalBundleAcceptedResponse:
    """
    建一份 QA 数据集并启动造题。

    **202 而不是 201**：资源确实建出来了（bundle 行已经落库），但这份
    数据集**还不可用** —— 造题是分钟级的后台任务。202 表达的是"受理了
    一个还没完成的动作"，而 201 会让调用方以为可以立刻去导出了。
    """

    try:
        return await service.create_bundle(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/bundles",
    response_model=EvalBundleListResponse,
)
async def list_retrieval_eval_bundles(
        status_: EvalBundleStatus | None = Query(
            default=None, alias="status",
        ),
        document_type_definition_id: UUID | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=200),
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> EvalBundleListResponse:
    """列出数据集，给选择器用。"""

    try:
        return await service.list_bundles(
            status=status_,
            document_type_definition_id=document_type_definition_id,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/bundles/{bundle_id}",
    response_model=EvalBundleDetail,
)
async def get_retrieval_eval_bundle(
        bundle_id: UUID,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> EvalBundleDetail:
    """取一份数据集的明细（含抽样/造题配置与最近一次造题状态）。"""

    try:
        return await service.get_bundle(bundle_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/bundles/{bundle_id}/cases",
    response_model=EvalCaseListResponse,
)
async def list_retrieval_eval_cases(
        bundle_id: UUID,
        document_type_definition_id: UUID | None = Query(default=None),
        validation_status: EvalCaseValidationStatus | None = Query(
            default=None,
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> EvalCaseListResponse:
    """
    预览数据集里的 QA。

    每条都带 `gold_chunk_ids`（含源 chunk）—— 一道逆向造出来的题可能在
    别的块里也有正确答案，复核时看得到全部金标。
    """

    try:
        return await service.list_cases(
            bundle_id,
            document_type_definition_id=document_type_definition_id,
            validation_status=validation_status,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/bundles/{bundle_id}/export",
    response_model=EvalBundleExport,
)
async def export_retrieval_eval_bundle(
        bundle_id: UUID,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> EvalBundleExport:
    """
    把一份**已就绪**的数据集导出成确定性 JSON。

    非 ready 会得到 409 而不是一份不完整的数据集 —— 拿不完整的数据集去
    跑评测，得到的偏低召回率来自数据集本身，而它在结果里与"检索真的差"
    长得一模一样。
    """

    try:
        return await service.export_bundle(bundle_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 造题进度
# ============================================================


@router.get(
    "/retrieval-eval/generation-runs/{generation_run_id}",
    response_model=GenerationRunDetail,
)
async def get_retrieval_eval_generation_run(
        generation_run_id: UUID,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> GenerationRunDetail:
    """取一次造题的进度，含每个窗口的状态与错误。"""

    try:
        return await service.get_generation_run(generation_run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/retrieval-eval/generation-runs/{generation_run_id}"
    "/retry-failed-windows",
    response_model=GenerationRetryAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_retrieval_eval_generation_windows(
        generation_run_id: UUID,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> GenerationRetryAcceptedResponse:
    """
    重跑这次造题里**还没成功**的窗口。

    ⚠️ 实际覆盖范围比名字宽一点：`queued` 也算。原因是造题一次执行有
    窗口数上限（`MAX_WINDOWS_PER_TASK_RUN`），跑不完的会留在 `queued`，
    而本层不发子任务 —— 再调一次这个端点就是继续跑它们的唯一途径。

    幂等：已 `completed` 的窗口不会再跑；真并发了的话，窗口认领是原子的，
    只有一个人能拿到（不会重复付模型的钱）。
    """

    try:
        return await service.retry_failed_windows(generation_run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 评测执行（尚未实现）
# ============================================================


@router.post(
    "/retrieval-eval/runs",
    response_model=CreateEvalRunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_retrieval_eval_run(
        request: CreateEvalRunRequest,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> CreateEvalRunAcceptedResponse:
    """
    在一份**已就绪**的数据集上启动一轮评测。

    逐条 case 各打一次它自己那个 DTD 的检索，判金标命中。**不调 LLM** ——
    造题才用模型，跑评测是纯 ID 比对。

    与建数据集同理是 202：case 多起来是分钟级的事，而返回体只说明"受理了"。
    """

    try:
        return await service.create_eval_run(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/runs/{eval_run_id}",
    response_model=EvalRunDetail,
)
async def get_retrieval_eval_run(
        eval_run_id: UUID,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> EvalRunDetail:
    """取一轮评测的状态与指标（含 Hit@1/5/K、MRR、延迟分位）。"""

    try:
        return await service.get_eval_run(eval_run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/runs/{eval_run_id}/stage-stats",
    response_model=StageStatsResponse,
)
async def get_retrieval_eval_stage_stats(
        eval_run_id: UUID,
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> StageStatsResponse:
    """
    宏观丢因统计：**哪一段丢最多、因为什么**。

    按 (stage_key, source_key) 分组，每组带 `drop_reason_counts`。归因
    规则是"进这一段前金标还在、出去后没了"才算这一段的锅 —— 所以
    `gold_drop_count` 是可以照着改配置的数字，而不是把上游的账也算进来。
    """

    try:
        return await service.get_stage_stats(eval_run_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/runs/{eval_run_id}/cases",
    response_model=EvalRunCaseListResponse,
)
async def list_retrieval_eval_run_cases(
        eval_run_id: UUID,
        hit: bool | None = Query(
            default=None,
            description=(
                "按命中与否筛选。**不传时连「没跑到」的那些一起返回** —— "
                "检索执行失败（gold_hit=null）与没命中是两回事，"
                "把它们并成 false 会让「检索挂了」看起来像「召回变差」。"
                "\n\n"
                "筛的是**折叠后**的结论（这条 case 最终找到没有），不是某"
                "一轮的结果：一条 case 第 1 轮没中、第 2 轮命中，它就属于"
                "`hit=true`。"
            ),
        ),
        document_type_definition_id: UUID | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> EvalRunCaseListResponse:
    """
    分页列出单条 case 的命中结果（含最终 top-5 与分数）。

    **一条 case 一行**：逐轮放宽让同一条 case 在库里有多行（一轮一行），
    这里给的是**代表行**（命中优先，再比名次）—— 于是 `total` 是 case 数，
    与轮次的 `total_cases` 同口径。哪一轮救回来的看轮次详情里的
    `by_attempt`；某一条的某一轮走 trace 那条（`attempt_no`）。
    """

    try:
        return await service.list_run_cases(
            eval_run_id,
            hit=hit,
            document_type_definition_id=document_type_definition_id,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/retrieval-eval/runs/{eval_run_id}/cases/{eval_case_id}/trace",
    response_model=EvalCaseTraceResponse,
)
async def get_retrieval_eval_case_trace(
        eval_run_id: UUID,
        eval_case_id: UUID,
        attempt_no: int | None = Query(
            default=None,
            ge=1,
            description=(
                "看**哪一轮**（逐轮放宽，一轮一行）。"
                "\n\n"
                "不传 = 代表行那一轮（命中优先，再比名次）—— 与列表、"
                "表头指标同一条，所以默认情况下三个页面上的「这条 case 的"
                "结果」是同一个。"
                "\n\n"
                "可选的轮次见响应里的 `available_attempts`：第 1 轮就命中"
                "的 case 只有 `[1]`，三轮都没命中的是 `[1, 2, 3]`。"
            ),
        ),
        service: RetrievalEvalService = Service[RetrievalEvalService],
) -> EvalCaseTraceResponse:
    """
    单条 case 的逐阶段痕迹：每一段进出的候选与分数，以及**金标在哪一段被丢**。

    **从库里读**，不重跑检索 —— 与 `POST /documents/retrieve` 带
    `include_stage_trace` 那条即时路径的分工是：那条当场返回、不留痕；
    这条读的是跑过之后存下来的。

    ⚠️ 一条 case 在这一轮里可能有**多行**（逐轮放宽，一轮一行），所以
    返回的痕迹必定带 `attempt_no` —— 第 3 轮才开精排，不说清是哪一轮
    的话这份痕迹会被读成「这条 case 的检索过程」。
    """

    try:
        return await service.get_case_trace(
            eval_run_id, eval_case_id, attempt_no=attempt_no
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
