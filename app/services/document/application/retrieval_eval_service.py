"""
第一层检索评测的**数据集侧**服务契约。

这一层做三件事：建一份数据集（抽样 + 造题，异步）、把它读出来 / 导出、
以及失败了再跑一遍。**评测执行不在这里** —— 那一层不用 LLM，也没有
窗口、没有重试，形态与这里完全不同，将来单独一个契约。

为什么建 bundle 只做一半就返回
------------------------------
建 bundle 要做的事分成两段，长度差三个数量级：

    同步段   校验 DTD、抽 N 个 chunk、切窗口、落库、提交任务
              —— 纯数据库与 CPU，几十毫秒
    异步段   跑 M 个窗口的 LLM 调用 —— 几分钟到几十分钟

所以接口返回 `202 + generation_run_id`，进度由任务侧写进
`retrieval_eval_generation_run` / `_window`。

⚠️ 这两段**不在同一个事务里**，是有意的
------------------------------------
同步段提交之后任务才可能被投递（`TaskService.submit` 挂的是 post_commit
钩子）。让它们同一个事务的话，"任务已经派出去了但 bundle 回滚了"会变成
一个必然出现的孤儿。现在的顺序（先落库、提交、再投递）保证任务读到的
一定是已经存在的数据。
"""

from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas.enum import EvalBundleStatus, EvalCaseValidationStatus
from ..domain.schemas.retrieval_eval import (
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

__all__ = ["RetrievalEvalService"]


class RetrievalEvalService(BaseService):
    """检索评测数据集服务。"""

    # ========================================================
    # 数据集
    # ========================================================

    @abstractmethod
    async def create_bundle(
            self,
            request: CreateEvalBundleRequest,
            *,
            tenant_id: UUID | None = None,
            created_by: UUID | None = None,
    ) -> CreateEvalBundleAcceptedResponse:
        """
        建一份数据集并**启动**造题，返回 202 的响应体。

        Raises:
            NotFoundError: 有 DTD 不存在或已逻辑删除。
            ConflictError: 有 DTD 不是 ACTIVE。
            ValueError: 抽到的 chunk 数不足以切出任何窗口（通常是选错了
                DTD，或者语料还没跑过写入管线）。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_bundles(
            self,
            *,
            status: EvalBundleStatus | None = None,
            document_type_definition_id: UUID | None = None,
            page: int = 1,
            page_size: int = 20,
            tenant_id: UUID | None = None,
    ) -> EvalBundleListResponse:
        """
        分页列出 bundle，给选择器用。

        `document_type_definition_id` 是"哪些 bundle 覆盖了这个 DTD" ——
        它筛的是 `selection_config` 里记的抽样范围，**不是**逐条 case 去
        join。一个 bundle 的 DTD 范围在建成时就固定了，用它筛比扫 case 表
        便宜得多。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_bundle(
            self,
            bundle_id: UUID,
    ) -> EvalBundleDetail:
        """
        取一份 bundle 的明细。

        Raises:
            NotFoundError: 不存在。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_cases(
            self,
            bundle_id: UUID,
            *,
            document_type_definition_id: UUID | None = None,
            validation_status: EvalCaseValidationStatus | None = None,
            page: int = 1,
            page_size: int = 50,
    ) -> EvalCaseListResponse:
        """
        分页预览 bundle 里的 QA。

        每条都带 `gold_chunk_ids`（含源 chunk）—— 这是"逆向提问可能别的
        chunk 也有正确答案"在**读**这一侧的落点：人在复核时会看到一道题
        有好几个金标，而不是只有一个。
        """
        raise NotImplementedError

    @abstractmethod
    async def export_bundle(
            self,
            bundle_id: UUID,
    ) -> EvalBundleExport:
        """
        导出成确定性的 JSON。

        **只有 `ready` 能导出。** `partial_failed` 也挡住：拿一份不完整的
        数据集去跑评测，得到的偏低召回率来自数据集本身，而那种偏低在结果
        里与"检索真的差"长得一模一样。

        Raises:
            NotFoundError: 不存在。
            ConflictError: 状态不是 ready。
        """
        raise NotImplementedError

    # ========================================================
    # 造题进度与重试
    # ========================================================

    @abstractmethod
    async def get_generation_run(
            self,
            generation_run_id: UUID,
    ) -> GenerationRunDetail:
        """
        取一次造题运行的进度（含每个窗口的状态与错误）。

        Raises:
            NotFoundError: 不存在。
        """
        raise NotImplementedError

    @abstractmethod
    async def retry_failed_windows(
            self,
            generation_run_id: UUID,
    ) -> GenerationRetryAcceptedResponse:
        """
        重跑这次运行里**还没成功**的窗口。

        ⚠️ 实际行为比名字宽：它处理的是 `queued` **加上** `failed` /
        `partial_failed`，而不只是失败的那些。原因是本层**不发子任务**，
        而一次执行有窗口数上限（见 impl 的 `MAX_WINDOWS_PER_TASK_RUN`）
        —— 跑不完的窗口会留在 `queued`，再提交一次是继续跑它们的唯一
        途径，所以它们必须在这个端点的覆盖范围内。

        幂等由两件事保证，不需要调用方操心：

            已经 `completed` 的窗口不会被再跑（列表里就没有它们）
            真并发了的话，`claim_window` 的条件更新只让一个人拿到它

        Raises:
            NotFoundError: 不存在。
            ConflictError: 这次运行已经是终态成功、或没有可跑的窗口。
        """
        raise NotImplementedError

    # ========================================================
    # 评测执行
    # ========================================================

    @abstractmethod
    async def create_eval_run(
            self,
            request: CreateEvalRunRequest,
            *,
            created_by: UUID | None = None,
    ) -> CreateEvalRunAcceptedResponse:
        """
        在一份**已就绪**的数据集上启动一轮评测。

        Raises:
            NotFoundError: bundle 不存在。
            ConflictError: bundle 不是 `ready`。
            ValueError: `max_k` 超过生效配置能返回的条数（那种情况下
                `hit_rate_at_k` 的分母是假的 —— 见 `CreateEvalRunRequest`
                里 `max_k` 的说明）。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_eval_run(self, eval_run_id: UUID) -> EvalRunDetail:
        """
        取一轮评测的状态与指标。

        Raises:
            NotFoundError: 不存在。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_stage_stats(
            self,
            eval_run_id: UUID,
    ) -> StageStatsResponse:
        """
        宏观丢因统计：**哪一段丢最多、因为什么**。

        从 `stage_observation` 按 (stage_key, source_key, drop_reason)
        聚合而来。`metrics` 里那份是缓存，这里是真值。

        Raises:
            NotFoundError: 轮次不存在。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_run_cases(
            self,
            eval_run_id: UUID,
            *,
            hit: bool | None = None,
            document_type_definition_id: UUID | None = None,
            page: int = 1,
            page_size: int = 50,
    ) -> EvalRunCaseListResponse:
        """
        分页列出单条 case 的命中结果。**一条 case 一行**。

        逐轮放宽让同一条 case 在一轮评测里有 2~3 行（一轮一行），所以这里
        先折叠成**代表行**（`collapse_by_case`）再分页 —— 于是 `total` 是
        **case 数**，与轮次的 `total_cases` 对得上，而不是行数。要看逐轮，
        走 `EvalRunDetail.metrics["by_attempt"]`；要钉某一条的某一轮，走
        `get_case_trace(..., attempt_no=)`。

        `hit` 是三值的：传 `None` 时连"没跑到"的那些一起返回。**不要**
        把 `None` 当成 `false` —— failed 与 miss 是两件事。

        筛选也在折叠**之后**做，这与「代表行」是同一件事：一条 case 第 1
        轮没中、第 2 轮命中，它的结论是**命中** —— 按行筛 `hit=false` 会
        把它捞出来，与列表里那一行显示的 `gold_hit=true` 自相矛盾。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_case_trace(
            self,
            eval_run_id: UUID,
            eval_case_id: UUID,
            *,
            attempt_no: int | None = None,
    ) -> EvalCaseTraceResponse:
        """
        单条 case 的逐阶段痕迹：每一段进出的候选、分数，以及金标的进出。

        **从库里读**（不重跑检索）—— 这正是不落库的那条即时调试路径
        （`POST /documents/retrieve` 带 `include_stage_trace`）与这条的
        分工：那条当场返回、不留痕；这条读的是跑过之后存下来的。

        `attempt_no` 指定看**哪一轮**；不传则取代表行 —— 与列表、表头
        同一套规则（`representative_result`），于是三个页面上的「这条 case
        的结果」是同一条。实际有哪些轮次看响应里的 `available_attempts`。

        Raises:
            NotFoundError: 轮次不存在、这条 case 在本轮没有结果，或者
                指定了 `attempt_no` 而这一轮没有它的结果。
        """
        raise NotImplementedError

    # ========================================================
    # 后台任务
    # ========================================================

    @abstractmethod
    async def generate(
            self,
            params: Any,
            ctx: Any,
    ) -> None:
        """
        `retrieval_eval.generate` 的 handler 主体。

        ⚠️ 签名必须是 `(params, ctx)` 两个**位置**参数 —— 注册期会校验，
        少了它在**启动时**报错（见 app/infra/task/registry.py）。

        什么算成功、什么算失败
        --------------------
            单个窗口失败           **不算失败** —— 记进窗口行与运行计数，
                                  bundle 最终落 `partial_failed` / `failed`。
                                  抛出去只会让整批重跑，把已经成功的窗口
                                  的模型调用再付一遍钱。
            取不到运行 / 落库抛    抛出去，让任务重试（那些是基建故障，
                                  而重试是安全的：窗口认领与 case 插入都幂等）
        """
        raise NotImplementedError

    @abstractmethod
    async def evaluate(
            self,
            params: Any,
            ctx: Any,
    ) -> None:
        """
        `retrieval_eval.evaluate` 的 handler 主体。

        逐条 case 调一次检索（**每条只打它自己那一个 DTD**），判金标命中，
        并把 `response.trace` 落成逐阶段的观察与候选行。

        ⚠️ 与 `generate` 相反的失败语义：这里**单条失败要吞**（记进
        `case_result.status=failed`、`gold_hit=NULL`），因为一条检索挂了
        不该让另外 999 条白跑 —— 而这一层不用 LLM，重跑便宜，却没有理由
        让一次网络抖动把整轮结果作废。
        """
        raise NotImplementedError
