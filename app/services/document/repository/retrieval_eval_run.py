from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import (
    RetrievalEvalCaseResult,
    RetrievalEvalRun,
    RetrievalEvalStageCandidate,
    RetrievalEvalStageObservation,
)
from ..domain.schemas.enum import EvalJobStatus

__all__ = ["RetrievalEvalRunRepository"]


class RetrievalEvalRunRepository(BaseRepository):
    """
    一次评测轮次及其四层结果。

    四张表一个仓储，与生成侧（bundle + run + window 同处一个仓储）同一
    取舍：它们**只被同一条链路写**，拆开只会让服务层注入四个对象、在四个
    之间来回传。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()，
    事务生命周期由 Service 层拥有。
    """

    # ========================================================
    # Run
    # ========================================================

    @abstractmethod
    async def add_run(self, run: RetrievalEvalRun) -> RetrievalEvalRun:
        """新增一次评测轮次。不 commit。"""
        raise NotImplementedError

    @abstractmethod
    async def get_run(self, eval_run_id: UUID) -> RetrievalEvalRun | None:
        """按主键查询。"""
        raise NotImplementedError

    @abstractmethod
    async def update_run(
            self,
            run: RetrievalEvalRun,
    ) -> RetrievalEvalRun:
        """
        更新轮次（状态、进度、指标缓存）。

        ⚠️ 计数与指标都由调用方**整体写回**，不累加 —— 任务是至少一次
        投递，累加在重跑时会算重（与生成侧的 `count_windows_by_status`
        同一条纪律，见那个方法的说明）。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_runs(
            self,
            *,
            bundle_id: UUID | None = None,
            status: EvalJobStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[RetrievalEvalRun], int]:
        """分页查询。返回 (当前页, 总数)。排序 `created_at DESC, id DESC`。"""
        raise NotImplementedError

    # ========================================================
    # Case result
    # ========================================================

    @abstractmethod
    async def add_case_results(
            self,
            results: list[RetrievalEvalCaseResult],
    ) -> int:
        """
        批量新增单条结果，返回实际写入条数。

        ⚠️ 冲突即跳过（`ON CONFLICT DO NOTHING`）：`(eval_run_id,
        eval_case_id)` 上有唯一约束，重跑一轮评测时已经判过的那些不该
        再插一遍 —— 靠应用层"先查再插"是挡不住并发的（见生成侧
        `add_cases` 的同一条说明）。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_case_results(
            self,
            *,
            eval_run_id: UUID,
            attempt_no: int | None = None,
            offset: int = 0,
            limit: int = 50,
    ) -> tuple[list[RetrievalEvalCaseResult], int]:
        """
        分页查询结果行。返回 (当前页, 总数)。**总数是行数**：一条 case
        走到第 2、3 轮就有 2~3 行（见 `add_case_results` 的唯一约束）。

        顺序按 case 身份（DTD、源 chunk、题号）再按 `attempt_no`，
        所以同一页里同一个 case 的多行是挨着的。

        `attempt_no` 传了就只看那一轮（逐轮指标与「这一轮跑过哪些」用）。

        ⚠️ 这里**没有** `hit` / DTD 过滤：`gold_hit` 是**逐轮**的，
        而「这条 case 到底命中了吗」只有把同一 case 的多行折叠掉才答得
        出来（`collapse_by_case`）。在 SQL 里筛 `hit` 会捞出一条 case 第
        1 轮 miss 的那一行 —— 与它第 2 轮命中的结论自相矛盾。筛选请放在
        折叠之后。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_case_result_briefs(
            self,
            eval_run_id: UUID,
    ) -> list[RetrievalEvalCaseResult]:
        """
        本轮**全部**结果行，但只有标量列。

        存在的理由是一次「先折叠再分页」的读：折叠只需要
        `eval_case_id` / `attempt_no` / `gold_hit` / `gold_rank`
        （`status` 与 `latency_ms` 顺手带上，指标也用得着），而
        **不折叠就没法分页** —— 所以这一步必须先把整个轮次过一遍。

        ⚠️ 不能拿 `list_case_results(limit=MAX_CASES_PER_RUN)` 凑合：
        那会把每行的 `retrieved_chunk_ids` 也读出来，而它是**带分数的
        top-N**（实测均值 7.8 KB、最大 20 KB）。1000 条 case 三轮就是
        3000 行、二十多 MB —— 为的只是从几个标量里挑出「哪一行代表这条
        case」，然后丢到只剩一页。翻一页收一次这个代价。

        返回的实体**只有上面那几列**有值：`retrieved_chunk_ids` /
        `degraded_routes` / `error_*` 都是默认值。它只够折叠与算指标，
        要整行请用 `list_results_by_identity` 或 `list_results_for_trace`。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_results_by_identity(
            self,
            keys: list[tuple[UUID, int]],
    ) -> list[RetrievalEvalCaseResult]:
        """
        按 `(eval_case_id, attempt_no)` 精确取若干**整行**。

        折叠之后挑出来的那一页就靠它取：按 `(case, 轮次)` 取而不是按
        `case` 取，是因为折叠已经定了要哪一轮 —— 按 case 取会把这条 case
        另外那两轮的 JSONB 也拖回来（3 倍）。空列表返回空。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_results_for_trace(
            self,
            eval_run_id: UUID,
            eval_case_id: UUID,
    ) -> list[RetrievalEvalCaseResult]:
        """
        取一条 case 在本轮的**全部轮次**结果，按 `attempt_no` 升序。
        单条 case 的痕迹页要用它。

        ⚠️ 不是「一行」。唯一约束是 `(eval_run_id, eval_case_id,
        attempt_no)` —— 逐轮放宽让走到第 2、3 轮的 case 有 2~3 行。所以
        这里的返回是**列表**：取哪一行（代表行，还是指定的某一轮）是
        调用方的决定，仓储只负责把这条 case 的行原样取回来。

        （早先这里返回单行、按 `(run, case)` 唯一定位；那个前提在
        `attempt_no` 加进来之后就不成立了，而它失败的方式是
        `MultipleResultsFound` 直接 500 —— 见
        tests/services/document/test_retrieval_eval_run_repository.py）
        """
        raise NotImplementedError

    # ========================================================
    # 逐阶段
    # ========================================================

    @abstractmethod
    async def add_stage_observations(
            self,
            observations: list[RetrievalEvalStageObservation],
    ) -> int:
        """批量新增逐阶段金标观察（幂等，冲突跳过）。"""
        raise NotImplementedError

    @abstractmethod
    async def add_stage_candidates(
            self,
            candidates: list[RetrievalEvalStageCandidate],
    ) -> int:
        """批量新增逐阶段候选（幂等，冲突跳过）。"""
        raise NotImplementedError

    @abstractmethod
    async def list_stage_observations(
            self,
            eval_case_result_id: UUID,
    ) -> list[RetrievalEvalStageObservation]:
        """取一条结果的全部阶段观察，按 `stage_key, source_key` 升序。"""
        raise NotImplementedError

    @abstractmethod
    async def list_stage_candidates(
            self,
            eval_case_result_id: UUID,
    ) -> list[RetrievalEvalStageCandidate]:
        """
        取一条结果的全部阶段候选。

        ⚠️ **不分页、也不截断**：这是单个 case 的诊断视图，行数由
        top-N ∪ 被盯住的上界决定（每阶段几十行）。它不参与宏观统计 ——
        那个走 `aggregate_stage_stats`。
        """
        raise NotImplementedError

    @abstractmethod
    async def aggregate_stage_stats(
            self,
            eval_run_id: UUID,
    ) -> list[dict]:
        """
        按 (stage_key, source_key) 聚合出一条轮次的宏观丢因统计。

        返回元素形状：

            {
              "stage_key": str, "source_key": str,
              "observed_cases": int, "executed_cases": int,
              "average_input_count": float | None,
              "average_output_count": float | None,
              "gold_present_before_count": int,
              "gold_present_after_count": int,
              "gold_drop_count": int,
              "drop_reason_counts": {reason: count},
            }

        ⚠️ **在 SQL 里聚合，不要把行捞进内存再算**：一轮 1000 条 case ×
        ~5 个阶段就是 5000 行，而"哪一段丢最多"只需要几个 group by。
        真值仍在 `stage_observation` 里（run 上的 `metrics` 只是缓存）。

        `gold_drop_count` 的口径必须是**进阶段前还在、出阶段后没了** ——
        聚合本身不判这个（它只数 `drop_reason` 非空的行），归因判断在写入
        观测时就已经做完。见 `RetrievalEvalStageObservation` 的说明。
        """
        raise NotImplementedError
