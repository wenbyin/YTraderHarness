from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import (
    RetrievalEvalBundle,
    RetrievalEvalGenerationRun,
    RetrievalEvalGenerationWindow,
)
from ..domain.schemas.enum import EvalBundleStatus, EvalJobStatus

__all__ = ["RetrievalEvalBundleRepository"]


class RetrievalEvalBundleRepository(BaseRepository):
    """
    bundle 及其造题尝试的持久化抽象。

    三张表一个仓储（bundle / generation_run / generation_window），与
    `PipelineRunRepository` 同时管 run 与 node_run 是同一取舍：它们**只
    被同一条链路写**，拆开只会让服务层注入三个对象、在三个之间来回传。

    窗口是其中唯一会被单独按状态查的表（重试要挑出失败的那些），所以
    它的读接口比另外两张细。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()，
    事务生命周期由 Service 层拥有。
    """

    # ========================================================
    # Bundle
    # ========================================================

    @abstractmethod
    async def add_bundle(
            self,
            bundle: RetrievalEvalBundle,
    ) -> RetrievalEvalBundle:
        """
        新增一份 bundle。

        只操作 PostgreSQL。不 commit。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_bundle(
            self,
            bundle_id: UUID,
    ) -> RetrievalEvalBundle | None:
        """按主键查询。不存在返回 None。"""
        raise NotImplementedError

    @abstractmethod
    async def update_bundle(
            self,
            bundle: RetrievalEvalBundle,
    ) -> RetrievalEvalBundle:
        """
        更新一份 bundle。

        ⚠️ 调用方负责**不**在 `ready` 之后改内容字段 —— 仓储不做这个
        判断。它是业务规则（"数据集不可变"），而业务规则住服务层；放这里
        会让"为什么这笔写入被拒"的答案藏在一个不起眼的地方。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_bundles(
            self,
            *,
            tenant_id: UUID | None = None,
            status: EvalBundleStatus | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[RetrievalEvalBundle], int]:
        """
        分页查询。返回 (当前页, 总数)。

        排序固定为 `created_at DESC, bundle_id DESC` —— 第二排序键是必需
        的：created_at 可能重复，缺少稳定的 tiebreaker 时翻页会出现记录
        重复或遗漏。

        ⚠️ **`created_at` 必须由调用方显式填，不能只靠列的 server_default。**
        ------------------------------------------------------------------
        PostgreSQL 的 `now()` 返回的是**事务开始时间**，不是语句执行的
        时刻。同一个事务里插入的多行因此拿到**完全相同**的 `created_at`
        （实测确认：在一个事务里建三份记录，三行的 `created_at` 到微秒
        都一致）。

        后果：排序退化成"由 `bundle_id` 这个**随机 UUID** 决定"，于是
        "最近创建的那一份"每次查询都可能不同 —— 而且不报错。

        这不只是排序难看：`get_latest_generation_run` 的语义就是
        "最近一次造题跑到哪了"，它在列表页与重试逻辑里都被依赖。

        所以服务层在构造实体时一律写
        `created_at=datetime.now(timezone.utc)`。列上的 `server_default`
        保留着，是给手工 SQL 与其它写入路径兜底的 —— 但**只靠它不够**。
        """
        raise NotImplementedError

    # ========================================================
    # Generation run
    # ========================================================

    @abstractmethod
    async def add_generation_run(
            self,
            run: RetrievalEvalGenerationRun,
    ) -> RetrievalEvalGenerationRun:
        """新增一次造题运行。不 commit。"""
        raise NotImplementedError

    @abstractmethod
    async def get_generation_run(
            self,
            generation_run_id: UUID,
    ) -> RetrievalEvalGenerationRun | None:
        """按主键查询。"""
        raise NotImplementedError

    @abstractmethod
    async def get_latest_generation_run(
            self,
            bundle_id: UUID,
    ) -> RetrievalEvalGenerationRun | None:
        """
        取某个 bundle 最近一次的造题运行。

        列表页每行都要显示"最近一次造题跑到哪了"，逐个 bundle 再查一次
        会变成 N+1。批量场景请用 `list_latest_generation_runs`。

        ⚠️ **「最近」是按 `created_at` 判的，而它必须由调用方显式填。**
        见 `list_bundles` 的同一条说明 —— PostgreSQL 的 `now()` 是
        **事务开始时间**，同一个事务里插的多行拿到的 `created_at`
        **一模一样**，于是排序会掉到 UUID 的 tiebreaker 上（随机），
        "最近一次"就成了一个每次查询都可能不同的答案。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_latest_generation_runs(
            self,
            bundle_ids: list[UUID],
    ) -> dict[UUID, RetrievalEvalGenerationRun]:
        """
        批量取每个 bundle 最近一次的造题运行。

        **存在的理由就是避免列表页的 N+1** —— 逐行调
        `get_latest_generation_run` 在 20 行的页面上就是 20 次往返。

        返回 bundle_id -> 运行。没有运行记录的 bundle 不出现在结果里
        （而不是映射到 None）—— 调用方用 `.get()` 取值，语义一样，
        但少一层 Optional 嵌套。
        """
        raise NotImplementedError

    @abstractmethod
    async def update_generation_run(
            self,
            run: RetrievalEvalGenerationRun,
    ) -> RetrievalEvalGenerationRun:
        """
        更新造题运行（状态、计数、错误）。

        ⚠️ 计数是**整体覆盖**，而且调用方应当从窗口行**重算**它们，
        而不是累加。见 `count_windows_by_status` 的说明。

        曾经有过一个 `increment_generation_run_counters`（走
        `UPDATE ... SET x = x + n`），已经删掉：任务基础设施是
        **至少一次投递**（见 app/common/task/models.py 的 `idempotency_key`
        说明），被 reclaimed 再跑一遍时累加会**算重** —— 症状是"产出的
        case 比列出来的多"，且只在重跑路径上出现。
        """
        raise NotImplementedError

    @abstractmethod
    async def count_windows_by_status(
            self,
            generation_run_id: UUID,
    ) -> dict[str, int]:
        """
        按状态数一次运行的窗口。返回 `{status_value: count}`。

        **这是计数的唯一正解，而不是累加。** 运行行上的
        `completed_windows` / `failed_windows` 必须等于这里数出来的值 ——
        因为两者都由窗口行推导，重复执行不会让它们漂开。

        累加为什么不行：任务被 reclaimed 后重跑是**预期内**的
        （at-least-once），而重跑会把同一批窗口的成功各记一次。
        """
        raise NotImplementedError

    # ========================================================
    # Generation window
    # ========================================================

    @abstractmethod
    async def add_windows(
            self,
            windows: list[RetrievalEvalGenerationWindow],
    ) -> int:
        """
        批量新增窗口。返回实际写入条数。

        **一批一次**，不是一个一个插：一个 100 个 chunk、L=5 的 bundle
        就是 20 个窗口，逐个 insert 是 20 次往返，而它们本来就在同一个
        事务里、同一次任务执行内。

        ⚠️ 窗口划分是**确定性的**（见 eval/windowing.py），而
        `(generation_run_id, window_index)` 上有唯一约束 —— 重跑同一次
        运行会撞它。这是有意的：撞上说明有人对同一次运行切了两次窗口，
        那是不该发生的事，静默忽略会让两套输入快照混在一起。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_windows(
            self,
            generation_run_id: UUID,
    ) -> list[RetrievalEvalGenerationWindow]:
        """
        取一次运行的全部窗口，按 `window_index` 升序。

        **不分页**：窗口数 = 抽中的 chunk 数 / L，量级是几十到几百，
        一次取完比翻页简单；而进度接口本来就要看全部窗口的状态分布。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_unfinished_windows(
            self,
            generation_run_id: UUID,
    ) -> list[RetrievalEvalGenerationWindow]:
        """
        取出**还没成功**的窗口（`queued` / `failed` / `partial_failed`），
        按 `window_index` 升序。

        ⚠️ `queued` **在里面**，而这与早期的一版相反
        -------------------------------------------------
        原先这里叫 `list_retryable_windows` 并**排除** `queued`，理由是
        "它只是还没跑，重复提交会让两个 worker 同时处理同一个窗口"。

        那条理由现在由 `claim_window` 承担 —— 因为一次执行**有窗口数
        上限**，跑不完时剩下的就留在 `queued`，而"再次提交同一个任务"
        正是继续跑它的唯一途径（本层不发子任务）。把 `queued` 排除掉的话
        那些窗口永远没人处理。

        也就是说：**过滤条件不再是并发保护，认领才是。** 不要因为
        "看起来更安全"就把 `queued` 加回来 —— 那会让上限之外的窗口
        彻底卡住。
        """
        raise NotImplementedError

    @abstractmethod
    async def claim_window(
            self,
            generation_window_id: UUID,
    ) -> bool:
        """
        原子地认领一个窗口：能跑则置 `running` 并返回 True，否则 False。

        实现是一条带条件的 UPDATE：

            UPDATE ... SET status='running', attempt_count = attempt_count + 1
            WHERE generation_window_id = :id
              AND status IN ('queued', 'failed', 'partial_failed')

        按 `rowcount` 判输赢。**这是"同一批窗口不会被跑两遍"唯一可靠的
        地方**：应用层"先查状态、再决定跑不跑"在两次执行并发时两边都会
        查到 `queued`，然后都去调模型 —— 而模型调用是按次计费的，重复
        那一遍的钱已经花掉了。库上的条件更新才排得掉这个竞态。

        什么时候会并发：调用方连点两次"重试"就会提交两条任务（幂等键在
        任务进终态时被释放，见 Task.reusable_key），两条各领一个 lease、
        各跑各的。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_window(
            self,
            generation_window_id: UUID,
    ) -> RetrievalEvalGenerationWindow | None:
        """按主键查询。"""
        raise NotImplementedError

    @abstractmethod
    async def update_window(
            self,
            window: RetrievalEvalGenerationWindow,
    ) -> RetrievalEvalGenerationWindow:
        """
        更新一个窗口（状态、原始返回、错误、产出计数）。

        ⚠️ `input_chunks` 与 `window_index` 一旦写入**不该再改**：它们是
        这个窗口的身份，重试靠它们复现当时的输入。要换输入就是另一个
        窗口，不是这一次的更新。
        """
        raise NotImplementedError
