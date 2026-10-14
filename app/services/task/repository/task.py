"""
Repository contract for the `tasks` table —— 也就是 spec 里的 **TaskStore**。

它是 Task 的事实来源在代码里的入口：状态机推进、租约、lineage 查询都
只经过这里。**MQ 与 Worker 都不碰 tasks 表**，它们只调用本契约。

两条贯穿全契约的规则
------------------
1. **从不 commit / rollback**（仓库通则，见 app/common/repository/base.py）。
   事务边界由调用方决定：

       TaskService.submit   跑在 ServiceFactory.scope() 的事务里，
                            因此 tasks + task_outbox 两个 INSERT 天然
                            同事务 —— 这正是 spec §5 要的原子性，
                            不需要显式 BEGIN。

2. **状态推进一律是条件 UPDATE（CAS）**，不是"读出来、改内存、写回去"。
   多 Worker 并发下后者会覆盖别人刚写入的结果，而这类覆盖**不会报错**，
   只会让某条任务的状态凭空倒退。

关于租约（lease）
----------------
claim() 会写下 lease_expires_at。它的存在只为一件事：**执行者崩了之后
任务还能回来**。没有它，claim 之后被杀的 Worker 会让那条任务永远停在
running —— 既不在可 claim 集合里（不会被重跑），也不在任何终态里
（不会有告警）。reclaim_expired() 就是把它们收回来的那一步。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository
from app.common.task.models import Task
from app.common.task.types import TaskStatus

__all__ = ["TaskStore"]


class TaskStore(BaseRepository):

    # ========================================================
    # 写入
    # ========================================================

    @abstractmethod
    async def add(
            self,
            task: Task,
    ) -> Task:
        """
        插入一条 Task，返回落库后的样子。

        幂等键冲突时**不抛异常**、也不覆盖既有行 —— 实现必须返回那条
        已存在的 Task（见 get_by_idempotency_key）。理由：至少一次投递
        意味着重复提交是常态不是异常，而"重复提交报错"会让调用方
        不得不先查一次再插一次（那中间仍有竞态）。

        所以调用方拿到的 task_id 可能是**既有**的那条，不一定是刚传进来的。
        """

        raise NotImplementedError

    @abstractmethod
    async def claim(
            self,
            task_id: UUID,
            *,
            leased_by: str,
            lease_seconds: float,
    ) -> Task | None:
        """
        把一条**已到期**且可执行的任务抢过来，置为 running。

        这是**并发保护的核心**：多实例同时收到同一条 task_id 的消息时，
        只有一条 UPDATE 能命中，其余拿到 None 并应当安静地放弃。重复
        投递因此不会导致重复执行。

        同时 attempt += 1、started_at = now()、写入租约。

        ⚠️ "可执行"包含**已到期**：status 可 claim **且**
        execute_at 为空或已到。execute_at 在未来的任务必须返回 None。

        这一条不能只放在 list_due 里。投递路径不止"扫描"一条 ——
        post-commit 钩子与 outbox relay 也会推，而它们在提交**当下**
        就会推一次（那时延迟任务还没到点）。到期若是只在扫描里判，
        延迟任务会在提交的瞬间被执行，定时语义彻底失效。

        所以到期判断是**执行前的最后一道闸**，必须在 claim 里。

        Returns:
            抢到了就是抢到之后的样子；没抢到（已终态 / 正被别人跑 /
            还没到点 / 不存在）返回 None —— 几种情况都不区分，
            调用方一律按"不用我管"处理。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_success(
            self,
            task_id: UUID,
    ) -> Task | None:
        """
        执行成功。置 success + finished_at，清租约。

        ⚠️ 只在当前是 running 时生效。任务可能在这期间被取消
        （cancel 把状态改成了 cancelled），那时本调用**必须无效** ——
        否则"用户取消过、却被执行者改回成功"，取消就形同虚设。

        Returns:
            生效了返回更新后的样子；没生效（状态已不是 running）返回 None。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_failure(
            self,
            task_id: UUID,
            *,
            error: str,
            retry_delay_seconds: float | None,
    ) -> Task | None:
        """
        执行失败。

        retry_delay_seconds 的含义是**调用方已经判完了**：
            一个数字     -> 还能再试，置 retryable，execute_at = now()+该值
            None        -> 不再重试，置 failed 并记 finished_at

        本方法刻意**不**看 attempt / max_attempts：那是 RetryPolicy 的事
        （见 infra/task/retry.py）。放在这里会让"能不能重试"有两处判断，
        而两处迟早不一致。

        error 落库后**不因后续成功而被清掉** —— 它回答的是
        "为什么重试过"，成功只说明最终成了。

        同样只在 running 时生效（并发取消保护）。

        Returns:
            更新后的样子；状态已不是 running 时返回 None。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_cancelled(
            self,
            task_id: UUID,
    ) -> bool:
        """
        取消一条任务。

        ⚠️ 取消是**协作式**的，这一条很容易被误解：
            - 还在 pending / retryable 的任务：状态直接改成 cancelled，
              它再也不会被 claim（因此不会执行）。
            - 正在 running 的任务：本方法照样把状态改成 cancelled，但
              **不会**打断已经跑起来的 Handler —— 那需要执行侧配合
              （Worker 侧的子任务取消）。执行者下次写回状态时会因为
              状态已不是 running 而无效（见 mark_success / mark_failure），
              于是最终停在 cancelled。
            - 已经终态的任务：不动，返回 False。取消来晚了不是错误。

        Returns:
            状态确实被改成了 cancelled 就返回 True；已是终态或不存在返回
            False（调用方据此决定是回 409 还是当作幂等成功）。
        """

        raise NotImplementedError

    @abstractmethod
    async def reschedule(
            self,
            task_id: UUID,
            *,
            execute_at: datetime | None,
    ) -> Task | None:
        """
        改一条**还没开始跑**的任务的执行时间，并把 generation + 1。

        这是"防抖的延迟任务"的落点：同一个逻辑定时被反复触发时，调用方
        每次都调它把时间推到最新，而不是取消旧的再提交一个新的 ——
        后者做不到原子，而且在"取消"与"新建"之间那条旧定时可能刚好到点。

        ⚠️ generation + 1 是**必须的**，不是附带效果
        ----------------------------------------
        已经生成的定时（正在睡的那个）醒来时会拿自己绑定的 generation
        与库里的比对：不一致就说明自己被取代了，于是**不投递**，改按新的
        执行时间重排（见 infra/task/dispatcher.py）。

        只比对 execute_at 是不够的 —— 把时间重排到**同一个时刻**是完全
        合法的一次重排（业务表达"按现在的时刻重算"），那时前后相等，
        只有计数器能区分新旧定时。

        语义边界：
            - 只在**可 claim**（pending / retryable）时生效。已经在跑的
              任务重排没有意义（甚至危险：它马上就要写回结果了），
              终态的更不能改。
            - 往**后**推也好、往**前**提也好，都算一次重排，都 +1。
            - attempt / error 一律不动：重排是"改时间"，不是"重试"。
              （要让失败的任务再跑一轮请用 schedule_retry。）

        Returns:
            重排之后的样子（generation 已递增）；当前状态不允许时返回 None。
        """

        raise NotImplementedError

    @abstractmethod
    async def next_scheduled(
            self,
    ) -> Task | None:
        """
        取**最近的、还没到点**的那条可跑任务，没有则 None。

        给定时器用来"绑定下一个闹钟"：拿到它就能算出该睡多久，从而在
        到点那一刻精确醒来，而不是固定节拍地空转（见
        infra/task/dispatcher.py）。

        为什么只取一条：调度器只需要知道**下一个**时刻。取一批回来再挑
        只是多传一些用不上的列。

        注意它与 list_due 的分工：
            list_due        已经到点的（该跑了）
            next_scheduled  还没到点的（下一个闹钟）
        两者条件互补，一个任务不会同时出现在两边。
        """

        raise NotImplementedError

    @abstractmethod
    async def schedule_retry(
            self,
            task_id: UUID,
    ) -> Task | None:
        """
        手动重跑一条**已终止**的任务（TaskService.retry 的落点）。

        与 mark_failure(retry_delay_seconds=...) 的区别是**语义**：
        那个是"这次失败后自动再试一次"，这个是"人决定再跑一轮"。
        因此这里把 attempt 清零、把 execute_at 置为立即，并清掉
        finished_at 与租约，让这条任务回到与刚提交时**一样**的可执行状态。

        ⚠️ attempt 清零是刻意的，不是疏忽
        --------------------------------
        不清零的话，一条已经用满 max_attempts 的任务被 retry 后，
        第一次再失败就会立刻又变成终态 —— 于是 retry 这个接口在最需要
        它的场景（"跑满次数失败了，人工修好数据，再跑一次"）恰好没用。

        代价是 attempt 的含义变成**本轮**执行次数，而不是历史总数。
        历史仍然查得到：error 与 finished_at 都保留着，query 接口能看到
        上一条失败原因。

        只允许从 failed / cancelled 进入（success 的重跑会产生第二次
        外部副作用，那应该是提交一条新任务的事，不是 retry）。
        当前状态不允许时返回 None。

        Returns:
            回到 pending 后的样子；状态不允许时返回 None。
        """

        raise NotImplementedError

    # ========================================================
    # 读取
    # ========================================================

    @abstractmethod
    async def get(
            self,
            task_id: UUID,
    ) -> Task | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_idempotency_key(
            self,
            idempotency_key: str,
    ) -> Task | None:
        """
        按幂等键取既有任务。

        add() 靠它实现"重复提交返回同一条"。唯一性由**部分唯一索引**
        保证（见 models/sqlalchemy/task.py），因此即使两个请求同时插，
        也只有一个能成功，另一个重新查一次即可 —— 本方法就是那一步。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_due(
            self,
            *,
            limit: int,
    ) -> list[UUID]:
        """
        捞出"现在就该跑"的任务 ID。

        条件：status 可 claim，且 execute_at 为空或已到。

        只回 task_id 不回整条任务：这是最热的查询，而拿到 ID 之后
        Worker 本来就要 claim 一次（那次会返回完整行），没必要在这里
        把 params_json 也读出来。**这也正是 MQ 消息只带 task_id 的
        同一条理由**（spec §6）：投递物越小，越大范围的状态以 DB 为准。

        排序是 execute_at 升序且 NULL 在前（NULL = 立即，就该排在
        定时任务前面），再按 created_at 保证稳定。
        """

        raise NotImplementedError

    @abstractmethod
    async def reclaim_expired(
            self,
            *,
            limit: int,
    ) -> list[UUID]:
        """
        收回租约已过期、状态还停在 running 的任务。

        它们是"执行者已经死了"的任务：放回 retryable 让别的 Worker 接管。

        ⚠️ attempt 已经用满的收回为 **failed**，不是 retryable
        ---------------------------------------------------
        否则每崩一次就多一个 attempt，而一个反复让 Worker 崩溃的任务
        （比如它一跑就 OOM）会永远在 running -> retryable -> running 之间
        循环，把实例一个接一个拖死。判成 failed 才停得下来。

        error 里会附一句"租约过期"，这是这种失败**唯一**的痕迹 ——
        执行者没机会写下任何东西，它已经死了。

        Returns:
            被收回的 task_id 列表（供调用方接着投递）。
        """

        raise NotImplementedError

    @abstractmethod
    async def list(
            self,
            *,
            status: TaskStatus | None = None,
            handler: str | None = None,
            root_task_id: UUID | None = None,
            parent_task_id: UUID | None = None,
            limit: int = 50,
            offset: int = 0,
    ) -> list[Task]:
        """
        按条件列任务。

        参数为 None 一律表示**不过滤**（与 capability 的 list 约定一致），
        不表示"取该字段为 NULL 的行"。

        排序是 created_at 倒序 —— 这是一个浏览用的接口，最近的最先看。
        要按链条顺序读用 list_lineage()。
        """

        raise NotImplementedError

    @abstractmethod
    async def count(
            self,
            *,
            status: TaskStatus | None = None,
            handler: str | None = None,
            root_task_id: UUID | None = None,
            parent_task_id: UUID | None = None,
    ) -> int:
        """满足同样筛选条件的总数（分页用）。"""

        raise NotImplementedError

    @abstractmethod
    async def list_lineage(
            self,
            root_task_id: UUID,
    ) -> list[Task]:
        """
        取整条执行链，按**产生顺序**（created_at 升序）排列。

        对应 spec §9 / §18 的"任务执行链追踪"：一个 Run、一次 HITL
        最终产生了哪些任务。升序是刻意的 —— 链条是因果叙事，
        倒着读没有意义。

        不分页：一条链的长度由 Handler 的派生深度决定，正常是几条到
        几十条。真出现几千条的链，那是设计出了问题，不该靠分页掩盖。
        """

        raise NotImplementedError
