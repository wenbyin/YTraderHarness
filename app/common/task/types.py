"""
Task 状态的取值，以及"哪些状态还能被 claim / 哪些已经是终态"。

为什么单独一个文件
----------------
同一条 Task 的状态被三处读，而且必须读成同一个意思：

    Dispatcher   按 status + execute_at 捞出"该跑"的那些行
    Worker       claim 时拿它做 CAS，结束时写终态
    查询接口     直接把它吐给前端

三处各写一份字符串比较，改一个值必然漏掉另外两处。取值集中在这里，
"可 claim"与"终态"两个判断也就只在这里写一次。

与 RunStatus 刻意分开
--------------------
RunStatus（app/services/session/domain/schemas/enum.py）回答的是
"一次 Agent 运行怎么样了"，这里回答"一次后台动作怎么样了"。
一次运行会派生好几条 Task（run.execute -> memory.l0.extract ->
memory.compress），两者不是同一个状态机 —— 合并会立刻需要一张
"用 Task 状态反推 Run 状态"的映射表，而那张表在 HITL 停在
waiting_input 时根本填不出来。
"""

from __future__ import annotations

from enum import Enum

__all__ = ["OutboxStatus", "TaskStatus"]


class TaskStatus(str, Enum):
    """
    一条 Task 的生命周期。

        PENDING ──claim──> RUNNING ──成功──────────────────> SUCCESS
                              │
                              ├──失败，还能再试──> RETRYABLE ──claim──┐
                              │                                        │
                              ├──失败，到上限了──> FAILED               │
                              │                                        │
                              └──被取消──────────> CANCELLED           │
                                                                       │
        RETRYABLE 从上面那条边回到 claim，是唯一的环。  <──────────────┘

    RETRYABLE 与 PENDING 刻意分开，而不是复用 PENDING
    ------------------------------------------------
    重试过的 Task 与从没跑过的 Task，在查询接口上必须能区分：
    "谁还在重试"是运维要看的第一件事。复用 PENDING 的话，前端与
    运维脚本都得靠 `attempt > 0 and status == 'pending'` 这种复合条件
    去猜 —— 一旦有地方要判第二次，就说明状态本身少了一个取值。
    """

    #: 已落库，等 Dispatcher 捞。execute_at 到了才该跑。
    PENDING = "pending"

    #: 已被某个 Worker claim 走，正在执行。
    RUNNING = "running"

    #: 上次执行失败，但 attempt < max_attempts，等下一次调度。
    RETRYABLE = "retryable"

    #: 执行成功。终态。
    SUCCESS = "success"

    #: 执行失败且不再重试。终态。
    FAILED = "failed"

    #: 被主动取消。终态。
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """终态：状态不会再变，也不会再被 claim。"""

        return self in _TERMINAL

    @property
    def is_claimable(self) -> bool:
        """
        还能被 Worker claim 的状态。

        FAILED / CANCELLED 不在其中是**刻意的**：想让它们再跑一次必须走
        TaskService.retry()，那是一步显式的、会记进 attempt 的操作。
        否则一个轮询恰好捞到一条 FAILED 的行就能让它重跑，
        而"失败后又被谁悄悄跑了一遍"是最难查的一类问题。
        """

        return self in _CLAIMABLE


#: 终态集合。Worker 靠它判断"写回状态时要不要一并记 finished_at"。
_TERMINAL = frozenset({
    TaskStatus.SUCCESS,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
})

#: 可被 claim 的集合。Dispatcher 的轮询条件与 Store 的 CAS 都用它。
_CLAIMABLE = frozenset({
    TaskStatus.PENDING,
    TaskStatus.RETRYABLE,
})


class OutboxStatus(str, Enum):
    """
    task_outbox 一行的投递状态。

    与 TaskStatus 是**两件事**，别混：
        TaskStatus     这条任务执行得怎么样了
        OutboxStatus   这条任务的**投递意图**落实了没有

    一条 Task 可以在 status=SUCCESS 的同时 outbox 还是 PENDING —— 那说明
    它是在本进程里被直接跑掉的（投递还没轮到），不是数据不一致。
    反过来 outbox=PUBLISHED 而 status=PENDING 才正常。

    之所以独立成表：Task 落库成功但投递失败（进程刚好在这两步之间被杀）
    时，任务不会丢 —— 补投的那条路径靠的就这张表（spec §5）。
    """

    #: 还没投递出去。OutboxRelay 会一直捞它。
    PENDING = "pending"

    #: 已经交给 Dispatcher 了。注意这只代表"投出去了"，
    #: 不代表任务跑完 —— 跑得怎么样看 Task.status。
    PUBLISHED = "published"

    #: 反复投递失败，超过上限。需要人工看一眼 ——
    #: 它是"本该被执行、但一直没被交出去"的唯一痕迹。
    FAILED = "failed"
