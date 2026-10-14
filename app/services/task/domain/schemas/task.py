"""
任务的对外契约。

对应 task.yaml 的 components/schemas。

分 Summary / Detail 两档，而不是一个模型走天下
--------------------------------------------
列表页可能一次返回几百条，而 `params` 是提交方写进去的任意结构（虽然
按约定只有业务 ID 与最小参数，但没有任何机制保证它小）。列表接口
不需要参数，就不该发。

因此：

    TaskSummary   列表行。**不含 params**，也不含 error 正文。
    TaskDetail    单条。含 params、error、租约信息。

这个划分同时是一道**信息的闸门**：params 与 error 里都可能带业务细节
（用户 ID、上游错误文本），只在单条查询里给出，比在列表里一次吐几百条
要好收。

泄露面刻意保留在单条接口上，而不是干脆不暴露
------------------------------------------
排查"这条任务为什么失败"时，error 与 params 是**唯一**能看的东西 ——
日志里没有（任务可能在任何实例上执行）、DB 也不该直连。所以 Detail
必须带上它们，这是可运维性的一部分，不是遗漏。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.common.schemas import APIModel
from app.common.task.types import TaskStatus

__all__ = [
    "TaskCancelResponse",
    "TaskDetail",
    "TaskLineage",
    "TaskPage",
    "TaskSummary",
]


class TaskSummary(APIModel):
    """列表里的一行任务。不含 params 正文。"""

    task_id: UUID = Field(description="任务 ID。")

    handler: str = Field(
        description="逻辑 Handler Name，如 hitl.timeout。",
    )

    status: TaskStatus = Field(
        description=(
            "pending / running / retryable / success / failed / cancelled。"
            "retryable 与 pending 分开，因此「重试中」的任务一眼可见。"
        ),
    )

    attempt: int = Field(
        description="本轮已执行次数。手动 retry 会把它清零。",
    )

    max_attempts: int = Field(
        description="本轮执行次数上限。",
    )

    generation: int = Field(
        description=(
            "定时器代际。提交时为 1，每重排一次 +1。"
            "大于 1 说明这条延迟任务的执行时间被改过 —— "
            "也是「被取代的那个定时不会执行」的依据。"
        ),
    )

    execute_at: datetime | None = Field(
        default=None,
        description="可执行时刻。为 null 表示立即执行。",
    )

    parent_task_id: UUID | None = Field(
        default=None,
        description="直接父任务。为 null 表示这是链条起点。",
    )

    root_task_id: UUID | None = Field(
        default=None,
        description="整条执行链的根任务 ID。按它取整条链。",
    )

    idempotency_key: str | None = Field(
        default=None,
        description="幂等键。带它的任务不会因为重复提交而多出一条。",
    )

    created_at: datetime = Field(description="创建时间。")

    updated_at: datetime = Field(description="最后更新时间。")

    started_at: datetime | None = Field(
        default=None,
        description="最近一次开始执行的时间。",
    )

    finished_at: datetime | None = Field(
        default=None,
        description="进入终态的时间。",
    )


class TaskDetail(TaskSummary):
    """
    单条任务的完整视图。

    继承 TaskSummary 而不是各写一份：两者对"同一条任务长什么样"必须
    只有一套说法，否则列表页与详情页会出现同一字段不同含义。
    """

    params: dict[str, Any] = Field(
        description=(
            "提交时写进去的业务参数（已反序列化成对象）。"
            "按约定只有业务 ID 与最小参数 —— 大状态在执行阶段按 ID 重新加载。"
        ),
    )

    causation_id: UUID | None = Field(
        default=None,
        description=(
            "触发源。可能是父任务 ID，也可能是业务事件 ID —— "
            "后者用来回答「这次 HITL 最终引发了哪些任务」。"
        ),
    )

    error: str | None = Field(
        default=None,
        description=(
            "最近一次失败的原因。重试成功后**仍然保留** —— "
            "它回答的是「它为什么重试过」。"
        ),
    )

    lease_expires_at: datetime | None = Field(
        default=None,
        description=(
            "当前这次执行的租约到期时刻。状态是 running 而它已过期，"
            "说明执行它的 Worker 崩了、任务正等着被收回重跑。"
        ),
    )

    leased_by: str | None = Field(
        default=None,
        description="持有租约的 Worker 标识。排查「卡住的这条谁在跑」。",
    )


class TaskReschedule(APIModel):
    """
    重排一条延迟任务的执行时间（防抖）。

    反复调它不会堆出多条任务：改的是**同一条**，并且每次都会让上一个
    已生成的定时失效（generation +1）。最后那一次的时间赢。
    """

    execute_at: datetime | None = Field(
        default=None,
        description=(
            "新的执行时间。传 null 表示改成「立即执行」。"
            "必须带时区（库里是 TIMESTAMPTZ）。"
        ),
    )


class TaskLineage(APIModel):
    """
    一条后台执行链。

    对应 spec §9：一个 Run / 一次 HITL 最终产生了哪些任务、某个 Task
    为什么产生。回答这些问题靠 root_task_id，而不是让调用方自己递归。
    """

    root_task_id: UUID = Field(
        description="链条根任务 ID。按它就能拿到这整条链。",
    )

    tasks: list[TaskSummary] = Field(
        description="链上的全部任务，按产生顺序（created_at 升序）排列。",
    )


class TaskPage(APIModel):
    """分页结果。"""

    items: list[TaskSummary] = Field(description="当前页的任务列表。")

    total: int = Field(
        description="满足筛选条件的总数（不只是当前页的条数）。",
    )


class TaskCancelResponse(APIModel):
    """
    取消的结果。

    ⚠️ 返回的是取消**之后**的状态，而不是无脑回 success。
    取消是协作式的：一条已经 running 的任务不会被强杀，执行它的 Worker
    要到下一次状态写回时才会看到 cancelled。因此这里如实回当前状态，
    调用方据此判断"停下来了"还是"还在收尾"。
    """

    task_id: UUID = Field(description="任务 ID。")

    status: TaskStatus = Field(
        description=(
            "取消之后的状态。为 cancelled 表示已拦下；"
            "仍为 running 表示正在收尾；已是终态则表示取消来晚了。"
        ),
    )
