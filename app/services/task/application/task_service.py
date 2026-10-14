"""
TaskService —— 所有业务创建异步任务的**唯一入口**（spec §4）。

业务代码只允许：

    await task_service.submit(task)

不允许自己碰底层：

    await redis.enqueue(...)          ✗
    await arq.enqueue_job(...)        ✗
    await celery.send_task(...)       ✗

这条纪律的价值不在"统一好看"，而在**可替换**：投递手段（DB 轮询 / MQ）、
执行方式（进程内 / 独立 Worker 进程）、重试策略全是本模块以下的事，
业务一行都不用改。反过来说，只要有一处业务直接用了 redis，那条路径就
把整个系统钉死在当时的技术选型上。

⚠️ 本契约是 ABC，不是 spec 里写的 Protocol
----------------------------------------
spec §4 把它写成 `class TaskService(Protocol)`。落到本仓库必须落成 ABC：
HTTP 层用 `Service[TaskService]` 取服务，而它走
app/common/introspect.py::resolve_impl —— 靠 `__subclasses__()` 找实现。
Protocol 没有子类，resolve_impl 对它必然失败。

共同抽象里因此只留**窄的那一半**（TaskSubmitter，只有 submit），供
TaskContext 与只需要排任务的业务模块使用。见
app/common/task/protocols.py 的说明。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from app.common.service.base import BaseService
from app.common.task.models import Task
from app.common.task.types import TaskStatus

__all__ = ["TaskService"]


class TaskService(BaseService):

    # ========================================================
    # 提交与生命周期
    # ========================================================

    @abstractmethod
    async def submit(
            self,
            task: Task,
    ) -> UUID:
        """
        提交一条 Task，返回它的 task_id。

        只做三件事：补齐 lineage、落库（tasks + task_outbox **同事务**）、
        登记一个提交后投递的钩子。**不等待执行** —— 这是异步任务的定义。

        lineage 在这里补齐，而不是靠调用方
        --------------------------------
        root_task_id 为空的根任务会被填成自己的 task_id（seal_root）。
        放在这里的原因是它是**唯一**知道"这条任务是不是链条起点"的地方：
        调用方传进来的 Task 可能来自 Handler 的 child()（已有 lineage），
        也可能来自一个 HTTP 请求（什么都没有）。

        handler 与 params 在这里就校验，不等执行
        --------------------------------------
        handler 拼错、params 类型不匹配，都在 submit 这一行报出来。
        不校验的话，错误要等 Worker 从库里读出来才发现 —— 而那已经是
        一次投递、一个事务、甚至一段等待之后，错误现场离原因很远。

        幂等
        ----
        task.idempotency_key 非空且已存在时，返回的是**既有**的 task_id，
        不会新建一条，也不会新增投递记录。

        Raises:
            UnknownHandlerError: handler 没有注册过。
            pydantic.ValidationError: params 与注册的参数类型不符。
        """

        raise NotImplementedError

    @abstractmethod
    async def cancel(
            self,
            task_id: UUID,
    ) -> TaskStatus:
        """
        取消一条任务，返回取消**之后**的状态。

        ⚠️ 返回状态而不是 None（spec §4 写的是 None）
        -------------------------------------------
        取消是协作式的，所以"发了取消"与"停下来了"是两件事：

            任务还没跑        -> 直接置 cancelled，它再也不会被 claim
            任务正在跑        -> 状态置 cancelled，但**不会**打断已经
                                 跑起来的 Handler；它下次写回结果时会被
                                 忽略（见 TaskStore.mark_cancelled）
            任务已经结束      -> 什么都不做

        调用方必须能区分这三种，否则界面上会显示"已取消"、而实际上
        那段代码还在跑。返回真实状态是唯一诚实的答法。

        已经终态时**不报错**：取消来晚了不是错误，是幂等。要报错就得先
        查一次状态再取消，而那中间仍有竞态 —— 结果是一个必然偶发的 409。

        Raises:
            NotFoundError: 任务不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            task_id: UUID,
    ) -> Task:
        """
        取一条任务。

        Raises:
            NotFoundError: 任务不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def retry(
            self,
            task_id: UUID,
    ) -> UUID:
        """
        人工重跑一条**已失败或已取消**的任务，返回它的 task_id。

        语义是"再跑一轮"，与自动重试不是一回事：attempt 会清零、
        execute_at 立即、上一条 error 保留（人工重跑前正要看它）。

        ⚠️ 不许重跑已成功的任务：那会产生**第二次外部副作用**
        （真发了一次通知、真扣了一次款）。需要重跑就该提交一条新任务，
        由调用方显式承担那个语义。

        Raises:
            NotFoundError: 任务不存在。
            TaskStateError: 任务在跑、在排队，或已经成功。
        """

        raise NotImplementedError

    @abstractmethod
    async def reschedule(
            self,
            task_id: UUID,
            *,
            execute_at: datetime | None,
    ) -> Task:
        """
        改一条**还没开始跑**的任务的执行时间（防抖的落点）。

        典型用法是审批超时：

            每来一次新动静，就把截止时间往后推一次
            -> 反复调它，最后那次赢

        与"取消旧的、再提交一条新的"相比，它的好处是**同一条任务**：
        没有"取消还没生效、新定时已经到点"的窗口，也不需要调用方维护
        一堆 task_id。

        ⚠️ 每次调用都会把 generation + 1
        ------------------------------
        已经生成的那个定时醒来时会发现自己的 generation 过期了，于是不
        投递、改按新的时间重排。这就是"防抖"在实现上的全部机制 ——
        见 TaskStore.reschedule 与 infra/task/dispatcher.py。

        Raises:
            NotFoundError: 任务不存在。
            TaskStateError: 任务已经在跑或已终结（重排没有意义：
                前者马上就写回结果，后者改了会让"已完成"变成"还会再跑"）。
        """

        raise NotImplementedError

    # ========================================================
    # 查询
    # ========================================================

    @abstractmethod
    async def list_tasks(
            self,
            *,
            status: TaskStatus | None = None,
            handler: str | None = None,
            root_task_id: UUID | None = None,
            parent_task_id: UUID | None = None,
            limit: int = 50,
            offset: int = 0,
    ) -> tuple[list[Task], int]:
        """
        按条件分页列任务，返回 (当前页, 满足条件的总数)。

        一次返回两者，而不是拆成 list + count 两个方法：分页**总是**
        两个都要（否则前端画不出页码），拆开只会让每个调用方都写两遍，
        还容易只改一处筛选条件 —— 那会让"总数"与"列表"对不上。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_lineage(
            self,
            root_task_id: UUID,
    ) -> list[Task]:
        """
        取以 root_task_id 为根的整条执行链，按产生顺序排列。

        对应 spec §9 / §18 的"任务执行链追踪"：一个 Run、一次 HITL
        最终产生了哪些任务。

        根任务不存在时返回**空列表**而不是 404：调用方问的是"这个根
        下面有哪些任务"，"一条都没有"是一个正当答案，与"这个根不存在"
        在运维排查时是同一个结论（都是"这里什么都没有"）。
        """

        raise NotImplementedError
