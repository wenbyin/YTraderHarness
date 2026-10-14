"""
Repository contract for the `task_outbox` table.

与 TaskStore 分成两个契约（而不是塞进一个仓储）的理由：两者的**读者**
不同。TaskStore 的服务对象是提交方与 Worker（任务本身怎么跑）；
本契约的服务对象是 OutboxRelay（怎么把"要投递"这件事交出去）。

⚠️ 但它们的**写**必须在同一个事务里
--------------------------------
spec §5 要求：

    BEGIN
      INSERT INTO tasks
      INSERT INTO task_outbox
    COMMIT

本契约自己不保证这一点 —— 它和 TaskStore 一样从不 commit。原子性来自
调用方把两次写入放在同一个事务里：

    ServiceFactory.scope(TaskService)  已经开好了事务
      TaskServiceImpl.submit()
        task_store.add(task)        -> 同一个 session
        outbox_store.add(task_id)   -> 同一个 session
      （离开 scope 时统一 COMMIT）

两个仓储共享同一个 session，是因为 RepositoryFactory.with_session()
把它们绑在同一个请求/作用域 session 上（见 common/repository/factory.py）。
所以"同事务"这件事**不需要显式的 BEGIN**，也不需要在这里再包一层。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import TaskOutbox

__all__ = ["TaskOutboxStore"]


class TaskOutboxStore(BaseRepository):

    @abstractmethod
    async def add(
            self,
            task_id: UUID,
    ) -> TaskOutbox:
        """
        登记一条待投递记录。**必须与 tasks 的 INSERT 同事务**（见模块
        docstring）。

        刻意不接受"整条 Task"：投递这件事只需要一个 task_id，给的多了
        就会有人开始往这里塞参数 —— 那正是 spec §6 明令禁止的
        （MQ 消息只携带 task_id）。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_unpublished(
            self,
            *,
            limit: int,
            max_retry_count: int,
    ) -> list[TaskOutbox]:
        """
        取还没投出去的记录，先来先投。

        含两类：
            pending                     还没试过，或试过但还没到放弃的程度
            failed 且 retry_count < 上限  曾经超限、后来上限被调大了

        第二类是刻意带上的：把"投递重试上限"当成可调参数时，调大之后
        那些 already-failed 的记录理应有机会被重投，否则只能手工改库。

        超过 max_retry_count 的 pending 记录**不返回**：让它们留在库里的
        目的只是别再浪费投递尝试，不是丢弃 —— 转成 failed 由
        mark_publish_failed 负责。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_published(
            self,
            outbox_id: UUID,
    ) -> None:
        """
        投递成功。记 published_at。

        ⚠️ 这一步与真正的投递**不可能同事务**（投递是外部副作用），
        所以"投出去了但还没标记"这个窗口必然存在，崩溃就落在这里。
        后果是重投一次 —— 至少一次投递，接收侧（Worker 的 claim CAS）
        必须幂等。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_publish_failed(
            self,
            outbox_id: UUID,
            *,
            error: str,
            max_retry_count: int,
    ) -> None:
        """
        投递失败。retry_count += 1，记 last_error；到上限转 failed。

        转 failed 而不是继续 pending 是为了让它**可见**：一直 pending 的
        记录会被下一次投递重试捞走，看起来一切正常，而实际上它可能已经
        失败了几十次。failed 是一次都不会再重试的终态，运维据此告警。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_by_task(
            self,
            task_id: UUID,
    ) -> TaskOutbox | None:
        """
        按任务取它的投递记录。

        排查"这条任务为什么没被跑"的第一个问题就是它：投出去了没有？
        失败了几次？最后一次报什么？
        """

        raise NotImplementedError
