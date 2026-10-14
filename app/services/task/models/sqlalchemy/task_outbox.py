from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


class TaskOutbox(Base):
    """
    Task 的**投递意图**。解决的是这一种不一致（spec §5）：

        DB commit 成功
        MQ publish 失败
        -> Task 永久丢失：库里有行，但没有任何人会被叫醒去跑它

    做法是把"要投递"这件事与 Task 本身写在**同一个事务**里：

        BEGIN
            INSERT INTO tasks        ...
            INSERT INTO task_outbox  ...
        COMMIT
        -- 提交之后才投递

    于是投递失败不再丢任务：那一行还是 pending，OutboxRelay 会一直捞它
    （见 app/infra/task/outbox.py）。反过来"先 commit 再 publish"则必然
    存在一个窗口，窗口里进程被杀就是永久丢失。

    ⚠️ 本仓库第一版没有 MQ，"投递"落在哪里？
    --------------------------------------
    落在 Dispatcher 的**进程内唤醒队列**上（infra/task/dispatcher.py）。
    这恰好让本表是**必需**的，而不是为将来预留的空壳：

        dispatch() 只是往内存队列里放一个 task_id
        进程重启 -> 队列没了，投递丢了

    丢掉的投递**只有本表能补**：重启后 OutboxRelay 从 pending 行重新
    投递一遍。既有 InProcessGraphDispatcher 的 docstring 里写的正是这个
    缺口（"跨重启的可靠性需要把队列换成 outbox 表 + worker 轮询"），
    本表就是那一步。

    因此"至少一次投递"是本表的前提而非意外：Relay 可能把同一条 task_id
    投两次（投完、标记 published 之前崩了）。接收侧必须幂等 ——
    Worker 的 claim 是 CAS，重复投递的第二次会失败并被丢弃。

    与 tasks.status 是两回事
    ----------------------
        本表 status     这条任务的**投递意图**落实了没有
        tasks.status    这条任务**执行**得怎么样了

    一条任务可以已经 SUCCESS 而本表仍是 pending（投递还没轮到、而它
    已经被直接跑掉了）—— 那不是不一致，见 OutboxStatus 的说明。
    """

    __tablename__ = "task_outbox"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="本行的 ID。与 task_id 分开，因为一条任务理论可投多次。",
    )

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "tasks.task_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="要投递的任务。",
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment=(
            "pending / published / failed。"
            "取值见 app/common/task/types.py::OutboxStatus，存 .value。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。Relay 按它排序，先来先投。",
    )

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="投递出去的时刻。NULL 表示还没投出去。",
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="投递尝试次数。超过上限转 failed，不再无限重投。",
    )

    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "最近一次投递失败的原因。它同时是「本该被执行、却一直没被"
            "交出去」这件事的唯一痕迹 —— 没有它，failed 状态就只是一个"
            "没有解释的枚举值。"
        ),
    )

    # ========================================================
    # Constraints / Indexes
    # ========================================================

    __table_args__ = (
        # Relay 的扫描条件：
        #   WHERE status = 'pending' ORDER BY created_at
        # 列顺序与之一致，(status, created_at) 可以直接走索引有序读出，
        # 不必再排一次序。
        Index("idx_task_outbox_pending", "status", "created_at"),

        # 按任务反查投递记录（排查"这条为什么没被跑"）。
        Index("idx_task_outbox_task", "task_id"),
    )


#: 跨模块引用时的别名，理由同 models/sqlalchemy/task.py 的 TaskModel。
OutboxModel = TaskOutbox
