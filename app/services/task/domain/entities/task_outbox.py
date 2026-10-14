"""
TaskOutbox 领域实体。

刻意做成极薄的一层：它没有业务规则，只是把 task_outbox 的一行搬到
Python 里，让 relay 不必 import SQLAlchemy 模型 —— 与其余模块
"实体不认识 ORM" 的分工一致。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.common.task.types import OutboxStatus

__all__ = ["TaskOutbox"]


class TaskOutbox(BaseModel):
    """一条投递意图。语义与字段说明见 domain/schemas 之外的表注释。"""

    model_config = ConfigDict(
        from_attributes=True,
    )

    outbox_id: UUID = Field(
        description="本行 ID。与 task_id 分开：一条任务理论上可投多次。",
    )

    task_id: UUID = Field(
        description="要投递的任务。",
    )

    status: OutboxStatus = Field(
        description="投递状态，与任务自身的执行状态是两件事。",
    )

    created_at: datetime = Field(
        description="创建时间。relay 按它先来先投。",
    )

    published_at: datetime | None = Field(
        default=None,
        description="投递出去的时刻。NULL 表示还没投出去。",
    )

    retry_count: int = Field(
        default=0,
        description="投递尝试次数。",
    )

    last_error: str | None = Field(
        default=None,
        description=(
            "最近一次投递失败的原因 —— "
            "「本该被执行却一直没被交出去」这件事的唯一痕迹。"
        ),
    )
