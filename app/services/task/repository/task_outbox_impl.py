"""
PostgreSQL implementation of TaskOutboxStore.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。

写法与 task_impl.py 一致：直接走 Core，不碰 ORM 的 identity map
（理由见那个文件的模块 docstring 第 2 条）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import case, func, insert, or_, select, update

from app.common.task.types import OutboxStatus

from ..domain.entities import TaskOutbox
from ..models.sqlalchemy.task_outbox import (
    TaskOutbox as TaskOutboxModel,
)
from .task_outbox import TaskOutboxStore

#: 表的 Core 句柄。
_T = TaskOutboxModel.__table__


def _to_entity(
        row: Mapping[str, Any],
) -> TaskOutbox:
    """一行 -> TaskOutbox 实体。

    注意列名 `id` 与实体字段 `outbox_id` 不同名：实体侧刻意不叫 id
    （在跨层传递的上下文里，一个裸的 `id` 说不清是谁的 ID）。
    """

    return TaskOutbox(
        outbox_id=row["id"],
        task_id=row["task_id"],
        status=OutboxStatus(row["status"]),
        created_at=row["created_at"],
        published_at=row["published_at"],
        retry_count=row["retry_count"],
        last_error=row["last_error"],
    )


class TaskOutboxStoreImpl(TaskOutboxStore):

    async def add(
            self,
            task_id: UUID,
    ) -> TaskOutbox:

        statement = (
            insert(_T)
            .values(
                id=uuid4(),
                task_id=task_id,
                status=OutboxStatus.PENDING.value,
                created_at=func.now(),
                retry_count=0,
            )
            .returning(*_T.c)
        )

        row = (await self._session.execute(statement)).one()

        return _to_entity(row._mapping)

    async def list_unpublished(
            self,
            *,
            limit: int,
            max_retry_count: int,
    ) -> list[TaskOutbox]:

        statement = (
            select(*_T.c)
            .where(
                _T.c.retry_count < max_retry_count,
                or_(
                    _T.c.status == OutboxStatus.PENDING.value,
                    # 曾经超限转 failed、后来上限被调大的记录。
                    # 带上它们，是为了让"调大重试上限"这个动作真能生效，
                    # 而不是只能去手工改库。
                    _T.c.status == OutboxStatus.FAILED.value,
                ),
            )
            # 先来先投。补一个 id 保证同刻插入的记录有稳定顺序。
            .order_by(_T.c.created_at.asc(), _T.c.id.asc())
            .limit(limit)
        )

        result = await self._session.execute(statement)

        return [_to_entity(r._mapping) for r in result]

    async def mark_published(
            self,
            outbox_id: UUID,
    ) -> None:

        await self._session.execute(
            update(_T)
            .where(_T.c.id == outbox_id)
            .values(
                status=OutboxStatus.PUBLISHED.value,
                published_at=func.now(),
                # 发布成功后把上次的失败原因清掉：它已经不再成立，
                # 留着会让"这条到底成功没有"变模糊。
                last_error=None,
            )
        )

    async def mark_publish_failed(
            self,
            outbox_id: UUID,
            *,
            error: str,
            max_retry_count: int,
    ) -> None:

        # 自增写在 SQL 里（不是读出来 +1）：同一个 relay 循环里可能
        # 并发处理多条，读改写会丢计数。
        next_count = _T.c.retry_count + 1

        await self._session.execute(
            update(_T)
            .where(_T.c.id == outbox_id)
            .values(
                retry_count=next_count,
                last_error=error,
                # 到上限就转 failed，让它**可见**：一直留在 pending 的话，
                # 下一次重试照样把它捞走，看起来一切正常，实际上它可能
                # 已经失败了几十次。
                status=(
                    # 在 SQL 里判断，避免"读了旧值再判断"的竞态：
                    # 只要 retry_count 已经到上限就转 failed。
                    case(
                        (
                            next_count >= max_retry_count,
                            OutboxStatus.FAILED.value,
                        ),
                        else_=OutboxStatus.PENDING.value,
                    )
                ),
            )
        )

    async def get_by_task(
            self,
            task_id: UUID,
    ) -> TaskOutbox | None:

        row = (await self._session.execute(
            select(*_T.c)
            .where(_T.c.task_id == task_id)
            # 一条任务理论上可有多行投递记录（投完没标记就崩了）。
            # 取最早的哪一条？取**最新**的：排查"为什么没被跑"时，
            # 最后一次尝试的结果才是相关的。
            .order_by(_T.c.created_at.desc(), _T.c.id.desc())
            .limit(1)
        )).one_or_none()

        return None if row is None else _to_entity(row._mapping)
