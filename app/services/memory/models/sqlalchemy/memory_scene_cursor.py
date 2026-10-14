"""
`memory_scene_cursor` —— L2 整合的**增量水位**，一个会话一行。

它回答三件事
---------
    last_atom_updated_time   这个会话的 L1 记忆，整合到哪一条了
    last_atom_ids            恰好落在边界时间戳上的那一批（见下）
    last_l2_at               这个会话上一次 L2 跑完是什么时候（触发 A 的 lastL2）

⚠️ 为什么不复用 `memory_atom_cursor`
--------------------------
那张表的键也是 session_id、也是"一个会话一行"，看着像能合并。不能：

    memory_atom_cursor.last_seq    读的是 **agent_message.seq** —— 消息的会话内序号
    本表.last_atom_updated_time 读的是 **memory_atom.updated_time** —— 记忆的写入时间

两个水位推进的时机与粒度都不同（L1 一次运行会写若干条 atom，它们在**同一个
事务**里写入、`now()` 相同），而且 L1 的水位由 L1 自己推进、本表由 L2 推进。
合成一张表就只能靠"哪几列归哪条链"来读，而那是读代码的人每处都要重新判断
一遍的东西。

⚠️ 水位为什么按 `updated_time` 而不是 record_id
----------------------------------------
L1 的 `update` / `merge` 是**破坏性**的：多对一合并里存活那条保留 id、被吃掉
的几条**整行消失**。所以"消费过哪些 id"这个集合会随 L1 的合并而失真 ——
按 id 记水位的话，一条被合并的记忆会让水位含义变成"曾经消费过一批现在
已经不存在的 id"。

`updated_time` 天然覆盖这件事：合并的**存活者**一定被改写、一定前移，
所以它会重新进入下一批增量 —— 而这正是我们要的（内容变了，画像该重看）。
代价说清楚：**被删掉的记忆不会**再出现，画像里由它衍生的内容只能靠模型
这一次的整合覆盖掉（它看得到现有场景全文，所以看得到那份旧内容）。

⚠️ `last_atom_ids` 不是冗余
---------------------
Postgres 的 `now()` 取的是**事务开始时刻**，L1 一次运行写的所有 atom 因此
共享同一个 `updated_time`。L2 按 `updated_time > 水位` 读，水位推到那个
时间戳之后，**恰好落在同一微秒**的另一批（并发提交）就会被 `>` 静默跳过
—— 漏一条记忆、不报错、也永远补不回来。

所以查询写成

    updated_time > :t OR (updated_time = :t AND record_id NOT IN :ids)

推进水位时把边界上**已经消费过**的那批 id 记在这里。代价是一个 JSONB 列
与一条 OR 条件，换掉一类静默丢数据。

⚠️ last_l2_at 为什么在这张表上
------------------------
触发 A 要算 `T = max(now + 90s, lastL2 + 900s)`，而 **profile 行上没有会话
维度**（一个 scope 一行，被该 scope 下所有会话共享），所以 lastL2 只能按
会话记 —— 就记在这一行。它与水位同事务更新。

⚠️ 行是**惰性创建**的：没有行等价于"这个会话还没整合过"（水位为 NULL、
last_l2_at 为 NULL）。不必在会话创建时插行 —— 那样反而要处理"会话建了但
L2 从没跑过"这种空行的语义。与 memory_atom_cursor 同一条。

⚠️ 推进时机：**只在整合成功并提交之后**
--------------------------------
调用方的作用域里，profile 行写完才推它。提前推 = 那批 L1 永远进不了画像，
且没有任何症状 —— 与 L1 那条最硬的约束同源。

⚠️ 没有外键
--------
与 memory 模块其它表一致（理由见 memory_atom.py 与 memory_atom_cursor.py 的
模块 docstring）。代价同样说清楚：会话可删时这里要显式清理。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

__all__ = ["MemorySceneCursor"]


class MemorySceneCursor(Base):
    """
    一个会话的 L2 增量水位。

    主键是会话 —— 与 `memory_atom_cursor` 同一粒度。理由是同一族的：被消费的
    L1 记忆是**会话内**抽出来的（`memory_atom.session_id`），跨会话没有
    可比的水位；而一张画像被多个会话共同书写，所以画像本身不能带水位。
    """

    __tablename__ = "memory_scene_cursor"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        comment=(
            "所属会话（agent_session.session_id）。无外键，见模块 docstring。\n\n"
            "⚠️ 注意它与**写入目标**不是一回事：水位按会话，画像按 "
            "`(agent_id, user_id|org_id)`。N 个会话的水位各自推进，最终都"
            "落进同几行画像 —— 那是刻意的（画像就是跨会话聚合的产物）。"
        ),
    )

    last_atom_updated_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "已经整合过的 L1 记忆的**最大** `memory_atom.updated_time`。下次"
            "按 `> 它`（并列上 `last_atom_ids` 的排除）取增量。\n\n"
            "NULL = 这个会话还没有任何 L1 进过画像（与”没有行“区分开：有行"
            "但为 NULL 是”L2 跑过、但没有新记忆可整合“）。\n\n"
            "⚠️ 只在整合成功、画像落库的**同一个事务**里推进 —— 提前推会让"
            "那批记忆永久跳过，而它没有任何症状。"
        ),
    )

    last_atom_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="[]",
        comment=(
            "恰好落在 `last_atom_updated_time` 这个时间戳上、**已经消费过**"
            "的 atom id（字符串形式）。\n\n"
            "存在的原因是”同一微秒“：L1 一次运行写的所有 atom 共享一个"
            " `now()`，而并发提交的那批会落在同一个值上。没有这一列时，"
            "水位推过去之后它们被 `>` 永久跳过（静默漏记忆）。见模块"
            " docstring。\n\n"
            "⚠️ 它只在那个**边界时间戳**上有意义，不要拿它当”消费过的全部"
            " id 集合”读 —— 那是一个会无限增长的集合，而这里刻意只留一批。"
        ),
    )

    last_l2_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "这个会话上一次 L2 **跑完**的时间（含”跑了但没东西可整合“那一次）。\n\n"
            "触发 A 用它算下一次的**最早**时间：`T = max(now + 90s, "
            "lastL2 + 900s)` —— 也就是”至少隔 900 秒再整合一次“。所以空转"
            "那一轮**也要**戳它，否则一个反复空转的会话会每 90 秒白跑一次。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "会话归属用户。**冗余**（权威在 agent_session），带上是为了让"
            "排查与将来的清理路径不必回表 —— 与 memory_atom_cursor.user_id 同一"
            "取舍。它**不参与**任何过滤。"
        ),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment=(
            "最后一次推进的时间。\n\n"
            "⚠️ `onupdate=func.now()` 只对 ORM 的 UPDATE 生效；core 的 "
            "`ON CONFLICT DO UPDATE` 要显式写（见 scene_impl）。"
        ),
    )

    __table_args__ = (
        # 按用户找回该用户全部会话的水位（排查 / 将来的清理路径）。
        Index("idx_memory_scene_cursor_user", "user_id"),
    )
