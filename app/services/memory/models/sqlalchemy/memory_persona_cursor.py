"""
`memory_persona_cursor` —— L3 画像生成的**scope 级水位**，一个 scope 一行。

它回答三件事
---------
    last_scene_updated_time   这个 scope 的场景，抽象到哪一个版本了
    last_scene_ids            恰好落在边界时间戳上的那一批（见下）
    last_l3_at                上一次画像生成是什么时候（触发条件 2 的基准）

⚠️ 为什么不能复用 `memory_scene_cursor`
--------------------------------
那张表的键是 **session_id**、这张的键是 **scope**。两者不是粒度不同而已，
是**维度不同**：

    memory_scene_cursor     "这个会话的 L1 记忆整合到哪一条了" —— L2 的增量
    memory_persona_cursor   "这个 scope 的场景抽象到哪一版了"   —— L3 的增量

L2 的产物按 scope 落行、水位按会话记（一个 scope 被 N 个会话共同书写）；
L3 的输入（`memory_scene` 的行）**没有 session_id**（那是刻意的，见
`_scope.MemoryFileColumns`）—— 所以 L3 这条链上根本凑不出会话这个维度，
它的水位只能是 scope 级。合成一张表就得让一半的列对另一半是 NULL。

⚠️ 主键为什么是 `(agent_id, scope_key)` 而不是三列
----------------------------------------------
业务键是 `agent_id + (user_id | org_id)`，看起来写成
`PRIMARY KEY (agent_id, user_id, org_id)` 最直观。**不行**：scope 那两列
恰有一列恒为 NULL（个人 / 团队二选一），而主键的每一列都必须是 NOT NULL
—— Postgres 会直接拒绝建表，或者更糟（按口径写成唯一约束时）把 NULL 视为
互不相同，于是**一条冲突都挡不住**（同样的坑见 `_scope.MemoryFileColumns`
里那段关于复合唯一约束的说明）。

所以键的形态跟着 `SceneScope.key` 走：它把两列折成一个**非空**字符串
（`user:123` / `org:456`），加上 `agent_id` 就是完整的业务键。两个真实列
仍然留着（要按 org 查、要能一眼看出这行是团队还是个人），并由同一条
CHECK 保证它们与 `scope_key` 说的是同一件事。

⚠️ 场景水位为什么按 `updated_at` 而不是场景 id
---------------------------------------
L2 的写入是**原地改写 + 真删**（`[DELETED]` 那些在 apply 阶段被删掉）。
按 id 记水位的话，"我消费过哪些场景"这个集合会随着删除失真 —— 一条被删的
场景让水位含义变成"曾经消费过一批现在已不存在的 id"。

`updated_at` 天然覆盖这件事：被改写的场景一定前移（`update_scene` 里
`updated_at=func.now()`），于是它重新进入下一批增量 —— 而这正是我们要的
（内容变了，画像该重看）。

⚠️ 代价说清楚：**删除不会进增量**
------------------------
一条被删掉的场景不会出现在"变化场景"里，画像里由它衍生的内容只能靠模型
这一次的生成覆盖掉（它**看得见现有的整篇 persona**，所以有机会删掉那段）。
与 L2 面对 L1 删除时是同一条取舍（见 memory_scene_cursor.py）。

⚠️ `last_scene_ids` 不是冗余
-----------------------
与 L2 的 `last_atom_ids` 同一个成因：Postgres 的 `now()` 取的是**事务开始
时刻**，一次 L2 整合写的所有场景共享同一个 `updated_at`。L3 按
`updated_at > 水位` 读，水位推到那个时间戳之后，**恰好落在同一微秒**的
另一批（并发提交）就会被 `>` 静默跳过 —— 少抽象一个场景、不报错。
所以查询写成

    updated_at > :t OR (updated_at = :t AND id NOT IN :ids)

⚠️ 行是**惰性创建**的
----------------
没有行等价于"这个 scope 还没生成过画像"（两个水位都为 NULL）。不必在
scope 第一次出现时插行 —— 那样反而要处理"建了行但 L3 从没跑过"的语义。
与 `memory_atom_cursor` / `memory_scene_cursor` 同一条。

⚠️ 推进时机：**只在画像成功落库并提交之后**
------------------------------------
调用方的作用域里，persona 行写完才推它。提前推 = 那批场景**永久**进不了
画像，且没有任何症状（下次按 `> 水位` 取增量）—— 与 L1 / L2 那条最硬的
约束同源。

⚠️ 没有外键
--------
与 memory 模块其它表一致（理由见 memory_atom.py 的模块 docstring）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

__all__ = ["MemoryPersonaCursor"]


class MemoryPersonaCursor(Base):
    """
    一个 scope 的 L3 生成水位。

    主键是 `(agent_id, scope_key)` —— 与 `memory_scene` 的业务键同一口径
    （见模块 docstring 里为什么不是三列）。
    """

    __tablename__ = "memory_persona_cursor"

    agent_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        comment=(
            "画像归属的 agent（业务键的一半）。取值口径与 memory_scene.agent_id "
            "逐字一致 —— 取 L1 记忆上的 duty_key，没有时回退会话的 agent_id，"
            "再没有就是 DEFAULT_AGENT_ID。\n\n"
            "⚠️ 它必须是主键的一部分：同一个 user 在不同 duty 下是**两份**画像"
            "（L3 的输入是 L2 的产物，而 L2 就是按 (agent, user) 聚合的）。"
        ),
    )

    scope_key: Mapped[str] = mapped_column(
        String(300),
        primary_key=True,
        comment=(
            "`SceneScope.key` —— `user:{user_id}` 或 `org:{org_id}`。\n\n"
            "⚠️ 带上 `user:` / `org:` 前缀是**必须的**（不是装饰）：两种命名"
            "空间共用这一列，不加前缀的话某个 user_id 恰好等于某个 org_id 时"
            "两份画像会算出同一个键，静默互相覆盖。同一条理由见 "
            "domain/entities/scene.py 的 `SceneScope.key`。\n\n"
            "长度 300 = 前缀 + 255（user_id / org_id 都是 String(255)）。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "个人画像的归属用户。与 org_id **恰有一个为 NULL**（不变量由 "
            "`ck_memory_persona_cursor_scope_exclusive` 兜底）。\n\n"
            "⚠️ 它在 `scope_key` 里已经有一份编码了，留着这一列是为了能按它"
            "查（部分索引）与人工排查时一眼看出这行是个人还是团队。"
        ),
    )

    org_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="团队画像的归属组织。与 user_id 恰有一个为 NULL，见上。",
    )

    last_scene_updated_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "已经抽象过的场景的**最大** `memory_scene.updated_at`。下次按 "
            "`> 它`（并列上 `last_scene_ids` 的排除）取变化场景。\n\n"
            "NULL = 这个 scope 还从来没有场景进过画像（与”没有行“区分开："
            "有行但为 NULL 是”L3 跑过、但当时没有变化的场景“）。"
        ),
    )

    last_scene_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="[]",
        comment=(
            "恰好落在 `last_scene_updated_time` 这个时间戳上、**已经抽象过**"
            " 的场景 id（字符串形式）。\n\n"
            "存在的原因是”同一微秒“：一次 L2 整合写的所有场景共享一个"
            " `now()`，而并发提交的那批会落在同一个值上。没有这一列时，"
            "水位推过去之后它们被 `>` 永久跳过（静默漏掉一批场景）。"
            "与 memory_scene_cursor.last_atom_ids 同一个成因。\n\n"
            "⚠️ 它只在那个**边界时间戳**上有意义，不要拿它当”抽象过的全部"
            " id 集合“读 —— 那是一个会无限增长的集合，而这里刻意只留一批。"
        ),
    )

    last_l3_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "这个 scope 上一次 L3 **跑完**的时间（含”跑了但没东西可更新“"
            "那一次）。\n\n"
            "触发条件 2（”距上次生成 >= 50 轮次“）用它当基准：在 L2 完成"
            "那一刻，数触发它的那个会话自这个时刻以来的 L0 用户轮次。\n\n"
            "⚠️ 它同时是条件 1 / 1.5 之外唯一一条”什么时候该再生成一次“的"
            "时间依据，所以空转那一轮**也要**戳它，否则一个反复空转的 scope"
            "会被每个 L2 完成重新排一次。"
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
            "`ON CONFLICT DO UPDATE` 要显式写（见 persona_impl）。"
        ),
    )

    __table_args__ = (
        # 与 memory_scene / memory_persona 同一条不变量。挡的是"scope 解析
        # 写错了"这类 bug —— 那种行（比如两列都为空）永远查不出来，且不报错。
        CheckConstraint(
            "(user_id IS NULL) <> (org_id IS NULL)",
            name="ck_memory_persona_cursor_scope_exclusive",
        ),
        # ⚠️ 这一条挡的是"scope_key 与两列说的不是同一件事"：比如写入方把
        # scope_key 填成 `user:123` 而 user_id 列留空。它不会导致任何查询
        # 失败（查询只按 scope_key 走），只会让人工排查时看到一行自相矛盾的
        # 记录 —— 而那种行一旦出现就没法判断该信哪一半。
        CheckConstraint(
            "scope_key <> ''",
            name="ck_memory_persona_cursor_scope_key_not_blank",
        ),
        # 与画像两张表一样的两个方向部分索引：复合索引对"只按 org 查"用不上。
        Index(
            "idx_memory_persona_cursor_user",
            "agent_id",
            "user_id",
            postgresql_where=text("user_id IS NOT NULL"),
        ),
        Index(
            "idx_memory_persona_cursor_org",
            "agent_id",
            "org_id",
            postgresql_where=text("org_id IS NOT NULL"),
        ),
    )
