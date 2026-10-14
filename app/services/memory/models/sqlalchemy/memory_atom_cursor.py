"""
`memory_atom_cursor` —— L1 抽取的**增量水位**，一个会话一行。

它回答什么
--------
"这个会话的记忆，抽到哪一条消息了“。`memory.l1.extract` 每次按
`agent_message.seq > last_seq` 取增量，成功落库之后才把这个值往前推。

⚠️ 为什么必须是**独立的一行**，不能从 memory_atom 反推
------------------------------------------------
原规格写的是"`updated_time` ← pipeline cursor 就靠这个做增量”，也就是
拿记忆表里最新的时间戳当水位。那在本仓库会错，因为 `update` / `merge`
是**破坏性**的（多对一合并要删掉被吃掉的记录）：

    被删的恰好是最新那条  ->  反推出来的水位**倒退**
    被删的行根本不在表里  ->  max(updated_time) 是“还活着的最新一条”，
                              不是“处理到哪了"

水位必须与记录的存亡无关。这也正是 L0 没这个问题的原因 ——
它是幂等全量重写，没有破坏性删除。

⚠️ 为什么键是 session_id 而不是 user_id
----------------------------------
`agent_message.seq` 是**会话内**单调递增的（三处一致：实体注释、
`(session_id, seq)` 索引、`CHECK seq > 0`），跨会话没有可比性。所以水位
只能按会话记。（去重的候选池也按会话收窄，粒度是一致的。）

⚠️ 推进时机：**只在抽取成功并提交之后**
--------------------------------
这是整条链上唯一不能让步的一条：

    提前推进  那批消息**永久跳过** —— 延时任务读的就是 `seq > last_seq`，
              救不回来
    丢了推进  下一轮收尾读到同样的增量，自然重来（自愈）

反过来说，**轮次计数丢了无所谓**：计数只决定”什么时候发任务“，抽取永远
按水位读全量。所以

    ✓  水位是唯一的事实来源，只在成功后前进
    ✗  不要为了”少查一次“把水位提前推进

⚠️ last_scene_name 存的是”上一个情境"
---------------------------------
抽取的第一步要“结合【上一个情境】判断继承还是切换”（提示词任务一）。
它**不能从 memory_atom 读**：承载那个情境的记录可能已经被一次 merge 删掉，
而合并后的记录带的是**新记忆**的情境（冲突提示词的输出里根本没有
scene_name）。所以它和“抽到哪了”一样，属于“必须放在不会被破坏性改写的
地方”的状态 —— 就放在这一行，与 last_seq 同事务更新。

⚠️ 没有外键 —— 与 memory 模块的其它表一致
------------------------------------
严格说这一行是**权威状态**而不是派生索引（与那三张表不同），所以
"会话没了、水位行还在“是一条会积累垃圾的路径。仍然不加外键的理由是：
本模块其余几张表都刻意没有外键（见 memory_atom.py 与 initializer.py 的
说明），单为这张表破例会让”这个模块到底有没有外键“变成要看表名才知道。

代价说清楚：**将来会话真的可删时，这里要显式清理**（一条 DELETE，或那时
再补外键 + ON DELETE CASCADE）。今天会话不可删，所以这条路径还不存在。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

__all__ = ["MemoryAtomCursor"]


class MemoryAtomCursor(Base):
    """
    一个会话的 L1 增量水位。

    行是**惰性创建**的：没有行等价于 `last_seq = 0`（这个会话还没抽过），
    所以不必在会话创建时插一行 —— 那样反而要处理”会话建了但 L1 从没跑过"
    这种空行的语义。
    """

    __tablename__ = "memory_atom_cursor"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        comment=(
            "所属会话（agent_session.session_id）。无外键，见模块 docstring。\n\n"
            "一个会话一行 —— 水位是**会话级**的，理由见模块 docstring。"
        ),
    )

    last_seq: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default="0",
        comment=(
            "已经抽过的最大 `agent_message.seq`。下次按 `seq > last_seq` "
            "取增量。\n\n"
            "⚠️ 只在抽取成功、记录落库提交**之后**才推进，理由见模块"
            " docstring —— 提前推进会让那批消息永久跳过。"
        ),
    )

    last_scene_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="",
        comment=(
            "上一次抽取结束时处在哪个情境 —— 提示词任务一“结合上一个情境”"
            "要用的输入。空串 = 这个会话还没有过情境。\n\n"
            "⚠️ 与 last_seq 同事务更新，否则会读到“上一个情境”与“抽到哪了”"
            "不匹配的状态。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "会话归属用户。**冗余**（权威在 agent_session），带上是为了让"
            "并发互斥与清理路径不必回表。"
        ),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后一次推进的时间。排查“这个会话为什么一直没抽”用的。",
    )

    __table_args__ = (
        CheckConstraint(
            "last_seq >= 0",
            name="ck_memory_atom_cursor_last_seq_non_negative",
        ),
        # 按用户找回该用户全部会话的水位（排查 / 将来的清理路径）。
        Index("idx_memory_atom_cursor_user", "user_id"),
    )
