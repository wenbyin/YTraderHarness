"""
`memory_atom_fts` —— `memory_atom` 的**关键词**索引（挂载表之一）。

它回答什么
--------
"哪条记忆里出现了这些词“。两种用途，用的是同一张表：

    去重的候选召回  拿一条新记忆的 content 当 query，找出可能冲突的旧记忆
    记忆检索        将来的自动召回 / 检索工具

⚠️ 只放召回要用的列，其余回 memory_atom 读
------------------------------------
这是 L0 那两张表立下的规矩，也是它们当初把 kind / status 删掉的理由：
**源表主键就在本表上，能回源表拿的东西不该在这里放第二份**（第二份会漂，
且漂了没人知道）。

所以原规格里那一长串 UNINDEXED 列（type / priority / scene_name /
version / 三个时间戳 / metadata_json …）**一个都不搬** —— 命中之后按
`record_id` 回 `memory_atom` 取即可，那是主键查。L0 的检索路径就是这么
做的（融合之后统一回表取正文与展示字段）。

唯一带上来的是**过滤键**：`session_id`（去重的候选池按它收窄）与
`user_id`（可见性口径，规则见 _scope.py）。它们在这里的意义和 L0 那边
一样 —— 让”过滤 + 排序“在同一张表上做完，不必 join 出去。

⚠️ 原规格的 `task_id` 列**去掉了**
----------------------------
那是”工作任务“的标识（work_* 四类记忆靠它判断 work 状态），本仓库没有
这个概念。而且系统里已经有一个 `tasks.task_id`（后台任务主键），
留一个同名不同物的列，早晚有人把它们对上。

⚠️ 没有外键、约束与索引两处同步
---------------------------
与本模块另外几张表同一条取舍，理由见 memory_atom.py 与 L0 的
agent_message_fts.py。
"""

from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Computed, Index, Text
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ._scope import MemoryAtomScopeColumns

__all__ = ["MemoryAtomFts"]


class MemoryAtomFts(MemoryAtomScopeColumns, Base):
    """
    一条记忆的关键词索引行。

    主键是 `record_id`：**一条记忆一行**。它跟着 memory_atom 的原地更新走
    —— 合并时存活行的内容变了，这里**必须一起重写**（tokens / original
    都是内容的派生）。写入方因此永远把两边放在同一个事务里。
    """

    __tablename__ = "memory_atom_fts"

    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        comment="指向 memory_atom.record_id。无外键，见模块 docstring。",
    )

    content_tokens: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "记忆正文分词后的词元串（空格连接），由 memory.tokenization"
            ".index_text 产出。**不是给人读的** —— 展示走 content_original。"
        ),
    )

    content_original: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "记忆正文原文。展示与高亮从这一列出。\n\n"
            "⚠️ 与 content_tokens 由同一次调用产出，且**必须与 memory_atom"
            ".content 逐字相同** —— 三处不一致时向量那侧就在回答另一个"
            "问题，而那种不一致不会有任何报错。"
        ),
    )

    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple'::regconfig, content_tokens)",
            persisted=True,
        ),
        comment=(
            "content_tokens 的 tsvector，生成列（IMMUTABLE 的两参形式）。\n\n"
            "用 simple 配置：中文分词已经在 tokens 里做完了，这里只需要"
            "一个「按空格切、不做词干还原」的配置。与 L0 的 agent_message_fts"
            "逐字相同 —— 换配置会让两边的相关性不可比。"
        ),
    )

    __table_args__ = (
        CheckConstraint(
            "content_tokens <> ''",
            name="ck_memory_atom_fts_tokens_not_blank",
        ),
        # ---- 召回主索引 --------------------------------------------------
        Index("idx_memory_atom_fts_tsv", "tsv", postgresql_using="gin"),
        # ---- 过滤索引 ----------------------------------------------------
        # 去重候选池的形状：按会话取。与向量表用同一组键，两路融合时
        # 两边筛出来的候选集才可比。
        Index("idx_memory_atom_fts_session", "session_id"),
        Index("idx_memory_atom_fts_user", "user_id"),
    )
