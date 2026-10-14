"""
`agent_message_fts` —— `agent_message` 的**关键词**索引（挂载表之一）。

它回答什么
--------
"哪条消息里出现了这些词"。写入的是**分好词的串**，不是一个搜索服务：
`tokens` 由应用侧的分词器产出（见 ../../tokenization.py），`tsv` 是它的
`tsvector`，GIN 索引落在 `tsv` 上。

⚠️ 为什么不给 agent_message 加一列
--------------------------------
三条，任何一条单独都够：

1. `create_all` 不会给已存在的表加东西。SQLAlchemy 的原话是
   "will not attempt to recreate tables already present"（schema.py 的
   `create_all` 文档）—— 加**列**和加**索引**都轮不到它，只能手写 ALTER。
   而 `agent_message.py` 的类 docstring 里正好留过一段 ALTER 配方和它踩
   的坑（"漏了回填那句，症状是读历史直接 500，且看起来像序列化问题"）。
   新表走 create_all 正规路径，索引与约束一次建齐。
2. `agent_message.updated_at` 挂着 `onupdate=func.now()`，任何 UPDATE 都会
   把它推走，而那个类 docstring 明写"`updated_at == created_at` 恒成立、
   等真出现编辑消息的能力它才会分叉"。异步回填索引列就是一次 UPDATE，
   于是那句话从真话变成假话。
3. session 模块会因此被迫依赖 memory 的分词器（那一列得由谁填？）。

⚠️ 原文另存一列，而且**必须**另存
----------------------------
`tokens` 是给人看的搜索结果里**不能**出现的形态（"东方 方财 财富"那样
空格连起来的一串），展示与高亮都要原文。所以 `original` 与 `tokens` 成对
写入，两者由同一次调用产出 —— 分开写会漂。

⚠️ tsv 是**生成列**，不是触发器
--------------------------
`to_tsvector('simple'::regconfig, tokens)` 这个**两参**形式是 IMMUTABLE
的（`pg_proc.provolatile = 'i'`，实测过），所以可以直接做生成列。一参形式
`to_tsvector(tokens)` 是 STABLE（依赖 `default_text_search_config`），
**不能**用在生成列里 —— 换成它会在建表时报错，而错误信息不会告诉你原因
是 volatile。

代价是 tokens 一变 tsv 就重算（那是我们要的）。收益是没有触发器、没有
"忘了建触发器所以索引一直空着"这种故障模式。

⚠️ 改了分词器就要重建
----------------
倒排只认写入时的切法。分词规则一改，新写入的 query 与旧索引里的词元
互不可见 —— 症状是"明明有这条却搜不到"，**不报错**。重建脚本见
models.sql 末尾。

⚠️ 这一张表**没有任何外键**
------------------------
`message_id` / `session_id` / `run_id` 都只是"指向谁"的记录，不是数据库
约束 —— 与 `agent_message` 自己那三条同样的取舍（见它的类 docstring）。
悬空行的代价由写入方承担：memory.l0.extract 任务按 run_id 重跑即可。

⚠️ 约束与索引在 ORM 上**也有一份**
-----------------------------
建表走的是 `Base.metadata.create_all`，它只认 ORM 上声明的东西，
**不读** models.sql。只在那份 .sql 里写约束，真实建出来的表就没有它们，
而 .sql 读起来像是有的 —— 最坏的一种不一致。两处必须同步。
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    CheckConstraint,
    Computed,
    Index,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ._scope import MemoryScopeColumns

__all__ = ["AgentMessageFts"]


class AgentMessageFts(MemoryScopeColumns, Base):
    """
    一条消息的关键词索引行。

    主键是 `message_id`（不是自增 id）：**一条消息一行**，这就是去重的
    全部机制。消息不可变，所以重跑一次索引是 upsert 而不是追加 ——
    任务被重试（至少一次投递意味着重复执行是常态）也不会产生第二行。

    与 `agent_message_vec` 的关系：两张表各自独立写入、各自可能失败。
    这不是缺陷 —— 关键词路不依赖 embedding 模型，向量模型没加载起来时
    它照常工作，而"两路都能用"由 degraded_routes 如实汇报（见 memory.yaml）。
    """

    __tablename__ = "agent_message_fts"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        comment="指向 agent_message.id。无外键，见模块 docstring。",
    )

    tokens: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "分词后的词元串（空格连接），由 memory.tokenization.index_text "
            "产出。**不是给人读的** —— 展示走 original。"
        ),
    )

    original: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "消息正文的原文（所有 text 块的拼接）。展示与高亮从这一列出。\n\n"
            "⚠️ 与 tokens 由同一次调用产出，不要分开写。"
        ),
    )

    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple'::regconfig, tokens)", persisted=True),
        comment=(
            "tokens 的 tsvector，生成列（IMMUTABLE 的两参形式）。\n\n"
            "用 simple 配置：中文分词已经在 tokens 里做完了，这里只需要"
            "一个「按空格切、不做词干还原」的配置 —— 换成 english 会把"
            "词尾砍掉，而中文词元被砍一刀就是另一个词。"
        ),
    )

    __table_args__ = (
        CheckConstraint(
            "seq > 0",
            name="ck_agent_message_fts_seq_positive",
        ),
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_agent_message_fts_role",
        ),
        # 与 agent_message 同一条理由：形状退化（空串、全是标点的正文）
        # 不该等到检索时才暴露成"什么都搜不到"。
        CheckConstraint(
            "tokens <> ''",
            name="ck_agent_message_fts_tokens_not_blank",
        ),
        # ---- 召回主索引 --------------------------------------------------
        # 关键词路唯一的索引。tsvector 上的 GIN 是 Postgres 全文检索的
        # 标准搭配（GiST 更小、更新更快，但查询慢且会假阳性重查）。
        Index(
            "idx_agent_message_fts_tsv",
            "tsv",
            postgresql_using="gin",
        ),
        # ---- 过滤索引 ----------------------------------------------------
        # scope=session 的形状：按会话取、按 seq 排序（"这一轮之前/之后"）。
        Index(
            "idx_agent_message_fts_session_seq",
            "session_id",
            "seq",
        ),
        # scope=user + 时间窗的形状。DESC 与"最新的优先"一致。
        Index(
            "idx_agent_message_fts_user_created",
            "user_id",
            text("created_at DESC"),
        ),
    )
