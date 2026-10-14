"""
`memory_atom_vec` —— `memory_atom` 的**向量**索引（挂载表之二）。

它回答什么
--------
"哪条记忆的意思和这句最接近“。与 `memory_atom_fts` 并列，两路融合（RRF）之后
得到去重的候选池 / 检索结果。

⚠️ 建这张表之前，pgvector 扩展必须先就位
------------------------------------
与 `agent_message_vec` 同一个前置条件：`create_all` 会直接发
`CREATE TABLE ... embedding vector(512)`，扩展没装时报的是
`type "vector" does not exist`，而那是**应用启动失败**。顺序见
agent_message_vec.py 的模块 docstring（它是这条链最早的记录）。

⚠️ 复用 L0 的 embedder 与维度
------------------------
512 维、cosine、HNSW 参数默认，与 `agent_message_vec` 逐字相同，走的是
同一个进程级 embedder 实例（`document` 模块那个 bge-small-zh-v1.5）。
不另开一个的理由与 document/memory 共用时那条一样：权重要几个 GB，
两份实例等于把内存翻倍，而且可能加载成不同的东西。

⚠️ 已知代价：content 可能是英文
---------------------------
提示词规定 `content` 的**语言跟随用户消息**（"英文用户 → The user (Maya)
is a senior product manager"）。而 bge-small-zh 是中文模型，英文语义
召回会退化 —— 且**不报错**，与”查股东户数搜出天气“是同一类静默失败。

这一条是**已知并接受**的取舍（用户 2026-10-10 定的：L1 的 embedder 同 L0）。
缓解在关键词那一路：英文专名与数字能被 FTS 精确召回。真要修就是换多语
模型，那时 L0 与 L1 要一起换（否则两边的相关性不可比）。

⚠️ 没有外键、约束与索引两处同步
---------------------------
同本模块其它几张表。
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ._scope import MemoryAtomScopeColumns

__all__ = ["MemoryAtomVec"]


class MemoryAtomVec(MemoryAtomScopeColumns, Base):
    """
    一条记忆的向量索引行。

    主键与 memory_atom_fts 一样是 `record_id`，写入时机也一致：合并改写了存活行的
    content 就必须重算向量。两路各自可能缺席（embedder 没起来时向量路
    不可用），这与 L0 的取舍相同 —— 谁缺席了由调用方如实汇报，不静默降级。
    """

    __tablename__ = "memory_atom_vec"

    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        comment="指向 memory_atom.record_id。无外键，见模块 docstring。",
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(512),
        nullable=False,
        comment=(
            "记忆正文的稠密向量（bge-small-zh-v1.5，512 维，已归一化）。\n\n"
            "⚠️ 正文必须与 memory_atom_fts.content_original、memory_atom.content 是"
            "同一段文本 —— 三处取不同的块（比如一边拼了情境名、一边没拼），"
            "两路召回就在回答不同的问题，而这种不一致不会有任何报错。\n\n"
            "⚠️ 查询侧用 embed_query（带指令前缀），写入侧用 embed（不带）。"
            "搞反的表现是召回质量下降，不报错。"
        ),
    )

    __table_args__ = (
        # --- 维度不做 CHECK ---
        # `Vector(512)` 这个类型本身就在插入时校验维度（不匹配直接报错），
        # 与 L0 的 agent_message_vec 同一条取舍 —— 那边也只有 seq / role
        # 两个 CHECK，没有维度约束。
        #
        # ---- 召回主索引 --------------------------------------------------
        # vector_cosine_ops 必须与查询里的 `<=>` 一致：写成 l2_ops 而查询
        # 用 `<=>` 时 Postgres 会**忽略**这个索引改用顺序扫描 —— 结果正确、
        # 只是慢，所以它是个不会报错的性能坑。
        Index(
            "idx_memory_atom_vec_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # ---- 过滤索引 ----------------------------------------------------
        # 与 memory_atom_fts 同一组形状（去重候选池按会话收窄）。向量路的过滤更
        # 要紧：HNSW 是近似索引，过滤性差时更容易漏召回，所以 scope 上
        # 必须有可用索引。
        Index("idx_memory_atom_vec_session", "session_id"),
        Index("idx_memory_atom_vec_user", "user_id"),
    )
