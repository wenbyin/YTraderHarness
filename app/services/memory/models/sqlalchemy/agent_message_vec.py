"""
`agent_message_vec` —— `agent_message` 的**向量**索引（挂载表之二）。

它回答什么
--------
"哪条消息的意思和这句最接近"。一行一条消息，`embedding` 由
`document` 模块那个稠密模型产出（bge-small-zh-v1.5，512 维），
查询侧用同一个模型、同一套指令前缀。

⚠️ 建这张表之前，扩展必须先就位
----------------------------
`initialize_database` 走的是 `Base.metadata.create_all`，它会直接发
`CREATE TABLE ... embedding vector(512)`。扩展没装时这条语句的报错是

    type "vector" does not exist

—— 而 `initializer.py` 的模型包列表是硬编码的，**只要这张表在列表里，
应用启动就会失败**。所以顺序是死的：

    ① 装扩展（PG 侧 vector.dll + .control + .sql，然后重启 PG）
    ② CREATE EXTENSION vector（在 ytrader_agent 这个库里）
    ③ 才把本模块的包加进 initializer.py 的 import_all_models 列表

反过来做（先加列表、后装扩展）拿到的是一个起不来的应用，而且报错指向
模型定义，离真正的原因很远。

Python 侧的 `pip install pgvector` **不**解决这件事：它给的是
SQLAlchemy 的 `Vector` 类型与 asyncpg 的 codec，服务端一个字节都没动。

⚠️ 为什么是 512 维 / cosine
------------------------
512 是 bge-small-zh-v1.5 的输出维度，与 `document/const.py` 里那个
"模型身份集中在一处"的常量是同一个值。**写入与查询必须同一个模型**，
换模型时维度可能不变、检索也照返回结果，只是排序没有意义 —— 那正是
const.py 那段注释要防的。

cosine 而不是 l2：两个模型输出的向量已经归一化，此时 cosine 与内积
等价，但 cosine 对"将来换成不归一化的模型"仍然成立。用错的表现是
召回质量下降，**不会报错**。

⚠️ HNSW 而不是 IVFFlat
--------------------
    HNSW     建索引慢、占内存多，查询快且**增量插入不掉质量**
    IVFFlat  建得快，但需要在**有数据之后**建，且要选 lists —— 而本表
             是持续追加的（每轮对话都写），列表数一选错就得重建

对话消息是持续追加的负载，HNSW 那一列更适合。参数用默认值（m=16,
ef_construction=64）：本表的规模是"一个用户的会话历史"，不是亿级语料。

⚠️ 那张表也**没有任何外键**，scope 列也是冗余的
------------------------------------------
与 `agent_message_fts` 同样的取舍，理由见它和 `_scope.py` 的 docstring。

⚠️ 约束与索引在 ORM 上**也有一份**
-----------------------------
同 fts 那张表：create_all 只认 ORM，models.sql 是给人读的 + 手工建库
脚本，两处必须同步。
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ._scope import MemoryScopeColumns

__all__ = ["AgentMessageVec", "EMBEDDING_DIMENSIONS"]


#: 向量维度 —— bge-small-zh-v1.5 的输出维度。
#:
#: 与 `app/services/document/const.py` 里的模型身份是同一个值。写成常量
#: 而不是在列上散着写 512：换模型时"哪里要改"必须能一眼看到，而漏改的
#: 表现是插入报维度不匹配（这个还算好）或者维度对得上但排序没意义
#: （这个没有任何提示）。
EMBEDDING_DIMENSIONS = 512


class AgentMessageVec(MemoryScopeColumns, Base):
    """
    一条消息的向量索引行。

    主键同样是 `message_id`，同样**一行一条消息**。upsert 的幂等性靠它：
    任务重试（至少一次投递意味着重复执行是常态）不会产生第二行。

    与 `agent_message_fts` 独立：embedding 模型没加载起来时关键词路照常
    工作，两条路各自可能缺席，而"谁缺席了"由响应里的 degraded_routes
    如实汇报（见 memory.yaml）。
    """

    __tablename__ = "agent_message_vec"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        comment="指向 agent_message.id。无外键，见模块 docstring。",
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=False,
        comment=(
            "消息正文的稠密向量（bge-small-zh-v1.5，512 维，已归一化）。\n\n"
            "⚠️ 正文与 fts 那张表的 original **必须是同一段文本** —— "
            "两边取不同的块（比如一边拼了工具调用、一边没拼），两路召回就"
            "在回答不同的问题，而这种不一致不会有任何报错。"
        ),
    )

    __table_args__ = (
        CheckConstraint(
            "seq > 0",
            name="ck_agent_message_vec_seq_positive",
        ),
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_agent_message_vec_role",
        ),
        # ---- 召回主索引 --------------------------------------------------
        # vector_cosine_ops 必须与查询里的 `<=>` 一致：写成 l2_ops 而查询
        # 用 `<=>` 时 Postgres 会**忽略**这个索引改用顺序扫描 ——
        # 结果正确、只是慢，所以它是个不会报错的性能坑。
        Index(
            "idx_agent_message_vec_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # ---- 过滤索引 ----------------------------------------------------
        # 与 fts 表同一组形状，理由见那一边。向量路的过滤更重要：HNSW 是
        # 近似索引，过滤掉的行不会命中，但扫描深度是按 top_k 定的 ——
        # 过滤性差时更容易漏召回，所以 scope 上要有可用的索引。
        Index(
            "idx_agent_message_vec_session_seq",
            "session_id",
            "seq",
        ),
        Index(
            "idx_agent_message_vec_user_created",
            "user_id",
            text("created_at DESC"),
        ),
    )
