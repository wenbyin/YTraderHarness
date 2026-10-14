"""
`memory_scene` —— L2 场景文件（一行 = 一个 `.md`）。

它回答什么
--------
"这个（用户 | 团队）+ agent 组合下，我们记住了哪些事，按什么场景分的“。
每行是 `memory.l2.extract` 任务产出的一份场景叙事文档（含 META 块）。

⚠️ 它是**权威数据**，与 L0 两张索引表相反
----------------------------------
与 `memory_atom` 同一立场：行里装的是 LLM 的结论（把碎片记忆整合成的
叙事）。从同一批 L1 重跑一次会得到**不同**的文档。所以

    ❌ 不要以为"删干净重建是安全的"（那句话只对 memory_atom_fts 那类派生表成立）
    ✓  可以重建的是它的下游（将来若有索引/导航，源都是本表）

⚠️ 与腾讯那版的两处结构性差异
------------------------
    没有文件系统         那边一行对应 `scene_blocks/` 下的一个真实文件，
                        模型用 read/write/edit 工具操作它；这边**只有表**，
                        模型一次回复里给出"要写哪些文件"，由工程侧翻译成行
                        （见 models/sqlalchemy/_scope.py 的 MemoryFileColumns）。
    拆表                 那边 L2/L3 共用一张 profiles、靠 `type` 列区分；
                        这边分成两句表名（profile / persona），因为两者的
                        **读取方与生命周期都不同**（L3 由另一个任务、
                        另一份提示词维护）。

⚠️ 没有虚表
--------
与 L1 的三张表不同，本表**没有**关键词 / 向量挂载表 —— 当前没有任何
召回路径，场景是整段拼进提示词的（场景清单 + 全文）。这是**刻意的**：
先让它能被读出来（下一个消费者是 L3 与主模型的上下文），检索等有明确
需求再做。

⚠️ 约束与索引在 ORM 上**也有一份**
-----------------------------
建表走 `Base.metadata.create_all`，它只认 ORM，**不读** models.sql。
两处必须同步，否则 .sql 读起来有而真实表没有 —— 最坏的一种不一致。
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ._scope import MemoryFileColumns

__all__ = ["MemoryScene"]


#: 这一行归属哪一列是业务键的一半。两个方向各自一个**部分索引**。
#:
#: ⚠️ 为什么不是一个 `(agent_id, user_id, org_id)` 复合索引
#: -----------------------------------------------
#: 那个形状对 `WHERE agent_id=? AND org_id=?` 用不上（中间那列是 NULL/被
#: 跳过），而 team 画像恰恰只按 org 查。两个部分索引各自精确命中一种 scope，
#: 代价只是多一条索引定义 —— 而画像表的行数按构造很小（一个 scope 十几行），
#: 排序字段也不参与过滤，所以这里的收益是”读起来准确“而不是性能。
_MEMORY_PROFILE_INDEXES = (
    Index(
        "idx_memory_scene_user",
        "agent_id",
        "user_id",
        postgresql_where=text("user_id IS NOT NULL"),
    ),
    Index(
        "idx_memory_scene_org",
        "agent_id",
        "org_id",
        postgresql_where=text("org_id IS NOT NULL"),
    ),
)


class MemoryScene(MemoryFileColumns, Base):
    """
    一个场景文件。

    主键 `id` 是**内容派生**的稳定身份（    文件名)`）—— 不是随机 UUID。为什么必须是派生的、为什么不能用复合唯一
    约束代替它，见 `_scope.MemoryFileColumns` 的类 docstring。
    """

    __tablename__ = "memory_scene"

    id: Mapped[str] = mapped_column(
        String(80),
        primary_key=True,
        comment=(
            "`scene:v1:` + sha256(scope_key \\0 filename) 的 hex"
            "（11 + 64 = 75 字符，留到 80）。\n\n"
            "⚠️ 是**文本**不是 UUID：与 memory_atom.record_id 的取舍不同，"
            "那边是随机身份（会出现在工具入参与引用里），这边是**业务键的"
            "编码** —— 同一个 scope 下的同一个文件名永远得到同一个值，"
            "这正是”一行一个场景文件“这个不变量成立的方式。\n\n"
            "⚠️ 别拿它当外键用（当前没有任何表引用它）。它的价值在于"
            "`ON CONFLICT (id)` 与 `WHERE id = ? AND version = ?` 两条语句"
            "能一次命中。"
        ),
    )

    __table_args__ = (
        CheckConstraint(
            "content <> ''",
            name="ck_memory_scene_content_not_blank",
        ),
        CheckConstraint(
            "version >= 0",
            name="ck_memory_scene_version_non_negative",
        ),
        # ⚠️ 业务键的**不变量**：个人画像与团队画像二选一，不许多也不许空。
        # 它挡的是”scope 解析写错了“这类 bug —— 那种行会让同一个 scope
        # 的查询少召回一条，而且不报错（没有它，user_id 与 org_id 同时为
        # NULL 的行永远查不出来）。
        CheckConstraint(
            "(user_id IS NULL) <> (org_id IS NULL)",
            name="ck_memory_scene_scope_exclusive",
        ),
        *_MEMORY_PROFILE_INDEXES,
    )
