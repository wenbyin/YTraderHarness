"""
`memory_persona` —— L3 用户画像（一个 scope 一行）。

它回答什么
--------
"这个（用户 | 团队）+ agent 组合**是个什么人**“。整篇 `persona.md`
（Archetype / 基本信息 / Chapter 1-4）内联在 `content` 里，由 L3 那条链
从场景文件（`memory_scene`）里二次抽象出来。

⚠️ 写入方是 `memory.l3.extract`，读取方还没接
-------------------------------------
产出它的是 L3 那条后台任务（`MemoryPersonaServiceImpl.memory_persona_extract`），
由 L2 完成触发（见 application/trigger.py 的 `submit_persona_extract`）。
消费侧 —— 画像怎么进主模型的上下文 —— 还没做，与 L2 的场景一样属"只有
生产侧"。

⚠️ 列集刻意与 memory_scene 一模一样
---------------------------
两张表装的是同一类东西（一行 = 一篇 markdown），共用
`_scope.MemoryFileColumns`。差别只有 filename 的取值域（L3 恒为
`persona.md`）与主键前缀（`scene:v1:` / `persona:v1:`）—— 于是"将来真要
合表"不必重算 id。**不要**单方面给它加 L2 没有的列。

⚠️ 一个 scope 只有一行（`filename` 恒为 `persona.md`）
-----------------------------------------------
与 L2 不同，画像不是"一个话题一行"，是**一整篇**。所以这个 scope 下
永远不会有第二个 filename，`id` 也就恒等于
`persona:v1:sha256(scope, "persona.md")` —— 仍然是派生键（同一套
编码，`kind` 段区分 L2/L3），仍然靠主键保证不出现第二行。

⚠️ 其余取舍全部照抄 memory_scene
-----------------------------
没有虚表、没有外键、CHECK 与索引在 ORM 与 models.sql 上各有一份、
列集与理由见 `_scope.MemoryFileColumns` 与 memory_scene.py 的模块 docstring。
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ._scope import MemoryFileColumns

__all__ = ["MemoryPersona"]

# ⚠️ 名字 `persona.md` **刻意不抽常量**：L3 那条链还没落地，此刻抽一个没人
#    引用的常量只会多一处”它到底被谁用“的疑问（grep 不到消费者，看起来像
#    漏接线）。L3 落地时它该住在 **domain 层**（`scene_id` 的旁边）——
#    模型层是被 domain 引用的那一侧，反过来就是分层倒置。名字本身写在模块
#    docstring 与类里的 id 注释中。


class MemoryPersona(MemoryFileColumns, Base):
    """一整篇用户画像。列的语义见 `_scope.MemoryFileColumns`。"""

    __tablename__ = "memory_persona"

    id: Mapped[str] = mapped_column(
        String(80),
        primary_key=True,
        comment=(
            # ⚠️ 这一行用单引号做定界符是**刻意的**：正文里要原样出现 ASCII
            # 双引号（`kind` 那一段是两个字符串字面量），用双引号定界就得
            # 逐个转义，而转义过的引号在读代码时与内容混在一起分不清。
            '`persona:v1:` + sha256(scope_key \\0 "persona.md") 的 hex。\n\n'
            "与 memory_scene.id 同一个编码、同一个长度约束，`kind` 那一段"
            "把两张表隔开（L2 是 `l2`）—— 于是两张表的主键**不会撞**，将来"
            "真要合表也不用重算。"
        ),
    )

    __table_args__ = (
        CheckConstraint(
            "content <> ''",
            name="ck_memory_persona_content_not_blank",
        ),
        CheckConstraint(
            "version >= 0",
            name="ck_memory_persona_version_non_negative",
        ),
        # 与 memory_scene 同一条不变量：个人 / 团队二选一。
        CheckConstraint(
            "(user_id IS NULL) <> (org_id IS NULL)",
            name="ck_memory_persona_scope_exclusive",
        ),
        Index(
            "idx_memory_persona_user",
            "agent_id",
            "user_id",
            postgresql_where=text("user_id IS NOT NULL"),
        ),
        Index(
            "idx_memory_persona_org",
            "agent_id",
            "org_id",
            postgresql_where=text("org_id IS NOT NULL"),
        ),
    )
