"""
`memory_atom` —— 从 L0 对话里提炼出来的记忆本体。

它回答什么
--------
"我们记住过哪些事实“。一行一条记忆，由 `memory.l1.extract` 任务从 L0 的原始
消息里抽出来（情境切分 + 三类抽取），再由同一个任务的第二步（去重）
与已有记忆合并。

⚠️ 它是**权威数据** —— 与 L0 那两张刚好相反
----------------------------------------
`agent_message_fts` / `agent_message_vec` 是纯派生的，删干净再从
`agent_message` 重建是安全的（`models.sql` 末尾那段就是这么写的，那张表
的 docstring 也这么说）。

**本表不是。** 它装的是 LLM 的结论：从同一批消息重跑一次会得到**不同**
的记忆（采样、候选池都在变）。所以：

    ❌ 不要照搬”删干净重建是安全的“那句话
    ✓  可以重建的是它的两张挂载表（memory_atom_fts / memory_atom_vec），源是本表

⚠️ 原地更新，不换 id
----------------
`update` / `merge` 的语义是”在存活的那一条上原地更新、version + 1"，
**不是**"删旧 id、插新 id"（原规格走的是后者）。三条理由：

    引用不断      指向这条记忆的引用不会因为一次合并而失效
    version 一致  全库只有一种 L1 更新语义，与“用户手工编辑”那条路径
                  对齐
    幂等          同一条记忆被反复写入不堆行 —— ON CONFLICT DO UPDATE
                  在这里是主路径，不是兜底

⚠️ 但被合并掉的**其余**记录仍然会消失
----------------------------------
提示词输出的 `target_ids` 是**数组**：多对一合并里存活一条、其余 N-1 条
要走删除。所以“引用不断”只对存活的那一条成立 —— 指向被吃掉那几条的引用
**照样断**。要彻底不断就得软删（加 superseded_by、行保留、查询侧过滤），
那是另一条路，当前没走。

⚠️ metadata_json 在库里叫 `meta`
-----------------------------
提示词与将来的接口都叫它 `metadata`，但那是 SQLAlchemy declarative 的
保留属性名（`Base.metadata`），不能直接做列名。按仓库既有口径统一在
**存储层**叫 `meta`，转换只发生在契约边界一次 —— 与 agent_session /
agent_run 的 `meta` 是同一个故事。

⚠️ 没有外键
--------
与 `agent_message` 同一条取舍：session_id / user_id 都只是“指向谁”的
记录。存亡由写入方负责（memory.l1.extract）。代价：会话被删时要显式清理这些行。

⚠️ 约束与索引在 ORM 上**也有一份**
-----------------------------
建表走 `Base.metadata.create_all`，它只认 ORM，**不读** models.sql。
两处必须同步，否则 .sql 读起来有而真实表没有 —— 最坏的一种不一致。
"""

from __future__ import annotations

import uuid
from datetime import datetime as dt
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ._scope import MemoryAtomScopeColumns

__all__ = ["MemoryAtom"]


#: 记忆类型词表 —— 当前**只作说明**，没有建 CHECK。
#:
#: ⚠️ 为什么不加 CHECK
#: ----------------
#: 这张词表是**会长**的：当前两份提示词只产 persona / episodic / instruction，
#: work_fact / work_task / work_method / work_artifact 四类要等另一套模板
#: （原规格里它们还依赖 task_id 识别 work 状态）。加 CHECK 等于让“加一类
#: 记忆”变成一次迁移，而漏改的表现是**写入直接失败**。
#:
#: 代价说清楚：写错的 type 会静默存下来，没有任何约束挡它。判据是“哪个
#: 更可能发生" —— 词表增长是确定的，拼错是可以在冒烟里发现的。
MEMORY_TYPES: tuple[str, ...] = (
    "persona",
    "episodic",
    "instruction",
    "work_fact",
    "work_task",
    "work_method",
    "work_artifact",
)


class MemoryAtom(MemoryAtomScopeColumns, Base):
    """
    一条提炼出来的记忆。

    主键 `record_id` 是**稳定身份**：合并时存活的那一条保留自己的 id，
    只把 `version` 加一。所以它不等于”一条内容“，而等于”一条记忆的
    生命史“—— 内容会被合并改写，id 不变。
    """

    __tablename__ = "memory_atom"

    # ⚠️ 原规格写的是 TEXT（`rec_xxx`）。
    #
    # 这里用 UUID：本仓库所有对外可见的身份都是 UUID（message_id /
    # session_id / run_id），而 record_id 会出现在提示词的输入输出里、
    # 将来也会出现在接口与 `memory_generation_refs` 之类的引用里。
    # 一个形态不同的 id 混进来，读代码的人每次都要确认”这个是不是外键
    # 能不能对上“，而收益只是几个字符的前缀。
    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="记忆身份。合并时存活行保留它，只加 version。无外键，见模块 docstring。",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "完整的、跳出当前对话依然成立的记忆陈述。**语言跟随用户消息**"
            "（提示词规定的）—— 所以本列可能是中文也可能是英文，而向量"
            "那侧用的是中文模型，见 memory_atom_vec.py。"
        ),
    )

    type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="",
        comment=(
            "记忆类型。取值见 MEMORY_TYPES，但**没有 CHECK** —— 词表会长，"
            "理由见那个常量的注释。"
        ),
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="50",
        comment=(
            "0-100，-1 = 全局死命令。\n\n"
            "⚠️ **不参与排序、不参与过滤**。它只有两个作用：抽取时给 LLM"
            " 自己当丢弃阈值（低于分档的直接不提取），以及工具检索时渲染成"
            " ` (priority: 85)` / ` (global instruction)` 给模型看。\n\n"
            "所以挂载表里**没有**这一列 —— 一旦冗余进去，早晚会有人开始"
            "拿它做 WHERE / ORDER BY，而那会造出”死命令排在第三位“这类"
            "没人解释得了的现象。"
        ),
    )

    scene_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="",
        comment=(
            "这条记忆所属的情境名。**由 LLM 生成，没有枚举**（提示词只要求"
            "它单句、约 30-50 字、全局唯一）。\n\n"
            "所以它是自由文本，不是外键、不做归一化 —— 归一化会打断 L2"
            "（情境的二次抽象）的归并依据。"
        ),
    )

    session_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default="",
        comment=(
            "会话的稳定键，当前等同于 `agent_session.thread_id`。\n\n"
            "⚠️ 与 session_id 冗余（本仓库里两者一对一）。留它是因为它才是"
            "模型侧认得的那个名字（LangGraph 的 thread_id），而 session_id"
            "是本仓库的内部主键。取值要点见 app/services/session —— thread_id"
            "是**惰性生成**的，没跑过运行的会话为 NULL。"
        ),
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
        comment=(
            "单调递增。update / merge 时取`被合并目标里最大的 version + 1`。\n\n"
            "它是”这条记忆被改写了几次“，不是内容的版本号 —— 内容与它不"
            "一一对应（同一版内容可能是合并来的）。"
        ),
    )

    # ---- 时间线（三个）------------------------------------------------
    #
    # ⚠️ 这三列**保持 TEXT**，与原规格一致，与下面两个时间列不同
    # ------------------------------------------------------
    # 它们是 LLM 从对话里**推出来的活动时间**（"用户上周三去了…"），可能
    # 是 "2018 年" 这种不能解析的半成品。真按 TIMESTAMPTZ 存，写入那一刻
    # 就会炸 —— 而那是模型的输出，不是我们的 bug。当字符串存，用什么格式
    # 由提示词约定。
    #
    # 后三个是一组：L1 的记忆会被 merge/update，时间线要**累积**（提示词
    # 要求 merged_timestamps 是所有新旧时间戳的并集），而 L0 的消息只有
    # 一个时间点，不需要区间。

    timestamp_str: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="",
        comment="时间线主值（提示词合并后的并集形式）。",
    )

    timestamp_start: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="",
        comment="活动时间下界。",
    )

    timestamp_end: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="",
        comment="活动时间上界。",
    )

    # ---- 系统生成的两个时间戳 -----------------------------------------
    #
    # ⚠️ 这两个是**真时间类型**，与原规格的 TEXT 不同
    # --------------------------------------------
    # 原规格是 SQLite，那边没有 datetime 类型。这边有：
    #
    #   updated_time  是**增量水位**（`memory_atom_cursor` 之外的第二处用途：按它
    #                 做跨会话的增量对账）。字符串比较做水位依赖字典序，
    #                 格式一漂移（时区、毫秒位数）就静默漏数据。
    #   created_time  与它成对，没有理由一个真一个假。

    created_time: Mapped[dt] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="这一行第一次写入的时间。合并更新**不动它**。",
    )

    updated_time: Mapped[dt] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment=(
            "最后一次被写入 / 合并的时间。**增量水位**。\n\n"
            "⚠️ 更新是原地进行的，所以这个值随每次 merge/update 前移 ——"
            "用它做对账时要按”键集 + 时间“分页，别只按时间（同一事务里"
            "多条记忆的时间戳可能完全相同）。"
        ),
    )

    # ---- 溯源与自由字段 ----------------------------------------------

    source_message_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="[]",
        comment=(
            "这条记忆是从哪几条 L0 消息来的（agent_message.message_id 的"
            "数组）。\n\n"
            "⚠️ 单独一列而不是塞进 meta：meta 的语义被钉成「episodic 的"
            "活动时间」了，混进去以后没人说得清里面有什么。\n\n"
            "⚠️ 没有外键，也不保证那些消息还在（会话可能被删）。它是"
            "**溯源线索**，不是约束。"
        ),
    )

    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment=(
            "episodic 的 activity_start_time / activity_end_time（提示词"
            "输出的 metadata 字段），其它类型为空对象。\n\n"
            "契约与提示词里叫 metadata，库里叫 meta —— 见模块 docstring。"
        ),
    )

    __table_args__ = (
        CheckConstraint(
            "version >= 0",
            name="ck_memory_atom_version_non_negative",
        ),
        CheckConstraint(
            "content <> ''",
            name="ck_memory_atom_content_not_blank",
        ),
        # ---- 增量水位 ----------------------------------------------------
        # 按 updated_time 做增量对账的形状。
        Index("idx_memory_atom_updated", "updated_time"),
        # ---- 会话 --------------------------------------------------------
        # 去重的候选池按会话收窄，这是它唯一的过滤键。
        Index("idx_memory_atom_session", "session_id"),
        # scope=user 的形状（跨会话召回）。
        Index("idx_memory_atom_user_updated", "user_id", "updated_time"),
    )
