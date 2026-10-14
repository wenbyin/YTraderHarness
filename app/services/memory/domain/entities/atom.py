"""
L1（memory_atom）的领域实体。

三种东西，分别对应三张表 + 一行状态：

    AtomCursorState   水位（memory_atom_cursor 的读模型）
    MemoryAtom          记忆本体（memory_atom 的写/读模型）
    AtomFtsRow         关键词索引行（memory_atom_fts）
    AtomVecRow         向量索引行（memory_atom_vec）

⚠️ 为什么索引行要单独两个模型，而不是让仓储从 MemoryAtom 自己派生
-----------------------------------------------------------
与 L0 同一条理由（见 application/_assembly.py 的模块 docstring）：进索引的
**必须是同一段文本**，而“哪段文本”这个决定要有唯一一处。向量还要过一遍
embedder —— 那是服务层才知道的事（它握着 embedder）。所以实体保持“数据"
的样子，索引行由服务层组装。

⚠️ 与 ORM 模型的关系
----------------
`MemoryAtom` 开了 `from_attributes`，所以能从 memory_atom 的 ORM 行直接
校验出来（仓储读路径用）。反方向（实体 -> ORM 行）走仓储里的 `_values()`
—— 那里是列名的唯一落点。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AtomFtsRow",
    "AtomRecall",
    "AtomVecRow",
    "MemoryAtom",
    "MemoryAtomQueryScope",
    "AtomCursorState",
]


class MemoryAtomQueryScope(BaseModel):
    """
    已解析、且**身份已经钉死**的过滤条件（L1 版）。

    与 `MemoryQueryScope`（L0）同一条分工：服务层算出它，仓储只按它拼
    WHERE —— 仓储因此不需要认识"scope 枚举“，也不需要知道”当前用户是谁"
    是从哪来的。越权因此变成一个在服务层就能看全的问题。

    ⚠️ 只有两列，没有时间窗
    -------------------
        session_id 有值   限定单个会话（去重的候选池 / “这次聊的”）
        session_id None   按 user_id 铺开（跨会话 —— 检索时 L1 事实的主场）
        user_id 有值      归属过滤：`= X OR IS NULL`
        user_id None      **只看无归属的行**（防御，见仓储里的说明）

    ⚠️ 去重的候选池走的是 `session_id` 那一支（原规格）
    --------------------------------------------
    所以跨会话的同一事实不会合并 —— 代价见 memory_atom_service_impl 的
    `_dedup_and_write` 里那段。

    L0 有时间窗（`time_from` / `time_to`），L1 **刻意没有**：记忆的时间来自
    提示词推出来的 `timestamp_str`，那是 TEXT（可能是"2018 年”这种），拿它
    做范围过滤要么报错要么静默错。要按时间筛，先得把时间线变成可信的类型。
    """

    model_config = ConfigDict(extra="forbid")

    session_id: UUID | None = Field(
        default=None,
        description="限定单个会话。None = 不按会话收窄（跨会话召回）。",
    )

    user_id: str | None = Field(
        default=None,
        description=(
            "归属用户。过滤规则是 `= X OR IS NULL`（NULL 是存量语义），"
            "与 agent_session 逐字一致。"
        ),
    )


class AtomRecall(BaseModel):
    """
    一次召回命中的一条记忆 —— **只有身份与分数**。

    与 L0 的 `MemoryRecall` 同形（那边带 `message_id`）。内容不在里面：
    命中之后由调用方按 id 回主表取（`MemoryAtomRepository.load`）—— 两路
    召回各自只需知道”哪条、多相关“，把正文一起搬回来会让融合阶段搬运
    用不上的数据。

    ⚠️ `score` 只在**同一次调用、同一条路**内可比
    -----------------------------------------
    关键词路是 `ts_rank`，向量路是余弦相似度 —— 不同量纲。融合因此按
    **名次**（RRF）而不是分数相加，见 `..fusion`。
    """

    model_config = ConfigDict(extra="forbid")

    record_id: UUID
    score: float


class AtomCursorState(BaseModel):
    """
    一个会话的 L1 水位（`memory_atom_cursor` 的读模型）。

    ⚠️ 没有对应行时返回的是**默认值**（0 / 空串），而不是 None
    ------------------------------------------------
    水位行的语义就是”没有行 = 还没抽过“，让调用方每次都判 None 等于把
    这条规则抄到每个调用点。仓储直接把它折成 `last_seq=0`。
    """

    model_config = ConfigDict(extra="forbid")

    last_seq: int = Field(
        default=0,
        description="已经抽过的最大 agent_message.seq。下次按 `seq > 它`取增量。",
    )

    last_scene_name: str = Field(
        default="",
        description=(
            "上一次抽取结束时处在哪个情境 —— 提示词任务一「结合上一个情境」"
            "的输入。空串 = 还没有过情境。"
        ),
    )


class MemoryAtom(BaseModel):
    """一条提炼出来的记忆。"""

    model_config = ConfigDict(from_attributes=True)

    record_id: UUID = Field(description="记忆身份。合并时存活行保留它。")

    content: str = Field(min_length=1)

    type: str = Field(default="")

    priority: int = Field(default=50)

    scene_name: str = Field(default="")

    session_key: str = Field(
        default="",
        description="会话的稳定键（当前 = agent_session.thread_id）。",
    )

    version: int = Field(default=0)

    timestamp_str: str = Field(default="")
    timestamp_start: str = Field(default="")
    timestamp_end: str = Field(default="")

    source_message_ids: list[Any] = Field(
        default_factory=list,
        description="来源消息 id 的**字符串**列表（解析得出 UUID 的那些）。",
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="提示词输出的 metadata（episodic 的活动时间）。",
    )

    created_time: datetime | None = Field(
        default=None,
        description=(
            "这条记忆**第一次**被写下来的时间（memory_atom.created_time）。\n\n"
            "与 `updated_time` 成对：合并会改写后者、前者不动。因此"
            "「这条记忆是什么时候开始存在的」看它，「现在这个样子是什么"
            "时候变成的」看 updated_time。带上提示词时两个都要 —— 只给"
            "一个，模型就分不出「早就有的事实」与「刚被合并改写过的事实」。\n\n"
            "⚠️ 与 `updated_time` 同一条：可以为 None（还没落库的那批）。"
        ),
    )

    updated_time: datetime | None = Field(
        default=None,
        description=(
            "最后一次写入 / 合并的时间（memory_atom.updated_time）。\n\n"
            "⚠️ **L2 的增量水位按它推**（见 MemoryAtomRepository."
            "list_session_delta）—— 它是这个实体上唯一一个”这条记忆"
            "什么时候变成现在这个样子”的信息，而 `update` / `merge` 是"
            "破坏性的（被吃掉的记录整行消失），所以按 id 记水位会随合并失真。\n\n"
            "⚠️ 可以为 None：**还没落库**的记忆（L1 抽取时刚组装出来的"
            "那批）没有这个值，它由数据库在插入时给。别拿它的 None 当"
            "”这条记忆很旧“来判 —— 那是两件事。"
        ),
    )

    # ---- scope 列（见 models/sqlalchemy/_scope.py）-------------------

    session_id: UUID

    business_agent_id: str | None = Field(
        default=None,
        description=(
            "主 agent 的**职责**（duty_key）—— 这条记忆是**为那份职责**而记的。"
            "取自本批消息上最近的那个 `agent_message.business_agent_id`；"
            "一条都没有（存量消息）时退回会话的 `agent_id`。\n\n"
            "⚠️ 与 L0 索引表上的同名列**是同一个口径**（都装 duty_key），"
            "但**粒度不同**：那边一列一消息，就是那条消息那一轮的职责；"
            "这里一条记忆跨很多轮，取的是这批增量讲完时在履职的那份职责。"
        ),
    )

    user_id: str | None = None
    org_id: str | None = None

    task_id: str | None = Field(
        default=None,
        description="当前恒为 None（留给 work_* 四类记忆的预留位）。",
    )


class AtomFtsRow(BaseModel):
    """一条记忆的关键词索引行。`original` 必须与 `MemoryAtom.content` 逐字相同。"""

    model_config = ConfigDict(extra="forbid")

    record_id: UUID
    session_id: UUID
    business_agent_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None

    tokens: str = Field(description="分词后的词元串（空格连接）。")
    original: str = Field(description="记忆正文原文。")


class AtomVecRow(BaseModel):
    """一条记忆的向量索引行。`embedding` 必须是对同一个 `content` 算的。"""

    model_config = ConfigDict(extra="forbid")

    record_id: UUID
    session_id: UUID
    business_agent_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None

    embedding: list[float]
