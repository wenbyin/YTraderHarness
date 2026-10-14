"""
memory_atom 的**对外契约**：检索请求 / 响应。

对应 memory_atom.yaml，同时也是暴露给主模型的工具入参 schema
（`memory_atom.search`）—— 与 memory.yaml 的 `MemorySearchRequest` 是
同一种双身份，理由见那个模块的 docstring。

⚠️ 与 `domain/schemas/atom.py` 不是一回事
-------------------------------------
那边是**模型输出**的解析形状（抽取 / 冲突检测的 JSON），只对那两份提示词
负责。这边是对调用方（HTTP、工具）负责的契约。两边字段名巧合地像，改一边
不必改另一边。

⚠️ 请求体里**没有身份**
------------------
`session_id` 走 query 参数（HTTP）或由运行时注入（工具），`user_id` 从
token / 调用者上下文取。**绝不能**进这个模型：它同时是模型能填的
bind_tools schema，身份一旦进去就是一条跨租户读的路径。

⚠️ 没有的字段，以及为什么
--------------------
    type 过滤     `type` 不在挂载表上（它不参与过滤，见 models 里的说明），
                  加这个筛选项就要么 join 回主表、要么把它冗余进索引 ——
                  为了一个没人要过的筛选项付这个代价不划算
    时间窗        `timestamp_*` 是 LLM 推出来的 TEXT（可能是"2018 年“），
                  拿它做范围过滤要么报错要么静默错
    priority 过滤  它明确不参与排序与过滤（只给 LLM 看），见 memory_atom 的列注释

要按这几样筛，得先让它们变成可信的数据 —— 那是另一个决定，不是加个字段。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DegradedAtomRoute",
    "MemoryAtomRecallRoute",
    "MemoryAtomSearchHit",
    "MemoryAtomSearchMode",
    "MemoryAtomSearchRequest",
    "MemoryAtomSearchResponse",
    "MemoryAtomSearchScope",
]


class MemoryAtomSearchScope(str, Enum):
    """
    召回铺多宽。

        session   只在本会话的记忆里找（"这次聊的”）
        user      跨会话 —— **L1 事实的主场**（memory.yaml 里那句
                  "the scope L1 facts live in"）
    """

    SESSION = "session"
    USER = "user"


class MemoryAtomSearchMode(str, Enum):
    """跑哪几路。`hybrid` 两路都跑并按名次融合。"""

    HYBRID = "hybrid"
    KEYWORD = "keyword"
    VECTOR = "vector"


class MemoryAtomRecallRoute(str, Enum):
    """一路召回。名字与两张挂载表对应。"""

    KEYWORD = "keyword"
    VECTOR = "vector"


class MemoryAtomSearchRequest(BaseModel):
    """一轮记忆检索。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description=(
            "自然语言查询，用用户的语言。\n\n"
            "它同时是关键词路要切词的文本、也是向量路要编码的文本 —— "
            "两侧用的分词器与模型与建索引时**必须同一套**（见 tokenization "
            "与 l1_vec 的说明），否则两路在回答不同的问题。"
        ),
    )

    scope: MemoryAtomSearchScope = Field(
        default=MemoryAtomSearchScope.SESSION,
        description="召回范围。默认只看当前会话。",
    )

    mode: MemoryAtomSearchMode = Field(
        default=MemoryAtomSearchMode.HYBRID,
        description="跑哪几路。这是**请求的**模式，实际哪几路跑了看 degraded_routes。",
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="最多返回几条。截断在**融合之后**，所以它限制的是响应条数，不是每路的召回深度。",
    )


class MemoryAtomSearchHit(BaseModel):
    """一条命中的记忆。"""

    record_id: UUID = Field(
        description=(
            "记忆身份。\n\n"
            "⚠️ 它只在**一次写入内**稳定：合并会保活下来的那条的 id、"
            "删掉被它吃掉的。拿它做长期引用要接受这一点。"
        ),
    )

    content: str = Field(
        description=(
            "记忆正文，**原样返回、不截断**。\n\n"
            "⚠️ 与 L0 返回片段（`SNIPPET_CHAR_LIMIT`）不同：那边是原始对话，"
            "长度是会话的性质；这边是已经提炼过的陈述，截断它可能把"
            "「用户喜欢 X，但**不能** Y」切掉后半句 —— 那比不给还糟。"
        ),
    )

    type: str = Field(
        default="",
        description="persona / episodic / instruction（词表会长，见 memory_atom 的说明）。",
    )

    priority: int = Field(
        default=50,
        description=(
            "0-100，-1 = 全局死命令。\n\n"
            "⚠️ 它**不参与排序也不参与过滤** —— 返回它是为了让调用方"
            "（尤其是模型）自己判断该重视哪条。排序由 score 决定。"
        ),
    )

    scene_name: str = Field(
        default="",
        description="这条记忆的情境名（LLM 生成的自由文本，没有枚举）。",
    )

    score: float = Field(
        description=(
            "融合分。**只在同一次响应内可比** —— 关键词路是 ts_rank、向量路"
            "是余弦相似度，融合按名次而非分数相加。跨响应或跨 mode 比它没有意义。"
        ),
    )

    matched_by: list[MemoryAtomRecallRoute] = Field(
        description=(
            "这条被哪几路召回到。只被向量路找到的那种最有信息量：它是"
            "关键词搜不到、换了个说法才捞回来的。"
        ),
    )

    created_time: datetime | None = Field(
        default=None,
        description=(
            "这条记忆第一次被写下来的时间（`memory_atom.created_time`）。\n\n"
            "⚠️ 字段名跟的是**源列**，与 L0 那条路的 `created_at` 不同名"
            " —— 两张表的列就叫这两个名字，中间不改写（见 "
            "domain/schemas/query.py 开头那段关于时间的说明）。"
        ),
    )

    updated_time: datetime | None = Field(
        default=None,
        description=(
            "最后一次写入 / 合并的时间（`memory_atom.updated_time`）。\n\n"
            "合并会改写它、`created_time` 不动 —— 两个一起给，读的人"
            "才分得出「早就有的事实」与「刚被合并改写过的事实」。"
        ),
    )


class DegradedAtomRoute(BaseModel):
    """
    一轮里没能参与的那一路。

    与 L0 的 `DegradedRoute` 同形（route + reason）：某一路挂掉不该让整轮
    失败，但**也不能静默跳过** —— 调用方要能分清“没召回到”和“那一路没跑”。
    """

    route: MemoryAtomRecallRoute
    reason: str


class MemoryAtomSearchResponse(BaseModel):
    """一轮记忆检索的结果。"""

    search_id: UUID = Field(description="本次调用，用于对齐日志与链路。")
    query: str = Field(description="原样回显，便于读结果时不必回滚上下文。")
    mode: MemoryAtomSearchMode
    scope: MemoryAtomSearchScope
    results: list[MemoryAtomSearchHit] = Field(
        description="按融合分降序。空数组且 degraded_routes 也为空 = 确实没有相关记忆（那不是错误）。",
    )
    degraded_routes: list[DegradedAtomRoute] = Field(default_factory=list)
