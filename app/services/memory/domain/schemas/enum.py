"""
Memory 检索的取值域。

对齐 memory.yaml 的 components/schemas 里那三个内联枚举。

⚠️ 为什么是 `str, Enum` 而不是 `Literal`
------------------------------------
与仓库其他地方一致（见 session/domain/schemas/enum.py 与
tool/domain/schemas/enum.py 的同一段说明）：值要进 query、要落日志、
要与 yaml 对齐，字符串最直白；Literal 只活在类型检查器里，运行时它
什么都不拦。

⚠️ role / kind **不在这里重新定义**
--------------------------------
`MemorySearchHit` 的 role / kind 就是 `agent_message` 的那两个取值域，
而它们已经在 `app/services/session/domain/schemas/enum.py` 里定义过
（`MessageRole` / `MessageKind`）。这里再抄一份的代价不是多几行：
两份定义会漂，而漂的方式是**静默**的 —— 一边加了新 kind（比如将来的
`summary`），另一边漏加，于是 memory 检索悄悄过滤掉那一类，没有任何
报错。所以 DTO 直接引用 session 那两个枚举。

跨模块 import 一个纯枚举（不碰库、不碰服务）是安全的：它不引入运行时
耦合，只有一处真相。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "MemoryRecallRoute",
    "MemorySearchMode",
    "MemorySearchScope",
]


class MemorySearchScope(str, Enum):
    """
    召回范围能铺多宽。

        SESSION   只搜当前会话（L0 自己的作用域）
        USER      搜该用户的**所有**会话 —— 这是 L1 事实住的那一层，
                  放在这里是为了让一个会话也能找到另一个会话里说过的话

    ⚠️ USER 是**真的放宽**，不等于"搜全部"
    ----------------------------------
    agent / team 从不被它放宽：两者始终钉死在调用者身上。也就是说
    scope=user 放开的是**会话维度**，不是用户维度 —— 否则它就是一条
    越权读别人数据的路径。

    工具侧能选这一档，但 user_id 本身由运行时注入（见 memory.yaml 的
    info.description）：模型能决定"要不要跨会话"，不能决定"跨谁的"。
    """

    SESSION = "session"
    USER = "user"


class MemorySearchMode(str, Enum):
    """
    跑哪几路召回。

        HYBRID   两路都跑，按排名融合
        KEYWORD  只跑关键词（tsvector + GIN）
        VECTOR   只跑向量（pgvector + HNSW）

    这是**请求**的模式，响应里原样回显。某一路实际没跑成时，它出现在
    `degraded_routes` 里，而不是被悄悄并进这个字段 —— "请求了混合但
    只有关键词在答"必须看得见。
    """

    HYBRID = "hybrid"
    KEYWORD = "keyword"
    VECTOR = "vector"


class MemoryRecallRoute(str, Enum):
    """
    单路召回。取值与两张索引表一一对应。

    `MemorySearchHit.matched_by` 与 `DegradedRoute.route` 都用它 ——
    "这条是向量路找到的"与"向量路这轮没跑"说的是同一套名字，所以是
    同一个枚举，不是两个同值枚举。
    """

    KEYWORD = "keyword"
    VECTOR = "vector"
