"""
Memory 索引行的实体。

⚠️ 为什么把 scope 提成一个基类
--------------------------
两张索引表的 scope 列完全相同（理由见 models/sqlalchemy/_scope.py），
读回来的行也需要同一组字段来组装命中。写成基类而不是嵌套字段，
是因为仓储与组装函数是**按列**用它们的（拼 WHERE、填 DTO），嵌一层
会让每一处都多一次 `.scope.`，而那一层不表达任何东西。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

__all__ = [
    "FtsIndexRow",
    "IndexedMessage",
    "MemoryQueryScope",
    "MemoryRecall",
    "MemoryScopeFields",
    "VecIndexRow",
]


class MemoryScopeFields(BaseModel):
    """
    两张索引表共用的那一组列，原样对应 `_scope.py` 里的 mixin。

    `role` 是**普通字符串**，不是枚举 —— 与 entities 层的 AgentMessage
    同一取舍：库里那一列是带 CHECK 的文本，实体只是把它搬过来。判取值域
    的地方在 DB 约束与 DTO 的注释上，中间这一层再拦一次只会多一处会漂的
    定义。

    ⚠️ 没有 kind / status：索引表里已经没有这两列了。要读就回
    `agent_message` 读（见 models/sqlalchemy/_scope.py 的说明）。
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    run_id: UUID | None = None
    seq: int
    role: str
    business_agent_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    created_at: datetime


class FtsIndexRow(MemoryScopeFields):
    """写进 agent_message_fts 的一行。"""

    message_id: UUID
    tokens: str
    original: str


class VecIndexRow(MemoryScopeFields):
    """写进 agent_message_vec 的一行。"""

    message_id: UUID
    embedding: list[float]


class IndexedMessage(MemoryScopeFields):
    """
    从索引表**读回来**的一行 —— 组装命中用的形状。

    带 `original`（原文）而不是 tokens：展示与片段都从原文出，检索用的
    分词串在读完就不再需要了。
    """

    message_id: UUID
    original: str


class MemoryRecall(BaseModel):
    """
    单路召回的一条结果 —— **只有 id 与分数**。

    正文不在这里：两路各自召回、融合、截断之后才去取原文（`load`）。
    先取正文会让"召回 200 条、最后只要 10 条"白读 190 条正文。

    ⚠️ score 是"这一路的原始分"，不是融合分
    -----------------------------------
    关键词路是 ts_rank、向量路是余弦距离（已转成"越大越相关"）。
    两者**不同量纲**，所以融合必须按**排名**做，不能拿这两个数相加 ——
    相加的结果会被量纲大的那一侧主宰，而那是哪一侧取决于实现细节。
    """

    message_id: UUID
    score: float


class MemoryQueryScope(BaseModel):
    """
    已解析、且**身份已经钉死**的过滤条件。

    它是服务层算出来的东西，仓储只按它拼 WHERE —— 仓储因此不需要认识
    `MemorySearchScope` 那个枚举，也不需要知道"当前用户是谁"是从哪来的。
    这条分工让"越权"变成一个在服务层就能看全的问题：仓储只能过滤
    `user_id`，它没有别的入口。

    ⚠️ user_id 与 session_id 的关系
    -----------------------------
        session_id 有值    限定单个会话（scope=session）。此时 user_id
                         **仍然生效**：归属校验不是"信 session_id"，
                         而是两个条件都要满足
        session_id None    scope=user，按 user_id 铺开

    两个条件的写法见仓储实现 —— `user_id = 调用者 OR user_id IS NULL`
    是**刻意**的（NULL 是存量语义），不是漏写等值。
    """

    session_id: UUID | None = None
    user_id: str | None = None
    roles: list[str] | None = None
    time_from: datetime | None = None
    time_to: datetime | None = None
