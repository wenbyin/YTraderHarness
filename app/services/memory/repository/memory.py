"""
Repository contract for the two memory index tables.

只管 `agent_message_fts` 与 `agent_message_vec` 两张表 —— 它们是纯派生的
（从 `agent_message` 重建即可），所以这个契约里**没有**"业务不变式"那一类
东西，只有写入、按 run 清理、两路召回。

⚠️ 为什么一个契约管两张表
--------------------
按表拆开（FtsRepository / VecRepository）看起来更整齐，但真实调用方
**总是同时**需要两个：写入两路一起写（同一次任务），检索两路一起跑
（同一次请求）。拆开只会让每个调用点都要注入两个、并且需要自己保证
"两路的过滤条件一致" —— 而过滤条件不一致的症状是两路召回回答不同的
问题，且不报错。

⚠️ 契约不认识分词器与枚举
----------------------
过滤条件进来的是 `MemoryQueryScope`（已解析、身份已钉死），关键词查询
进来的是**已经切好的 tsquery 字符串**。理由：分词器属于应用侧
（`memory/tokenization.py`），"没有可用词元时这一路降级"这个决定属于
服务层（它要写 `degraded_routes`）。仓储只做它唯一该做的事：按条件
查两张表。

这条分工也让越权在服务层就能看全：仓储只能按 `user_id` 过滤，
它没有别的入口。
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import (
    FtsIndexRow,
    IndexedMessage,
    MemoryQueryScope,
    MemoryRecall,
    VecIndexRow,
)

__all__ = ["MemoryIndexRepository"]


class MemoryIndexRepository(BaseRepository):

    # ========================================================
    # 写入
    # ========================================================

    @abstractmethod
    async def upsert_fts(
            self,
            rows: Sequence[FtsIndexRow],
    ) -> int:
        """
        写入/覆盖若干关键词索引行，返回写入条数。

        ⚠️ 必须是 **upsert**，不是 insert
        -------------------------------
        主键是 `message_id`，而任务会被重复执行（至少一次投递意味着重复
        是常态，租约过期被回收也会导致重跑）。用 insert 的话第二次就撞
        主键报错 —— 而那是**预期内的重复**，不该是一次失败。

        空序列是合法的 no-op（本轮一条消息都不该进索引时就是这种），
        调用方不必自己判空。
        """

        raise NotImplementedError

    @abstractmethod
    async def upsert_vec(
            self,
            rows: Sequence[VecIndexRow],
    ) -> int:
        """
        写入/覆盖若干向量索引行，返回写入条数。语义与 `upsert_fts` 相同。

        ⚠️ 与 fts 是**两次独立调用**，不在一个方法里
        ---------------------------------------
        两路的输入不同（一个要分词、一个要过模型），算出来的时机也不同：
        embedding 模型没加载起来时关键词路照样能写。合并成一个方法会让
        "向量那半失败"变成"两半都不写"。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_by_run(
            self,
            run_id: UUID,
    ) -> int:
        """
        删掉某次运行的全部索引行，返回删除条数。

        给"重跑一次运行的索引"用：先删后写，或者直接 upsert（消息不可变，
        重跑的结果相同）。留着它是因为**改了分词器/换了模型**时要先把
        旧行清掉 —— 那种情况下重写不一定会覆盖到同一条消息（过滤规则可能
        变了），残留的旧行会以错误的切法继续被搜到。
        """

        raise NotImplementedError

    # ========================================================
    # 召回
    # ========================================================

    @abstractmethod
    async def search_keyword(
            self,
            *,
            tsquery: str,
            scope: MemoryQueryScope,
            recall_limit: int,
    ) -> list[MemoryRecall]:
        """
        关键词路召回，按 `ts_rank` 降序。

        `tsquery` 是应用侧切好并 OR 连接好的串（见
        tokenization.tsquery_text）。它**不能为空** —— 空串会让
        `to_tsquery` 抛语法错误，而"查询里没有可用词元"是输入问题，
        由调用方在进来之前判掉（判掉了就把它记进 degraded_routes）。

        ⚠️ recall_limit 不是响应里的 limit
        ------------------------------
        融合需要比最终条数**更多的候选**：只看每路前 10 条来融合，等于
        假定两路的前 10 条高度重叠。所以这一层的深度由服务层单独给
        （通常是最终 limit 的若干倍），截断在融合之后做。
        """

        raise NotImplementedError

    @abstractmethod
    async def search_vector(
            self,
            *,
            embedding: Sequence[float],
            scope: MemoryQueryScope,
            recall_limit: int,
    ) -> list[MemoryRecall]:
        """
        向量路召回，按余弦距离升序（转成"越大越相关"后返回），
        走 `idx_agent_message_vec_embedding_hnsw`。

        维度必须与建索引时一致（512）。不一致时 Postgres 会直接报错
        （不是静默返回空），那是**要好于**静默的失败方式。

        ⚠️ HNSW 是近似索引
        ---------------
        它保证"接近正确"而不是"一定正确"。过滤条件（scope）把候选集收得
        很窄时，近似性会让召回更容易漏 —— 这是 recall_limit 要给得比最终
        条数大的另一个理由。
        """

        raise NotImplementedError

    @abstractmethod
    async def load(
            self,
            message_ids: Sequence[UUID],
    ) -> dict[UUID, IndexedMessage]:
        """
        按 id 取回索引行（scope 列 + 原文），给组装命中用。

        返回**字典**而不是列表：调用方手里是一串"按分数排好的 id"，要按
        那个顺序拼结果，因此需要能按 id 查。返回列表会逼调用方自己建索引，
        而且还要处理"某些 id 没查到"（两路都召回、但另一路的那一行被删了）
        —— 那种情况下**丢掉那一条命中**是对的，字典天然表达这一点。

        ⚠️ 从 fts 表读，不是 vec 表
        ------------------------
        原文只存在 fts 表（vec 表没有正文列）。因此向量路**独自**能找到
        而 fts 表里没有的行，会在这里被丢掉。这不是 bug 而是刻意的取舍：
        两路写的是同一批消息，只有关键字表被单独清理过的场合才会出现，
        而那时展示不出原文的命中本来也无法呈现。
        """

        raise NotImplementedError
