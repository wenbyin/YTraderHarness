"""
PostgreSQL implementation of MemoryIndexRepository.

Transaction rule
----------------
与另外几个仓储一致：本仓储**从不**调用 session.commit() / rollback()。
调用方决定事务边界（写入路径是 memory.l0.extract 任务的作用域，检索路径是
请求级作用域）。

⚠️ 两路搜索的排序表达式必须与索引的操作符**逐字对应**
-----------------------------------------------
    keyword   ORDER BY ts_rank(tsv, q)        ← 前置条件 `tsv @@ q` 让 GIN 生效
    vector    ORDER BY embedding <=> :vec     ← 让 HNSW 生效

向量那一路尤其容易踩：写成 `ORDER BY 1 - (embedding <=> :vec) DESC` 语义
相同，但优化器**不一定**能把它认回索引可用的形状 —— 认不出来的表现是
退化成顺序扫描：结果正确、只是慢。所以这里排序按原始距离升序，把
"越大越相关"的换算放到 Python 侧做。

⚠️ 过滤条件用的是 `user_id = 调用者 OR user_id IS NULL`
--------------------------------------------------
不是等值。NULL 是刻意的存量语义（`agent_session.user_id` 加那一列之前的
会话对谁都可见），memory 这边跟着 session 模块走 —— 否则同一个用户会看到
"消息列表里有、记忆检索搜不到"，而用户只会认为搜索坏了。
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import (
    ColumnElement,
    delete,
    func,
    or_,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..domain.entities import (
    FtsIndexRow,
    IndexedMessage,
    MemoryQueryScope,
    MemoryRecall,
    VecIndexRow,
)
from ..models.sqlalchemy.agent_message_fts import AgentMessageFts
from ..models.sqlalchemy.agent_message_vec import AgentMessageVec
from ._tsquery import (
    ts_match as _ts_match,
    ts_rank as _ts_rank,
    tsquery_expr as _tsquery_expr,
)
from .memory import MemoryIndexRepository

__all__ = ["MemoryIndexRepositoryImpl"]


#: tsvector 配置名 —— 与 ORM 上生成列的表达式、以及
#: `tokenization.tsquery_text` 的调用方**必须是同一个**。
#:
#: 写成常量而不是在三处各写一遍 'simple'：改配置时漏一处，症状是索引用
#: 一个配置建、查询用另一个，于是永远搜不到（**不报错**）。


class MemoryIndexRepositoryImpl(MemoryIndexRepository):

    # ========================================================
    # 写入
    # ========================================================

    async def upsert_fts(
            self,
            rows: Sequence[FtsIndexRow],
    ) -> int:

        if not rows:
            return 0

        values = [
            self._fts_values(row)
            for row in self._dedupe(rows, key=lambda r: r.message_id)
        ]

        statement = pg_insert(AgentMessageFts).values(values)

        # DO UPDATE 的 set_ 里**不含** tsv：它是生成列，写它会报
        # "cannot insert a non-DEFAULT value into column"。tokens 一变
        # tsv 自动重算 —— 那正是要的。
        statement = statement.on_conflict_do_update(
            index_elements=[AgentMessageFts.message_id],
            set_={
                column: statement.excluded[column]
                for column in _FTS_MUTABLE_COLUMNS
            },
        )

        await self._session.execute(statement)

        return len(values)

    async def upsert_vec(
            self,
            rows: Sequence[VecIndexRow],
    ) -> int:

        if not rows:
            return 0

        values = [
            self._vec_values(row)
            for row in self._dedupe(rows, key=lambda r: r.message_id)
        ]

        statement = pg_insert(AgentMessageVec).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[AgentMessageVec.message_id],
            set_={
                column: statement.excluded[column]
                for column in _VEC_MUTABLE_COLUMNS
            },
        )

        await self._session.execute(statement)

        return len(values)

    async def delete_by_run(
            self,
            run_id: UUID,
    ) -> int:
        """
        两张表一起删。

        ⚠️ 用 `run_id` 而不是 `session_id`：一次运行的粒度才是"重跑这一轮"
        的单位。按会话删会把同一会话里别的轮次也清掉，而重跑是从 agent_run
        出发的，没有"重跑某个会话"这个动作。
        """

        deleted = 0

        for model in (AgentMessageFts, AgentMessageVec):

            result = await self._session.execute(
                delete(model).where(model.run_id == run_id)
            )

            deleted += result.rowcount or 0

        return deleted

    # ========================================================
    # 召回
    # ========================================================

    async def search_keyword(
            self,
            *,
            tsquery: str,
            scope: MemoryQueryScope,
            recall_limit: int,
    ) -> list[MemoryRecall]:

        query = _tsquery_expr(tsquery)

        rank = _ts_rank(AgentMessageFts, query)

        statement = (
            select(
                AgentMessageFts.message_id,
                rank.label("score"),
            )
            .where(
                _ts_match(AgentMessageFts, query),
                *self._conditions(AgentMessageFts, scope),
            )
            .order_by(rank.desc())
            .limit(recall_limit)
        )

        rows = await self._session.execute(statement)

        return [
            MemoryRecall(message_id=row.message_id, score=float(row.score))
            for row in rows
        ]

    async def search_vector(
            self,
            *,
            embedding: Sequence[float],
            scope: MemoryQueryScope,
            recall_limit: int,
    ) -> list[MemoryRecall]:

        distance = AgentMessageVec.embedding.cosine_distance(
            list(embedding)
        )

        statement = (
            select(
                AgentMessageVec.message_id,
                distance.label("distance"),
            )
            .where(*self._conditions(AgentMessageVec, scope))
            .order_by(distance.asc())
            .limit(recall_limit)
        )

        rows = await self._session.execute(statement)

        return [
            MemoryRecall(
                message_id=row.message_id,
                # cosine 距离 -> 相似度。放在 Python 侧算，理由见模块
                # docstring：排序表达式必须与索引的操作符逐字一致。
                score=1.0 - float(row.distance),
            )
            for row in rows
        ]

    async def load(
            self,
            message_ids: Sequence[UUID],
    ) -> dict[UUID, IndexedMessage]:

        if not message_ids:
            # 空 IN () 走一次无意义的往返。空结果是空字典，与"都查不到"
            # 同一种返回，调用方不必分辨。
            return {}

        # 去重：调用方手里是"两路召回拼起来的 id 串"，同一条消息被两路
        # 都召回到是**常态**（hybrid 的意义就在这里）。不去重只是让 IN
        # 列表变长，但重复一多列表会明显变长 —— 而去重在这里是一行。
        unique_ids = list(dict.fromkeys(message_ids))

        rows = await self._session.scalars(
            select(AgentMessageFts).where(
                AgentMessageFts.message_id.in_(unique_ids)
            )
        )

        # 原文只存在 fts 表，理由见契约里 `load` 的说明。
        return {
            row.message_id: IndexedMessage.model_validate(row)
            for row in rows
        }

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _dedupe(
            rows: Sequence,
            *,
            key,
    ) -> list:
        """
        按主键去重，**保留最后一条**。

        ⚠️ 这一步不是优化，是防错
        ----------------------
        `INSERT ... ON CONFLICT DO UPDATE` 在一个 batch 里出现重复主键时
        Postgres 直接报错：

            ON CONFLICT DO UPDATE command cannot affect row a second time

        而"同一批里有重复"在真实路径上可能出现：任务按 run 重跑、或者
        调用方把两次查询的结果拼在一起。保留最后一条（而不是第一条）是
        因为后写的通常是最新的那次计算。
        """

        deduped: dict = {}

        for row in rows:
            deduped[key(row)] = row

        return list(deduped.values())

    @staticmethod
    def _fts_values(row: FtsIndexRow) -> dict:
        return {
            "message_id": row.message_id,
            "session_id": row.session_id,
            "run_id": row.run_id,
            "seq": row.seq,
            "role": row.role,
            "business_agent_id": row.business_agent_id,
            "user_id": row.user_id,
            "org_id": row.org_id,
            "created_at": row.created_at,
            "tokens": row.tokens,
            "original": row.original,
        }

    @staticmethod
    def _vec_values(row: VecIndexRow) -> dict:
        return {
            "message_id": row.message_id,
            "session_id": row.session_id,
            "run_id": row.run_id,
            "seq": row.seq,
            "role": row.role,
            "business_agent_id": row.business_agent_id,
            "user_id": row.user_id,
            "org_id": row.org_id,
            "created_at": row.created_at,
            "embedding": list(row.embedding),
        }

    @staticmethod
    def _conditions(
            model,
            scope: MemoryQueryScope,
    ) -> list[ColumnElement[bool]]:
        """
        scope -> WHERE 条件。两路搜索共用，这是**刻意的**：

        两路的过滤条件一旦分叉，向量路与关键词路就在回答不同的问题，
        而融合会把两个不同问题的答案按分数混在一起 —— 不报错，只是结果
        没法解释。所以这里只有一份。
        """

        conditions: list[ColumnElement[bool]] = []

        if scope.session_id is not None:
            conditions.append(model.session_id == scope.session_id)

        if scope.user_id is not None:
            # ⚠️ `OR IS NULL` 不是冗余兜底，见模块 docstring。
            conditions.append(
                or_(
                    model.user_id == scope.user_id,
                    model.user_id.is_(None),
                )
            )
        else:
            # ⚠️ 没有身份时**必须**加这一条，不能什么都不加
            # ------------------------------------------
            # "user_id 为 None 就不拼条件"看起来像"不过滤"，实际是
            # **搜出所有人**的会话 —— 一条静默的跨租户读。而这个函数是
            # 过滤条件的唯一产地，写错一次就没有第二处能拦住。
            #
            # 加上这一条之后语义与 `user_id is not None` 那一支的右半边
            # 一致：没有身份的人只看得到**无归属的存量行**（NULL 是存量
            # 语义，本来就对谁都可见），看不到任何人的数据。
            #
            # 服务层现在要求身份必填（见 _resolve_scope），所以这条正常
            # 路径上走不到 —— 它是**防御**，不是功能：将来多一处调用点、
            # 或者有人直接拿仓储去查，唯一可能的后果是"查不到"，
            # 不是"查到别人的"。
            conditions.append(model.user_id.is_(None))

        if scope.roles:
            conditions.append(model.role.in_(list(scope.roles)))

        if scope.time_from is not None:
            conditions.append(model.created_at >= scope.time_from)

        if scope.time_to is not None:
            conditions.append(model.created_at <= scope.time_to)

        return conditions

# ============================================================
# `tsv @@ q` / `ts_rank` / `to_tsquery` 三个表达式构造搬去了 _tsquery
# ============================================================
#
# ⚠️ 搬走的原因是 L1 的候选召回要用同一套（尤其那个 regconfig 必须与两张
# 表的生成列逐字一致）。下面按原名引进本模块，调用点因此一行没动。




#: upsert 时真正会被覆盖的列（不含主键与生成列）。
#:
#: 提成常量而不是在 set_ 里手写：漏一列的症状是"重跑索引之后某一列还是
#: 旧值"，而那一列可能是 user_id（越权的过滤键）—— 悄悄留旧值的后果比
#: 报错严重得多。
_FTS_MUTABLE_COLUMNS = (
    "session_id",
    "run_id",
    "seq",
    "role",
    "business_agent_id",
    "user_id",
    "org_id",
    "created_at",
    "tokens",
    "original",
)

_VEC_MUTABLE_COLUMNS = (
    "session_id",
    "run_id",
    "seq",
    "role",
    "business_agent_id",
    "user_id",
    "org_id",
    "created_at",
    "embedding",
)
