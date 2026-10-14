"""
PostgreSQL implementation of MemoryAtomRepository.

不 commit（见契约的模块 docstring）：每个方法只发语句，事务边界归调用方
那一段 `scope`。因此"记录 + 两张挂载表 + 水位"能被调用方放进**同一个
事务**里 —— 那是 `advance_cursor` 那条约束成立的前提。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, and_, delete, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..domain.entities.atom import (
    AtomFtsRow,
    AtomRecall,
    AtomVecRow,
    MemoryAtom,
    MemoryAtomQueryScope,
    AtomCursorState,
)
from ..models.sqlalchemy.memory_atom import MemoryAtom as MemoryAtomModel
from ..models.sqlalchemy.memory_atom_vec import MemoryAtomVec
from ..models.sqlalchemy.memory_atom_cursor import MemoryAtomCursor
from ..models.sqlalchemy.memory_atom_fts import MemoryAtomFts
from ._tsquery import ts_match, ts_rank, tsquery_expr
from ._upsert import dedupe_by
from .atom import MemoryAtomRepository

__all__ = ["MemoryAtomRepositoryImpl"]


#: 冲突时要覆盖的列。
#:
#: ⚠️ 两个**不在**里面，都不是疏忽：
#:
#:     record_id     它就是要冲突的那一列（主键）
#:     created_time  "这一行第一次写入的时间"，原地改写不该动它
#:
#: 其余全列覆盖：一次合并可能改内容、类型、优先级、情境、版本、时间线、
#: 溯源、meta —— 逐列挑"哪些会变"等于给未来留一个"某列悄悄没更新"的坑。
_ATOM_MUTABLE_COLUMNS: tuple[str, ...] = (
    "content",
    "type",
    "priority",
    "scene_name",
    "session_key",
    "version",
    "timestamp_str",
    "timestamp_start",
    "timestamp_end",
    "source_message_ids",
    "meta",
    "session_id",
    "business_agent_id",
    "user_id",
    "org_id",
    "task_id",
    "updated_time",
)

#: 索引行的可变列（除了主键 `record_id` 全是）。
_FTS_MUTABLE_COLUMNS: tuple[str, ...] = (
    "content_tokens",
    "content_original",
    "session_id",
    "business_agent_id",
    "user_id",
    "org_id",
)

#: 同上。⚠️ **不含** `tsv` —— 它是生成列，写它会报
#: "cannot insert a non-DEFAULT value into column"；content_tokens 一变
#: 它自动重算，那正是要的。
_VEC_MUTABLE_COLUMNS: tuple[str, ...] = (
    "embedding",
    "session_id",
    "business_agent_id",
    "user_id",
    "org_id",
)


class MemoryAtomRepositoryImpl(MemoryAtomRepository):

    # ========================================================
    # 水位
    # ========================================================

    async def get_cursor(
            self,
            session_id: UUID,
    ) -> AtomCursorState:

        row = await self._session.scalar(
            select(MemoryAtomCursor).where(
                MemoryAtomCursor.session_id == session_id
            )
        )

        # 没有行 = 还没抽过。折成默认值，而不是把 None 抛给调用方 ——
        # 见契约里 get_cursor 的说明。
        if row is None:
            return AtomCursorState()

        return AtomCursorState(
            last_seq=row.last_seq,
            last_scene_name=row.last_scene_name,
        )

    async def advance_cursor(
            self,
            session_id: UUID,
            *,
            last_seq: int,
            last_scene_name: str,
            user_id: str | None = None,
    ) -> None:

        statement = pg_insert(MemoryAtomCursor).values(
            session_id=session_id,
            last_seq=last_seq,
            last_scene_name=last_scene_name,
            user_id=user_id,
        )

        statement = statement.on_conflict_do_update(
            index_elements=[MemoryAtomCursor.session_id],
            set_={
                "last_seq": statement.excluded.last_seq,
                "last_scene_name": statement.excluded.last_scene_name,
                "user_id": statement.excluded.user_id,
                # ⚠️ 显式写：`onupdate=func.now()` 只对 ORM 的 UPDATE 生效，
                # 对 core 的 ON CONFLICT DO UPDATE **不**生效。漏了它，
                # updated_at 会一直停在第一次插入的时刻。
                "updated_at": statement.excluded.updated_at,
            },
        )

        await self._session.execute(statement)

    # ========================================================
    # 本体与挂载表
    # ========================================================

    async def upsert_atoms(
            self,
            atoms: Sequence[MemoryAtom],
    ) -> int:

        if not atoms:
            return 0

        values = [_atom_values(atom) for atom in dedupe_by(
            atoms, key=lambda atom: atom.record_id
        )]

        statement = pg_insert(MemoryAtomModel).values(values)

        statement = statement.on_conflict_do_update(
            index_elements=[MemoryAtomModel.record_id],
            set_={
                column: statement.excluded[column]
                for column in _ATOM_MUTABLE_COLUMNS
            },
        )

        await self._session.execute(statement)

        return len(values)

    async def upsert_fts(
            self,
            rows: Sequence[AtomFtsRow],
    ) -> int:

        if not rows:
            return 0

        values = [_fts_values(row) for row in dedupe_by(
            rows, key=lambda row: row.record_id
        )]

        statement = pg_insert(MemoryAtomFts).values(values)

        statement = statement.on_conflict_do_update(
            index_elements=[MemoryAtomFts.record_id],
            set_={
                column: statement.excluded[column]
                for column in _FTS_MUTABLE_COLUMNS
            },
        )

        await self._session.execute(statement)

        return len(values)

    async def upsert_vec(
            self,
            rows: Sequence[AtomVecRow],
    ) -> int:

        if not rows:
            return 0

        values = [_vec_values(row) for row in dedupe_by(
            rows, key=lambda row: row.record_id
        )]

        statement = pg_insert(MemoryAtomVec).values(values)

        statement = statement.on_conflict_do_update(
            index_elements=[MemoryAtomVec.record_id],
            set_={
                column: statement.excluded[column]
                for column in _VEC_MUTABLE_COLUMNS
            },
        )

        await self._session.execute(statement)

        return len(values)

    # ========================================================
    # 召回与删除
    # ========================================================

    async def recall_keyword(
            self,
            scope: MemoryAtomQueryScope,
            *,
            tsquery: str,
            limit: int,
    ) -> list[AtomRecall]:

        query = tsquery_expr(tsquery)

        rank = ts_rank(MemoryAtomFts, query)

        statement = (
            select(
                MemoryAtomFts.record_id,
                rank.label("score"),
            )
            .where(
                ts_match(MemoryAtomFts, query),
                *self._conditions(MemoryAtomFts, scope),
            )
            .order_by(rank.desc())
            .limit(limit)
        )

        rows = await self._session.execute(statement)

        return [
            AtomRecall(record_id=row.record_id, score=float(row.score))
            for row in rows
        ]

    async def recall_vector(
            self,
            scope: MemoryAtomQueryScope,
            *,
            embedding: Sequence[float],
            limit: int,
    ) -> list[AtomRecall]:

        distance = MemoryAtomVec.embedding.cosine_distance(
            list(embedding)
        )

        statement = (
            select(
                MemoryAtomVec.record_id,
                distance.label("distance"),
            )
            .where(*self._conditions(MemoryAtomVec, scope))
            .order_by(distance.asc())
            .limit(limit)
        )

        rows = await self._session.execute(statement)

        return [
            AtomRecall(
                record_id=row.record_id,
                # cosine 距离 -> 相似度。放在 Python 侧算：排序表达式必须
                # 与索引的操作符逐字一致（见契约里 recall_vector 的说明）。
                score=1.0 - float(row.distance),
            )
            for row in rows
        ]

    async def load(
            self,
            record_ids: Sequence[UUID],
    ) -> dict[UUID, MemoryAtom]:

        if not record_ids:
            return {}

        # 去重：两路召回拼起来的 id 串里，同一条被两路都召回到是常态。
        unique_ids = list(dict.fromkeys(record_ids))

        rows = await self._session.scalars(
            select(MemoryAtomModel).where(
                MemoryAtomModel.record_id.in_(unique_ids)
            )
        )

        return {
            row.record_id: MemoryAtom.model_validate(row)
            for row in rows
        }

    async def delete_batch(
            self,
            record_ids: Sequence[UUID],
            *,
            session_id: UUID,
    ) -> int:

        if not record_ids:
            return 0

        unique_ids = list(dict.fromkeys(record_ids))

        # 三张表一起删，同一个事务（调用方的作用域）。顺序按"派生 -> 权威"：
        # 没有外键，顺序不影响正确性，但这样读起来与数据层级一致。
        for model in (MemoryAtomFts, MemoryAtomVec):

            await self._session.execute(
                delete(model).where(
                    model.record_id.in_(unique_ids),
                    model.session_id == session_id,
                )
            )

        result = await self._session.execute(
            delete(MemoryAtomModel).where(
                MemoryAtomModel.record_id.in_(unique_ids),
                # ⚠️ 归属校验在**每一条** DELETE 上，不是装饰 —— 见契约里
                # delete_batch 的说明（target_ids 来自模型输出，不可信），
                # 以及"粒度必须与候选池一致"那一条。
                MemoryAtomModel.session_id == session_id,
            )
        )

        return result.rowcount or 0

    async def list_session_delta(
            self,
            session_id: UUID,
            *,
            updated_after: datetime | None,
            consumed_ids: Sequence[UUID],
            limit: int,
    ) -> list[MemoryAtom]:

        # ⚠️ `limit <= 0` 必须在这里挡住
        # --------------------------
        # Postgres 把**负的** LIMIT 当"不限量"：`limit(-1)` 会返回整个会话
        # 的全部记忆，而调用方以为自己传了"不要任何东西"。返回空列表是
        # 唯一说得清的语义。
        if limit <= 0:
            return []

        conditions: list[ColumnElement[bool]] = [
            MemoryAtomModel.session_id == session_id,
        ]

        if updated_after is not None:

            # ⚠️ 边界那一批要按 id 排除，见契约里的说明：同一事务写入的
            # atom 共享一个 `now()`，水位推过去之后它们会被 `>` 漏掉。
            boundary = MemoryAtomModel.updated_time == updated_after

            if consumed_ids:
                boundary = and_(
                    boundary,
                    MemoryAtomModel.record_id.not_in(
                        list(dict.fromkeys(consumed_ids))
                    ),
                )

            conditions.append(
                or_(MemoryAtomModel.updated_time > updated_after, boundary)
            )

        rows = await self._session.scalars(
            select(MemoryAtomModel)
            .where(*conditions)
            # 升序 —— 取**最早**的那一批。降序会让调用方把水位推到最后一条，
            # 中间那段永远不再出现（静默丢数据）。见契约。
            .order_by(
                MemoryAtomModel.updated_time.asc(),
                # 同一时刻内的顺序也要稳定：否则"水位推到哪一条"取决于
                # 数据库当时怎么排的，两次调用可能给出不同的批次边界。
                MemoryAtomModel.record_id.asc(),
            )
            .limit(limit)
        )

        return [MemoryAtom.model_validate(row) for row in rows]

    async def list_by_source_message(
            self,
            message_id: UUID,
            *,
            session_id: UUID,
            limit: int,
    ) -> list[MemoryAtom]:

        if limit <= 0:
            return []

        rows = await self._session.scalars(
            select(MemoryAtomModel)
            .where(
                # session_id 在前：它走 idx_memory_atom_session，把下面的
                # JSONB 包含判断限制在这一个会话的记忆上（见契约里的说明
                # —— 这一列没有 GIN 索引）。
                MemoryAtomModel.session_id == session_id,
                # ⚠️ 比的是**字符串**：列里存的是 message_id 的字符串形式
                # （见 domain/schemas/atom.py 那个字段的说明）。传 UUID
                # 会一条都匹配不上，而且不报错。
                MemoryAtomModel.source_message_ids.contains([str(message_id)]),
            )
            # 降序：这是"看当下"，最近被改写/合并过的排前面 —— 与
            # list_session_delta 的升序（增量必须一条不丢）是两个目的。
            .order_by(
                MemoryAtomModel.updated_time.desc(),
                MemoryAtomModel.record_id.desc(),
            )
            .limit(limit)
        )

        return [MemoryAtom.model_validate(row) for row in rows]

    async def list_page(
            self,
            *,
            user_id: str,
            record_type: str | None,
            priority: int | None,
            scene_name: str | None,
            updated_from: datetime | None,
            updated_to: datetime | None,
            text_contains: str | None,
            order: str,
            limit: int,
            after: Sequence[Any] | None,
    ) -> tuple[list[MemoryAtom], bool]:

        if limit <= 0:
            return [], False

        conditions: list[ColumnElement[bool]] = [
            # 可见性口径与 `_conditions` 逐字相同（NULL 是存量语义）。
            or_(
                MemoryAtomModel.user_id == user_id,
                MemoryAtomModel.user_id.is_(None),
            ),
        ]

        if record_type is not None:
            conditions.append(MemoryAtomModel.type == record_type)

        if priority is not None:
            conditions.append(MemoryAtomModel.priority == priority)

        if scene_name is not None:
            conditions.append(MemoryAtomModel.scene_name == scene_name)

        # 范围两端都是**含**（契约里写的 inclusive）—— `>=` / `<=` 与
        # `between` 等价，但分成两条更容易与"只给了下界"组合。
        if updated_from is not None:
            conditions.append(MemoryAtomModel.updated_time >= updated_from)

        if updated_to is not None:
            conditions.append(MemoryAtomModel.updated_time <= updated_to)

        if text_contains:
            conditions.append(
                MemoryAtomModel.content.ilike(
                    self._like_pattern(text_contains),
                    escape="\\",
                )
            )

        # ---- 排序与 keyset ------------------------------------------
        #
        # 每个 order 的 keyset 条件都必须**同时**用上唯一键（record_id）：
        # `updated_time` 会被合并原地改写、`priority` 大量重复 —— 只按第一
        # 列比较，边界上的一批会在下一页里再出现一次（或者被跳过），两个
        # 方向都不报错。
        if order in ("updated_at_desc", "updated_at_asc"):
            sort_col = MemoryAtomModel.updated_time
        else:
            sort_col = MemoryAtomModel.priority

        descending = order.endswith("_desc")

        if after is not None:
            conditions.append(
                self._keyset(
                    sort_col=sort_col,
                    tie_col=MemoryAtomModel.record_id,
                    key=after[0],
                    tie=after[1],
                    descending=descending,
                )
            )

        if descending:
            order_by = (sort_col.desc(), MemoryAtomModel.record_id.desc())
        else:
            order_by = (sort_col.asc(), MemoryAtomModel.record_id.asc())

        # 多取一条判"还有没有下一页"：取回 limit 条的话，"刚好取满"与
        # "还有更多"长得一样，客户端会多翻一次空页。
        rows = await self._session.scalars(
            select(MemoryAtomModel)
            .where(*conditions)
            .order_by(*order_by)
            .limit(limit + 1)
        )

        models = list(rows)

        has_more = len(models) > limit

        return (
            [MemoryAtom.model_validate(row) for row in models[:limit]],
            has_more,
        )

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _keyset(
            *,
            sort_col: Any,
            tie_col: Any,
            key: Any,
            tie: Any,
            descending: bool,
    ) -> ColumnElement[bool]:
        """
        `(排序键, 唯一键) > (上一页最后一条)` —— 方向按 `descending` 翻转。

        写成通用的一段而不是四份字面条件：四个 order 的差别只有"哪一列"
        与"哪个方向"，抄四遍的话总有一份把 `<` 写成 `>`，而它的症状是
        "翻页翻出一个环"（或者提前结束），不报错。
        """

        if descending:
            return or_(
                sort_col < key,
                and_(sort_col == key, tie_col < tie),
            )

        return or_(
            sort_col > key,
            and_(sort_col == key, tie_col > tie),
        )

    @staticmethod
    def _like_pattern(value: str) -> str:
        """转义 LIKE 通配符后构造包含匹配 pattern（与其它模块同一份实现）。"""

        escaped = (
            value
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        return f"%{escaped}%"

    @staticmethod
    def _conditions(
            model,
            scope: MemoryAtomQueryScope,
    ) -> list[ColumnElement[bool]]:
        """
        scope -> WHERE 条件。**两路召回共用**，这是刻意的：条件一旦分叉，
        向量路与关键词路就在回答不同的问题，而融合会把两个不同问题的答案
        按名次混在一起 —— 不报错，只是结果没法解释。

        与 L0 的 `_conditions` 同一套口径（含 `user_id` 那条 `OR IS NULL`
        与"没有身份时只看无归属行"的防御），少一个时间窗 —— 理由见
        `MemoryAtomQueryScope` 的 docstring。
        """

        conditions: list[ColumnElement[bool]] = []

        if scope.session_id is not None:
            conditions.append(model.session_id == scope.session_id)

        if scope.user_id is not None:
            # NULL 是存量语义（加那一列之前的行对谁都可见），
            # 规则与 agent_session 逐字一致。
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
            # **搜出所有人**的记忆 —— 一条静默的跨租户读。服务层要求身份
            # 必填，所以正常路径走不到；这是防御：漏判的后果是"查不到"，
            # 而不是"查到别人的"。
            conditions.append(model.user_id.is_(None))

        return conditions


# ============================================================
# 实体 -> 列值
# ============================================================


def _atom_values(atom: MemoryAtom) -> dict:
    return {
        "record_id": atom.record_id,
        "content": atom.content,
        "type": atom.type,
        "priority": atom.priority,
        "scene_name": atom.scene_name,
        "session_key": atom.session_key,
        "version": atom.version,
        "timestamp_str": atom.timestamp_str,
        "timestamp_start": atom.timestamp_start,
        "timestamp_end": atom.timestamp_end,
        "source_message_ids": list(atom.source_message_ids),
        "meta": dict(atom.meta),
        "session_id": atom.session_id,
        "business_agent_id": atom.business_agent_id,
        "user_id": atom.user_id,
        "org_id": atom.org_id,
        "task_id": atom.task_id,
    }


def _fts_values(row: AtomFtsRow) -> dict:
    return {
        "record_id": row.record_id,
        "content_tokens": row.tokens,
        "content_original": row.original,
        "session_id": row.session_id,
        "business_agent_id": row.business_agent_id,
        "user_id": row.user_id,
        "org_id": row.org_id,
    }


def _vec_values(row: AtomVecRow) -> dict:
    return {
        "record_id": row.record_id,
        "embedding": row.embedding,
        "session_id": row.session_id,
        "business_agent_id": row.business_agent_id,
        "user_id": row.user_id,
        "org_id": row.org_id,
    }
