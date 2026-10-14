"""
PostgreSQL implementation of MemorySceneRepository.

不 commit（见契约的模块 docstring）：每个方法只发语句，事务边界归调用方
那一段 `scope`。因此"画像行 + 水位"能被调用方放进**同一个事务**里 ——
那是 `advance_l2_cursor` 那条约束成立的前提。

⚠️ 两条写路径的时间戳都是**显式写**的
-----------------------------
`onupdate=func.now()` 只在 ORM 的 UPDATE 上生效（那是 SQLAlchemy 在发
UPDATE 语句时替你带上 `updated_at = now()`，不是数据库触发器）。这里
两条写路径都是 core 语句，所以各自显式写：

    advance_l2_cursor   set_={"updated_at": statement.excluded.updated_at}
    update_scene        values(updated_at=func.now())

漏掉前者的症状是那一行的 `updated_at` 一直停在第一次插入的时刻 ——
而它是排查用的列，失真了没人会立刻发现。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import ColumnElement, and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..domain.entities.scene import (
    SceneCursorState,
    SceneScope,
    SceneRow,
)
from ..models.sqlalchemy.memory_scene_cursor import MemorySceneCursor
from ..models.sqlalchemy.memory_scene import MemoryScene as MemorySceneModel
from ._upsert import dedupe_by
from .scene import MemorySceneRepository

__all__ = ["MemorySceneRepositoryImpl"]


#: 插入时**不**带的列。
#:
#:     created_at / updated_at   交给服务端默认值（`now()`）。带上就得在
#:                               应用侧取一次时间，而那一份时间与数据库的
#:                               不是同一个 —— "这一行什么时候写的"会有
#:                               两个答案。
#:
#: 其余列（含 `version`）都带：`version` 带上是为了让调用方能显式指定起始
#: 版本（当前恒为 0，但显式比"靠默认值"少一处要查的东西）。
_INSERT_COLUMNS: tuple[str, ...] = (
    "id",
    "filename",
    "content",
    "content_md5",
    "agent_id",
    "user_id",
    "org_id",
    "version",
)


class MemorySceneRepositoryImpl(MemorySceneRepository):

    # ========================================================
    # 水位
    # ========================================================

    async def get_l2_cursor(
            self,
            session_id: UUID,
    ) -> SceneCursorState:

        row = await self._session.scalar(
            _fresh(select(MemorySceneCursor)).where(
                MemorySceneCursor.session_id == session_id
            )
        )

        # 没有行 = 还没整合过。折成默认值，而不是把 None 抛给调用方 ——
        # 见契约里 get_l2_cursor 的说明。
        if row is None:
            return SceneCursorState()

        return SceneCursorState(
            last_atom_updated_time=row.last_atom_updated_time,
            last_atom_ids=list(row.last_atom_ids or []),
            last_l2_at=row.last_l2_at,
        )

    async def advance_l2_cursor(
            self,
            session_id: UUID,
            *,
            last_atom_updated_time: datetime | None,
            last_atom_ids: Sequence[str],
            last_l2_at: datetime,
            user_id: str | None = None,
    ) -> None:

        statement = pg_insert(MemorySceneCursor).values(
            session_id=session_id,
            last_atom_updated_time=last_atom_updated_time,
            # 去重：调用方给的是"边界上那批 id"，模型/查询都可能给出重复项，
            # 而这一列只是集合语义 —— 存重复值会让它随轮次慢慢涨大。
            last_atom_ids=list(dict.fromkeys(last_atom_ids)),
            last_l2_at=last_l2_at,
            user_id=user_id,
        )

        statement = statement.on_conflict_do_update(
            index_elements=[MemorySceneCursor.session_id],
            set_={
                "last_atom_updated_time": (
                    statement.excluded.last_atom_updated_time
                ),
                "last_atom_ids": statement.excluded.last_atom_ids,
                "last_l2_at": statement.excluded.last_l2_at,
                "user_id": statement.excluded.user_id,
                # ⚠️ 显式写：`onupdate=func.now()` 只对 ORM 的 UPDATE 生效，
                # 对 core 的 ON CONFLICT DO UPDATE **不**生效。见模块 docstring。
                "updated_at": statement.excluded.updated_at,
            },
        )

        await self._session.execute(statement)

    # ========================================================
    # 场景文件
    # ========================================================

    async def list_scenes(
            self,
            scope: SceneScope,
    ) -> list[SceneRow]:

        rows = await self._session.scalars(
            _fresh(select(MemorySceneModel))
            .where(*self._scope_conditions(scope))
            # 升序是给提示词缓存用的（见契约）。`updated_at` 每次写入都会变，
            # 拿它排会让同一份清单每次读出来顺序不同。
            .order_by(MemorySceneModel.filename.asc())
        )

        return [SceneRow.model_validate(row) for row in rows]

    async def list_scene_delta(
            self,
            scope: SceneScope,
            *,
            updated_after: datetime | None,
            consumed_ids: Sequence[str],
            limit: int,
    ) -> list[SceneRow]:

        # ⚠️ `limit <= 0` 必须在这里挡住：Postgres 把**负的** LIMIT 当"不限量"，
        # 而调用方以为自己传了"不要任何东西"。与 list_session_delta 同一条。
        if limit <= 0:
            return []

        conditions: list[ColumnElement[bool]] = list(
            self._scope_conditions(scope)
        )

        if updated_after is not None:

            # ⚠️ 边界那一批要按 id 排除：一次 L2 整合写的所有场景共享一个
            # `now()`，水位推过去之后它们会被 `>` 漏掉（静默丢一批）。
            boundary = MemorySceneModel.updated_at == updated_after

            if consumed_ids:
                boundary = and_(
                    boundary,
                    MemorySceneModel.id.not_in(
                        list(dict.fromkeys(consumed_ids))
                    ),
                )

            conditions.append(
                or_(MemorySceneModel.updated_at > updated_after, boundary)
            )

        rows = await self._session.scalars(
            _fresh(select(MemorySceneModel))
            .where(*conditions)
            # 升序 —— 取**最早**的那一批。降序会让调用方把水位推到最后一条，
            # 中间那段永远不再出现（静默丢数据）。见契约。
            .order_by(
                MemorySceneModel.updated_at.asc(),
                # 同一时刻内的顺序也要稳定：否则"水位推到哪一条"取决于数据库
                # 当时怎么排的，两次调用可能给出不同的批次边界。
                MemorySceneModel.id.asc(),
            )
            .limit(limit)
        )

        return [SceneRow.model_validate(row) for row in rows]

    async def count_scenes(
            self,
            scope: SceneScope,
    ) -> int:

        total = await self._session.scalar(
            select(func.count())
            .select_from(MemorySceneModel)
            .where(*self._scope_conditions(scope))
        )

        return int(total or 0)

    async def insert_scenes(
            self,
            rows: Sequence[SceneRow],
    ) -> int:

        if not rows:
            return 0

        values = [
            _insert_values(row)
            for row in dedupe_by(rows, key=lambda row: row.id)
        ]

        statement = pg_insert(MemorySceneModel).values(values)

        # ⚠️ DO NOTHING 而不是 DO UPDATE：冲突意味着**另一个 L2 任务**刚为
        # 同一个 scope + 同一个文件名建了行（id 是内容派生的，所以撞上就是
        # 同一件事）。覆盖等于悄悄丢掉另一次整合的结论 —— 见契约里
        # insert_scenes 的说明：让调用方看到"少插了几行"，由它重来。
        statement = statement.on_conflict_do_nothing(
            index_elements=[MemorySceneModel.id],
        )

        result = await self._session.execute(statement)

        return result.rowcount or 0

    async def update_scene(
            self,
            scene_id: str,
            *,
            content: str,
            content_md5: str,
            expected_version: int,
    ) -> int:

        statement = (
            update(MemorySceneModel)
            .where(
                MemorySceneModel.id == scene_id,
                # ⚠️ 这一句就是乐观锁。少了它，两个 L2 任务会互相覆盖，
                # 而"覆盖"丢掉的是一次完整的模型调用结果。
                MemorySceneModel.version == expected_version,
            )
            .values(
                content=content,
                content_md5=content_md5,
                # version 由这里自增，不由调用方传 —— 见契约里的说明。
                version=MemorySceneModel.version + 1,
                updated_at=func.now(),
            )
        )

        result = await self._session.execute(statement)

        return result.rowcount or 0

    async def delete_scenes(
            self,
            scope: SceneScope,
            *,
            filenames: Sequence[str],
    ) -> int:

        if not filenames:
            return 0

        unique_names = list(dict.fromkeys(filenames))

        result = await self._session.execute(
            delete(MemorySceneModel).where(
                # ⚠️ 归属校验在**每一条** DELETE 上，不是装饰：文件名来自
                # 模型的输出（content 写成 [DELETED] 的那些），是不可信输入。
                # 见契约里 delete_scenes 的说明。
                *self._scope_conditions(scope),
                MemorySceneModel.filename.in_(unique_names),
            )
        )

        return result.rowcount or 0

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _scope_conditions(
            scope: SceneScope,
    ) -> list[ColumnElement[bool]]:
        """
        scope -> WHERE 条件。**四条语句共用**（读、插不走它、改按主键、
        删又用它），这是刻意的：口径一旦分叉，就会出现"读得出来、删不掉"
        或反过来的半截行为，而两边都不报错。

        ⚠️ 只匹配非空的那一侧
        -----------------
        `user_id` 与 `org_id` 恰有一个为 NULL（不变量由 CHECK 兜底，
        见 models/sqlalchemy/_scope.py），所以判断哪一侧有值就够了 ——
        再补一句 `另一侧 IS NULL` 是重复的，而重复的条件会让人以为
        它可以不成立。
        """

        conditions: list[ColumnElement[bool]] = [
            MemorySceneModel.agent_id == scope.agent_id,
        ]

        if scope.org_id is not None:
            conditions.append(MemorySceneModel.org_id == scope.org_id)
        else:
            conditions.append(MemorySceneModel.user_id == scope.user_id)

        return conditions


# ============================================================
# 实体 -> 列值
# ============================================================


def _insert_values(row: SceneRow) -> dict:
    """新行的列值。不含时间戳（交给服务端默认值），见 `_INSERT_COLUMNS`。"""

    return {
        column: getattr(row, column)
        for column in _INSERT_COLUMNS
    }


def _fresh(statement):
    """
    强制这次查询**从库里重新读**，即使同一个 session 的 identity map 里
    已经有这个对象。

    ⚠️ 不加它的话读到的是**过期值**，而且不报错
    ----------------------------------------
    SQLAlchemy 的 identity map 让"同一个主键在同一个 session 里只有一个
    对象"。默认行为是：再次 `select` 命中同一个主键时，**返回那个已经加载
    的实例、不覆盖它已有的属性**（要用 `.populate_existing()` 才会刷新）。

    本模块的两处读恰好都是"必须以库里的当前状态为准"的：

        get_l2_cursor   水位 —— 读到旧值 = 那批记忆被**重复**喂给模型
                        （多花钱、结论被覆盖），或更糟：水位看着没动，
                        而库里其实已经推过（取决于谁先读）
        list_scenes     写入的基线 —— 读到旧内容 = 乐观锁用的 version
                        偏旧，于是正常路径上凭空出现 ConflictError

    生产路径上每条任务各自开一个 scope（新 session），所以这个坑平时不
    显形；而**冒烟脚本会在同一个 session 里连着调几次 handler**（那正是
    它要验的"第二轮只喂增量"），于是第一次读到的东西会一直跟着它。
    这类"只在测试里出现"的差异最容易把人引到错误的方向上去 —— 与其让
    下一个读代码的人重新踩一遍，不如把意图写死在查询上。
    """

    return statement.execution_options(populate_existing=True)
