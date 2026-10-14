"""
PostgreSQL implementation of MemoryPersonaRepository.

不 commit（见契约的模块 docstring）：每个方法只发语句，事务边界归调用方
那一段 `scope`。因此"画像行 + 水位"能被调用方放进**同一个事务**里 —— 那是
`advance_l3_cursor` 那条约束成立的前提。

⚠️ 两条写路径的时间戳都是**显式写**的
-----------------------------
与 scene_impl 同一条成因：`onupdate=func.now()` 只在 ORM 的 UPDATE 上生效
（那是 SQLAlchemy 替你带上 `updated_at = now()`，不是数据库触发器）。这里
两条写路径都是 core 语句，所以各自显式写：

    advance_l3_cursor   set_={"updated_at": statement.excluded.updated_at}
    update_persona      values(updated_at=func.now())

⚠️ 水位的键是**复合的**（agent_id + scope_key）
----------------------------------------
不能写成 `(agent_id, user_id, org_id)`：scope 那两列恰有一列恒为 NULL，
而 `ON CONFLICT (…)` 与主键都要求非空（理由见 memory_persona_cursor.py
的模块 docstring）。所以冲突判定走 `[agent_id, scope_key]`。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..domain.entities.persona import (
    PersonaCursorState,
    PersonaRow,
)
from ..domain.entities.scene import SceneScope
from ..models.sqlalchemy.memory_persona import MemoryPersona as MemoryPersonaModel
from ..models.sqlalchemy.memory_persona_cursor import (
    MemoryPersonaCursor as MemoryPersonaCursorModel,
)
from .persona import MemoryPersonaRepository

__all__ = ["MemoryPersonaRepositoryImpl"]


#: 插入时**不**带的列。
#:
#:     created_at / updated_at   交给服务端默认值（`now()`）。带上就得在
#:                               应用侧取一次时间，而那一份时间与数据库的
#:                               不是同一个 —— "这一行什么时候写的"会有
#:                               两个答案。
#:
#: 其余列（含 `version`）都带：`version` 带上是为了让调用方能显式指定起始
#: 版本（当前恒为 0，但显式比"靠默认值"少一处要查的东西）。与 scene_impl
#: 的 `_INSERT_COLUMNS` 逐字同形（两张表的列集一样）。
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


class MemoryPersonaRepositoryImpl(MemoryPersonaRepository):

    # ========================================================
    # 画像本体
    # ========================================================

    async def get_persona(
            self,
            scope: SceneScope,
    ) -> PersonaRow | None:

        row = await self._session.scalar(
            _fresh(select(MemoryPersonaModel)).where(
                *self._scope_conditions(scope)
            )
        )

        if row is None:
            return None

        return PersonaRow.model_validate(row)

    async def insert_persona(
            self,
            row: PersonaRow,
    ) -> int:

        statement = pg_insert(MemoryPersonaModel).values(_insert_values(row))

        # ⚠️ DO NOTHING 而不是 DO UPDATE：冲突意味着**另一个 L3 任务**刚为
        # 同一个 scope 建了行（id 是 scope 派生的，撞上就是同一件事）。覆盖
        # 等于悄悄丢掉另一次生成的结论 —— 见契约里 insert_persona 的说明：
        # 让调用方看到 0，由它重来。
        statement = statement.on_conflict_do_nothing(
            index_elements=[MemoryPersonaModel.id],
        )

        result = await self._session.execute(statement)

        return result.rowcount or 0

    async def update_persona(
            self,
            persona_id: str,
            *,
            content: str,
            content_md5: str,
            expected_version: int,
    ) -> int:

        statement = (
            update(MemoryPersonaModel)
            .where(
                MemoryPersonaModel.id == persona_id,
                # ⚠️ 这一句就是乐观锁。少了它，两个 L3 任务会互相覆盖，
                # 而"覆盖"丢掉的是一次完整的模型调用结果（**整篇**画像）。
                MemoryPersonaModel.version == expected_version,
            )
            .values(
                content=content,
                content_md5=content_md5,
                # version 由这里自增，不由调用方传 —— 见契约里的说明。
                version=MemoryPersonaModel.version + 1,
                updated_at=func.now(),
            )
        )

        result = await self._session.execute(statement)

        return result.rowcount or 0

    # ========================================================
    # 水位
    # ========================================================

    async def get_l3_cursor(
            self,
            scope: SceneScope,
    ) -> PersonaCursorState:

        row = await self._session.scalar(
            _fresh(select(MemoryPersonaCursorModel)).where(
                *self._cursor_conditions(scope)
            )
        )

        # 没有行 = 还没生成过。折成默认值，而不是把 None 抛给调用方 ——
        # 见契约里 get_l3_cursor 的说明。
        if row is None:
            return PersonaCursorState()

        return PersonaCursorState(
            last_scene_updated_time=row.last_scene_updated_time,
            last_scene_ids=list(row.last_scene_ids or []),
            last_l3_at=row.last_l3_at,
        )

    async def advance_l3_cursor(
            self,
            scope: SceneScope,
            *,
            last_scene_updated_time: datetime | None,
            last_scene_ids: Sequence[str],
            last_l3_at: datetime,
    ) -> None:

        statement = pg_insert(MemoryPersonaCursorModel).values(
            agent_id=scope.agent_id,
            scope_key=scope.key,
            user_id=scope.user_id,
            org_id=scope.org_id,
            last_scene_updated_time=last_scene_updated_time,
            # 去重：调用方给的是"边界上那批 id"，而这一列只是集合语义 ——
            # 存重复值会让它随轮次慢慢涨大。与 advance_l2_cursor 同一条。
            last_scene_ids=list(dict.fromkeys(last_scene_ids)),
            last_l3_at=last_l3_at,
        )

        statement = statement.on_conflict_do_update(
            index_elements=[
                MemoryPersonaCursorModel.agent_id,
                MemoryPersonaCursorModel.scope_key,
            ],
            set_={
                # ⚠️ scope 的两个数据列也一起更新：它们在正常情况下不会变
                # （scope_key 就是由它们算出来的），但存量行可能是更早的版本
                # 写下的、或者被人工订正过。写回一份与 scope_key 一致的副本，
                # 比留着一份自相矛盾的数据好 —— 而且 `_scope_conditions` 读的
                # 是这两个列（不是 scope_key），它们漂了会让查询查不到自己
                # 刚写的行。
                "user_id": statement.excluded.user_id,
                "org_id": statement.excluded.org_id,
                "last_scene_updated_time": (
                    statement.excluded.last_scene_updated_time
                ),
                "last_scene_ids": statement.excluded.last_scene_ids,
                "last_l3_at": statement.excluded.last_l3_at,
                # ⚠️ 显式写：`onupdate=func.now()` 只对 ORM 的 UPDATE 生效，
                # 对 core 的 ON CONFLICT DO UPDATE **不**生效。见模块 docstring。
                "updated_at": statement.excluded.updated_at,
            },
        )

        await self._session.execute(statement)

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _scope_conditions(
            scope: SceneScope,
    ) -> list[ColumnElement[bool]]:
        """
        scope -> 画像行的 WHERE 条件。与 `scene_impl._scope_conditions`
        逐字同形（两张表的列集与那一条不变量完全一样）。

        ⚠️ 只匹配非空的那一侧
        -----------------
        `user_id` 与 `org_id` 恰有一个为 NULL（不变量由 CHECK 兜底），所以
        判断哪一侧有值就够了 —— 再补一句 `另一侧 IS NULL` 是重复的。
        """

        conditions: list[ColumnElement[bool]] = [
            MemoryPersonaModel.agent_id == scope.agent_id,
        ]

        if scope.org_id is not None:
            conditions.append(MemoryPersonaModel.org_id == scope.org_id)
        else:
            conditions.append(MemoryPersonaModel.user_id == scope.user_id)

        return conditions

    @staticmethod
    def _cursor_conditions(
            scope: SceneScope,
    ) -> list[ColumnElement[bool]]:
        """
        scope -> 水位行的 WHERE 条件。

        ⚠️ 这里是**主键的两半**，不是那两个数据列
        ---------------------------------
        水位表的键是 `(agent_id, scope_key)`（那两列必有一列为 NULL，进不了
        主键）。拿 `user_id` / `org_id` 去查在这张表上也**能**查到，但那样
        会绕开主键的唯一性 —— 一个写入方把 scope_key 拼错时，查询仍然命中，
        于是错误被藏起来，直到某天两行的数据列相同而主键不同。
        按主键查 + 那条 `scope_key <> ''` 的 CHECK 一起兜住这件事。
        """

        return [
            MemoryPersonaCursorModel.agent_id == scope.agent_id,
            MemoryPersonaCursorModel.scope_key == scope.key,
        ]


# ============================================================
# 实体 -> 列值
# ============================================================


def _insert_values(row: PersonaRow) -> dict:
    """新行的列值。不含时间戳（交给服务端默认值），见 `_INSERT_COLUMNS`。"""

    return {
        column: getattr(row, column)
        for column in _INSERT_COLUMNS
    }


def _fresh(statement):
    """
    强制这次查询**从库里重新读**，即使同一个 session 的 identity map 里
    已经有这个对象。

    ⚠️ 与 scene_impl._fresh 同一条理由，一个字都没变
    ----------------------------------------
    SQLAlchemy 的 identity map 让"同一个主键在同一个 session 里只有一个
    对象"，而默认行为是**返回已加载的实例、不覆盖它已有的属性**。

    本模块的两处读恰好都是"必须以库里的当前状态为准"的：

        get_l3_cursor   水位 —— 读到旧值 = 那批场景被**重复**喂给模型
        get_persona     写入的基线 —— 读到旧内容 = 乐观锁用的 version 偏旧，
                        于是正常路径上凭空出现 ConflictError

    生产路径上每条任务各自开一个 scope（新 session），所以这个坑平时不显形；
    而**冒烟脚本会在同一个 session 里连着调几次 handler**（那正是它要验的
    "第二轮只喂增量"）。与其让下一个读代码的人重新踩一遍，不如把意图写死在
    查询上。
    """

    return statement.execution_options(populate_existing=True)
