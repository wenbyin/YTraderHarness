"""
PostgreSQL implementation of ToolProcessorRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
事务由调用方（ServiceFactory.scope()）持有 —— 一次链替换必须
整体成功或整体回滚，中途失败留下"链只写了一半"比报错更糟。

为什么所有写操作都要走 _apply_order
-----------------------------------
`uq_tool_processor_tool_sequence` 是 (tool_id, sequence) 上的
**非延迟唯一索引**：PostgreSQL 逐行立即检查，没有"语句结束时
才校验"这回事。而链的顺序就是语义 —— 插入、删除、重排都必然
让一些行的 sequence 变成另一些行**当前**占着的值。

举例：把 [A=1, B=2] 重排成 [B, A]，直接 `UPDATE ... SET
sequence = 新值` 会在把 A 改成 2 的那一刻撞上 B 还占着的 2。

`_apply_order` 用两段式解决：先把整条链抬到一个**不可能与终值
相撞**的高位区间，再落到 1..N。两段各一条语句，不改约束、
不在事务内留中间态（同一个事务里别人看不到中间态，但唯一索引
看得到 —— 所以必须真的绕开它，而不是指望延迟检查）。

被否掉的几种做法
----------------
    挪到负区间              ❌ 撞 CHECK (sequence >= 1)
    按倒序逐行 UPDATE       ❌ PostgreSQL 不保证 UPDATE 的行处理顺序，
                              得拆成 N 条单独语句
    可延迟唯一约束          ❌ 偏离契约（sql 里写的是 CREATE UNIQUE INDEX），
                              且丢掉部分索引的表达力
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import case, delete, select, update

from app.common.exceptions import NotFoundError

from ..domain.entities import ToolProcessor
from ..domain.schemas.enum import ExecutionMode, ProcessorHook
from ..models.sqlalchemy.tool_processor import (
    ToolProcessor as ToolProcessorModel,
)
from .tool_processor import ToolProcessorRepository

__all__ = ["ToolProcessorRepositoryImpl"]


#: 把整条链抬到高位时用的偏移。
#:
#: 只要它大于任何真实 sequence 即可。链的长度是人配出来的（几条到
#: 十几条），而所有写操作结束时都会重编号成 1..N，因此库里的
#: sequence 永远不会接近这个量级。
_TEMP_OFFSET = 1_000_000

#: 新行入库时的**临时** sequence。
#:
#: 取一半：它既不会与现有行的 1..N 相撞，也不会与 _apply_order
#: 抬走后的 [1_000_001, 1_000_000+n] 区间相撞。
_NEW_ROW_TEMP = 500_000


class ToolProcessorRepositoryImpl(ToolProcessorRepository):

    # ========================================================
    # Read
    # ========================================================

    async def list_for_tool(
            self,
            tool_id: UUID,
            *,
            enabled_only: bool = False,
    ) -> list[ToolProcessor]:

        statement = select(ToolProcessorModel).where(
            ToolProcessorModel.tool_id == tool_id
        )

        if enabled_only:
            statement = statement.where(
                ToolProcessorModel.enabled.is_(True)
            )

        statement = statement.order_by(
            ToolProcessorModel.sequence
        ).execution_options(
            # 写操作走的是 Core 的 update()/delete()，身份映射里的
            # 对象不会跟着变（SQLAlchemy 不知道它们被改过）。
            # 不强制重读的话，"插完再读回来"会拿到一组过期的
            # sequence —— 表现是响应里的顺序与库里不一致。
            populate_existing=True,
        )

        rows = (
            await self._session.execute(statement)
        ).scalars().all()

        return [self._to_entity(row) for row in rows]

    async def get(
            self,
            tool_id: UUID,
            processor_id: UUID,
    ) -> ToolProcessor | None:

        model = await self._session.get(
            ToolProcessorModel, processor_id
        )

        # 必须再核一次 tool_id：路径是 /tools/{tool_id}/processors/
        # {processor_id}，只按主键取会让"拿 A 工具的 id 去改 B 工具的
        # 行"变成一次成功的越界写。
        if model is None or model.tool_id != tool_id:
            return None

        return self._to_entity(model)

    async def count_for_tool(self, tool_id: UUID) -> int:

        return len(await self._ids_of(tool_id))

    # ========================================================
    # Write
    # ========================================================

    async def insert(
            self,
            entity: ToolProcessor,
            *,
            position: int | None = None,
    ) -> ToolProcessor:

        model = self._to_model(entity)

        # 临时高位入库，位置由 _apply_order 落定 ——
        # 直接写目标 sequence 会在"插到中间"时与现有行撞唯一索引。
        model.sequence = _NEW_ROW_TEMP

        self._session.add(model)

        await self._session.flush()

        order = await self._ids_of(entity.tool_id)

        # position 是最终的 1-based 位置；None = 追加到末尾。
        index = (
            len(order) - 1
            if position is None
            else max(0, min(position - 1, len(order) - 1))
        )

        order.remove(model.processor_id)
        order.insert(index, model.processor_id)

        await self._apply_order(entity.tool_id, order)

        # ⚠️ 必须读回来再映射。
        #
        # _apply_order 走的是 Core 的 update()，SQLAlchemy 会把**被
        # 匹配到的**对象标记为过期（synchronize_session 的默认行为）：
        # 之后再访问 model.sequence 会触发一次懒加载，而在 pydantic
        # 的同步属性访问里没有 greenlet 上下文，直接抛
        # MissingGreenlet（错误信息指向 pydantic，离原因很远）。
        await self._session.refresh(model)

        return self._to_entity(model)

    async def update(self, entity: ToolProcessor) -> ToolProcessor:

        model = await self._session.get(
            ToolProcessorModel, entity.processor_id
        )

        if model is None or model.tool_id != entity.tool_id:
            raise NotFoundError(
                f"挂载记录不存在：{entity.processor_id}"
            )

        # 只覆盖可改的四个字段。processor_key 与 sequence 各有专门的
        # 操作（删了重挂 / PUT order），在这里一起改会让"我只想调个
        # 参数"变成一次隐式的链重排。
        model.execution_mode = entity.execution_mode.value
        model.hook = entity.hook.value
        model.config = dict(entity.config)
        model.enabled = entity.enabled

        # updated_at 由列上的 onupdate=func.now() 维护（见 ORM 的
        # 说明：不镜像 sql 里的触发器）。
        await self._session.flush()

        return self._to_entity(model)

    async def delete(
            self,
            tool_id: UUID,
            processor_id: UUID,
    ) -> int:

        result = await self._session.execute(
            delete(ToolProcessorModel).where(
                ToolProcessorModel.tool_id == tool_id,
                ToolProcessorModel.processor_id == processor_id,
            )
        )

        await self._session.flush()

        if not result.rowcount:
            return 0

        # 补洞：剩下的重编号成 1..N。不补的话链会留下空洞
        # （1, 3, 4），而"从 1 开始连续"是这套设计的约定
        # （见 tool_processor.sql 的注释）。
        remaining = await self._ids_of(tool_id)

        if remaining:
            await self._apply_order(tool_id, remaining)

        return result.rowcount

    async def replace_all(
            self,
            tool_id: UUID,
            entities: list[ToolProcessor],
    ) -> list[ToolProcessor]:

        # ⚠️ 必须用 Core 的 delete **先删并 flush**
        # ------------------------------------------
        # 用 ORM 的 session.delete(obj) + session.add(new) 会炸：
        # 一次 flush 内部的顺序是 INSERT 先、DELETE 后，新行的
        # (tool_id, 1..N) 会在旧行还没删掉时插进去，直接撞唯一索引。
        await self._session.execute(
            delete(ToolProcessorModel).where(
                ToolProcessorModel.tool_id == tool_id
            )
        )

        await self._session.flush()

        for index, entity in enumerate(entities, start=1):

            model = self._to_model(entity)

            # 忽略实体带来的 sequence，一律按列表下标重算 ——
            # 顺序只有一个来源（数组顺序），不让两个来源打架。
            model.sequence = index

            self._session.add(model)

        await self._session.flush()

        return await self.list_for_tool(tool_id)

    async def reorder(
            self,
            tool_id: UUID,
            ordered_ids: list[UUID],
    ) -> int:

        await self._apply_order(tool_id, ordered_ids)

        return len(ordered_ids)

    # ========================================================
    # internals
    # ========================================================

    async def _ids_of(self, tool_id: UUID) -> list[UUID]:
        """
        取该 Tool 的处理器 ID，**按 sequence 升序**。

        只取 ID 而不是整行：挪位与重编号只关心顺序，
        把 config 这些一并读出来纯属浪费（而且 Core update 之后
        它们本来就是过期的）。
        """

        statement = (
            select(ToolProcessorModel.processor_id)
            .where(ToolProcessorModel.tool_id == tool_id)
            .order_by(ToolProcessorModel.sequence)
        )

        return list(
            (await self._session.execute(statement)).scalars().all()
        )

    async def _apply_order(
            self,
            tool_id: UUID,
            ordered_ids: list[UUID],
    ) -> None:
        """
        把整条链按 ordered_ids 的顺序重编号成 1..N（两段式）。

        第一段：整体抬到 [1_000_001, 1_000_000+n]。
            所有行同加一个常数，彼此之间仍然互不相等，因此不会
            撞唯一索引；终值是 1..N，全都落在这个区间之外，
            因此第二段也不会撞到"还没轮到的行"。

        第二段：一次 UPDATE + CASE 落终值。
            写成 N 条单独语句也对（此时已经没有冲突可能），
            但那要 N 次往返，而链的顺序变动是常规操作。
        """

        current = await self._ids_of(tool_id)

        if set(current) != set(ordered_ids):
            # 顺序表必须是现有行的**全集**：少一个会让那一行留在
            # 高位区间（脱离 1..N），多一个则是调用方拿着过期的链
            # 在操作。两种情况都该显式失败，而不是尽量猜。
            raise ValueError(
                "重排的顺序表必须是该 Tool 现有行的全集："
                f"现有 {len(current)} 行，给了 {len(ordered_ids)} 个 ID。"
                "少给会让没被点到的行留在临时高位。"
            )

        if not ordered_ids:
            return

        await self._session.execute(
            update(ToolProcessorModel)
            .where(ToolProcessorModel.tool_id == tool_id)
            .values(
                sequence=(
                    ToolProcessorModel.sequence + _TEMP_OFFSET
                )
            )
        )

        mapping = {
            processor_id: index
            for index, processor_id in enumerate(
                ordered_ids, start=1
            )
        }

        await self._session.execute(
            update(ToolProcessorModel)
            .where(ToolProcessorModel.tool_id == tool_id)
            .values(
                sequence=case(
                    mapping,
                    value=ToolProcessorModel.processor_id,
                )
            )
        )

        await self._session.flush()

    # ========================================================
    # mapping
    # ========================================================

    @staticmethod
    def _to_entity(model: ToolProcessorModel) -> ToolProcessor:

        return ToolProcessor.model_validate(model)

    @staticmethod
    def _to_model(entity: ToolProcessor) -> ToolProcessorModel:
        """
        实体 -> ORM 行。

        不写 `**entity.model_dump()`：枚举字段（execution_mode /
        hook）在 python 模式下 dump 出来是枚举对象，而列是 String，
        显式取 .value 才不会把"靠 SQLAlchemy 恰好认得 str 子类"
        当成正确性依据。
        """

        return ToolProcessorModel(
            processor_id=entity.processor_id,
            tool_id=entity.tool_id,
            processor_key=entity.processor_key,
            sequence=entity.sequence,
            execution_mode=ExecutionMode(
                entity.execution_mode
            ).value,
            hook=ProcessorHook(entity.hook).value,
            config=dict(entity.config),
            enabled=entity.enabled,
            # 两个时间戳由调用方给（与 ToolDiscoveryServiceImpl 造
            # ToolDefinition 实体时同一做法）。不给的话列上的
            # server_default 会兜住，但那样实体里带的值就成了摆设，
            # 而"写进去的"与"以为写进去的"不一致是最费时间的一类
            # 排查。insert() 里 sequence 被覆盖成临时值，这里照抄
            # 无妨 —— 终值由 _apply_order 落定。
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )
