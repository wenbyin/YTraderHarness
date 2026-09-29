"""
PostgreSQL implementation of ToolDefinitionRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。
事务边界由 ServiceFactory.scope() 持有的那段事务决定。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, or_, select, update

from ..domain.entities import ToolDefinition
from ..domain.schemas.enum import SourceState, SourceType
from ..models.sqlalchemy.tool_classification import (
    ToolClassification as ToolClassificationModel,
)
from ..models.sqlalchemy.tool_classification import (
    ToolDefinitionClassification as LinkModel,
)
from ..models.sqlalchemy.tool_definition import (
    ToolDefinition as ToolDefinitionModel,
)
from .tool_definition import ToolDefinitionRepository


class ToolDefinitionRepositoryImpl(ToolDefinitionRepository):

    # ========================================================
    # Write
    # ========================================================

    async def add(
            self,
            definition: ToolDefinition,
    ) -> ToolDefinition:

        model = ToolDefinitionModel(
            tool_id=definition.tool_id,
            tool_key=definition.tool_key,
            name=definition.name,
            source_type=definition.source_type.value,
            source_ref=definition.source_ref,
            entrypoint=definition.entrypoint,
            input_schema=dict(definition.input_schema),
            output_schema=(
                None
                if definition.output_schema is None
                else dict(definition.output_schema)
            ),
            source_docstring=(
                None
                if definition.source_docstring is None
                else dict(definition.source_docstring)
            ),
            manifest_hash=definition.manifest_hash,
            source_state=definition.source_state.value,
            discovered_at=definition.discovered_at,
            last_synced_at=definition.last_synced_at,
            created_at=definition.created_at,
            updated_at=definition.updated_at,
        )

        self._session.add(model)
        await self._session.flush()

        return self._to_entity(model)

    async def update(
            self,
            definition: ToolDefinition,
    ) -> ToolDefinition:
        """
        整行覆盖（除主键与 discovered_at）。

        discovered_at 刻意保留原值 —— 它记的是"首次发现时间"，
        每次同步都刷成现在就等于没有这个字段了。
        """

        await self._session.execute(
            update(ToolDefinitionModel)
            .where(ToolDefinitionModel.tool_id == definition.tool_id)
            .values(
                tool_key=definition.tool_key,
                name=definition.name,
                source_type=definition.source_type.value,
                source_ref=definition.source_ref,
                entrypoint=definition.entrypoint,
                input_schema=dict(definition.input_schema),
                output_schema=(
                    None
                    if definition.output_schema is None
                    else dict(definition.output_schema)
                ),
                source_docstring=(
                    None
                    if definition.source_docstring is None
                    else dict(definition.source_docstring)
                ),
                manifest_hash=definition.manifest_hash,
                source_state=definition.source_state.value,
                last_synced_at=definition.last_synced_at,
                updated_at=definition.updated_at,
            )
        )

        await self._session.flush()

        refreshed = await self.get(definition.tool_id)

        return refreshed if refreshed is not None else definition

    async def mark_missing(
            self,
            tool_ids: list[UUID],
            *,
            now: datetime,
    ) -> int:

        if not tool_ids:
            return 0

        result = await self._session.execute(
            update(ToolDefinitionModel)
            .where(ToolDefinitionModel.tool_id.in_(tool_ids))
            .values(
                source_state=SourceState.MISSING.value,
                last_synced_at=now,
                updated_at=now,
            )
        )

        await self._session.flush()

        return result.rowcount or 0

    async def touch_synced_at(
            self,
            tool_ids: list[UUID],
            *,
            now: datetime,
    ) -> int:

        if not tool_ids:
            return 0

        result = await self._session.execute(
            update(ToolDefinitionModel)
            .where(ToolDefinitionModel.tool_id.in_(tool_ids))
            .values(last_synced_at=now)
        )

        await self._session.flush()

        return result.rowcount or 0

    async def delete_many(
            self,
            tool_ids: list[UUID],
    ) -> int:

        if not tool_ids:
            return 0

        result = await self._session.execute(
            delete(ToolDefinitionModel).where(
                ToolDefinitionModel.tool_id.in_(tool_ids)
            )
        )

        await self._session.flush()

        return result.rowcount or 0

    # ========================================================
    # Read
    # ========================================================

    async def get(
            self,
            tool_id: UUID,
    ) -> ToolDefinition | None:

        model = await self._session.get(
            ToolDefinitionModel, tool_id
        )

        return None if model is None else self._to_entity(model)

    async def get_by_key(
            self,
            tool_key: str,
    ) -> ToolDefinition | None:

        result = await self._session.execute(
            select(ToolDefinitionModel).where(
                ToolDefinitionModel.tool_key == tool_key
            )
        )

        model = result.scalar_one_or_none()

        return None if model is None else self._to_entity(model)

    async def list_by_keys(
            self,
            tool_keys: list[str],
    ) -> list[ToolDefinition]:

        if not tool_keys:
            return []

        result = await self._session.execute(
            select(ToolDefinitionModel).where(
                ToolDefinitionModel.tool_key.in_(tool_keys)
            )
        )

        return [self._to_entity(m) for m in result.scalars()]

    async def list_active(self) -> list[ToolDefinition]:

        result = await self._session.execute(
            select(ToolDefinitionModel).where(
                ToolDefinitionModel.source_state
                == SourceState.ACTIVE.value
            )
        )

        return [self._to_entity(m) for m in result.scalars()]

    async def list_all(
            self,
            *,
            source_type: SourceType | None = None,
    ) -> list[ToolDefinition]:

        statement = select(ToolDefinitionModel)

        if source_type is not None:
            statement = statement.where(
                ToolDefinitionModel.source_type
                == source_type.value
            )

        result = await self._session.execute(
            statement.order_by(ToolDefinitionModel.tool_key)
        )

        return [self._to_entity(m) for m in result.scalars()]

    async def list_paged(
            self,
            *,
            keyword: str | None = None,
            source_type: SourceType | None = None,
            source_state: SourceState | None = None,
            classification_code: str | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[ToolDefinition], int]:

        conditions = []

        if keyword:
            # 关键词同时匹配 key 与展示名：调用方搜 "document"
            # 时两者都该命中，只匹配一个会让人以为数据丢了。
            pattern = f"%{keyword}%"
            conditions.append(
                or_(
                    ToolDefinitionModel.tool_key.ilike(pattern),
                    ToolDefinitionModel.name.ilike(pattern),
                )
            )

        if source_type is not None:
            conditions.append(
                ToolDefinitionModel.source_type
                == source_type.value
            )

        if source_state is not None:
            conditions.append(
                ToolDefinitionModel.source_state
                == source_state.value
            )

        if classification_code:
            # 用 EXISTS 而不是 join：一个 Tool 可能同时挂着多个分类，
            # join 会让它在结果里出现多行，分页与 total 就都错了。
            conditions.append(
                select(LinkModel.tool_id)
                .join(
                    ToolClassificationModel,
                    ToolClassificationModel.classification_id
                    == LinkModel.classification_id,
                )
                .where(
                    LinkModel.tool_id
                    == ToolDefinitionModel.tool_id,
                    ToolClassificationModel.code
                    == classification_code,
                )
                .exists()
            )

        count_statement = select(
            func.count()
        ).select_from(ToolDefinitionModel)

        page_statement = select(ToolDefinitionModel)

        if conditions:
            count_statement = count_statement.where(*conditions)
            page_statement = page_statement.where(*conditions)

        total = (
            await self._session.execute(count_statement)
        ).scalar_one()

        result = await self._session.execute(
            page_statement
            .order_by(ToolDefinitionModel.tool_key)
            .offset(offset)
            .limit(limit)
        )

        return [self._to_entity(m) for m in result.scalars()], total

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_entity(
            model: ToolDefinitionModel,
    ) -> ToolDefinition:
        """
        ORM -> 领域实体。

        实体带 from_attributes，source_type / source_state 两个
        枚举字段由 pydantic 从字符串自动转换，不必手工映射。
        """

        return ToolDefinition.model_validate(model)
