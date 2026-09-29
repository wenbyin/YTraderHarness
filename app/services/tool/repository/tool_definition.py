"""
Repository contract for ToolDefinition.

Transaction rule
----------------
实现不得调用 session.commit() / session.rollback()，
事务边界由调用方（ServiceFactory.scope() 持有的那段事务）决定。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import ToolDefinition
from ..domain.schemas.enum import SourceState, SourceType

__all__ = ["ToolDefinitionRepository"]


class ToolDefinitionRepository(BaseRepository):
    """
    ToolDefinition 的持久化契约。

    这一层的调用方只有两个：
        ToolQueryService      读（列表 / 详情）
        ToolDiscoveryService  写（同步）
    """

    # ========================================================
    # Write （只由 discovery 调用）
    # ========================================================

    @abstractmethod
    async def add(
            self,
            definition: ToolDefinition,
    ) -> ToolDefinition:
        """新建一行。调用方负责保证 tool_key 不与既有行冲突。"""

        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            definition: ToolDefinition,
    ) -> ToolDefinition:
        """
        整行覆盖。

        可以安全地整行覆盖，是因为这一行的每个字段都由代码产生、
        没有任何人工写入口（见 domain/entities/tool_definition.py）。
        """

        raise NotImplementedError

    @abstractmethod
    async def mark_missing(
            self,
            tool_ids: list[UUID],
            *,
            now: datetime,
    ) -> int:
        """
        把这些行标成 MISSING，返回受影响行数。

        不删除是刻意的：Tool 可能已被编排进某个 Agent / 工作流，
        行删了引用就断了、报错也指不到原因。
        """

        raise NotImplementedError

    @abstractmethod
    async def touch_synced_at(
            self,
            tool_ids: list[UUID],
            *,
            now: datetime,
    ) -> int:
        """
        只刷新 last_synced_at，返回受影响行数。

        给"内容没变、但这次确实同步过"的行用（unchanged）。

        为什么值得单独一个方法：last_synced_at 的语义是
        "最近一次同步时间"，那就该对每个扫到的工具都刷新 ——
        否则它实际记的是"最近一次内容变化的时间"，与字段名和
        注释都对不上。但逐个 update() 意味着几百次往返，
        因此做成**一条** IN 语句。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_many(
            self,
            tool_ids: list[UUID],
    ) -> int:
        """
        物理删除，返回受影响行数。

        只由 sync 的 remove_missing=true 走到，而那条路默认是关的。
        关联的说明行与分类关联由数据库 CASCADE 带走。
        """

        raise NotImplementedError

    # ========================================================
    # Read
    # ========================================================

    @abstractmethod
    async def get(
            self,
            tool_id: UUID,
    ) -> ToolDefinition | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_key(
            self,
            tool_key: str,
    ) -> ToolDefinition | None:
        """同步按它判断 created / updated / unchanged。"""

        raise NotImplementedError

    @abstractmethod
    async def list_by_keys(
            self,
            tool_keys: list[str],
    ) -> list[ToolDefinition]:
        """
        按 key 批量取，供同步比对用。

        批量而不是逐个 get_by_key：一次同步可能扫出几百个工具，
        逐个查就是几百次往返。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_active(self) -> list[ToolDefinition]:
        """
        取全部 ACTIVE 的行。

        同步用它做反向比对：这些 key 里没被本次扫到的，就是
        "代码里已经没有了"，要转成 MISSING。
        """

        raise NotImplementedError

    @abstractmethod
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
        """
        分页查询，返回 (本页数据, 符合条件的总数)。

        classification_code 需要连到关联表与分类表上过滤 ——
        因此它不是简单的一列条件，见实现里的 join。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_all(
            self,
            *,
            source_type: SourceType | None = None,
    ) -> list[ToolDefinition]:
        """不分页取全部，供内部聚合用。"""

        raise NotImplementedError
