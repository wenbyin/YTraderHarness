"""
Repository contract for tool documentation.

一个契约覆盖两张表：
    tool_documentation            整工具的说明（一对一）
    tool_parameter_documentation  各参数的说明（一对多）

理由：它们在同一段事务里被同一个接口整体替换
（PUT /tools/{id}/documentation），拆成两个仓储只会让调用方
每次都成对地取、成对地写，还要自己保证两者一致。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import (
    ParameterDocumentation,
    ToolDocumentation,
)

__all__ = ["ToolDocumentationRepository"]


class ToolDocumentationRepository(BaseRepository):

    # ========================================================
    # Write
    # ========================================================

    @abstractmethod
    async def upsert(
            self,
            tool_id: UUID,
            *,
            description_override: str | None,
            return_description_override: str | None,
            updated_by: str | None,
            now: datetime,
    ) -> ToolDocumentation:
        """
        写入整工具的说明覆盖。没有行就建，有就覆盖。

        第一次保存时才建行 —— 一个从没被人工改过的 Tool
        不该在库里留一行全 null 的说明（那会让"改没改过"
        这个判断变得不可靠）。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_parameter_documentations(
            self,
            tool_id: UUID,
            entries: list[tuple[str, str | None]],
            *,
            updated_by: str | None,
            now: datetime,
    ) -> None:
        """
        整体替换某 Tool 的参数说明。

        entries 是 (parameter_path, description_override) 的列表。

        语义是替换不是追加：没列出来的参数，其覆盖会被清掉
        （回到代码里的原始说明）。这样前端可以直接拿
        PUT 的 body 全量提交，不必自己算增删 ——
        与 ToolClassificationUpdate 同一套取舍。

        description_override 为 None 的条目**照样入库**还是跳过，
        由实现决定，但必须与 get() 的读法一致，否则会出现
        "写完读回来不一样"。
        """

        raise NotImplementedError

    # ========================================================
    # Read
    # ========================================================

    @abstractmethod
    async def get(
            self,
            tool_id: UUID,
    ) -> ToolDocumentation | None:
        """
        取整工具的说明（含全部参数说明）。

        从没被人工改过时返回 None —— 不是"返回一个空壳"。
        调用方据此知道该回退到代码里的原始说明。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_parameter_documentations(
            self,
            tool_id: UUID,
    ) -> list[ParameterDocumentation]:
        """只取参数说明，不取整工具的说明。"""

        raise NotImplementedError

    @abstractmethod
    async def map_description_overrides(
            self,
            tool_ids: list[UUID],
    ) -> dict[UUID, str | None]:
        """
        批量取工具级的描述覆盖（列表页用），返回 tool_id -> 覆盖值。

        单独开一个"只取这一列"的方法，而不是复用 get()：
        列表页每页 20 条，get() 会把参数说明也一并查出来
        （多一次查询 + 多一堆用不上的行），而列表根本用不到参数说明。

        没被人工改过的 tool_id 不会出现在返回的 dict 里，
        调用方请用 .get(tool_id) 取值。
        """

        raise NotImplementedError

    @abstractmethod
    async def map_parameter_overrides(
            self,
            tool_ids: list[UUID],
    ) -> dict[UUID, dict[str, str | None]]:
        """
        批量取**参数级**的描述覆盖，返回 tool_id -> {参数路径: 覆盖值}。

        与 list_parameter_documentations 的关系，就是
        map_description_overrides 与 get 的关系：前者一个工具一次、
        后者一批一次。执行那条路（把人工说明叠进 bind 给模型的工具）
        要的正是批量版 —— 逐个工具查就是经典的 N+1，而它跑在
        每一轮对话的路径上。

        ⚠️ 返回的**内层 dict 里保留值为 None 的行**（人为建了行但没写
        覆盖）。上层要判"算不算覆盖"请用自己的口径（空串与 None 都
        算没有），别指望这里替你滤 —— 滤掉之后"有行但空"与"没有行"
        就分不开了，而两者排查方向不同。

        没有任何参数说明的 tool_id 不会出现在返回的 dict 里，
        调用方请用 .get(tool_id) 取值。
        """

        raise NotImplementedError

    @abstractmethod
    async def count_for_tools(
            self,
            tool_ids: list[UUID],
    ) -> int:
        """
        统计这些 Tool 名下有多少**说明行**（含参数说明）。

        同步在物理删除定义前用它填 ToolSyncResult
        .orphaned_documentations —— 那些行会被外键 CASCADE 静默带走，
        至少让调用方知道一共没了多少条，而不是事后才发现。

        两张表都算：整工具的说明与各参数的说明都是人写的字，
        只报其中一张会让这个数字偏小、失去警示意义。
        """

        raise NotImplementedError
