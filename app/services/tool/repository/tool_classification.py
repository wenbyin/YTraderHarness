"""
Repository contract for ToolClassification.

一个契约覆盖两张表：
    tool_classification              分类树
    tool_definition_classification   Tool ↔ 分类的关联

理由与 ToolDocumentationRepository 相同：分类与它的关联
总是被同一个接口一起读写（GET/PUT /tools/{id}/classifications），
拆开只会让调用方自己保证一致性。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import ToolClassification

__all__ = ["ToolClassificationRepository"]


class ToolClassificationRepository(BaseRepository):

    # ========================================================
    # Taxonomy
    # ========================================================

    @abstractmethod
    async def add(
            self,
            classification: ToolClassification,
    ) -> ToolClassification:
        """新建一个分类节点。code 冲突由调用方先查过。"""

        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            classification: ToolClassification,
    ) -> ToolClassification:
        """
        整行覆盖。

        实现必须**保留 code**（或由调用方保证传进来的 code 没变）——
        code 是稳定标识，改了会让已有的 tool 关联与外部引用对不上。
        """

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            classification_id: UUID,
    ) -> ToolClassification | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_code(
            self,
            code: str,
    ) -> ToolClassification | None:
        raise NotImplementedError

    @abstractmethod
    async def list(
            self,
            *,
            parent_id: UUID | None = None,
            keyword: str | None = None,
            is_active: bool | None = None,
    ) -> list[ToolClassification]:
        """
        按条件取分类。

        parent_id 为 None 表示**不过滤**（取全部层级的节点），
        而不是"只取一级分类" —— 后者是另一个查询，需要时再单独加
        一个方法，别把两种语义塞进一个参数。
        """

        raise NotImplementedError

    # ========================================================
    # Tool ↔ Classification
    # ========================================================

    @abstractmethod
    async def list_for_tool(
            self,
            tool_id: UUID,
    ) -> list[ToolClassification]:
        raise NotImplementedError

    @abstractmethod
    async def list_for_tools(
            self,
            tool_ids: list[UUID],
    ) -> dict[UUID, list[ToolClassification]]:
        """
        批量取（列表页用），返回 tool_id -> 分类列表。

        批量而不是逐个 list_for_tool：列表页一页 20 条，
        逐个查就是 20 次往返（经典的 N+1）。

        没有关联的 tool_id **不一定**出现在返回的 dict 里，
        调用方请用 .get(tool_id, []) 取值。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_tool_classifications(
            self,
            tool_id: UUID,
            classification_ids: list[UUID],
    ) -> None:
        """
        整体替换一个 Tool 的分类关联。

        语义是替换不是追加，与 ToolClassificationUpdate 的
        契约一致：没列出来的会被解除关联。
        """

        raise NotImplementedError

    @abstractmethod
    async def count_links(
            self,
            classification_ids: list[UUID],
    ) -> int:
        """
        统计这些分类被多少个 Tool 引用。

        删分类前用它给出"还有 N 个工具在用"的提示 ——
        CASCADE 会静默解除关联，调用方至少该有机会先知道。
        """

        raise NotImplementedError
