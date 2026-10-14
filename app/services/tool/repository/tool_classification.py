"""
Repository contract for the Tool ↔ capability association.

只管 tool_definition_classification 一张关联表：
    读：某个/某批工具关联了哪些能力（列表页批量取，避免 N+1）
    写：整体替换某个工具的关联集合

⚠️ 能力**本体**的读写在 app/services/capability/repository/，
不在这里。这张关联表当初与能力树挤在一个契约里，理由是"两者总是
被同一个接口一起读写"；现在能力树独立成模块，那个理由不成立了 ——
关联的主语是工具，工具侧只需要"挂上/摘掉"。

关联的**存在性校验**（分类 id 是否存在）由调用方先做：那个判断要查
能力表，属于 capability 模块，不该在这里再开一条跨模块的读路径。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository
from app.services.capability.domain.entities import Capability

__all__ = ["ToolClassificationRepository"]


class ToolClassificationRepository(BaseRepository):

    @abstractmethod
    async def list_for_tool(
            self,
            tool_id: UUID,
    ) -> list[Capability]:
        raise NotImplementedError

    @abstractmethod
    async def list_for_tools(
            self,
            tool_ids: list[UUID],
    ) -> dict[UUID, list[Capability]]:
        """
        批量取（列表页用），返回 tool_id -> 能力列表。

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
        整体替换一个 Tool 的能力关联。

        语义是替换不是追加，与 ToolClassificationUpdate 的
        契约一致：没列出来的会被解除关联。
        """

        raise NotImplementedError
