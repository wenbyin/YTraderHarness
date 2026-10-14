"""
Repository contract for Capability.

只管 capability 表本身（能力树），加上一条"我被多少工具引用"
的反查 —— 后者是删除前的必要判断，属于能力自己的生命周期，
因此留在本契约里（见 count_links）。

⚠️ 关联的**读写**不在这里。它们是 tool 模块的事，契约是
app/services/tool/repository/tool_classification.py ——
那条路径的主语是工具，删工具就该顺带解除关联。本模块只问
"能不能删"，不改关联。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import Capability

__all__ = ["CapabilityRepository"]


class CapabilityRepository(BaseRepository):

    # ========================================================
    # Taxonomy
    # ========================================================

    @abstractmethod
    async def add(
            self,
            capability: Capability,
    ) -> Capability:
        """新建一个能力节点。code 冲突由调用方先查过。"""

        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            capability: Capability,
    ) -> Capability:
        """
        整行覆盖。

        实现必须**保留 code**（或由调用方保证传进来的 code 没变）——
        code 是稳定标识，改了会让已有的 tool 关联与外部引用对不上。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete(
            self,
            capability_id: UUID,
    ) -> None:
        """
        物理删除一个能力节点。

        只删这一个节点，**不做**子树级联 —— 与 ORM 里自引用外键
        刻意不加 CASCADE 是同一条规则（见 models/sqlalchemy/
        capability.py）。有子节点时实现会撞外键约束并抛错，
        因此调用方必须先确认 count_children() == 0。

        指向 Tool 的关联行由数据库的 ON DELETE CASCADE 一并清掉。
        """

        raise NotImplementedError

    @abstractmethod
    async def count_children(
            self,
            capability_id: UUID,
    ) -> int:
        """
        直接子节点的个数（不含后代）。

        删能力前用它挡住"下面还挂着东西"的情况 —— 那一步必然失败，
        与其让调用方撞一个外键报错，不如先给出一句能读懂的话。
        """

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            capability_id: UUID,
    ) -> Capability | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_code(
            self,
            code: str,
    ) -> Capability | None:
        raise NotImplementedError

    @abstractmethod
    async def list(
            self,
            *,
            parent_id: UUID | None = None,
            keyword: str | None = None,
            is_active: bool | None = None,
    ) -> list[Capability]:
        """
        按条件取能力。

        parent_id 为 None 表示**不过滤**（取全部层级的节点），
        而不是"只取一级能力" —— 后者是另一个查询，需要时再单独加
        一个方法，别把两种语义塞进一个参数。
        """

        raise NotImplementedError

    # ========================================================
    # 反查引用
    # ========================================================

    @abstractmethod
    async def count_links(
            self,
            capability_ids: list[UUID],
    ) -> int:
        """
        统计这些能力被多少个 Tool 引用。

        删能力前用它给出"还有 N 个工具在用"的提示 ——
        CASCADE 会静默解除关联，调用方至少该有机会先知道。

        它读的是 tool 模块的关联表（tool_definition_classification）。
        读在这里是刻意的：问的是"这个能力能不能删"，主语是能力；
        而**写**关联（挂上/摘掉）仍然只发生在 tool 模块。
        """

        raise NotImplementedError
