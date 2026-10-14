"""
能力树（taxonomy）的读写契约。

对应 capability.yaml：
    GET    /capabilities
    POST   /capabilities
    GET    /capabilities/{capability_id}
    PATCH  /capabilities/{capability_id}
    PUT    /capabilities/{capability_id}
    DELETE /capabilities/{capability_id}

⚠️ 工具 ↔ 能力的**关联**不在这里 —— 它对应
/tools/{tool_id}/classifications，契约是 tool 模块的
ToolClassificationService，因为那个路径的主语是工具。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    Capability,
    CapabilityCreate,
    CapabilityReplace,
    CapabilityUpdate,
)

__all__ = ["CapabilityService"]


class CapabilityService(BaseService):

    # ========================================================
    # 能力树
    # ========================================================

    @abstractmethod
    async def list_capabilities(
            self,
            *,
            parent_id: UUID | None = None,
            keyword: str | None = None,
            is_active: bool | None = None,
    ) -> list[Capability]:
        """
        按条件列出能力。

        parent_id 不传表示**不过滤**（全部层级的节点都在结果里），
        不是"只取一级能力"。要拿某个节点的直接子节点就传它的 id。
        """

        raise NotImplementedError

    @abstractmethod
    async def create_capability(
            self,
            request: CapabilityCreate,
    ) -> Capability:
        """
        Raises:
            ConflictError: code 已被占用；或 parent_id 不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_capability(
            self,
            capability_id: UUID,
    ) -> Capability:
        """
        Raises:
            NotFoundError: 能力不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def update_capability(
            self,
            capability_id: UUID,
            request: CapabilityUpdate,
    ) -> Capability:
        """
        局部更新。code 不可改 —— 它是稳定标识。

        Raises:
            NotFoundError: 能力不存在。
            ConflictError: 想把自己挂到自己或自己的后代下面（成环）。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_capability(
            self,
            capability_id: UUID,
            request: CapabilityReplace,
    ) -> Capability:
        """
        整体替换。没传的可空字段按 null 处理，不是保持原值。

        code 保持不动 —— 与局部更新一样，它是稳定标识。

        Raises:
            NotFoundError: 能力不存在。
            ConflictError: 父能力不存在；或想挂到自己或自己的后代下面。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_capability(
            self,
            capability_id: UUID,
            *,
            force: bool = False,
    ) -> None:
        """
        物理删除一个能力节点，连带解除它与 Tool 的关联。

        force 只覆盖**工具引用**这一关：确认后把关联行一起清掉。
        它**不能**覆盖子能力 —— 有子节点时无条件失败，先挪走或删掉
        它们再删本节点（自引用外键刻意不加 CASCADE，理由见
        models/sqlalchemy/capability.py）。

        Raises:
            NotFoundError: 能力不存在。
            ConflictError: 还有子能力；或仍被 Tool 引用且未给 force。
        """

        raise NotImplementedError
