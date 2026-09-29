"""
分类与「工具 <-> 分类」关联的读写契约。

对应 tool.yaml：
    GET    /classifications
    POST   /classifications
    GET    /classifications/{classification_id}
    PATCH  /classifications/{classification_id}
    GET    /tools/{tool_id}/classifications
    PUT    /tools/{tool_id}/classifications
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    Classification,
    ClassificationCreate,
    ClassificationUpdate,
    ToolClassificationUpdate,
)

__all__ = ["ClassificationService"]


class ClassificationService(BaseService):

    # ========================================================
    # 分类树
    # ========================================================

    @abstractmethod
    async def list_classifications(
            self,
            *,
            parent_id: UUID | None = None,
            keyword: str | None = None,
            is_active: bool | None = None,
    ) -> list[Classification]:
        """
        按条件列出分类。

        parent_id 不传表示**不过滤**（全部层级的节点都在结果里），
        不是"只取一级分类"。要拿某个节点的直接子节点就传它的 id。
        """

        raise NotImplementedError

    @abstractmethod
    async def create_classification(
            self,
            request: ClassificationCreate,
    ) -> Classification:
        """
        Raises:
            ConflictError: code 已被占用；或 parent_id 不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_classification(
            self,
            classification_id: UUID,
    ) -> Classification:
        """
        Raises:
            NotFoundError: 分类不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def update_classification(
            self,
            classification_id: UUID,
            request: ClassificationUpdate,
    ) -> Classification:
        """
        局部更新。code 不可改 —— 它是稳定标识。

        Raises:
            NotFoundError: 分类不存在。
            ConflictError: 想把自己挂到自己或自己的后代下面（成环）。
        """

        raise NotImplementedError

    # ========================================================
    # 工具 <-> 分类
    # ========================================================

    @abstractmethod
    async def get_tool_classifications(
            self,
            tool_id: UUID,
    ) -> list[Classification]:
        """
        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_tool_classifications(
            self,
            tool_id: UUID,
            request: ToolClassificationUpdate,
    ) -> list[Classification]:
        """
        整体替换一个 Tool 的分类关联。

        Raises:
            NotFoundError: tool_id 或某个 classification_id 不存在。
        """

        raise NotImplementedError
