"""
工具 ↔ 能力关联的读写契约。

对应 tool.yaml：
    GET /tools/{tool_id}/classifications
    PUT /tools/{tool_id}/classifications

主语是**工具**：读的是"这个工具关联了哪些能力"，写的是"把这个工具的
关联整体换成这一批"。因此它留在 tool 模块，而能力树本体的契约是
app/services/capability 的 CapabilityService（对应 capability.yaml）。

术语刻意保留 classification（不叫 capability）：能力本体改名了，
但这两个路径描述的是工具这一侧的关联，没有理由跟着改。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService
from app.services.capability.domain.schemas import Capability

from ..domain.schemas import ToolClassificationUpdate

__all__ = ["ToolClassificationService"]


class ToolClassificationService(BaseService):

    @abstractmethod
    async def get_tool_classifications(
            self,
            tool_id: UUID,
    ) -> list[Capability]:
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
    ) -> list[Capability]:
        """
        整体替换一个 Tool 的能力关联。

        Raises:
            NotFoundError: tool_id 或某个 classification_id 不存在。
        """

        raise NotImplementedError
