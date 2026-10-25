"""
Tool 运行时开关的写契约。

对应 tool.yaml：
    PATCH /tools/{tool_id}/enabled

工具查询、说明覆盖之外，这是**唯一**能改运行期行为的地方。
`tool_definition` 仍然归代码，本服务不碰它。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import ToolEnabledState

__all__ = ["ToolDefinitionStateService"]


class ToolDefinitionStateService(BaseService):

    @abstractmethod
    async def set_enabled(
            self,
            tool_id: UUID,
            enabled: bool,
            *,
            updated_by: str | None = None,
    ) -> ToolEnabledState:
        """
        显式禁用 / 恢复启用一个工具。

        语义是"有行 = 禁用"：
            enabled=False  写入一行（已经禁用则刷新审计字段）
            enabled=True   删除那一行（没有行是幂等的，不报错）

        只验证 tool_id 存在；MISSING 的工具也可以被禁用 / 恢复 ——
        那正是"代码里删了再回来"时保持开关的必要条件。

        写成功后把进程内快照置为未加载（post_commit），下一轮
        工具解析才会重新查库。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError
