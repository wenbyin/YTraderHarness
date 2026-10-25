"""
Repository contract for tool_definition_state.

这张表只存**显式禁用**的 Tool：有行 = 禁用，无行 = 启用。
因此这里没有 update_enabled(bool)，只有 disable / enable 两个动词 ——
"恢复启用"在持久层就是删除那一行。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

from app.common.repository.base import BaseRepository

from ..domain.entities import ToolDefinitionState

__all__ = ["ToolDefinitionStateRepository"]


class ToolDefinitionStateRepository(BaseRepository):

    # ========================================================
    # Write
    # ========================================================

    @abstractmethod
    async def disable(
            self,
            tool_key: str,
            *,
            disabled_by: str | None,
            now: datetime,
    ) -> ToolDefinitionState:
        """
        写入一条禁用记录。已经禁用时更新审计字段（幂等）。

        只有真的执行"禁用"才建行 —— 启用状态不在库里留任何记录，
        这样"哪些工具被显式禁用过"一眼可查。
        """

        raise NotImplementedError

    @abstractmethod
    async def enable(
            self,
            tool_key: str,
    ) -> bool:
        """
        删除禁用记录，恢复启用。返回是否真的删掉了一行。

        没有行时返回 False —— 这是幂等的"恢复"，不是错误。
        """

        raise NotImplementedError

    # ========================================================
    # Read
    # ========================================================

    @abstractmethod
    async def list_disabled_keys(self) -> set[str]:
        """
        全量禁用 key。进程内缓存启动时调用一次（见 registry/state.py）。
        """

        raise NotImplementedError

    @abstractmethod
    async def disabled_keys_for(
            self,
            tool_keys: list[str],
    ) -> set[str]:
        """
        `tool_keys` 里被显式禁用的那些。管理接口（列表 / 详情）用它
        把 `enabled` 贴着这一页的工具算出来，而不是信进程内缓存 ——
        缓存只在运行期那条热路径上用，管理页应当看到库里的当前值。
        """

        raise NotImplementedError
