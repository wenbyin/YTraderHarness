"""
ToolDefinitionStateService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext

from ..domain.schemas import ToolEnabledState
from ..registry.state import get_disabled_tool_cache
from ..repository.tool_definition import ToolDefinitionRepository
from ..repository.tool_definition_state import (
    ToolDefinitionStateRepository,
)
from .tool_definition_state_service import ToolDefinitionStateService

__all__ = ["ToolDefinitionStateServiceImpl"]


class ToolDefinitionStateServiceImpl(ToolDefinitionStateService):

    def __init__(
            self,
            *,
            tool_definition_repository: ToolDefinitionRepository,
            tool_definition_state_repository: (
                ToolDefinitionStateRepository
            ),
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._definitions = tool_definition_repository
        self._states = tool_definition_state_repository

    async def set_enabled(
            self,
            tool_id: UUID,
            enabled: bool,
            *,
            updated_by: str | None = None,
    ) -> ToolEnabledState:

        definition = await self._definitions.get(tool_id)

        if definition is None:
            raise NotFoundError(f"Tool 定义不存在：{tool_id}")

        if enabled:
            await self._states.enable(definition.tool_key)

        else:
            await self._states.disable(
                definition.tool_key,
                disabled_by=updated_by,
                now=datetime.now(timezone.utc),
            )

        # ⚠️ 提交后失效，不是现在。事务可能回滚 —— 现在清快照的话，
        # 一个并发请求会在回滚前重新加载旧值，看起来像"开关没生效"。
        # post_commit 由 ServiceFactory.scope() 在提交成功后执行
        # （与 document 的产物清理同一条机制）。
        self._ctx.post_commit.append(_invalidate_disabled_cache)

        return ToolEnabledState(
            tool_key=definition.tool_key,
            enabled=enabled,
        )


async def _invalidate_disabled_cache() -> None:
    """丢进程内快照；下一次工具解析才重新查库。"""

    get_disabled_tool_cache().invalidate()
