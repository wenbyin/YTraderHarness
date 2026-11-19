"""
工具展示描述符的读写契约。

对应 tool.yaml：
    PUT    /tools/{tool_id}/presentation
    DELETE /tools/{tool_id}/presentation

⚠️ 这里只决定**显示什么**
------------------------
它不决定模型能不能调用（那是 ToolDefinitionStateService），
也不控制时机 / 合并 / 消失（那些属于客户端）。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    ToolPresentation,
    ToolPresentationState,
)

__all__ = ["ToolPresentationService"]


class ToolPresentationService(BaseService):

    @abstractmethod
    async def set_presentation(
            self,
            tool_id: UUID,
            request: ToolPresentation,
            *,
            updated_by: str | None = None,
    ) -> ToolPresentationState:
        """
        写入或整份替换一个工具的展示描述符。

        **整份替换**（PUT 语义）：请求体就是完整体。
        写入即意味着「显示」—— 没有单独的 presented 布尔。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def clear_presentation(
            self,
            tool_id: UUID,
            *,
            updated_by: str | None = None,
    ) -> ToolPresentationState:
        """
        删除 presenter，回到默认隐藏。

        工具仍然可以被模型调用 —— 「不显示」不等于「下线」。
        没有 presenter 时也成功（幂等）。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError

    # 运行期读取不在这里：投影层注入直接走仓储的 list_all()
    # （见 agent_run_dispatch 的 _load_tool_presentations）。服务层
    # 只负责 HTTP 契约里的写路径。
