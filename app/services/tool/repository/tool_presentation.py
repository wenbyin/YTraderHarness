"""
Repository contract for tool_presentation.

语义与表一致：**有行 = 显示，无行 = 不显示（默认）**。
因此没有 update_visible(bool)，只有 upsert / delete 两个动词 ——
「关闭展示」在持久层就是删除那一行。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any

from app.common.repository.base import BaseRepository

from ..domain.entities import ToolPresentation

__all__ = ["ToolPresentationRepository"]


class ToolPresentationRepository(BaseRepository):

    # ========================================================
    # Write
    # ========================================================

    @abstractmethod
    async def upsert(
            self,
            tool_key: str,
            *,
            renderer: str,
            display_name: str | None,
            summary: str | None,
            show_input: bool,
            show_output: bool,
            props: dict[str, Any],
            updated_by: str | None,
            now: datetime,
    ) -> ToolPresentation:
        """
        写入或整份替换一个 presenter。

        没有行时插入；有行时**整份覆盖**（PUT 语义）。不做部分更新：
        描述符是一个整体，缺字段由服务层在调用前补默认值。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete(
            self,
            tool_key: str,
    ) -> bool:
        """
        删除 presenter，回到默认隐藏。返回是否真的删掉了一行。

        没有行时返回 False —— 幂等的「关闭」，不是错误。
        """

        raise NotImplementedError

    # ========================================================
    # Read
    # ========================================================

    @abstractmethod
    async def get(
            self,
            tool_key: str,
    ) -> ToolPresentation | None:
        """单个 presenter；没有行返回 None。"""

        raise NotImplementedError

    @abstractmethod
    async def list_all(self) -> list[ToolPresentation]:
        """
        全量 presenter。

        运行期每个 run 读一次（投影层按 run 注入快照）。表很小，
        一次查询换「投影器不碰数据库」是划算的。
        """

        raise NotImplementedError

    @abstractmethod
    async def map_for_keys(
            self,
            tool_keys: list[str],
    ) -> dict[str, ToolPresentation]:
        """
        `tool_keys` 里有 presenter 的那些。

        管理页（列表 / 详情）用它贴着这一页算 `presented`，
        而不是拉全表。
        """

        raise NotImplementedError
