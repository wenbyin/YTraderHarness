"""
挂在 Tool 上的处理器链的读写契约。

对应 tool_processor.yaml：
    GET    /tools/{tool_id}/processors
    POST   /tools/{tool_id}/processors
    PATCH  /tools/{tool_id}/processors/{processor_id}
    DELETE /tools/{tool_id}/processors/{processor_id}
    PUT    /tools/{tool_id}/processors
    PUT    /tools/{tool_id}/processors/order

一个 Tool 一条链
----------------
链就是该 Tool 在 tool_processor 表里的行，按 sequence 排序。
顺序即语义 —— 谁先谁后决定执行顺序，因此每个写方法都要维护
"写完之后 sequence 恰好是 1..N"这个不变式（见仓储的说明）。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    CreateToolProcessorRequest,
    ReorderToolProcessorsRequest,
    ReplaceToolProcessorChainRequest,
    ToolProcessor,
    ToolProcessorChain,
    UpdateToolProcessorRequest,
)

__all__ = ["ToolProcessorService"]


class ToolProcessorService(BaseService):

    # ========================================================
    # 读
    # ========================================================

    @abstractmethod
    async def list_chain(
            self,
            tool_id: UUID,
            *,
            enabled_only: bool = False,
    ) -> ToolProcessorChain:
        """
        取一条链。

        enabled_only=True 时只返回启用的行（给"执行时真正会跑的那
        几条"用）；管理面一律用 False —— 关掉的行也要看得见，
        否则人会以为它被删了。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError

    # ========================================================
    # 写
    # ========================================================

    @abstractmethod
    async def add_processor(
            self,
            tool_id: UUID,
            request: CreateToolProcessorRequest,
    ) -> ToolProcessor:
        """
        挂一条记录。

        request.sequence 只是**位置提示**（插到"所有 sequence 小于它"
        的行之后），落库前整条链会重编号成 1..N。

        Raises:
            NotFoundError: tool 不存在；或 processor_key 没有注册实现。
            ConflictError: hook / execution_mode 与实现不相容，
                或该插入会让 AROUND 的区间交叉（洋葱模型）。
            ValidationError: config 不满足实现的 config_model（→400）。
        """

        raise NotImplementedError

    @abstractmethod
    async def update_processor(
            self,
            tool_id: UUID,
            processor_id: UUID,
            request: UpdateToolProcessorRequest,
    ) -> ToolProcessor:
        """
        改一条记录（不改 processor_key，也不改位置）。

        改了 hook / execution_mode 之后会重新跑一遍相容性校验 ——
        否则一条合法的 AROUND+SYNC 被改成 ASYNC 就能绕过挂载校验，
        然后在执行期撞上 DB 的 CHECK。

        Raises:
            NotFoundError: tool 不存在；或该行不存在。
            ConflictError: 改完与实现不相容，或洋葱模型被破坏。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_processor(
            self,
            tool_id: UUID,
            processor_id: UUID,
    ) -> None:
        """
        摘掉一条记录，剩下的重编号成 1..N（补洞）。

        Raises:
            NotFoundError: tool 不存在；或该行不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_chain(
            self,
            tool_id: UUID,
            request: ReplaceToolProcessorChainRequest,
    ) -> ToolProcessorChain:
        """
        整体替换一条链。数组顺序即执行顺序，允许空数组（= 全摘掉）。

        ⚠️ 行是**全删重建**的（processor_id 全部重新生成）。

        Raises:
            NotFoundError: tool 不存在；或某个 processor_key 没注册。
            ConflictError: 某个 hook / mode 不相容，或洋葱模型被破坏。
            无 ValidationError 之外的 400：config 由 pydantic 模型校验。
        """

        raise NotImplementedError

    @abstractmethod
    async def reorder(
            self,
            tool_id: UUID,
            request: ReorderToolProcessorsRequest,
    ) -> ToolProcessorChain:
        """
        按给定顺序重排整条链。

        request.processor_ids 必须是现有行的**全集**（含 disabled）。

        Raises:
            NotFoundError: 该 tool_id 不存在。
            ConflictError: 给的 ID 集合与现有行不一致（链已过期），
                或重排后洋葱模型被破坏。
        """

        raise NotImplementedError
