"""
MemoryAtomService —— L1（memory_atom）：把对话提炼成结构化记忆。

**没有 HTTP 端点。** 这个服务只有一条路：`memory.l1.extract` 任务的
handler。它的调用方是 Worker，不是请求（见 app/bootstrap/task.py 的
注册处）。

与 `MemoryService`（L0）分开而不是并进去
------------------------------------
两者的**输入与产物都不同**：

    L0   输入一次运行的全部消息，产物是“原样可检索”的索引行，幂等重写
    L1   输入一个会话里还没抽过的那一段，产物是 LLM 提炼的记忆，
         且要付一次（或多次）模型调用

并在一起会让 `MemoryService` 同时依赖 embedder、模型注册表、提示词，
而 L0 的检索路径一个都不需要 —— 那条路径在最热的查询上。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas.memory_atom import (
    MemoryAtomSearchRequest,
    MemoryAtomSearchResponse,
)
from ..domain.tasks import MemoryAtomExtractParams

__all__ = ["MemoryAtomService"]


class MemoryAtomService(BaseService):

    # ========================================================
    # 工具入口（暴露给主模型的那一个）
    # ========================================================

    @abstractmethod
    async def search_atom(
            self,
            request: MemoryAtomSearchRequest,
    ) -> MemoryAtomSearchResponse:
        """
        暴露给主模型的工具 `memory_atom.search`。

        与 `search` 的区别**只在身份怎么来**：这个只收一个模型参数（工具
        基建要求的形状，见 tool/registry/schema.py 的 `resolve_request_model`），
        身份从进程内的调用者上下文取（app/common/tool_caller.py）；
        `search` 的身份是显式入参，给 HTTP 路径用。

        Raises:
            ValueError: 拿不到调用者身份，或 scope=session 而这次调用不挂在
                        任何会话上。两种情况都**拒绝**而不是放行 —— 记忆检索
                        必须知道以谁的身份搜。
        """

        raise NotImplementedError

    # ========================================================
    # 检索（HTTP 路径与工具共用的实现）
    # ========================================================

    @abstractmethod
    async def search(
            self,
            request: MemoryAtomSearchRequest,
            *,
            session_id: UUID | None = None,
            user_id: str | None = None,
    ) -> MemoryAtomSearchResponse:
        """
        一轮记忆检索（关键词 + 向量，按名次融合）。

        **不是工具** —— 它是 HTTP 路径（`POST /memory-atom/search`）与上面
        那个工具包装共用的实现。身份是显式入参，因为两条入口确认身份的方式
        不同：HTTP 从 token，工具从运行时上下文。
        """

        raise NotImplementedError


    @abstractmethod
    async def memory_atom_extract(
            self,
            params: MemoryAtomExtractParams,
            ctx: Any,
    ) -> None:
        """
        `memory.l1.extract` 的 handler 主体。

        ⚠️ 签名必须是 `(params, ctx)` 两个位置参数 —— 注册期会校验，
        少了它在**启动时**报错（见 app/infra/task/registry.py）。不需要
        ctx 也要声明它（将来去重那一步会用到它派生子任务）。

        这一步做完什么
        -----------
            按水位取增量消息 -> 渲染提示词 -> 调模型 -> 解析 JSON
            -> 写 memory_atom + 两张挂载表 -> **推进水位**

        什么算成功、什么算失败
        ------------------
            没有增量消息        正常结束（那一段已经被别人抽走了）
            模型调用抛          抛出去，让任务重试
            输出解析不了        抛出去，让任务重试
            写库失败            抛出去，让任务重试（upsert 幂等）
            一切成功            推水位，提交

        ⚠️ 水位**只在最后推**，且与记录同事务 —— 提前推会让那一段消息
        永久跳过，而它没有任何症状。
        """

        raise NotImplementedError
