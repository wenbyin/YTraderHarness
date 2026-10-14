"""
MemoryService —— L0 记忆的检索与索引。

对应 memory.yaml：POST /memory/search。
另外还有一条**不是 HTTP 端点**的方法（`extract`）：它是
`memory.l0.extract` 任务的 handler（方法名 message_extract）。

⚠️ 身份是**显式入参**，不是从请求体里读的
------------------------------------
`search` 的 `session_id` / `user_id` 是关键字参数，而它们**不在**
`MemorySearchRequest` 里。这个请求模型同时是暴露给模型的工具入参
schema（见 domain/schemas/memory.py），身份一旦进去就是一条越权路径。

所以分工是：请求模型说"搜什么"，调用方（HTTP 层从 token、工具层从运行时）
说"以谁的身份搜"。服务把两者拼成 `MemoryQueryScope` —— 拼装是**唯一**
一处同时看到两边的地方，因此越权与否在这一个函数里就能看全。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import MemorySearchRequest, MemorySearchResponse
from ..domain.tasks import MemoryExtractParams

__all__ = ["MemoryService"]


class MemoryService(BaseService):

    # ========================================================
    # 检索
    # ========================================================

    # ========================================================
    # 工具入口
    # ========================================================

    @abstractmethod
    async def search_memory(
            self,
            request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        """
        暴露给主模型的工具 `memory.search`。

        与 `search` 的区别**只在身份怎么来**：这个只收一个模型参数（工具
        基建要求的形状，见 tool/registry/schema.py 的
        `resolve_request_model`），身份从进程内的调用者上下文取
        （app/common/tool_caller.py）；`search` 的身份是显式入参，给 HTTP
        路径用。

        声明在契约上（而不是只在实现里）是与 document 的
        `retrieve_documents` 同一条约定：**工具是服务对外的一部分**，
        不写在这里就等于说它是个内部方法。

        Raises:
            ValueError: 拿不到调用者身份，或 scope=session 而这次调用不
                        挂在任何会话上。两种情况都**拒绝**而不是放行 ——
                        记忆检索必须知道以谁的身份搜。
        """

        raise NotImplementedError

    # ========================================================
    # 检索（HTTP 路径与工具共用的实现）
    # ========================================================

    @abstractmethod
    async def search(
            self,
            request: MemorySearchRequest,
            *,
            session_id: UUID | None = None,
            user_id: str | None = None,
    ) -> MemorySearchResponse:
        """
        一轮混合召回。

        Parameters
        ----------
        session_id
            `request.scope == SESSION` 时**必填**；它由调用方给
            （HTTP 层取 query 参数，工具层由运行时注入）。给了就要校验
            归属 —— 不属于 `user_id` 的会话是 NotFoundError，与"这个会话
            不存在"同一个响应（不泄露某个 id 是否存在）。
        user_id
            调用者身份。None 表示"没有身份"，此时**只能搜到 user_id 为
            NULL 的存量行**（过滤规则是 `= 调用者 OR IS NULL`，两边与
            session 模块一致）。它不是一个"不过滤"的口子。

        Raises:
            ValueError: scope=SESSION 但没给 session_id，或者
                        scope=USER 却给了 session_id（两种都是调用方
                        用错了这个接口，越早报越好）。
            NotFoundError: 会话不存在，或不属于 user_id。
        """

        raise NotImplementedError

    # ========================================================
    # 索引（后台任务）
    # ========================================================

    @abstractmethod
    async def message_extract(
            self,
            params: MemoryExtractParams,
            ctx: Any,
    ) -> None:
        """
        `memory.l0.extract` 任务的 handler（方法名 message_extract）：把一次运行产出的消息写进两张索引表。

        签名 `(params, ctx)` 是任务基建要求的（见
        app/infra/task/registry.py 的 `_assert_signature`），`ctx` 是
        `TaskContext` —— 本方法**不需要**派生新任务，所以用不到它。

        ⚠️ 必须是**幂等**的
        -----------------
        至少一次投递意味着重复执行是常态（提交后投递失败会由 relay 补投、
        租约过期会被判"执行者已死"而重跑）。两处保证幂等：

            主键    message_id，仓储走 ON CONFLICT DO UPDATE
            读入    `list_by_run(run_id)` 取的是**那一轮的全部**消息，
                    重跑写出同一批行、覆盖同一批值

        ⚠️ 不吞异常
        ---------
        失败要抛出去，让重试策略接手（base_delay 5s、倍投、最多 3 次）。
        把异常吞掉会让"这次没索引上"变成一件**没人知道**的事 —— 而它的
        后果是长期记忆里永久缺一轮对话。

        只有一种情况是**正常返回**：那一轮没有任何值得索引的消息
        （纯工具调用、模型返回空串、运行连一条消息都没产出）。那不是失败，
        没有数据就是没有数据。
        """

        raise NotImplementedError
