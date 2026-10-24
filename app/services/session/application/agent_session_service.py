"""
会话的读写契约。

对应 session.yaml：
    POST   /sessions                  建会话
    GET    /sessions                  列本人的会话
    GET    /sessions/{session_id}     查会话
    DELETE /sessions/{session_id}     软删（归档）

⚠️ 删除是**软删**
--------------
DELETE 只把 `status` 置为 archived，不物理删除行，也不清理 checkpoint /
消息 / 记忆索引。归档后：

    列表默认不再返回它（要 status=archived 才看得到）
    按 id 仍能读到（status 是 archived）
    不能再开新运行（AgentRunService.create_run 的 409）

归属口径：
    列表 / 删除   严格 `user_id == 当前用户`
    按 id 读      本人或 NULL（存量公共会话，见仓储 get 的说明）

存量 NULL 会话因此不在任何人的列表里、也不能被任何用户删除；它们仍可按
id 读。这是刻意的取舍 —— 无主会话不属于任何一个人。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    CreateSessionRequest,
    Session,
    SessionPage,
    SessionStatus,
)

__all__ = ["AgentSessionService"]


class AgentSessionService(BaseService):

    @abstractmethod
    async def create_session(
            self,
            request: CreateSessionRequest,
            *,
            user_id: str | None,
            org_id: str | None = None,
    ) -> Session:
        """
        建一个会话，初始状态 active。

        user_id 是会话**归属**（Casdoor 的 sub），由 api 层从 access token
        取 —— 刻意不从请求体来，见 domain/schemas/session.py 的说明。
        允许 None（token 里没有 sub 时归一成 NULL = 公共，见 api 层）。

        org_id 是会话所属的**组织**（JWT 的 org claim），同样由 api 层从
        token 取。它与 user_id 有两处不同，别把两者当成一回事：

            它不是过滤键   —— 可见性只由 user_id 决定（见仓储的 get）
            空置是正常的   —— token 里没有组织信息就存 NULL，
                              Casdoor 各版本 claim 配置不一

        Raises:
            ValueError: agent_id 不合规（由 DTO 校验，见 CreateSessionRequest）。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_session(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> Session:
        """
        Raises:
            NotFoundError: 会话不存在，**或它不属于 user_id** ——
                           两者是同一个响应，不泄露 id 是否存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_sessions(
            self,
            *,
            user_id: str,
            status: SessionStatus | None = SessionStatus.ACTIVE,
            agent_id: str | None = None,
            updated_after: datetime | None = None,
            updated_before: datetime | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> SessionPage:
        """
        列**本人**的会话，按最后活动时间倒序。

        Args:
            status: 默认 active —— 软删后的会话要显式传 archived 才出现。
                    传 None 表示不过滤（两个状态一起返回；接口层默认不透传
                    None）。
            agent_id: Agent 类型精确过滤。
            updated_after / updated_before: 闭区间的最后活动时间。
                    **必须带时区**；naive datetime 是 400 而不是“按服务器
                    时区猜一个”。
            page / page_size: 页码从 1 开始。

        ⚠️ 空 user_id 返回空页，不查库
        ---------------------------
        没有身份的人没有“我的会话”。空串与 `user_id = ''` 比较虽然也什么都
        匹配不到，但显式短路更清楚地表达了这条语义，也省掉一次必然为空的
        查询。

        Raises:
            ValueError: 时间不带时区，或 updated_after 晚于 updated_before。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_session(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> None:
        """
        软删（归档）一个会话，并先结束它的未终态运行。

            会话 status -> archived
            queued / running    有进程内任务则请求取消（异步）
            waiting_input 等    事务内直接记为 cancelled

        ⚠️ 只允许删除**严格本人**的会话
        ---------------------------
        别人的、以及 `user_id` 为 NULL 的存量公共会话都抛
        `NotFoundError`（-> 404）。不泄露 id 是否存在。

        ⚠️ 已归档时幂等
        ------------
        直接返回，不推进 `updated_at`。但**仍会扫描并结束**残留的未终态
        运行 —— 上一次删除如果在提交后钩子那一步丢失了取消请求，这次可以
        补上。

        ⚠️ 取消正在执行的运行是异步的
        --------------------------
        返回只表示归档已提交、取消已请求；图要在下一个 await 点才会停下。
        与 `POST /runs/{run_id}/cancel` 的“受理不等于完成”一致。

        Raises:
            NotFoundError: 会话不存在、不属于 user_id，或是 NULL 存量会话。
        """

        raise NotImplementedError
