"""
Repository contract for AgentSession.

只管 agent_session 一张表。

⚠️ 刻意**不**与 agent_run 合成一个契约
------------------------------------
tool 模块有过一次相反的取舍（分类树与关联表合成一个契约），理由是
"两者总是被同一个接口一起读写"。这里不成立：

    会话   长期存在，跨多次运行。写少读多，生命周期是"创建 / 归档"。
    运行   一次执行，追加型历史。写多，生命周期是"排队 / 跑 / 收尾"。

它们的读写在**接口层面**也不总是一起（查会话不带运行，开运行要读会话
存在性）—— 合成一个契约只会让两边互相牵制。所以是两个契约、两张表。

用**会话存在性**做校验的那一步在服务层（run_service_impl.py），
它注入的是本契约而不是直接查表。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import AgentSession

__all__ = ["AgentSessionRepository"]


class AgentSessionRepository(BaseRepository):

    @abstractmethod
    async def add(
            self,
            session: AgentSession,
    ) -> AgentSession:
        """新建一个会话。主键冲突由调用方先查过。"""

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSession | None:
        """
        按 id 取会话，**并且**要求它对 `user_id` 可见。

            可见 = 会话的 user_id 等于它，或会话的 user_id 是 NULL

        后者是存量语义：加 user_id 那一列之前的会话对**所有**登录用户
        可见（见 models.sql 末尾）。取不到一律返回 None，调用方据此抛
        NotFoundError —— "不存在"与"不属于你"共用同一个响应，不泄露
        某个 id 是否存在。

        ⚠️ user_id 是**必填**，不是"None 表示不过滤"
        ----------------------------------------
        默认不过滤意味着"谁忘了传就静默把别人的会话读出来"，而本仓储
        是读会话的**唯一**入口（三个调用点全在 HTTP 路径上）。将来若真
        出现"没有用户"的读（后台派发今天完全不读会话），**另加一个名字
        不同的方法**（如 get_any）并在那里写明为什么可以不过滤，
        不要在这里开一个默认口子。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_any(
            self,
            session_id: UUID,
    ) -> AgentSession | None:
        """
        按 id 取会话，**不做可见性过滤**。

        ⚠️ 它是给后台路径开的，不是"少传一个参数的 get"
        ------------------------------------------
        `get` 的 docstring 已经写明了这个口径："将来若真出现『没有用户』的
        读（后台派发今天完全不读会话），**另加一个名字不同的方法**（如
        get_any）并在那里写明为什么可以不过滤，不要在那里开一个默认口子。"

        现在真出现了：`memory.extract` 任务给索引行回填 `user_id` /
        `org_id` 两列，而它跑在**调度器的后台 worker** 里 —— 那一刻没有
        请求、没有 token，因此没有"当前用户"可以与会话归属比对。它需要
        的是**这一行的归属值**，不是"这个人能不能看"。

        可以不过滤的理由只有一条，而且必须成立：**读到的值被原样写进
        索引行，而索引行自己带 user_id 过滤**。也就是说这里不做鉴权不会
        让任何数据变得可见 —— 可见性在检索那侧由 `user_id = 调用者
        OR IS NULL` 决定。哪天调用方把读到的会话**内容**返回给某个用户，
        这个理由就不成立了，那时必须换回 `get`。

        Raises:
            无 —— 取不到返回 None（会话可能已经不存在，见调用方对
            "会话没了"的处理：宁可整轮不索引，也不写出无归属的行）。
        """

        raise NotImplementedError

    @abstractmethod
    async def touch(
            self,
            session_id: UUID,
    ) -> None:
        """
        把会话的 updated_at 推到当下 —— 「这个会话刚刚有过动静」。

        ⚠️ 为什么需要一个专门的方法，不让列的 onupdate 自己管
        --------------------------------------------------
        `onupdate=func.now()` 只在**真的改了这一行的某个字段**时才生效，
        而"往会话里发一条消息"改的是 agent_message 那张表 ——
        agent_session 这一行一个字都没变。不显式碰一下，会话的 updated_at
        会永远停在创建（或首次生成 thread_id）的那一刻，"按最后活动排序"
        读到的就全是错的答案。

        没有任何别的字段要改，所以这里是**故意显式**写 updated_at 的：
        没有字段变动，onupdate 就没有可依附的 UPDATE 语句。

        ⚠️ 会话不存在时**静默无操作**，不抛异常
        -------------------------------------
        调用方是"写消息"那条路径，而此刻运行行与消息都写好了。为一个
        时间戳把整个请求打成 500 不值得 —— 那会让"用户发的那句话"跟着
        一起失败。实际也几乎不可能发生：调用方在同一事务里刚读过这个
        会话（create_run 开头的存在性校验）。
        """

        raise NotImplementedError

    @abstractmethod
    async def ensure_thread_id(
            self,
            session_id: UUID,
            candidate: str,
    ) -> str:
        """
        取得该会话**生效的** thread_id，必要时把 `candidate` 写进去。

            已有值  -> 原样返回（candidate 被丢弃）
            是 NULL -> 把 candidate 写进去并返回

        ⚠️ 为什么不是简单的"读出来是空就赋值"
        ----------------------------------
        两个请求同时给一个还**没有** thread_id 的历史会话发运行时，
        会各自生成一个 candidate。先读后写的做法下两边都读到 NULL，
        于是各写各的 —— 结果同一个会话有了两条 thread，多轮历史被劈成
        两半，而两边看起来都成功了。

        因此写这一步必须是**条件更新**（`WHERE thread_id IS NULL`）：
        并发时只有一个能改到行，输的那个读到赢家写下的值。
        返回"生效的那个"而不是"我写成功了吗"，就是为了让调用方无论
        输赢都直接用它，不必自己再读一次（再读还可能读到旧快照）。

        Returns:
            该会话生效的 thread_id。

        Raises:
            ValueError: 会话不存在。
        """

        raise NotImplementedError
