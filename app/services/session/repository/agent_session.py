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
from datetime import datetime
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
        默认不过滤意味着"谁忘了传就静默把别人的会话读出来"。本仓储读会话
        有几种口径，各自名字不同，别混用：

            get / get_for_update        本人或 NULL（按 id 读）
            get_any                     后台无身份读，不做过滤
            get_owned_for_update        严格本人 + 行锁（删除）
            list_page / archive         严格本人（列表 / 归档）

        将来若再出现"没有用户"的读，**另加一个名字不同的方法**并在那里写明
        为什么可以不过滤，不要在这里开一个默认口子。
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
    async def get_for_update(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSession | None:
        """
        按 id 取会话并**锁住这一行**，可见性与 `get` 相同（本人或 NULL）。

        谁用：`create_run` / `resume_run`。它们要在“会话是否已归档”这个
        判断上与会话删除串行化 —— 只读一次再写，归档可能恰好插在中间：

            create_run 读到 active  ->  归档提交  ->  create_run 插入运行

        结果是给一个已归档的会话开了新运行。`SELECT ... FOR UPDATE` 让两边
        在会话行上排队：先拿到锁的那一方先完成，后一方重读时看到的是新状态。

        ⚠️ 必须在事务里调用（请求级 `Service[...]` 会开）。离开事务的
        `FOR UPDATE` 没有意义，锁随语句结束就放掉了。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_owned_for_update(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSession | None:
        """
        **严格本人**地取会话并锁行，供 DELETE 使用。

        与 `get_for_update` 唯一的差别是**不含** `user_id IS NULL`：存量公共
        会话不属于任何人，因此不能被任何一个用户删除。取不到一律 None，
        调用方按 404 处理 —— 别人的、无主的、不存在的共用同一个响应。

        ⚠️ 空 user_id 不应该在这里被“宽容处理”：空串与 `user_id = ''`
        比较的结果是什么都匹配不到（写路径把空串归一成了 NULL），因此自然
        404。绝不能把空串换成 None 交给仓储 —— 那会变成 `IS NULL`，正好把
        存量公共会话变成可删除的。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_page(
            self,
            *,
            user_id: str,
            status: str | None,
            agent_id: str | None,
            updated_after: datetime | None,
            updated_before: datetime | None,
            offset: int,
            limit: int,
    ) -> tuple[list[AgentSession], int]:
        """
        取当前用户的一页会话，并返回满足条件的总数。

        ⚠️ 是**严格本人**，不是 `get` 的“本人或 NULL”
        --------------------------------------------
        列表回答的是“**我的**会话”。无主的存量会话不属于任何人，混进每个人
        的列表会让分页与 total 失去意义；而如果允许它们出现却又不允许删除，
        又会形成“看得见、删不掉”的不一致。

        ⚠️ 一个方法返回两者，而不是 list + count 两个方法
        ---------------------------------------------
        分页**总是**两个都要（否则前端画不出页码）。拆开会让每个调用方都写
        两遍筛选条件，还容易只改一处 —— 那会让 total 与 items 对不上。条件
        在本方法内部构造一次、两条语句共用，见实现里的 `_conditions`。

        Args:
            status: 会话状态精确匹配；None 表示不过滤（默认值由服务层决定，
                    仓储不替业务做“默认 active”）。
            agent_id: Agent 类型精确匹配；None 表示不过滤。
            updated_after / updated_before: 最后活动时间的**闭区间**；
                    None 表示该侧不过滤。调用方负责时区归一。
            offset / limit: 已经换算好的偏移与条数。
        """

        raise NotImplementedError

    @abstractmethod
    async def archive(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> bool:
        """
        把**本人**的会话置为 archived，并把 updated_at 推到当下。

        归属条件进 SQL（`WHERE session_id = ... AND user_id = ...`），因此
        即使服务层忘了先校验归属，这里也不会改到别人的行。返回是否改到了
        行（False = 不存在或不归本人）。

        ⚠️ 已经是 archived 时调用方应**跳过本方法**：重复归档会把
        updated_at 推前，让一条早已删除的会话跳到列表顶部。幂等判断在服务
        层做（它已经拿到了锁住的会话行）。
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
