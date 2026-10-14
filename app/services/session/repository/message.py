"""
Repository contract for AgentMessage.

只管 agent_message 一张表。与 AgentSessionRepository / AgentRunRepository 分开
的理由同那两者：读写形状不同（消息是纯追加型的不可变历史），合成一个契约只会
让三方互相牵制。

⚠️ 与另外两个契约不同的地方，都是刻意的
------------------------------------
1. **写是批量的**（`add_many`），没有单条的 `add`。
   两个写入点都是"一次性写下若干条"：建运行写 1 条用户消息，派发 handler 写
   本轮产出的全部 assistant 消息（0..N 条）。给单条再开一个方法等于让 seq
   分配与插入的逻辑出现两份。

2. **seq 由调用方分配，用 `next_seq` 取号**。
   它必须在**同一个事务**里与 `add_many` 一起调用，否则并发下会取到同一个号
   —— 表上的 `uq_agent_message_session_seq` 会拦下来（不会静默写坏），但报出来的
   是一次唯一键冲突。串行化靠 thread 锁，见
   application/agent_run_service_impl.py 与 agent_run_dispatch.py 的写入点。

   "取号"不能内联进 add_many：实体的 seq 是必填字段（它落的是 NOT NULL 列，
   契约里也是必填），调用方要先有号才能造出实体。

3. **翻页返回的是三元组**，不是仓库别处的 `(items, total)`。
   keyset 分页给不出 total（见 domain/schemas/agent_message.py 的说明），代替它的是
   两个"还能不能往那个方向翻"的布尔量。
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import AgentMessage

__all__ = ["MessageRepository"]


class MessageRepository(BaseRepository):

    @abstractmethod
    async def next_seq(
            self,
            session_id: UUID,
    ) -> int:
        """
        该会话下一个可用的 seq（`COALESCE(MAX(seq), 0) + 1`）。

        ⚠️ 必须与紧随其后的 `add_many` 在**同一个事务**里调用。
        分两次事务时两个写入点可能取到同一个号 —— 会被唯一约束拦下，而
        不是静默写坏，但那是一次本可避免的 500。

        注意"下一个"不等于"上一条+1"：中间可能有空洞（写入失败）。这不影响
        任何调用方 —— 翻页按 keyset 走，谁都不靠连续性。
        """

        raise NotImplementedError

    @abstractmethod
    async def add_many(
            self,
            messages: Sequence[AgentMessage],
    ) -> list[AgentMessage]:
        """
        批量写入。调用方负责填好 seq（见 `next_seq`）。

        空序列是合法的 no-op：一次运行完全没产出消息（图一进来就失败）时，
        调用方不必自己判空。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_page(
            self,
            session_id: UUID,
            *,
            limit: int,
            before_seq: int | None = None,
            after_seq: int | None = None,
    ) -> tuple[list[AgentMessage], bool, bool]:
        """
        取一页历史。返回 `(messages, has_more_before, has_more_after)`，
        messages **按 seq 升序**（旧 -> 新）。

        三个入参组合的语义（`before_seq` 与 `after_seq` 不同时给，
        同时给由服务层拦成 400，走不到这里）：

            都不给        最新的 limit 条
            before_seq    它**之前**（更旧）的 limit 条
            after_seq     它**之后**（更新）的 limit 条

        两个标志的含义就是字面意思：还存在更旧 / 更新的行。
        空页时两个都是 False —— 客户端据此收手，而不是拿着空游标接着翻。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_by_run(
            self,
            run_id: UUID,
    ) -> list[AgentMessage]:
        """
        某次运行写下的**全部**消息，按 seq 升序。

        ⚠️ 为什么没有 limit
        ----------------
        一轮产出多少条是运行自己的性质（模型调了几次工具就几条），不是
        调用方能裁的量。记忆索引要的是"这一轮的全部"，截断它会**静默地**
        漏掉对话的尾巴 —— 漏掉的部分不留任何迹象。

        ⚠️ 为什么加在这里而不是让别的模块自己查 agent_message
        --------------------------------------------------
        agent_message 的归属是 session 模块。memory 模块要按 run 取消息
        （记忆索引按 run 重跑），走这个契约而不是自己去 join 那张表，
        是为了让"消息长什么样、有多少条"只有一个出口。

        走 `idx_agent_message_run_seq`（部分索引，`run_id IS NOT NULL`）
        —— 这条查询形状就是那个索引存在的理由。

        一次运行没写过任何消息（图一进来就失败、或输入在进图之前就被
        拒了）时返回**空列表**，不是 None：调用方问的是"这次运行产出了
        哪些消息"，"一条都没有"是一个正当答案。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_by_ids(
            self,
            message_ids: Sequence[UUID],
            *,
            session_id: UUID,
    ) -> list[AgentMessage]:
        """
        按 id 批量取消息，按 seq 升序。取不到的不在返回里（不是错误）。

        ⚠️ 这是**按 id 反查**的唯一入口，`session_id` 是必填的收窄
        --------------------------------------------------
        调用方是记忆溯源（`memory.query.l1_sources`：一条 L1 记忆的
        `source_message_ids` 指向哪几句原话）。那些 id 来自 **LLM 的输出**
        并落进 `memory_atom.source_message_ids`，是**不可信输入** ——
        与 `MemoryAtomRepository.delete_batch` 面对的是同一类数据。带上
        会话之后，一个幻觉出来的 id 至多在这个会话里查一遍，拿不到别的
        会话的消息。

        它同时走 `idx_agent_message_session_seq`（session_id 等值 + seq
        排序），所以收窄不是"为了安全白付一次过滤"。

        ⚠️ 取不到的那些**不报错**
        ----------------------
        `source_message_ids` 上没有外键、也不保证消息还在（见那个列的
        注释）。调用方（溯源服务）据此把差集作为 `missing_message_ids`
        回报 —— "记得有 5 条来源、只找到 3 条"必须是看得见的事实，而不是
        静默地少几条。
        """

        raise NotImplementedError

    @abstractmethod
    async def browse(
            self,
            *,
            user_id: str,
            session_id: UUID | None,
            roles: Sequence[str] | None,
            created_from: datetime | None,
            created_to: datetime | None,
            text_contains: str | None,
            order: str,
            limit: int,
            after: Sequence[Any] | None,
    ) -> tuple[list[AgentMessage], bool]:
        """
        管理端浏览：按 scope + 过滤条件取**一页**原始消息，返回 `(items, has_more)`。

        给 `memory.query.l0` 用。与 `list_page`（按会话翻历史，给会话视图用）
        是两个方法，因为**作用域不同**：`list_page` 必须给 `session_id`
        （它是"这个会话的历史"），而这里可以只给 `user_id` 跨会话查 ——
        `agent_message` 上没有 `user_id` 列，归属在**会话**上，所以跨会话
        那条路要 join `agent_session`。

        ⚠️ 可见性规则与会话读取**逐字一致**
        -------------------------------
        `agent_session.user_id == :uid OR agent_session.user_id IS NULL`。
        后者是加那一列之前的存量语义（见 `AgentSessionRepository.get`）。
        浏览另写一条规则的话，"这个会话读得到、它的消息却不在列表里"会
        变成一件查不出原因的事。

        ⚠️ `text_contains` 只匹配**文本块**
        -----------------------------
        它等价于 `_assembly.message_text`（memory 模块那一个"消息正文怎么
        拼"的定义：只取 `type=text` 的块）在 SQL 里的写法。**不要**改成
        `content::text ILIKE`：那样 JSON 的**键名**也会被匹配 —— 搜 "text"
        会命中每一条消息，而它不报错。

        代价说清楚：它是那份 Python 定义的**镜像**，改一处必须改两处
        （镜像换来的是"过滤条件与返回的正文是同一件事"，且不必让
        session 模块去 import memory 模块的索引表）。

        ⚠️ `order` 决定 `after` 的形状
        ------------------------
            created_at_desc / created_at_asc    after = (created_at, message_id)
            session_seq_asc / session_seq_desc  after = (session_id, seq)

        `created_at` 在同一事务里写的多条消息上**完全相同**，所以必须带上
        `message_id` 做 tie-breaker，否则翻页会在边界上重复或漏掉一整批。
        `session_seq_*` 的第二项是 `seq`（会话内唯一，与 `session_id` 合起来
        就是唯一键）。

        ⚠️ 不返回 total
        ------------
        游标分页给不出总数，而且调用方要的是"还要不要继续翻"（`has_more`）。
        """

        raise NotImplementedError

    @abstractmethod
    async def user_message_id(
            self,
            run_id: UUID,
    ) -> UUID | None:
        """
        某次运行写下的那条用户消息的 id。

        给 assistant 消息回填 `reply_to_message_id` 用。取不到时返回 None
        而不是抛：那条 user 行由**另一个事务**写下（建运行的请求事务），
        它不在的唯一合理解释是"这次运行是被直接喂进库的"（冒烟测试、
        将来的补跑工具），那时回复关联本就无从谈起。
        """

        raise NotImplementedError

    @abstractmethod
    async def count_user_turns(
            self,
            session_id: UUID,
            *,
            after_seq: int | None = None,
    ) -> int:
        """
        该会话的**用户轮次条数**（`role='user'` 且 `kind='chat'`）。

        `after_seq` 给定时只数**那条消息之后**的（`seq > after_seq`）；
        不给时是整个会话的生命周期计数。

        ⚠️ 为什么"轮次"就等于"用户消息条数"
        ------------------------------
        本仓库对"一轮"的定义只有一条，写在 context builder 的
        `split_turns` 里：**每条 Human 消息开始一轮，直到下一条 Human
        之前**。所以数用户消息就是数轮 —— assistant 的旁白与工具活动都
        归进当前这一轮，不额外产生轮。

        ⚠️ 为什么只算 kind='chat'
        ----------------------
        kind 还可能是 interaction / notice / tool_activity，那些不是用户
        说的话。L0 的索引与 L1 的抽取都只收 chat，这个计数必须与它们**同
        档** —— 否则"每 N 轮抽一次"与"抽的是哪几条消息"会按两个集合算，
        而两者不一致不会有任何报错。

        ⚠️ `after_seq` 是**过滤条件**，不是拿来相减的数
        -------------------------------------------
        两种调用方的形状要分清：

            count_user_turns(sid)                 生命周期计数（诊断 / 展示）
            count_user_turns(sid, after_seq=X)    "X 之后还有几轮"

        L1 的触发要的是后者（自上次抽取以来攒了几轮 —— 见
        `trigger.submit_atom_extract`）。**别把它写成 `总数 - 水位`**：水位
        是 `agent_message.seq`（每一条消息都占号，user / assistant /
        tool_activity 都在内），而这里是**用户消息的条数**，两者单位不同。
        相减的结果会随"assistant 有多啰嗦"浮动 —— 5 轮阈值静默变成 20 轮，
        而且不报错。

        ⚠️ 取值必须**精确**（不能像 `_exists` 那样 LIMIT 1）—— 见实现里的
        说明：调用方拿它跟阈值比，少一条就整体平移。
        """

        raise NotImplementedError

    @abstractmethod
    async def count_user_turns_in_scope(
            self,
            *,
            agent_id: str,
            user_id: str | None,
            org_id: str | None,
            since: datetime,
            default_agent_id: str,
    ) -> int:
        """
        **一个 scope（跨会话）**自 `since` 以来的用户轮次条数。

        ⚠️ 与 `count_user_turns` 是两个问题，别互相替代
        ----------------------------------------
            count_user_turns(sid, ...)              **一个会话**里几轮
            count_user_turns_in_scope(scope, ...)   **一组会话**里几轮

        L1 与 L2 要的是前者（它们的粒度是会话）；**L3 要的是后者** ——
        画像按 scope 聚合、被该 scope 下所有会话共同书写，所以"距上次生成
        攒了多少轮"必须跨会话数。只数触发它的那一个会话的话，一个开很多
        短会话的用户永远到不了阈值（每次 L2 完成时那个会话都只有几轮），
        于是那条触发条件**永远不会成立**，而且不报错。

        ⚠️ `agent_id` 匹配的是**消息上的 duty_key**，不是会话的 agent_id
        ----------------------------------------------------
        这是本方法最容易写错的一处。scope 的 `agent_id` 来自
        `SceneScope.resolve`，它是 **duty_key**：

            memory_atom.business_agent_id  ->  agent_message.business_agent_id
              -> 兜底：agent_session.agent_id  ->  兜底：DEFAULT_AGENT_ID

        而 `agent_session.agent_id` 是**另一个列**（更粗的角色标识，本仓库
        还没接线）。所以这里匹配的是 `agent_message.business_agent_id`，
        并且**照抄那条兜底链**（`coalesce(business_agent_id, session.agent_id,
        :default_agent_id)`）—— 否则 `business_agent_id` 为 NULL 的存量消息
        会把整个计数算成 0，而症状是"画像永远不更新"。

        ⚠️ 个人 scope 要求会话**没有 org**
        ------------------------------
        `SceneScope.resolve` 的规则是"有 org 就丢掉 user"：带组织的会话
        聚合出的是**团队**画像。所以个人 scope 的计数必须排除带 org 的会话
        —— 否则同一个人的团队会话轮次会漏进他的个人画像计数里，症状是
        "个人画像更新得莫名其妙地勤"。

        ⚠️ **空串按 NULL 处理**
        -------------------
        `agent_session.org_id` 是 JWT 的 claim，历史数据里出现过空串（与
        NULL 是两种不同的"没有"）。`SceneScope.resolve` 已经把它归一成
        NULL 了，这里必须跟上 —— 少这一句，"有 org" 的判定就与 scope 的
        口径不一致，而那一批会话会被两边各算一次或都不算。

        ⚠️ 为什么必须 join（而不是直接查 agent_message）
        ----------------------------------------
        `agent_message` 上**没有** user_id / org_id（见 memory.yaml 里那
        段说明），归属只在 `agent_session` 上。两张表同属 session 模块，
        所以这是一次模块内的 join。

        ⚠️ 取值必须**精确**（同 `count_user_turns`）。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_chat_after(
            self,
            session_id: UUID,
            seq: int,
    ) -> list[AgentMessage]:
        """
        该会话里 `seq` **之后**的全部 `kind='chat'` 消息，按 seq 升序。

        这是 L1（memory_atom）抽取的输入形状：水位
        （`memory_atom_cursor.last_seq`）是它的入参，取出来的这批就是"还没抽过
        的那一段"。

        ⚠️ 为什么没有 limit
        ----------------
        与 `list_by_run` 同一条理由：一次要塞进去多少条是数据的性质（用户
        一口气说了很多、或上次抽取失败积压了），不是调用方能裁的量。截断
        它会**静默地**漏掉对话的尾巴 —— 漏掉的部分不留任何迹象。

        ⚠️ 为什么只收 kind='chat'
        ---------------------
        与 L0 的索引、以及 `count_user_turns` **同档** —— 工具结果与
        assistant 的动手旁白属于 trace，不属于对话。三处口径必须一致，
        否则"抽了几轮"与"抽的是哪几条"会按两个集合算，而两者不一致不会
        报错。

        ⚠️ 不过滤 role
        -----------
        用户与 assistant 都要：记忆的主体既可能是"用户喜欢 X"，也可能是
        "AI 答应过 Y"。哪个消息能提取出什么由提示词判（它要求主体是
        「用户（姓名）」或「AI」），不由这里判。

        没有更多消息时返回**空列表**，不是 None —— 与 `list_by_run` 同一
        条取舍："一条都没有"是正当答案（那一段已经被别的任务抽走了）。
        """

        raise NotImplementedError
