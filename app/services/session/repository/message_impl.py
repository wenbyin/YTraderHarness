"""
PostgreSQL implementation of MessageRepository.

Transaction rule
----------------
本仓储**从不**调用 session.commit() / session.rollback()。

两个调用场景：
    请求路径（建运行写用户消息）  组合根给的请求级 session，随请求提交
    后台路径（写 assistant 消息） RepositoryFactory.scope() 自开的短事务，
                                  见 application/agent_run_dispatch.py

⚠️ 翻页只发两次查询，不是三次
--------------------------
方向那一侧用 `LIMIT limit + 1` 探一下（多出来的那一行说明还有），
反方向那一侧才需要一句 EXISTS。因此：

    无游标    1 次（反方向必然没有更新的 —— 取的就是最新那段）
    before    2 次
    after     2 次

那条 EXISTS 不是"怕客户端给个不存在的游标"才写的：游标可能真的落在
数据之外（客户端自己编的 seq、或者上一次翻页之后才有新消息），这时
"还有更新的"就不成立，而想当然地写 True 会让客户端多翻一次空页。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, and_, column, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB

from ..domain.entities import AgentMessage
from ..models.sqlalchemy.agent_message import AgentMessage as AgentMessageModel
from ..models.sqlalchemy.agent_session import AgentSession as AgentSessionModel
from .message import MessageRepository


class MessageRepositoryImpl(MessageRepository):

    async def next_seq(
            self,
            session_id: UUID,
    ) -> int:

        # COALESCE 而不是 Python 侧判 None：空会话时 MAX 回的是 NULL，
        # 拉到应用层再转成 0 会多一次往返，也让"取不到就是 1"这条规则
        # 散到两个地方。
        current = await self._session.scalar(
            select(func.max(AgentMessageModel.seq)).where(
                AgentMessageModel.session_id == session_id
            )
        )

        return (current or 0) + 1

    async def add_many(
            self,
            messages: Sequence[AgentMessage],
    ) -> list[AgentMessage]:

        if not messages:
            # 空 batch 是合法输入（一次运行什么都没产出），因此这里直接返回，
            # 而不是让 flush 去做一次无意义的往返。
            return []

        models = [
            AgentMessageModel(
                message_id=message.message_id,
                session_id=message.session_id,
                run_id=message.run_id,
                seq=message.seq,
                role=message.role,
                kind=message.kind,
                business_agent_id=message.business_agent_id,
                content=list(message.content),
                # ⚠️ 这一行漏了的症状是**静默写 NULL**：列可空、构造是逐字段
                # 的，没有任何一处会报错 —— 表现为"思维链明明有，历史里却
                # 没有"。加字段时记得它也在这张清单上。
                reasoning_content=message.reasoning_content,
                status=message.status,
                client_message_id=message.client_message_id,
                reply_to_message_id=message.reply_to_message_id,
                meta=dict(message.meta),
                created_at=message.created_at,
                updated_at=message.updated_at,
            )
            for message in messages
        ]

        self._session.add_all(models)
        await self._session.flush()

        return [self._to_entity(model) for model in models]

    async def list_page(
            self,
            session_id: UUID,
            *,
            limit: int,
            before_seq: int | None = None,
            after_seq: int | None = None,
    ) -> tuple[list[AgentMessage], bool, bool]:

        # 多取一条当探针。它不进结果，只看"有没有"。
        probe = limit + 1

        if after_seq is not None:
            # 往新翻：升序取，取到的就是客户端要的顺序，不必反转。
            rows = await self._fetch(
                session_id,
                direction="asc",
                bound=AgentMessageModel.seq > after_seq,
                limit=probe,
            )

            has_more_after = len(rows) > limit
            rows = rows[:limit]

            # 空页时两个标志都回 False（见契约）：客户端据此收手。
            has_more_before = (
                await self._exists_before(session_id, rows[0].seq)
                if rows
                else False
            )

            return rows, has_more_before, has_more_after

        # 无游标与 before 是同一个形状：都按倒序取"最新的一段"，
        # 只是多一个"必须比游标更旧"的条件。
        bound = (
            AgentMessageModel.seq < before_seq
            if before_seq is not None
            else None
        )

        rows = await self._fetch(
            session_id,
            direction="desc",
            bound=bound,
            limit=probe,
        )

        has_more_before = len(rows) > limit

        # 探针那一条（如果有）是最旧的，丢掉它 —— 它属于下一页。
        rows = rows[:limit]

        # 倒序取出来的要翻回正序：契约承诺的是"旧 -> 新"。
        rows.reverse()

        if before_seq is None:
            # 取的就是最新那段 —— 更新的必然不存在。
            # 少这一句 EXISTS 是有依据的，不是省事。
            has_more_after = False

        else:
            has_more_after = (
                await self._exists_after(session_id, rows[-1].seq)
                if rows
                else False
            )

        return rows, has_more_before, has_more_after

    async def list_by_run(
            self,
            run_id: UUID,
    ) -> list[AgentMessage]:

        # 不设 limit：一轮的条数是运行自己的性质，调用方裁不了它。
        # 排序按 seq —— 同一事务里写的多条消息时间戳可能完全相同，
        # 时间戳排不出先后（与 list_page 同一条理由）。
        models = await self._session.scalars(
            select(AgentMessageModel)
            .where(AgentMessageModel.run_id == run_id)
            .order_by(AgentMessageModel.seq.asc())
        )

        return [self._to_entity(model) for model in models]

    async def user_message_id(
            self,
            run_id: UUID,
    ) -> UUID | None:

        return await self._session.scalar(
            select(AgentMessageModel.message_id)
            .where(
                AgentMessageModel.run_id == run_id,
                AgentMessageModel.role == "user",
            )
            # 理论上一次运行只有一条用户消息。加上排序与 LIMIT 1 是为了
            # 万一有多条时结果**确定**，而不是"数据库随便挑一条"。
            .order_by(AgentMessageModel.seq)
            .limit(1)
        )

    async def count_user_turns(
            self,
            session_id: UUID,
            *,
            after_seq: int | None = None,
    ) -> int:

        # ⚠️ 这里**必须**是 COUNT(*)，不能用 _exists 那种 LIMIT 1 的探法
        # ---------------------------------------------------------------
        # 与 `_exists` 的取舍相反（那里明确写了"只问有没有，计数会扫完
        # 整段索引"）。差别在于调用方要的是**精确值**：L1 的触发拿它跟阈值
        # 比（`>= N 就抽`），少一条就整体平移 —— 而症状只是"抽得早一点/
        # 晚一点"，不会报错。
        #
        # 代价可控：扫描范围被 session_id 前缀卡住（走
        # idx_agent_message_session_seq），而它数的是**一个会话**的用户
        # 消息，量级是几十到几百行。
        conditions = _user_turn_conditions() + [
            AgentMessageModel.session_id == session_id,
        ]

        # ⚠️ `seq >` 是**过滤**，不是把水位从计数里减掉
        # ----------------------------------------
        # 两种量单位不同：`seq` 是"第几条消息"（每一条都占号），这里是
        # "用户轮次条数"。写成本模块**之外**的 `总数 - 水位` 会得到一个大
        # 得多的差额，而那正是"阈值静默变大"的成因。见契约里的说明。
        if after_seq is not None:
            conditions.append(AgentMessageModel.seq > after_seq)

        total = await self._session.scalar(
            select(func.count())
            .select_from(AgentMessageModel)
            .where(*conditions)
        )

        return int(total or 0)

    async def count_user_turns_in_scope(
            self,
            *,
            agent_id: str,
            user_id: str | None,
            org_id: str | None,
            since: datetime,
            default_agent_id: str,
    ) -> int:

        # ⚠️ 这里**必须**是 COUNT(*)：调用方拿它跟阈值比（">= 50 才生成
        # 画像"），少一条就整体平移。理由与 `count_user_turns` 那段逐字相同。
        #
        # 代价比那一个高：这是一次 **join**，而且作用在 agent_message 这张
        # 热表上（按 session_id 走 idx_agent_message_session_seq，再回
        # agent_session 的主键）。时间界 `created_at > since` 把扫描范围
        # 卡在"上次生成画像之后的那段" —— 所以它随窗口线性，不随会话总量。
        conditions = _user_turn_conditions() + [
            AgentMessageModel.created_at > since,
            # ⚠️ 兜底链必须与 `SceneScope.resolve` **逐字一致**（见契约）。
            # 写成 `== agent_id` 的话，`business_agent_id` 为 NULL 的存量
            # 消息会让整个计数变成 0 —— 症状是"画像永远不更新"，不报错。
            func.coalesce(
                AgentMessageModel.business_agent_id,
                AgentSessionModel.agent_id,
                default_agent_id,
            ) == agent_id,
        ]

        # ⚠️ 归属过滤在**会话**表上：agent_message 没有 user_id / org_id
        # ----------------------------------------------------------
        # 两张表同属 session 模块，所以这是一次模块内的 join（见契约里的
        # 说明）—— 不是"memory 模块伸手去够别人的表"。
        if org_id is not None:

            # 团队 scope：只看带这个组织的会话。
            conditions.append(AgentSessionModel.org_id == org_id)

        else:

            # ⚠️ 个人 scope：**必须**排除带组织的会话
            # ---------------------------------
            # `SceneScope.resolve` 的规则是"有 org 就丢掉 user"——带组织的
            # 会话聚合出的是**团队**画像。不排的话，同一个人的团队会话轮次
            # 会漏进他的个人画像计数，症状是"个人画像更新得莫名其妙地勤"。
            conditions += [
                AgentSessionModel.user_id == user_id,
                # ⚠️ 空串按 NULL 处理：历史数据里 org_id 可能是空串（与 NULL
                # 是两种不同的"没有"），而 `SceneScope.resolve` 已经把它归一
                # 成 NULL 了。少这一句，这批会话在两边各算一次或都不算。
                or_(
                    AgentSessionModel.org_id.is_(None),
                    AgentSessionModel.org_id == "",
                ),
            ]

        total = await self._session.scalar(
            select(func.count())
            .select_from(AgentMessageModel)
            .join(
                AgentSessionModel,
                AgentSessionModel.session_id == AgentMessageModel.session_id,
            )
            .where(*conditions)
        )

        return int(total or 0)

    async def list_chat_after(
            self,
            session_id: UUID,
            seq: int,
    ) -> list[AgentMessage]:

        # 走 idx_agent_message_session_seq：session_id 等值 + seq 范围，
        # 两个条件正好是那条索引的形状。kind 是回表后再筛的 —— 反过来
        # 把它放进索引前缀会让这条索引只对"会话内的 chat"有用，而
        # list_page 那条路要的是会话内的**全部**消息。
        models = await self._session.scalars(
            select(AgentMessageModel)
            .where(
                AgentMessageModel.session_id == session_id,
                AgentMessageModel.seq > seq,
                AgentMessageModel.kind == "chat",
            )
            .order_by(AgentMessageModel.seq.asc())
        )

        return [self._to_entity(model) for model in models]

    async def list_by_ids(
            self,
            message_ids: Sequence[UUID],
            *,
            session_id: UUID,
    ) -> list[AgentMessage]:

        # ⚠️ 空集合直接返回
        # --------------
        # `IN ()` 在 Postgres 里是语法错误，而"这条记忆没有来源"是一个
        # 完全正常的输入（模型没给 source_message_ids）。
        if not message_ids:
            return []

        # 去重：`source_message_ids` 是模型给的列表，同一条消息被引用两次
        # 不奇怪（同一句原话支撑了同一个情境里的两条记忆）。不去重的话
        # 返回里就会出现重复条目，而调用方按 id 配对时会把它当成两条。
        unique_ids = list(dict.fromkeys(message_ids))

        models = await self._session.scalars(
            select(AgentMessageModel)
            .where(
                # session_id 在前 —— 这条查询的形状就是
                # idx_agent_message_session_seq（等值 + seq 排序）。
                # 只按 id 查会退化成主键扫 + 内存里筛会话。
                AgentMessageModel.session_id == session_id,
                AgentMessageModel.message_id.in_(unique_ids),
            )
            # 按 seq 升序而不是入参顺序：溯源要看的是"这几句话**当时**是
            # 怎么说的"，会话里的先后才是那句话的语境。入参顺序是模型
            # 输出的顺序，没有意义。
            .order_by(AgentMessageModel.seq.asc())
        )

        return [self._to_entity(model) for model in models]

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

        if limit <= 0:
            return [], False

        conditions: list[ColumnElement[bool]] = [
            # 可见性在**会话**上（agent_message 没有 user_id 列）。
            # `OR IS NULL` 是存量语义，与 AgentSessionRepository.get 一致。
            or_(
                AgentSessionModel.user_id == user_id,
                AgentSessionModel.user_id.is_(None),
            ),
        ]

        if session_id is not None:
            conditions.append(AgentMessageModel.session_id == session_id)

        if roles:
            # 去重：调用方可能把 user 传两遍（契约里是 uniqueItems，但契约
            # 拦不住所有调用方）。
            conditions.append(
                AgentMessageModel.role.in_(list(dict.fromkeys(roles)))
            )

        if created_from is not None:
            conditions.append(AgentMessageModel.created_at >= created_from)

        if created_to is not None:
            conditions.append(AgentMessageModel.created_at <= created_to)

        if text_contains:
            conditions.append(self._text_contains(text_contains))

        # ---- 排序与 keyset ------------------------------------------
        #
        # 两族 order 的差别只有"哪两列"：座标轴都是
        # （主排序列，唯一/近唯一键），所以 keyset 条件共用 `_keyset`。
        if order in ("created_at_desc", "created_at_asc"):
            sort_col = AgentMessageModel.created_at
            tie_col = AgentMessageModel.message_id
        else:
            # session_seq_*：`(session_id, seq)` 就是唯一键，所以"主列"
            # 是 session_id、"次列"是 seq。
            sort_col = AgentMessageModel.session_id
            tie_col = AgentMessageModel.seq

        descending = order.endswith("_desc")

        if after is not None:
            conditions.append(
                self._keyset(
                    sort_col=sort_col,
                    tie_col=tie_col,
                    key=after[0],
                    tie=after[1],
                    descending=descending,
                )
            )

        if descending:
            order_by = (sort_col.desc(), tie_col.desc())
        else:
            order_by = (sort_col.asc(), tie_col.asc())

        # 多取一条判"还有没有下一页"（与仓储里其它分页同一条手法）。
        models = await self._session.scalars(
            select(AgentMessageModel)
            # join 只为一件事：拿会话的归属。session_id 是 agent_session 的
            # 主键，所以是 1:1，不会把消息行放大。
            .join(
                AgentSessionModel,
                AgentSessionModel.session_id == AgentMessageModel.session_id,
            )
            .where(*conditions)
            .order_by(*order_by)
            .limit(limit + 1)
        )

        rows = [self._to_entity(model) for model in models]

        has_more = len(rows) > limit

        return rows[:limit], has_more

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _keyset(
            *,
            sort_col: Any,
            tie_col: Any,
            key: Any,
            tie: Any,
            descending: bool,
    ) -> ColumnElement[bool]:
        """
        `(主列, 次列) > (上一页最后一条)` —— 方向按 `descending` 翻转。

        与 memory 模块的 `MemoryAtomRepositoryImpl._keyset` 是同一段逻辑
        （两个模块各有自己的仓储实现，没有共享的基类可放）。
        """

        if descending:
            return or_(
                sort_col < key,
                and_(sort_col == key, tie_col < tie),
            )

        return or_(
            sort_col > key,
            and_(sort_col == key, tie_col > tie),
        )

    @staticmethod
    def _text_contains(value: str) -> ColumnElement[bool]:
        """
        正文包含（只匹配 `type=text` 的块）—— `_assembly.message_text` 的 SQL 镜像。

        ⚠️ 为什么不用 `content::text ILIKE`
        ------------------------------
        JSONB 转文本会把**键名**一起暴露给匹配：`content::text` 里恒有
        `"type"` / `"text"` 这两个键，于是搜 "text" 命中所有消息 —— 而且
        不报错。所以这里按块拆开，只比 `->>'text'`。

        ⚠️ 为什么是 EXISTS 而不是 join
        --------------------------
        join `jsonb_array_elements(...)` 会把一条消息**放大成多行**（每个
        文本块一行），于是 `LIMIT limit + 1` 数的是块不是消息 —— 分页会
        少给或多给，而且不报错。EXISTS 只回答"有没有这么一块"。
        """

        escaped = (
            value
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        block = (
            func.jsonb_array_elements(AgentMessageModel.content)
            .table_valued(column("value", JSONB))
            .alias("block")
        )

        return (
            select(1)
            .select_from(block)
            .where(
                and_(
                    block.c.value["type"].astext == "text",
                    block.c.value["text"].astext.ilike(
                        f"%{escaped}%",
                        escape="\\",
                    ),
                )
            )
            .exists()
        )

    async def _fetch(
            self,
            session_id: UUID,
            *,
            direction: str,
            bound: ColumnElement[bool] | None,
            limit: int,
    ) -> list[AgentMessage]:
        """
        取一页，`direction` 决定排序方向。

        `bound` 是额外的 where 条件（这里只有 `seq < c` / `seq > c` 两种），
        为 None 时表示不加。写成参数而不是在调用处拼语句，是为了让两个方向
        共用同一段"选列 + 排序 + 限流"，只差一个方向。
        """

        order = (
            AgentMessageModel.seq.asc()
            if direction == "asc"
            else AgentMessageModel.seq.desc()
        )

        statement = select(AgentMessageModel).where(
            AgentMessageModel.session_id == session_id
        )

        if bound is not None:
            statement = statement.where(bound)

        models = await self._session.scalars(
            statement.order_by(order).limit(limit)
        )

        return [self._to_entity(model) for model in models]

    async def _exists_before(
            self,
            session_id: UUID,
            seq: int,
    ) -> bool:
        """还有比 seq 更旧的行吗。"""

        return await self._exists(session_id, AgentMessageModel.seq < seq)

    async def _exists_after(
            self,
            session_id: UUID,
            seq: int,
    ) -> bool:
        """还有比 seq 更新的行吗。"""

        return await self._exists(session_id, AgentMessageModel.seq > seq)

    async def _exists(
            self,
            session_id: UUID,
            bound: ColumnElement[bool],
    ) -> bool:
        """
        有没有满足条件的行。

        用 `SELECT ... LIMIT 1` 取一列而不是 `COUNT(*)`：这里只问"有没有"，
        计数会把整段索引扫完。走的是 idx_agent_message_session_seq，取到第一条
        就停。
        """

        found = await self._session.scalar(
            select(AgentMessageModel.seq)
            .where(
                AgentMessageModel.session_id == session_id,
                bound,
            )
            .limit(1)
        )

        return found is not None

    @staticmethod
    def _to_entity(
            model: AgentMessageModel,
    ) -> AgentMessage:

        return AgentMessage.model_validate(model)


# ============================================================
# internals
# ============================================================


def _user_turn_conditions() -> list[ColumnElement[bool]]:
    """
    "一条用户轮次"的判据 —— **两份计数共用的那一份**。

    ⚠️ 为什么必须共用
    ------------
    "轮次"的定义在本仓库只有一条（见 `count_user_turns` 的契约）：`role
    = 'user'` 且 `kind = 'chat'`。两个计数方法各写一遍的话，某天改了一处
    （比如把 `kind` 收得更紧），另一处就漂 —— 症状是"每 N 轮"在两条链上
    是两个意思，而两者不一致不报错。

    返回的是**条件列表**（不是完整 WHERE）：两个调用方各自还要叠自己的
    过滤（会话 / scope + 时间），而它们的差异正是它们存在的理由。
    """

    return [
        AgentMessageModel.role == "user",
        AgentMessageModel.kind == "chat",
    ]
