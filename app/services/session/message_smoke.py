"""
会话消息读路径的冒烟测试：**真库、真 SQL、真分页**。

⚠️ 与仓库其余 smoke 脚本不同：这一个需要连得上的 PostgreSQL
--------------------------------------------------------
别的脚本（agent_run_smoke / pipeline_smoke / tool_smoke…）都把仓储换成内存
实现，不需要任何外部依赖。本脚本**刻意不那样做**。

理由是这一段的风险全在 SQL 里：

    keyset 的 ORDER BY 方向 + LIMIT limit+1 探针 + 反向反转
    两个 has_more_* 的推导（一个靠探针、一个靠 EXISTS）
    部分索引与 CHECK 约束**到底有没有真的建出来**

内存实现验的是"我对这段逻辑的理解"，而它对不对取决于我有没有写错同一处
理解 —— 那是最没有价值的一种测试。因此这里连真库、发真 SQL，断言的是
契约承诺的那几条不变量。

代价说清楚：它会往开发库（app/config/datasource/postgre.yaml 指的那个）里写几行，
跑完在 finally 里删干净。中途被杀会留下一个孤儿会话 —— 没有外键，因此不会
牵连别的表，手工删掉即可。

⚠️ 表不存在时**自己建**
--------------------
启动时会走一次 `initialize_database`（create_all）。这里也走一次 ——
它顺带验证了"ORM 上声明的 CHECK / 索引真的建出来了"，而那正是
models.sql 与 ORM 两份声明最容易悄悄不一致的地方。

跑法：
    python -m app.services.session.message_smoke
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import delete, text

from app.common.exceptions import NotFoundError
from app.infra.datasource.initializer import initialize_database
from app.infra.datasource.loader import DataSourceConfigLoader
from app.infra.datasource.models import (
    PostgreSQLDataSourceDefinition,
)
from app.infra.datasource.postgresql import PostgreSQLDataSource

from .application.message_service_impl import MessageServiceImpl
from .domain.entities import AgentMessage, AgentSession
from .domain.schemas import (
    MessagePage,
    MessageRole,
    MessageStatus,
    SessionStatus,
    decode_cursor,
    encode_cursor,
)
from .domain.schemas import AgentMessage as AgentMessageDTO
from .models.sqlalchemy.agent_session import AgentSession as AgentSessionModel
from .repository.agent_session_impl import AgentSessionRepositoryImpl
from .repository.message_impl import MessageRepositoryImpl

#: 配置目录。与其他脚本一致，相对项目根。
_CONFIG_DIRECTORY = "./app/config/datasource/"

#: 造多少条消息。刻意取一个不是默认 limit(50) 倍数的数，好让"有没有下一页"
#: 两种结果都出现。
_TOTAL = 7

#: 本脚本扮演的调用者。服务现在要求每次读都带 user_id（归属过滤，见
#: repository/agent_session.py 的 get）。
#:
#: ⚠️ 它会匹配到 `_seed` 造的那些会话，靠的是**存量语义**：那些会话的
#: user_id 是 NULL，而过滤条件里有 `OR user_id IS NULL` —— 不是为了这个
#: 脚本方便，是加列之前的历史会话对谁都可见（见 models.sql 末尾）。
#: "自己的会话"与"别人的会话"两条路各有一个用例专门验（见"归属过滤"段）。
_SMOKE_USER_ID = "smoke-user"


# ============================================================
# 断言工具（与 agent_run_smoke.py 同一套）
# ============================================================


def check(label: str, condition: object) -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title: str) -> None:
    print(f"\n{title}")


# ============================================================
# 装配
# ============================================================


async def _build_datasource() -> PostgreSQLDataSource:
    """只建 postgres 一个数据源 —— 不必因此把 ES / Milvus / Neo4j 都拉起来。"""

    definitions = DataSourceConfigLoader(
        Path(_CONFIG_DIRECTORY)
    ).load_all()

    for definition in definitions:

        # 按**定义**类型挑，不是按数据源实例 —— 这一步拿到的全是配置对象。
        if isinstance(definition, PostgreSQLDataSourceDefinition):
            return PostgreSQLDataSource(definition)

    raise RuntimeError(
        f"{_CONFIG_DIRECTORY} 下没有 postgresql 数据源定义。"
    )


def _service(session: object) -> MessageServiceImpl:
    """
    用**真的**仓储与**真的**服务 —— 本脚本要验的就是它们。

    仓储拿到的是同一个 session，因此服务里的会话查询、翻页查询与用例自己
    的清理都在一个事务里。
    """

    return MessageServiceImpl(
        message_repository=MessageRepositoryImpl(session),  # type: ignore[arg-type]
        agent_session_repository=AgentSessionRepositoryImpl(session),  # type: ignore[arg-type]
    )


async def _seed(session: object, session_id: UUID) -> None:
    """
    造一个会话 + `_TOTAL` 条消息。用的也是真仓储（顺带验 next_seq / add_many）。
    """

    now = datetime.now(timezone.utc)

    # user_id 刻意**留空**（NULL）：本脚本的用例全都走"存量公共会话"这条路
    # —— 过滤条件里的 `OR user_id IS NULL` 正是靠它验的（见 _SMOKE_USER_ID）。
    # 归属过滤的另外两面在下面的"归属过滤"段单独验。
    await AgentSessionRepositoryImpl(session).add(  # type: ignore[arg-type]
        AgentSession(
            session_id=session_id,
            agent_id="smoke",
            status=SessionStatus.ACTIVE.value,
            meta={},
            created_at=now,
            updated_at=now,
        )
    )

    repository = MessageRepositoryImpl(session)  # type: ignore[arg-type]

    # 取一次号，然后连着分配 —— 与 message_recorder.to_entities 同一手法。
    start = await repository.next_seq(session_id)

    run_id = uuid4()

    await repository.add_many([
        AgentMessage(
            message_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            seq=start + index,
            role=(
                MessageRole.USER.value
                if index % 2 == 0
                else MessageRole.ASSISTANT.value
            ),
            kind="chat",
            content=[{"type": "text", "text": f"第 {index + 1} 条"}],
            status=MessageStatus.COMPLETED.value,
            meta={},
            created_at=now,
            # 与 created_at 同一个值：消息不可变，见 ORM 的说明。
            updated_at=now,
        )
        for index in range(_TOTAL)
    ])


async def _cleanup(session: object, session_id: UUID) -> None:
    """
    删掉本脚本写下的东西。

    两句话而不是一句：agent_message 表**没有外键**（刻意的，见 models.sql），
    删会话不会连带删消息。反过来说，一个会话被删了而它的消息还在是完全可能
    的 —— 这正是"没有引用完整性"的代价，写在这里免得被当成 bug。
    """

    await session.execute(  # type: ignore[attr-defined]
        text("DELETE FROM agent_message WHERE session_id = :sid"),
        {"sid": session_id},
    )

    await session.execute(  # type: ignore[attr-defined]
        delete(AgentSessionModel).where(
            AgentSessionModel.session_id == session_id
        )
    )


# ============================================================
# 用例
# ============================================================


async def main() -> int:

    datasource = await _build_datasource()

    engine = datasource.engine
    session_factory = datasource.session_factory

    # 建表（幂等）。顺带证明 ORM 上声明的 CHECK / 部分索引真的建得出来。
    await initialize_database(engine)

    session_id = uuid4()

    try:
        async with session_factory() as session:

            async with session.begin():
                await _seed(session, session_id)

            service = _service(session)

            # ------------------------------------------------
            section("默认页：最新的一段，且按 seq 升序")

            page = await service.list_messages(
                session_id,
                user_id=_SMOKE_USER_ID,
                limit=3,
            )

            check("返回的是契约 DTO，不是实体", type(page) is MessagePage)
            check(
                "页里的每一条也是 DTO",
                all(type(item) is AgentMessageDTO for item in page.messages),
            )
            check(
                "取到 3 条（最新的一段）",
                [item.seq for item in page.messages] == [5, 6, 7],
            )
            check(
                "升序 —— 最旧在前（客户端可以直接顺序渲染）",
                [item.seq for item in page.messages]
                == sorted(item.seq for item in page.messages),
            )
            check("前面还有更旧的", page.pagination.has_more_before is True)
            check("后面没有更新的了", page.pagination.has_more_after is False)
            check(
                "两个游标都指向本页的首尾",
                decode_cursor(page.pagination.before_cursor or "")
                == 5
                and decode_cursor(page.pagination.after_cursor or "") == 7,
            )
            check(
                "契约里有 metadata（库里那一列叫 meta）",
                page.messages[0].metadata == {},
            )
            check(
                "契约里没有 session_id 之外的会话字段",
                page.session_id == session_id,
            )

            # ------------------------------------------------
            section("before：往旧翻")

            page = await service.list_messages(
                session_id,
                user_id=_SMOKE_USER_ID,
                limit=3,
                before=page.pagination.before_cursor,
            )

            check(
                "取到游标之前的 3 条",
                [item.seq for item in page.messages] == [2, 3, 4],
            )
            check("前面还有更旧的", page.pagination.has_more_before is True)
            check("后面还有更新的", page.pagination.has_more_after is True)

            # ------------------------------------------------
            section("before：翻到最早一段，has_more_before 收敛")

            page = await service.list_messages(
                session_id,
                user_id=_SMOKE_USER_ID,
                limit=3,
                before=page.pagination.before_cursor,
            )

            check(
                "只剩 1 条 —— 因为只有 7 条，不是硬凑 3 条",
                [item.seq for item in page.messages] == [1],
            )
            check(
                "前面没有了（客户端据此收手）",
                page.pagination.has_more_before is False,
            )
            check("后面还有更新的", page.pagination.has_more_after is True)

            # ------------------------------------------------
            section("after：往新翻")

            # 承接上一段：此刻 page 是**最早**那一段（只有 seq=1），
            # 它的 after_cursor 指向 1。
            page = await service.list_messages(
                session_id,
                user_id=_SMOKE_USER_ID,
                limit=2,
                after=page.pagination.after_cursor,
            )

            check(
                "取到游标之后的 2 条",
                [item.seq for item in page.messages] == [2, 3],
            )
            check("前面还有更旧的", page.pagination.has_more_before is True)
            check("后面还有更新的", page.pagination.has_more_after is True)

            # ------------------------------------------------
            section("after：翻到最新一段，has_more_after 收敛")

            page = await service.list_messages(
                session_id,
                user_id=_SMOKE_USER_ID,
                limit=2,
                after=page.pagination.after_cursor,
            )

            check(
                "取到 4、5",
                [item.seq for item in page.messages] == [4, 5],
            )
            check("后面还有更新的", page.pagination.has_more_after is True)

            page = await service.list_messages(
                session_id,
                user_id=_SMOKE_USER_ID,
                limit=2,
                after=page.pagination.after_cursor,
            )

            check(
                "取到 6、7 —— 正好收尾",
                [item.seq for item in page.messages] == [6, 7],
            )
            check(
                "后面没有了（客户端据此收手）",
                page.pagination.has_more_after is False,
            )

            # ------------------------------------------------
            section("整段历史：一条不漏、一条不重")

            # 从最新一页开始，一路 before 翻到底 —— 把拼起来的 seq 与
            # 1..7 比。翻页最容易出的错（重复、漏读、顺序反了）都会在这里露出来。
            collected: list[int] = []
            cursor: str | None = None
            guard = 0

            while True:

                guard += 1

                if guard > _TOTAL + 2:  # 防死循环：翻不完就是有 bug
                    raise AssertionError("before 翻页没有收敛")

                page = await service.list_messages(
                    session_id,
                    user_id=_SMOKE_USER_ID,
                    limit=2,
                    before=cursor,
                )

                collected = [
                    item.seq for item in page.messages
                ] + collected

                if not page.pagination.has_more_before:
                    break

                cursor = page.pagination.before_cursor

            check(
                "翻出来的正好是全部消息，且不重不漏",
                collected == list(range(1, _TOTAL + 1)),
            )

            # 反向再来一遍：从最早那一段开始一路 after 翻到底。两个方向都验，
            # 因为"漏读"在两个方向上的成因并不一样（一个错在边界含不含，
            # 一个错在排序方向）。
            #
            # ⚠️ 起点只能是**一条真实消息**的游标
            # ---------------------------------
            # 没有一个"第 0 条"的游标可用（decode_cursor 要求 seq >= 1）：
            # 游标是**位置**，而位置由一条存在的消息给出。这不是缺口 ——
            # 默认页给的就是最新一段，往前翻用 before，after 是给"页面开着
            # 的时候又来了新消息"用的。上面那个 before 循环正好停在最早
            # 那一段，直接拿它当起点。
            forward: list[int] = [item.seq for item in page.messages]
            cursor = page.pagination.after_cursor
            guard = 0

            while page.pagination.has_more_after:

                guard += 1

                if guard > _TOTAL + 2:
                    raise AssertionError("after 翻页没有收敛")

                page = await service.list_messages(
                    session_id,
                    user_id=_SMOKE_USER_ID,
                    limit=2,
                    after=cursor,
                )

                forward.extend(
                    item.seq for item in page.messages
                )

                cursor = page.pagination.after_cursor

            check(
                "after 方向翻出来的也是全部消息，顺序一致",
                forward == list(range(1, _TOTAL + 1)),
            )

            # ------------------------------------------------
            section("会话存在但还没有消息：空页")

            empty_id = uuid4()

            # ⚠️ 这里用 commit() 而不是 `async with session.begin()`
            # -----------------------------------------------
            # 上面的读已经让 session 自动开了一段事务（SQLAlchemy 的
            # autobegin），此时再 begin() 会报 "A transaction is already
            # begun on this Session"。读路径不需要显式事务，写路径
            # 自己 commit 即可。
            await _seed_empty_session(session, empty_id)
            await session.commit()

            try:
                page = await service.list_messages(
                    empty_id, user_id=_SMOKE_USER_ID,
                )

                check("数组是空的", page.messages == [])
                check(
                    "两个标志都是 false（客户端据此收手，不会死循环）",
                    page.pagination.has_more_before is False
                    and page.pagination.has_more_after is False,
                )
                check(
                    "两个游标都是 None（没有可指向的）",
                    page.pagination.before_cursor is None
                    and page.pagination.after_cursor is None,
                )
                check("limit 原样回显", page.pagination.limit == 50)

            finally:
                await _cleanup(session, empty_id)
                await session.commit()

            # ------------------------------------------------
            section("参数与资源错误")

            for label, kwargs in (
                ("两个游标同时给 -> ValueError", {"before": encode_cursor(3), "after": encode_cursor(3)}),
                ("游标解析不出来 -> ValueError", {"before": "不是一个游标"}),
                ("游标里的序号越界 -> ValueError", {"after": encode_cursor(0)}),
            ):
                try:
                    await service.list_messages(
                        session_id,
                        user_id=_SMOKE_USER_ID,
                        **kwargs,
                    )
                    check(label, False)

                except ValueError:
                    check(label, True)

            check(
                "会话不存在 -> NotFoundError（不是空数组）",
                await _expect_not_found(service, uuid4()),
            )

            # ------------------------------------------------
            section("归属过滤")

            # 三种归属各验一次。NULL 那种已经被上面**所有**用例覆盖了
            # （_seed 刻意不写 user_id，过滤靠 `OR user_id IS NULL` 放行），
            # 这里补上另外两种 —— 归属只有在这两种情况下才真的会拦人。
            own_id = uuid4()
            other_id = uuid4()

            await _seed_empty_session(
                session, own_id, user_id=_SMOKE_USER_ID,
            )
            await _seed_empty_session(
                session, other_id, user_id="someone-else",
            )
            await session.commit()

            try:
                check(
                    "自己的会话照常可读",
                    (
                        await service.list_messages(
                            own_id, user_id=_SMOKE_USER_ID,
                        )
                    ).messages == [],
                )
                check(
                    "别人的会话 -> NotFoundError（与不存在同一个响应）",
                    await _expect_not_found(service, other_id),
                )

                # 反向再验一次：拿另一个身份去看 _SMOKE_USER_ID 的会话，同样
                # 该是 404 —— 否则上面那条"可读"可能是靠"谁都放行"蒙对的。
                check(
                    "换个身份看同一个会话 -> NotFoundError",
                    await _expect_not_found(
                        service, own_id, user_id="someone-else",
                    ),
                )

            finally:
                await _cleanup(session, own_id)
                await _cleanup(session, other_id)
                await session.commit()

            # ------------------------------------------------
            section("库里的约束真的建出来了")

            # 这几条验的是 models.sql 与 ORM 两份声明有没有落地。它们**只**
            # 声明在 models.sql 里是没用的（建表走 ORM 的 create_all），
            # 而"约束没建出来"在应用层完全看不出来 —— 只有在这里插一条坏数据
            # 才知道。
            check(
                "role 的 CHECK 拦得住越界取值",
                await _rejected(
                    session,
                    session_id,
                    "INSERT INTO agent_message (id, session_id, seq, role, kind, content, status, meta)"
                    " VALUES (gen_random_uuid(), :sid, 9001, 'system', 'chat', '[]'::jsonb, 'completed', '{}'::jsonb)",
                    session_id,
                ),
            )
            check(
                "content 的 CHECK 拦得住不是数组的内容",
                await _rejected(
                    session,
                    session_id,
                    "INSERT INTO agent_message (id, session_id, seq, role, kind, content, status, meta)"
                    " VALUES (gen_random_uuid(), :sid, 9002, 'user', 'chat', '\"一段文本\"'::jsonb, 'completed', '{}'::jsonb)",
                    session_id,
                ),
            )
            check(
                "(session_id, seq) 的唯一约束拦得住重号",
                await _rejected(
                    session,
                    session_id,
                    "INSERT INTO agent_message (id, session_id, seq, role, kind, content, status, meta)"
                    " VALUES (gen_random_uuid(), :sid, 1, 'user', 'chat', '[]'::jsonb, 'completed', '{}'::jsonb)",
                    session_id,
                ),
            )

            # ------------------------------------------------
            section("清理")

            await _cleanup(session, session_id)
            await session.commit()

            leftover = await session.scalar(
                text("SELECT count(*) FROM agent_message WHERE session_id = :sid"),
                {"sid": session_id},
            )

            check("本脚本写下的消息已删干净", leftover == 0)

    finally:
        await datasource.close()

    print("\n全部通过。")
    return 0


async def _seed_empty_session(
        session: object,
        session_id: UUID,
        *,
        user_id: str | None = None,
) -> None:
    """
    一个没有任何消息的会话（"刚建的会话"就是这个状态）。

    user_id 默认 None = 加这一列之前的存量会话（对谁都可见）。传值可以造出
    "属于某人"的会话 —— 归属过滤那一段两个方向都要用。
    """

    now = datetime.now(timezone.utc)

    await AgentSessionRepositoryImpl(session).add(  # type: ignore[arg-type]
        AgentSession(
            session_id=session_id,
            agent_id="smoke",
            status=SessionStatus.ACTIVE.value,
            meta={},
            user_id=user_id,
            created_at=now,
            updated_at=now,
        )
    )


async def _expect_not_found(
        service: MessageServiceImpl,
        session_id: UUID,
        user_id: str = _SMOKE_USER_ID,
) -> bool:
    try:
        await service.list_messages(
            session_id, user_id=user_id,
        )

    except NotFoundError:
        return True

    except ValueError:
        return False

    return False


async def _rejected(
        session: object,
        session_id: UUID,
        statement: str,
        sid: UUID,
) -> bool:
    """
    插一条应该被约束拒掉的坏数据。返回"确实被拒了"。

    ⚠️ 必须自己回滚
    --------------
    PostgreSQL 里一条语句失败会把**整个事务**标记成 aborted，后面所有语句
    都报 "current transaction is aborted"。因此每条用例自己开一段事务、
    失败就回滚 —— 不这么做的话第一条用例之后所有用例都会以同一个与原因
    无关的报错失败。

    ⚠️ 调用方的 statement 里**必须**带齐那些"非空又没有 DEFAULT"的列
    -----------------------------------------------------------
    最典型的是 meta：它在库里是 NOT NULL 而 **没有** DEFAULT
    （models.sql 写了 DEFAULT，ORM 只声明了 Python 侧的 default=dict ——
    两处并不一致，那是另一个问题）。statement 漏掉它，插入会因为 meta
    为空被拒，于是本函数返回 True，**无论 CHECK 在不在** —— 一条永远
    绿的假用例。本文件那三条用例因此都显式写了 meta，别删。
    """

    try:
        await session.execute(  # type: ignore[attr-defined]
            text(statement),
            {"sid": sid},
        )
        await session.commit()  # type: ignore[attr-defined]

    except Exception:
        await session.rollback()  # type: ignore[attr-defined]
        return True

    # 没被拒 = 约束没建出来。把它插进去的脏数据删掉，免得污染后面的断言。
    await session.execute(  # type: ignore[attr-defined]
        text(
            "DELETE FROM agent_message WHERE session_id = :sid AND seq >= 9000"
        ),
        {"sid": sid},
    )
    await session.commit()  # type: ignore[attr-defined]

    return False


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
