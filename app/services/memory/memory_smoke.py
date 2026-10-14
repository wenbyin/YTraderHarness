"""
Memory 模块的冒烟测试：分词 -> 建索引 -> 两路召回 -> 融合 -> 越权。

直接用真数据库跑（本地 Postgres 里有 pgvector），但 **embedder 是假的** ——
本脚本要验的是这个模块的读写与过滤，不是本地模型的下载与前向。假 embedder
把文本映射成确定性的 512 维向量，于是"最近邻"这件事可以被断言。

    .venv/Scripts/python.exe -m app.services.memory.memory_smoke

⚠️ 用过即清
--------
脚本自己造 session / run / message，跑完按 run_id 删索引行、按 id 删消息 /
运行 / 会话。**失败路径也清**（finally），否则库里会留下垃圾行，而它们
会在下次跑别的冒烟脚本时以"多出来几条"的形式出现。

⚠️ 它绕过了 HTTP 层
----------------
与仓库其他冒烟脚本一样，这里直接调服务 —— 因此**不能**用它证明
"接口在 HTTP 上通"（响应模型、鉴权、序列化都不经过）。那几条要走真应用
（起 uvicorn 打 /api/v1/memory/search）。这里证明的是服务与数据层。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import sys
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from app.services.document.embedder.base import EmbeddingOutput
from app.services.memory.application.memory_service_impl import (
    MemoryServiceImpl,
)
from app.services.memory.domain.entities import MemoryQueryScope
from app.services.memory.domain.schemas import (
    MemorySearchMode,
    MemorySearchRequest,
    MemorySearchScope,
)
from app.services.memory.domain.tasks import MemoryExtractParams
from app.services.memory.models.sqlalchemy.agent_message_fts import (
    AgentMessageFts,
)
from app.services.memory.models.sqlalchemy.agent_message_vec import (
    AgentMessageVec,
)
from app.services.memory.repository.memory_impl import (
    MemoryIndexRepositoryImpl,
)
from app.services.memory.tokenization import index_text, tokenize, tsquery_text
from app.services.session.domain.entities import (
    AgentMessage as AgentMessageEntity,
)
from app.services.session.domain.entities import AgentSession, AgentRun
from app.services.session.models.sqlalchemy.agent_message import (
    AgentMessage as AgentMessageModel,
)
from app.services.session.models.sqlalchemy.agent_run import AgentRun as AgentRunModel
from app.services.session.models.sqlalchemy.agent_session import (
    AgentSession as AgentSessionModel,
)
from app.services.session.repository.agent_session_impl import (
    AgentSessionRepositoryImpl,
)
from app.services.session.repository.message_impl import (
    MessageRepositoryImpl,
)

DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

#: 两次运行：一次给"我的会话"，一次给"别人的会话"（越权断言要用）。
USER_A = "smoke-user-a"
USER_B = "smoke-user-b"

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ✓ {label}")
        return

    _FAILURES.append(label)
    print(f"  ✗ {label}{(' —— ' + detail) if detail else ''}")


# ============================================================
# 假的 embedder：确定性向量，让"最近邻"可断言
# ============================================================


class FakeEmbedder:
    """
    文本 -> 512 维 one-hot（按内容哈希定位）。

    为什么用 one-hot：两条**相同**文本的余弦相似度是 1.0，不同文本是 0.0。
    于是"查询与某条消息最像"这件事有一个可以写死的答案，而真模型给出的
    相似度是连续的、只能"大致"断言。
    """

    async def embed(self, texts, *, config):
        return [self._vector(text) for text in texts]

    async def embed_query(self, text, *, config):
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * 512
        vector[hash(text) % 512] = 1.0
        return vector


class FakeRegistry:
    def __init__(self) -> None:
        self._embedder = FakeEmbedder()

    def embedder_for(self, kind):
        return self._embedder if kind is EmbeddingOutput.DENSE else None


# ============================================================
# 造数据 / 清理
# ============================================================


async def _seed(session, *, user_id: str) -> tuple[uuid.UUID, uuid.UUID, list[uuid.UUID]]:
    """造一个会话 + 一次运行 + 三条消息，返回 (session_id, run_id, message_ids)。"""

    now = dt.datetime.now(dt.timezone.utc)
    session_id = uuid.uuid4()
    run_id = uuid.uuid4()

    session.add(AgentSessionModel(
        session_id=session_id,
        agent_id="react",
        status="active",
        meta={},
        thread_id=f"smoke-{session_id}",
        user_id=user_id,
        org_id="smoke-org",
        created_at=now,
        updated_at=now,
    ))

    # ⚠️ **先 flush 会话**，再插运行
    # ----------------------------
    # `agent_run.session_id` 上有一条**真外键**（`agent_run_session_id_fkey`，
    # ORM 里就有 `ForeignKey("agent_session.session_id")`）—— 与
    # `agent_message` 刻意没有外键正好相反。两者放在同一次 flush 里时
    # 插入顺序不由这里决定，实测会先插 agent_run 而撞外键。
    #
    # 刻意**不**用 `relationship()` 让 SQLAlchemy 自己排序：那会给这三个
    # 模型之间加上 ORM 级的关联，而它们现在只有"列"这一层关系 ——
    # 冒烟脚本不该顺手改动被测模块的模型定义。
    await session.flush()

    session.add(AgentRunModel(
        run_id=run_id,
        session_id=session_id,
        status="completed",
        input={},
        meta={},
        caller_identity={},
        user_id=user_id,
        duty_key="stock_research",
        created_at=now,
        updated_at=now,
    ))

    # ⚠️ 四条消息，每条都在钉一个过滤规则
    # ---------------------------------
    #     (文本, kind, 期望)  为什么
    #     有文本 / chat        ✅  正常对话，进索引
    #     有文本 / chat        ✅  与查询无关的那条，用来验召回集合
    #     空文本 / chat        ✗   规格里"content 为空则丢"
    #     有文本 / tool_activity ✗  **动手前的旁白**（带 tool_call 的
    #                              assistant 消息）—— 它曾经被索引进去，
    #                              这正是线上那条 5 条消息的运行里
    #                              "索引到的是中间思考、而最终答复没进去"
    #                              的一半原因（另一半是幂等键，见
    #                              agent_run_dispatch 里 waiting_input
    #                              那一段）。用断言把它钉住。
    fixtures = [
        ("东方财富的股东户数是多少", "chat"),
        ("今天天气不错，适合出门", "chat"),
        ("", "chat"),
        ("我来查一下平台当前实际注册的工具和能力清单。", "tool_activity"),
    ]

    message_ids: list[uuid.UUID] = []

    for index, (text, kind) in enumerate(fixtures, start=1):
        message_id = uuid.uuid4()
        message_ids.append(message_id)
        session.add(AgentMessageModel(
            message_id=message_id,
            session_id=session_id,
            run_id=run_id,
            seq=index,
            role="user" if index == 1 else "assistant",
            kind=kind,
            business_agent_id="stock_research",
            content=[{"type": "text", "text": text}] if text else [],
            status="completed",
            reply_to_message_id=None,
            meta={},
            created_at=now,
            updated_at=now,
        ))

    await session.flush()

    return session_id, run_id, message_ids


async def _cleanup(session, *, session_ids: list[uuid.UUID]) -> None:
    """按会话清干净（索引行按 run_id 关联，这里一并按会话删）。"""

    run_ids = list(await session.scalars(
        select(AgentRunModel.run_id).where(
            AgentRunModel.session_id.in_(session_ids)
        )
    ))

    if run_ids:
        await session.execute(
            delete(AgentMessageFts).where(AgentMessageFts.run_id.in_(run_ids))
        )
        await session.execute(
            delete(AgentMessageVec).where(AgentMessageVec.run_id.in_(run_ids))
        )

    await session.execute(
        delete(AgentMessageModel).where(
            AgentMessageModel.session_id.in_(session_ids)
        )
    )
    await session.execute(
        delete(AgentRunModel).where(AgentRunModel.session_id.in_(session_ids))
    )
    await session.execute(
        delete(AgentSessionModel).where(
            AgentSessionModel.session_id.in_(session_ids)
        )
    )
    await session.flush()


# ============================================================
# 各段断言
# ============================================================


def smoke_tokenizer() -> None:
    """
    分词对齐 ES 的 `cjk` 分析器 —— 与 es.py 注释里那份**实测输出**逐字比对。

    这一段不是"看起来对就算过"：`es.py:131-147` 那段注释记的是真在 ES 上
    跑出来的结果，本模块宣称"对齐"就该拿它当基准。
    """

    print("\n[1] 分词器对齐 ES 的 cjk 分析器")

    check(
        "东方财富股东户数 -> 与 es.py 的实测输出逐字相同",
        tokenize("东方财富股东户数")
        == ["东方", "方财", "财富", "富股", "股东", "东户", "户数"],
        str(tokenize("东方财富股东户数")),
    )

    check(
        "长度 1 的 CJK 串保留单字（ES 的 unigram 规则）",
        tokenize("股 abc 东") == ["股", "abc", "东"],
        str(tokenize("股 abc 东")),
    )

    check(
        "全角压半角 + 小写",
        tokenize("Ｆｕｌｌｗｉｄｔｈ　ＡＢＣ１２３") == ["fullwidth", "abc123"],
        str(tokenize("Ｆｕｌｌｗｉｄｔｈ　ＡＢＣ１２３")),
    )

    check(
        "纯标点切不出词元 -> tsquery 为 None（调用方据此降级）",
        tsquery_text("、。！") is None,
    )

    check(
        "tsquery 用空格连接 + 小写，且是 OR",
        tsquery_text("东方财富") == "东方 | 方财 | 财富",
        str(tsquery_text("东方财富")),
    )


async def smoke_flow(session) -> None:
    print("\n[2] extract：真写进两张索引表")

    session_a, run_a, messages_a = await _seed(session, user_id=USER_A)
    session_b, _run_b, _messages_b = await _seed(session, user_id=USER_B)

    service = MemoryServiceImpl(
        index=MemoryIndexRepositoryImpl(session=session),
        messages=MessageRepositoryImpl(session=session),
        sessions=AgentSessionRepositoryImpl(session=session),
        pipeline_registry=FakeRegistry(),
    )

    await service.message_extract(
        MemoryExtractParams(run_id=run_a, session_id=session_a), None
    )

    fts_rows = list(await session.scalars(
        select(AgentMessageFts).where(AgentMessageFts.run_id == run_a)
    ))
    vec_rows = list(await session.scalars(
        select(AgentMessageVec).where(AgentMessageVec.run_id == run_a)
    ))

    check(
        "只有 chat 的两条进了索引（tool_activity 与空正文都被跳过）",
        len(fts_rows) == 2 and len(vec_rows) == 2,
        f"fts={len(fts_rows)} vec={len(vec_rows)}",
    )

    indexed_ids = {row.message_id for row in fts_rows}

    check(
        "动手前的旁白（kind=tool_activity，有正文）**没有**进索引",
        messages_a[3] not in indexed_ids,
        f"索引里 {[str(i)[:8] for i in indexed_ids]}",
    )

    check(
        "用户那句话与最终答复都在索引里",
        messages_a[0] in indexed_ids and messages_a[1] in indexed_ids,
    )

    check(
        "scope 列从**会话**上回填（agent_message 没有 user_id）",
        all(row.user_id == USER_A for row in fts_rows),
        str({row.user_id for row in fts_rows}),
    )

    check(
        "tokens 是分词串、original 是原文（两者不同源就是 bug）",
        any(row.tokens != row.original for row in fts_rows)
        and any("东方" in row.original for row in fts_rows),
    )

    check(
        "tsv 是生成列：tokens 写进去了它就有值",
        all(row.tsv for row in fts_rows),
    )

    # 幂等：重跑一次不该多出行、也不该报错
    await service.message_extract(
        MemoryExtractParams(run_id=run_a, session_id=session_a), None
    )

    again = list(await session.scalars(
        select(AgentMessageFts).where(AgentMessageFts.run_id == run_a)
    ))

    check("重跑幂等（行数不变）", len(again) == 2, f"fts={len(again)}")

    print("\n[3] search：两路召回 + 融合")

    response = await service.search(
        MemorySearchRequest(query="股东户数"),
        session_id=session_a,
        user_id=USER_A,
    )

    check("关键词路召回到含该词的消息", len(response.results) >= 1)
    check(
        "无关的那条（天气）没被召回",
        all("天气" not in hit.text for hit in response.results),
        str([hit.text for hit in response.results]),
    )
    check(
        "没有降级（两路都跑了）",
        response.degraded_routes == [],
        str(response.degraded_routes),
    )
    check(
        "命中标注了来自哪几路",
        any(hit.matched_by for hit in response.results),
    )

    # 向量路：用与第二条消息**完全相同**的文本当查询 -> 它应当是最近邻
    response_vec = await service.search(
        MemorySearchRequest(query="今天天气不错，适合出门"),
        session_id=session_a,
        user_id=USER_A,
    )

    check(
        "向量路把最像的那条排在第一位",
        bool(response_vec.results)
        and response_vec.results[0].text.startswith("今天天气不错"),
        response_vec.results[0].text if response_vec.results else "(空)",
    )

    # 只跑关键词模式
    keyword_only = await service.search(
        MemorySearchRequest(
            query="股东户数", mode=MemorySearchMode.KEYWORD
        ),
        session_id=session_a,
        user_id=USER_A,
    )

    check(
        "mode=keyword 时不出向量路标记",
        all(
            all(route.value == "keyword" for route in hit.matched_by)
            for hit in keyword_only.results
        ),
    )

    print("\n[4] 越权与降级")

    # 别人的会话：应当 404（同一个服务、换 user_id）
    from app.common.exceptions import NotFoundError

    try:
        await service.search(
            MemorySearchRequest(query="股东户数"),
            session_id=session_a,
            user_id=USER_B,
        )
        check("别人的会话被拒", False, "居然搜到了")
    except NotFoundError:
        check("别人的会话被拒（NotFoundError）", True)

    # scope=user：只能搜到自己的（B 的会话不该出现）
    cross = await service.search(
        MemorySearchRequest(
            query="股东户数", scope=MemorySearchScope.USER
        ),
        user_id=USER_B,
    )

    check(
        "scope=user 只铺开自己的会话（B 搜不到 A 的行）",
        all(hit.session_id != session_a for hit in cross.results),
        str([str(hit.session_id) for hit in cross.results]),
    )

    # 查询里没有可切分的词元 -> 关键词路降级，向量路照常
    punctuation = await service.search(
        MemorySearchRequest(query="、。！？"),
        session_id=session_a,
        user_id=USER_A,
    )

    check(
        "纯标点查询：关键词路降级并如实汇报",
        any(
            route.route.value == "keyword"
            for route in punctuation.degraded_routes
        ),
        str(punctuation.degraded_routes),
    )

    # 参数校验
    for label, request in (
        (
            "scope=session 缺 session_id -> 报错",
            MemorySearchRequest(query="x"),
        ),
    ):
        try:
            await service.search(request, user_id=USER_A)
            check(label, False, "没有报错")
        except ValueError:
            check(label, True)

    try:
        await service.search(
            MemorySearchRequest(
                query="x", scope=MemorySearchScope.USER
            ),
            session_id=session_a,
            user_id=USER_A,
        )
        check("scope=user 却给了 session_id -> 报错", False, "没有报错")
    except ValueError:
        check("scope=user 却给了 session_id -> 报错", True)

    # ⚠️ 没有身份 -> 两条路都必须拒绝，而不是"不过滤"
    # 这一天亏在仓储侧重演过一次：条件不拼 = 搜出所有人。
    for scope in (MemorySearchScope.SESSION, MemorySearchScope.USER):
        try:
            await service.search(
                MemorySearchRequest(query="股东户数", scope=scope),
                session_id=session_a if scope is MemorySearchScope.SESSION
                else None,
                user_id=None,
            )
            check(f"没有身份（scope={scope.value}）-> 拒绝", False, "居然放行了")
        except ValueError:
            check(f"没有身份（scope={scope.value}）-> 拒绝", True)

    # 仓储层兜底：直接拿一个 user_id=None 的 scope 去查，只能得到无归属行
    naked = await MemoryIndexRepositoryImpl(session=session).search_keyword(
        tsquery="东方",
        scope=MemoryQueryScope(session_id=session_a, user_id=None),
        recall_limit=10,
    )
    check(
        "仓储兜底：user_id=None 时搜不到有归属的行（不越权）",
        naked == [],
        f"拿到 {len(naked)} 条",
    )

    print("\n[5] 清理")
    await _cleanup(session, session_ids=[session_a, session_b])

    left = await session.scalar(
        select(AgentMessageFts).where(AgentMessageFts.run_id == run_a)
    )
    check("索引行已清干净", left is None)


async def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    engine = create_async_engine(DSN)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    smoke_tokenizer()

    async with maker() as session:
        try:
            await smoke_flow(session)
            await session.commit()
        finally:
            await session.rollback()

    await engine.dispose()

    print()

    if _FAILURES:
        print(f"FAILED（{len(_FAILURES)} 条）：")
        for label in _FAILURES:
            print(f"  - {label}")
        return 1

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
