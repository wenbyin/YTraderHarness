"""
AgentRun 仓储**写入路径**的冒烟测试：真库、真 SQL、真往返。

⚠️ 与仓库其余 smoke 脚本不同：这一个需要连得上的 PostgreSQL
--------------------------------------------------------
理由与 `message_smoke.py` 那条一样：内存实现验的是"我对这段逻辑的
理解"，而它对不对取决于我有没有写错同一处理解 —— 那是最没有价值的一种
测试。这里要验的恰恰是**仓储有没有把实体上的每个字段真的写进 SQL**，
那是内存桩结构上不可能验的东西。

⚠️ 它盯的是一个**已经发生过**的 bug
---------------------------------
`AgentRunRepositoryImpl.add()` 建 ORM 模型时漏了 `caller_identity`。
服务层设了（`agent_run_service_impl.create_run`）、ORM 模型声明了、
`_to_entity` 那侧也是自动的（`model_validate(model)` 按属性名映射）——
**断的正是中间这一跳**。

列有 `DEFAULT '{}'::jsonb`，所以漏传**不报错**，只静默写空对象。症状是
工具级 checkauth 永远看到"调用者持有 []"（没有身份），而排查会一路往
token / Casdoor claim 那边找 —— 离原因很远。

所以本脚本的核心不是"caller_identity 能存"，而是**填满每一个字段再
往返**：将来谁再漏映射一个字段，这里会直接红。

跑法：
    python -m app.services.session.agent_run_repo_smoke
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import delete

from app.infra.datasource.initializer import initialize_database
from app.infra.datasource.loader import DataSourceConfigLoader
from app.infra.datasource.models import (
    PostgreSQLDataSourceDefinition,
)
from app.infra.datasource.postgresql import PostgreSQLDataSource

from .domain.entities import AgentRun, AgentSession
from .domain.schemas import RunStatus, SessionStatus
from .models.sqlalchemy.agent_session import AgentSession as AgentSessionModel
from .repository.agent_run_impl import AgentRunRepositoryImpl
from .repository.agent_session_impl import AgentSessionRepositoryImpl

#: 配置目录。与其他脚本一致，相对项目根。
_CONFIG_DIRECTORY = "./app/config/datasource/"


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
    """只建 postgres 一个数据源 —— 不必因此把 ES / Milvus 都拉起来。"""

    definitions = DataSourceConfigLoader(
        Path(_CONFIG_DIRECTORY)
    ).load_all()

    for definition in definitions:

        if isinstance(definition, PostgreSQLDataSourceDefinition):
            return PostgreSQLDataSource(definition)

    raise RuntimeError(
        f"{_CONFIG_DIRECTORY} 下没有 postgresql 数据源定义。"
    )


#: 每个字段都给一个**非默认**值 —— 这是本脚本的全部要点。
#:
#: 用默认值就分不出"写进去了"与"列默认值恰好等于它"，而那正是漏映射
#: 那个 bug 的藏身处。
_SNAPSHOT = {
    "user_id": "smoke-user",
    "groups": ["Y-Trader-User", "y-trader/Y-Trader-User"],
    "roles": ["Y-Trader-Admin"],
}


def _run_entity(session_id: UUID, now: datetime, **overrides: object) -> AgentRun:
    payload: dict[str, object] = {
        "run_id": uuid4(),
        "session_id": session_id,
        "status": RunStatus.QUEUED.value,
        "input": {"text": "冒烟：把每个字段都填上"},
        "meta": {"smoke": True},
        "error_type": "SmokeError",
        "error_message": "冒烟用的错误文本",
        "caller_identity": dict(_SNAPSHOT),
        "user_id": "smoke-user",
        # 后加的列（trace 开关）。给非空值，漏映射 add() 时能红。
        "trace_id": "smoke-trace",
        "created_at": now,
        "updated_at": now,
    }
    payload.update(overrides)

    return AgentRun(**payload)  # type: ignore[arg-type]


# ============================================================
# 用例
# ============================================================


async def main() -> int:
    datasource = await _build_datasource()

    engine = datasource.engine
    session_factory = datasource.session_factory

    # 建表（幂等）—— 顺带证明 ORM 上声明的东西真的建得出来。
    await initialize_database(engine)

    session_id = uuid4()
    now = datetime.now(timezone.utc)

    try:
        async with session_factory() as session:

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

            repository = AgentRunRepositoryImpl(session)  # type: ignore[arg-type]

            # --------------------------------------------------------
            section("每个字段都往返（漏一个映射就会红）")

            full = _run_entity(session_id, now)
            await repository.add(full)
            await session.commit()

            back = await repository.get(full.run_id)

            check("读得回来", back is not None)

            assert back is not None

            for field in (
                "run_id", "session_id", "status", "input", "meta",
                "error_type", "error_message", "user_id",
                "caller_identity", "trace_id",
            ):
                check(
                    f"{field} 往返一致",
                    getattr(back, field) == getattr(full, field),
                )

            # ⚠️ 单独再钉一条：上面那个循环挂掉时你看不出**是哪个**字段
            # 没写进去，而这一条是要命的那个（列有 DEFAULT，漏了不报错）。
            check(
                "caller_identity 不是空对象（回归：add() 曾经漏映射它）",
                back.caller_identity == _SNAPSHOT,
            )

            # --------------------------------------------------------
            section("空快照也要能往返（默认值那条路）")

            empty = _run_entity(
                session_id, now, caller_identity={}, error_type=None,
                error_message=None, user_id=None, trace_id=None,
            )
            await repository.add(empty)
            await session.commit()

            empty_back = await repository.get(empty.run_id)

            check("读得回来", empty_back is not None)
            check(
                "空 caller_identity 往返仍是 {}",
                empty_back is not None
                and empty_back.caller_identity == {},
            )
            check(
                "可空列（error_* / user_id / duty_key）往返是 None",
                empty_back is not None
                and empty_back.error_type is None
                and empty_back.error_message is None
                and empty_back.user_id is None
                # duty_key 是后加的列，且服务层**总是**会写它 —— 因此
                # "不写时读回来是 None"这条只有这里盯得住。仓储 add() 里
                # 漏掉那一句映射的话，这一条会红（否则静默写 NULL）。
                and empty_back.duty_key is None
                # trace_id 同理：空 = 未开启 Trace。
                and empty_back.trace_id is None,
            )

            print("\n全部通过。")
            return 0

    finally:
        # 删会话即删它的全部运行（agent_run 有 ON DELETE CASCADE，与
        # agent_message 不同 —— 后者刻意没有外键，见 models.sql）。
        async with session_factory() as session:
            await session.execute(
                delete(AgentSessionModel).where(
                    AgentSessionModel.session_id == session_id
                )
            )
            await session.commit()

        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
