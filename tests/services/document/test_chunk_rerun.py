"""
重跑（reprocess）时切分产出的清场：**删得准、删完能插**。

为什么这个用例必须连真库
------------------------
它要焊住的东西**只有 PostgreSQL 知道**：

    UNIQUE(document_version_id, chunker_key, chunker_config, chunk_index)

而 `delete_by_identity` 的过滤条件就是这条约束去掉 chunk_index。
用假仓储测等于把过滤条件抄一遍再断言它等于自己 —— 抄错一个字段，
重跑照样撞约束，用例却全绿。

（同一个道理：这条约束在真库上是不是**存在**，也只有真库答得出来。
模型上写了约束、库里没建的情形，正是一类只在重跑时才暴露的问题。）

连不上库就跳过，不失败
----------------------
这套用例是给本地与 CI 的**回归**用的，不是"必须有库才能跑测试"的
前置条件 —— 其余用例都只用假替身，不该被一条库连接拖住。

全程在一个**会回滚**的事务里
----------------------------
用例会往 document_chunk 插行（这正是它要验的动作）。为了不污染
开发库，所有写入都发生在一个显式事务里，结束时无条件 rollback。
因此它不改动任何既有数据，也不需要清理步骤。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.services.document.repository.document_chunk_impl import (
    DocumentChunkRepositoryImpl,
)

#: 真实的数据源配置。从 __file__ 往上推而不是写相对路径：
#: 相对路径只在 CWD 恰好是仓库根时成立（见 tests/conftest.py 的说明）。
POSTGRE_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "app" / "config" / "datasource" / "postgre.yaml"
)


def _dsn() -> str:
    raw = yaml.safe_load(POSTGRE_CONFIG.read_text(encoding="utf-8"))
    return raw["config"]["dsn"]


async def _connect_or_skip():
    engine = create_async_engine(_dsn())

    try:
        connection = await engine.connect()
        await connection.close()

    except Exception as exc:  # noqa: BLE001 —— 连不上就跳过，原因记进 skip 里
        await engine.dispose()
        pytest.skip(f"连不上本地 Postgres，跳过重跑回归：{exc}")

    return engine


@pytest.fixture
async def session():
    """
    一个用完就回滚的会话。

    不用 `engine.begin()`：那样退出时是 **commit**，用例会往开发库里
    留下垃圾行。这里显式 rollback。
    """

    engine = await _connect_or_skip()
    session = AsyncSession(engine, expire_on_commit=False)

    try:
        yield session

    finally:
        await session.rollback()
        await session.close()
        await engine.dispose()


async def _sample_identity(session) -> dict:
    """
    找一个已有的切分身份作为样本。

    没数据就跳过 —— 这条用例验的是"重跑不撞约束"，而库里一条 chunk
    都没有时它无从谈起（那不是失败，是没素材）。
    """

    row = (
        await session.execute(
            text(
                """
                SELECT document_version_id,
                       chunker_key,
                       chunker_config::text AS chunker_config,
                       count(*) AS n
                FROM document_chunk
                GROUP BY 1, 2, 3
                ORDER BY n DESC
                LIMIT 1
                """
            )
        )
    ).mappings().first()

    if row is None:
        pytest.skip("document_chunk 里没有样本数据，跳过重跑回归。")

    return dict(row)


_INSERT = text(
    """
    INSERT INTO document_chunk (
        document_chunk_id, document_version_id, chunker_key,
        chunker_config, document_processor_definition_id, chunk_index,
        start_index, end_index, content, content_hash,
        token_count, char_count, locator, metadata
    )
    VALUES (
        :chunk_id, :version_id, :chunker_key,
        CAST(:chunker_config AS jsonb), NULL, :chunk_index,
        0, 1, 'x', 'h',
        NULL, NULL, '{}'::jsonb, '{}'::jsonb
    )
    """
)


async def test_identity_is_the_unique_key(session) -> None:
    """
    先钉住"为什么要有这个删除"：**同身份同序号插第二行会撞约束**。

    没有这条，下面那个用例会显得像在验一件本来就不会发生的事 ——
    而它恰恰是重跑失败的根因。
    """

    sample = await _sample_identity(session)

    with pytest.raises(IntegrityError):
        await session.execute(
            _INSERT,
            {
                "chunk_id": uuid.uuid4(),
                "version_id": sample["document_version_id"],
                "chunker_key": sample["chunker_key"],
                "chunker_config": sample["chunker_config"],
                "chunk_index": 0,
            },
        )

    # 事务被这条失败弄脏了，回滚一下再还给 fixture。
    await session.rollback()


async def test_delete_then_insert_survives_a_rerun(session) -> None:
    """
    重跑的完整动作：**先按身份删、再插同身份同序号的行** —— 必须成功。

    这就是重跑路径上原本会炸的那一步。删与插在这里是连续两个动作，
    与 chunking 节点在**同一个事务**里做的那两件事一致。
    """

    sample = await _sample_identity(session)

    version_id = sample["document_version_id"]
    chunker_key = sample["chunker_key"]
    chunker_config = sample["chunker_config"]

    # 另一个身份（同版本、不同配置）。它的行**不该**被这次删除波及 ——
    # 同一版本下有多份切分产出时，并发分支各自删自己的那份。
    other_config = '{"chunk_size": 123, "marker": "not-mine"}'

    await session.execute(
        _INSERT,
        {
            "chunk_id": uuid.uuid4(),
            "version_id": version_id,
            "chunker_key": chunker_key,
            "chunker_config": other_config,
            "chunk_index": 0,
        },
    )

    repository = DocumentChunkRepositoryImpl(session)

    removed = await repository.delete_by_identity(
        document_version_id=version_id,
        chunker_key=chunker_key,
        chunker_config=yaml.safe_load(chunker_config),
    )

    assert removed == sample["n"], (
        f"按身份删除应删掉 {sample['n']} 行，实际 {removed} 行 —— "
        "过滤条件与唯一约束对不上时，重跑会在插入时撞约束。"
    )

    surviving = (
        await session.execute(
            text(
                """
                SELECT count(*) FROM document_chunk
                WHERE document_version_id = :v
                  AND chunker_key = :k
                  AND chunker_config = CAST(:c AS jsonb)
                """
            ),
            {"v": version_id, "k": chunker_key, "c": other_config},
        )
    ).scalar_one()

    assert surviving == 1, (
        "删除波及其他切分身份了 —— 那种丢失不会报错，只表现为"
        "某一类 chunk 少了一批。"
    )

    # 这一步在原实现下会抛 UniqueViolationError。
    await session.execute(
        _INSERT,
        {
            "chunk_id": uuid.uuid4(),
            "version_id": version_id,
            "chunker_key": chunker_key,
            "chunker_config": chunker_config,
            "chunk_index": 0,
        },
    )
