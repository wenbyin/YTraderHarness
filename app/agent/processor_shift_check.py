"""
两段式挪位的**真库**验证。

它与 tool_processor_smoke.py 是**互补**的，两个都要跑
----------------------------------------------------
    tool_processor_smoke.py    内存假仓储 -> 验证**服务层**逻辑
    processor_shift_check.py   真 Postgres -> 验证**挪位 SQL**

假仓储永远测不出 SQL 正确性，而这恰恰是最容易炸的地方：
`uq_tool_processor_tool_sequence` 是非延迟唯一索引，逐行立即检查，
"把 A 改成 B 当前占着的值"这种顺序会当场撞上。反过来，连真库的
用例覆盖不了洋葱模型、过期链判断那些纯逻辑分支。

全程在一个事务里跑，最后**回滚**，不留下任何测试数据。
建表（create_all）是唯一的持久化副作用，那是功能本身该做的事 ——
因此本脚本也会顺带确认新表建得出来。

跑法：python -m app.agent.processor_shift_check
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

import yaml

from app.common.utils import expand_env

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from app.common.models import Base
from app.infra.datasource.initializer import import_all_models
from app.services.tool.domain.entities import ToolProcessor
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)
from app.services.tool.models.sqlalchemy.tool_definition import (
    ToolDefinition as ToolDefinitionModel,
)
from app.services.tool.repository.tool_processor_impl import (
    ToolProcessorRepositoryImpl,
)

def _dsn() -> str:
    """
    从数据源配置取 DSN。

    以前这里写死了本地口令；DSN 属于部署事实，只应该有一份
    （`app/config/datasource/postgre.yaml`），并用 `${POSTGRES_DSN}`
    覆盖。
    """

    path = (
        Path(__file__).resolve().parents[1]
        / "config" / "datasource" / "postgre.yaml"
    )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return str(expand_env(raw["config"])["dsn"])

_failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
    if not ok:
        _failures.append(label)


def processor(tool_id, key: str, sequence: int = 1) -> ToolProcessor:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    return ToolProcessor(
        processor_id=uuid4(),
        tool_id=tool_id,
        processor_key=key,
        sequence=sequence,
        execution_mode=ExecutionMode.SYNC,
        hook=ProcessorHook.AROUND,
        config={"k": key},
        enabled=True,
        created_at=now,
        updated_at=now,
    )


async def main() -> int:

    import_all_models()

    engine = create_async_engine(_dsn())

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    session = session_factory()

    try:
        from datetime import datetime, timezone

        tool_id = uuid4()
        now = datetime.now(timezone.utc)

        session.add(
            ToolDefinitionModel(
                tool_id=tool_id,
                tool_key=f"shift.check.{tool_id}",
                name="shift-check",
                source_type="local_method",
                source_ref="__main__",
                entrypoint="__main__:check",
                input_schema={},
                manifest_hash="0" * 64,
                source_state="ACTIVE",
                # 这两列是 NOT NULL 且没有 server_default
                # （它们的语义是"发现于"与"同步于"，不是入库时间）。
                discovered_at=now,
                last_synced_at=now,
            )
        )
        await session.flush()

        repo = ToolProcessorRepositoryImpl(session)

        async def keys() -> list[str]:
            return [
                row.processor_key
                for row in await repo.list_for_tool(tool_id)
            ]

        async def sequences() -> list[int]:
            return [
                row.sequence
                for row in await repo.list_for_tool(tool_id)
            ]

        print("\n追加三条")
        for name in ("A", "B", "C"):
            await repo.insert(processor(tool_id, name))
        check("顺序 A B C", await keys() == ["A", "B", "C"])
        check("编号 1 2 3", await sequences() == [1, 2, 3])

        print("\n插到最前（最容易撞唯一索引的一种）")
        await repo.insert(processor(tool_id, "X"), position=1)
        check("顺序 X A B C", await keys() == ["X", "A", "B", "C"])
        check("编号 1..4", await sequences() == [1, 2, 3, 4])

        print("\n插到中间")
        await repo.insert(processor(tool_id, "Y"), position=3)
        check("顺序 X A Y B C", await keys() == ["X", "A", "Y", "B", "C"])

        print("\n位置超出链长 -> 落到末尾")
        await repo.insert(processor(tool_id, "Z"), position=99)
        check("Z 在末尾", (await keys())[-1] == "Z")

        print("\n整体倒序重排")
        rows = await repo.list_for_tool(tool_id)
        reversed_ids = [r.processor_id for r in reversed(rows)]
        await repo.reorder(tool_id, reversed_ids)
        check(
            "顺序完全反过来",
            await keys() == list(reversed(
                ["X", "A", "Y", "B", "C", "Z"]
            )),
        )
        check("编号仍是 1..6", await sequences() == [1, 2, 3, 4, 5, 6])

        print("\n相邻互换（终值正好是对方当前值）")
        rows = await repo.list_for_tool(tool_id)
        swapped = [r.processor_id for r in rows]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        await repo.reorder(tool_id, swapped)
        check("前两位换了", (await keys())[:2] == ["C", "Z"])

        print("\n删中间一条 -> 补洞")
        rows = await repo.list_for_tool(tool_id)
        victim = rows[2]
        deleted = await repo.delete(tool_id, victim.processor_id)
        check("删掉一行", deleted == 1)
        check("编号补成 1..5", await sequences() == [1, 2, 3, 4, 5])
        check("被删的不在了", victim.processor_key not in await keys())

        print("\n重复删同一行 -> 0，不报错")
        check(
            "第二次删返回 0",
            await repo.delete(tool_id, victim.processor_id) == 0,
        )

        print("\n整体替换（全删重建）")
        old_ids = {r.processor_id for r in await repo.list_for_tool(tool_id)}
        fresh = [processor(tool_id, n) for n in ("P", "Q", "R")]
        await repo.replace_all(tool_id, fresh)
        check("顺序 P Q R", await keys() == ["P", "Q", "R"])
        check("编号 1 2 3", await sequences() == [1, 2, 3])
        new_ids = {r.processor_id for r in await repo.list_for_tool(tool_id)}
        check("id 全部重生", not (old_ids & new_ids))

        print("\n替换成空数组 -> 清空")
        await repo.replace_all(tool_id, [])
        check("链空了", await keys() == [])

        print("\n重排顺序表不是全集 -> 显式报错")
        await repo.insert(processor(tool_id, "M"))
        await repo.insert(processor(tool_id, "N"))
        rows = await repo.list_for_tool(tool_id)
        try:
            await repo.reorder(tool_id, [rows[0].processor_id])
            check("应当报错", False)
        except ValueError:
            check("报 ValueError", True)

        print("\nAROUND + ASYNC -> 被 DB 的 CHECK 拦下")
        from sqlalchemy.exc import IntegrityError

        bad = processor(tool_id, "BAD")
        bad.hook = ProcessorHook.AROUND
        bad.execution_mode = ExecutionMode.ASYNC
        try:
            await repo.insert(bad)
            check("应当被 CHECK 拦下", False)
        except IntegrityError:
            check("抛 IntegrityError（服务层必须先拦，否则是 500）", True)

    finally:
        # 验证完整体回滚：不留任何测试数据。
        await session.rollback()
        await session.close()
        await engine.dispose()

    print()
    if _failures:
        print(f"失败 {len(_failures)} 项：{_failures}")
        return 1

    print("两段式挪位在真库上全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
