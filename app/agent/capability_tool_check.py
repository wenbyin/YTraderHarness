"""
"能力 code -> 工具" 那条查询的**真库**验证。

它与 tool_smoke.py 是**互补**的，两个都要跑
------------------------------------------------
    tool_smoke.py                 内存假仓储 -> 验证服务层逻辑
    capability_tool_check.py      真 Postgres -> 验证那条 EXISTS 子查询

假仓储永远测不出 SQL 正确性，而这条查询的语义恰恰全在 SQL 里：

    精确匹配 code，**不沿 parent_id 递归**
    只取 ACTIVE（MISSING 的工具不该被暴露、更不能被执行）
    code 列表为空 -> 空结果（不能落成 `IN ()`）
    一个工具挂多个能力时**只出现一次**（EXISTS 而不是 join）

最后一条尤其容易在假仓储里"碰巧正确"：内存实现天然去重，而 SQL 里
写成 join 会让同一个工具出多行 —— 于是模型看到的工具列表里有重复项，
token 白花，且没有任何报错。

全程在一个事务里跑，最后**回滚**，不留下任何测试数据。

跑法：python -m app.agent.capability_tool_check
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from app.common.utils import expand_env

from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from app.common.models import Base
from app.infra.datasource.initializer import import_all_models
from app.services.capability.models.sqlalchemy.capability import (
    Capability as CapabilityModel,
)
from app.services.tool.models.sqlalchemy.tool_classification import (
    ToolDefinitionClassification as LinkModel,
)
from app.services.tool.models.sqlalchemy.tool_definition import (
    ToolDefinition as ToolDefinitionModel,
)
from app.services.tool.repository.tool_definition_impl import (
    ToolDefinitionRepositoryImpl,
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


def _tool(tool_key: str, state: str, now: datetime) -> ToolDefinitionModel:
    return ToolDefinitionModel(
        tool_id=uuid4(),
        tool_key=tool_key,
        name=tool_key,
        source_type="local_method",
        source_ref="__main__",
        entrypoint="__main__:check",
        input_schema={},
        manifest_hash="0" * 64,
        source_state=state,
        # 这两列 NOT NULL 且没有 server_default（语义是"发现于"
        # 与"同步于"，不是入库时间）。
        discovered_at=now,
        last_synced_at=now,
    )


def _capability(code: str, now: datetime, parent_id=None):
    return CapabilityModel(
        capability_id=uuid4(),
        code=code,
        name=code,
        parent_id=parent_id,
        sort_order=0,
        is_active=True,
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
        now = datetime.now(timezone.utc)
        tag = uuid4().hex[:8]

        parent = _capability(f"check.parent.{tag}", now)
        child = _capability(f"check.child.{tag}", now, parent.capability_id)
        other = _capability(f"check.other.{tag}", now)

        session.add_all([parent, child, other])

        # 四个工具，覆盖四种情形：
        #   active    挂在父能力上，ACTIVE
        #   missing   也挂父能力，但 MISSING（代码里已经没有了）
        #   in_child  挂在**子**能力上（精确匹配不该把它带出来）
        #   multi     同时挂父与另一个能力（去重）
        active = _tool(f"check.active.{tag}", "ACTIVE", now)
        missing = _tool(f"check.missing.{tag}", "MISSING", now)
        in_child = _tool(f"check.inchild.{tag}", "ACTIVE", now)
        multi = _tool(f"check.multi.{tag}", "ACTIVE", now)

        session.add_all([active, missing, in_child, multi])

        session.add_all([
            LinkModel(
                tool_id=active.tool_id,
                classification_id=parent.capability_id,
            ),
            LinkModel(
                tool_id=missing.tool_id,
                classification_id=parent.capability_id,
            ),
            LinkModel(
                tool_id=in_child.tool_id,
                classification_id=child.capability_id,
            ),
            # 一个工具挂两个能力 —— 命中一次就够了。
            LinkModel(
                tool_id=multi.tool_id,
                classification_id=parent.capability_id,
            ),
            LinkModel(
                tool_id=multi.tool_id,
                classification_id=other.capability_id,
            ),
        ])

        await session.flush()

        repository = ToolDefinitionRepositoryImpl(session)

        print("\n按 code 列表取工具")

        rows = await repository.list_active_by_classification_codes(
            [parent.code]
        )
        keys = [row.tool_key for row in rows]

        check("命中挂在父能力上的 ACTIVE 工具", active.tool_key in keys)
        check(
            "**不**递归到子能力（精确匹配）",
            in_child.tool_key not in keys,
        )
        check("MISSING 的工具被排除", missing.tool_key not in keys)
        check(
            "一个工具挂多个能力时只出现一次",
            keys.count(multi.tool_key) == 1,
        )
        check(
            "结果按 tool_key 排序（顺序稳定）",
            keys == sorted(keys),
        )

        print("\n边界")

        check(
            "code 列表为空 -> 空结果（不落成 IN ()）",
            await repository.list_active_by_classification_codes([]) == [],
        )
        check(
            "认不出的 code -> 空结果，不抛",
            await repository.list_active_by_classification_codes(
                [f"check.nope.{tag}"]
            ) == [],
        )

        many = await repository.list_active_by_classification_codes(
            [parent.code, other.code]
        )

        check(
            "一次给多个 code：两个都命中且整体去重",
            sorted(row.tool_key for row in many)
            == sorted([active.tool_key, multi.tool_key]),
        )

    finally:
        # 全程不 commit —— 回滚掉本次插入的一切。
        await session.rollback()
        await session.close()
        await engine.dispose()

    if _failures:
        print(f"\n失败 {len(_failures)} 条：")
        for label in _failures:
            print(f"  - {label}")
        return 1

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
