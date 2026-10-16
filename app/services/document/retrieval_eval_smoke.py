"""
检索评测冒烟：**走一遍完整 lifespan**，验装配而不是验逻辑。

跑法：
    .venv/Scripts/python.exe -m app.services.document.retrieval_eval_smoke

为什么这个脚本与其它冒烟脚本不同
------------------------------
别的冒烟脚本（`task_store_smoke`、`memory_*_smoke`）**刻意跳过 lifespan**
——它们只验自己那一层，自己 create_all 建表，跑得快。本脚本要验的恰恰是
**lifespan 本身能不能起来**：

    数据源装配      五个后端（PG / Redis / ES / Milvus / Neo4j）各自握手
    建表            initialize_database 的 create_all 真把九张表建出来
    提示词注册表    两个新 model_role 的提示词被反射发现
    任务 handler    造题 handler 在集中注册点登记上了
    模型注册表      两个新 (agent_type, model_role) 绑定的 runtime 能解析
    服务装配        检索评测服务能造出来、能真查一次库

这几件在此之前**都是分开手工验的**（各自在一个临时脚本里），而"合起来
能不能起来"从来没跑过 —— 装配问题的典型形态就是每一件单独看都对、
拼在一起才暴露出顺序或依赖错。

⚠️ 它会真的建表（create_all 幂等，已存在就跳过），但不写任何业务数据。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

__all__ = ["main"]

_FAILS: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:

    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{detail}", flush=True)

    if not ok:
        _FAILS.append(f"{label}{detail}")


def _dsn() -> str:
    """从配置里读 Postgres DSN（与 datasource 加载器同一份来源）。"""

    import yaml

    raw = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "config" / "datasource" / "postgre.yaml"
        ).read_text(encoding="utf-8")
    )

    return raw["config"]["dsn"]


async def _check_tables() -> None:
    """九张表真的建出来了吗（这是**首次**真建）。"""

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(_dsn())

    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE tablename LIKE 'retrieval_eval%' "
                        "ORDER BY 1"
                    )
                )
            ).scalars().all()

    finally:
        await engine.dispose()

    _check("建表", len(rows) == 9, f"：{len(rows)}/9 张")

    for name in rows:
        print(f"          {name}", flush=True)


async def _check_prompts() -> None:
    """两个新 role 的提示词被反射发现了吗。"""

    from app.agent.model import ModelRole
    from app.agent.prompt import get_prompt_provider

    provider = get_prompt_provider()

    for role in (
        ModelRole.EVAL_QA_GENERATION,
        ModelRole.EVAL_CROSS_CHUNK_JUDGE,
    ):
        try:
            text = provider.node_prompt(role)
            _check(f"提示词 {role}", bool(text), f"：{len(text)} 字")

        except Exception as exc:  # noqa: BLE001 —— 冒烟要把原因打出来
            _check(f"提示词 {role}", False, f"：{type(exc).__name__}: {exc}")


def _check_handler() -> None:
    """造题 handler 在集中注册点上吗。"""

    from app.bootstrap.task import get_task_handler_registry
    from app.services.document.domain.tasks import (
        RETRIEVAL_EVAL_GENERATE_HANDLER,
    )

    known = list(get_task_handler_registry().known())

    _check(
        "造题 handler 注册",
        RETRIEVAL_EVAL_GENERATE_HANDLER in known,
        f"：已注册 {known}",
    )


def _check_model_roles() -> None:
    """两个新绑定能解析出 runtime 吗。"""

    from app.agent.model import ModelRole, get_model_registry

    registry = get_model_registry()

    for agent_type, role in (
        ("retrieval_eval_generate", ModelRole.EVAL_QA_GENERATION),
        ("retrieval_eval_adjudicate", ModelRole.EVAL_CROSS_CHUNK_JUDGE),
    ):
        try:
            registry.get_runtime(agent_type=agent_type, model_role=role)
            _check(f"runtime {agent_type}", True)

        except Exception as exc:  # noqa: BLE001
            _check(
                f"runtime {agent_type}",
                False,
                f"：{type(exc).__name__}: {exc}",
            )


async def _check_service() -> None:
    """检索评测服务能装配、能真查一次库吗。"""

    from app.common.factories import get_service_factory
    from app.services.document.application.retrieval_eval_service import (
        RetrievalEvalService,
    )

    try:
        factory = get_service_factory()

        async with factory.scope(RetrievalEvalService) as service:

            listed = await service.list_bundles(page=1, page_size=1)

            _check(
                "服务装配 + 真查库",
                True,
                f"：{type(service).__name__}, total={listed.total}",
            )

    except Exception as exc:  # noqa: BLE001
        _check(
            "服务装配 + 真查库",
            False,
            f"：{type(exc).__name__}: {exc}",
        )


async def main() -> int:
    """
    跑完整 lifespan，逐项验装配，然后干净退出。

    ⚠️ **不进服务、不接请求** —— 起完就关。这样能一次暴露数据源/建表/
    注册表/装配这几层里拼在一起才会出现的问题，而不用先开一个端口等它。
    """

    from app.main import app

    print("启动 lifespan（reranker 与云端模型预热会花几分钟）……", flush=True)

    async with app.router.lifespan_context(app):

        print("lifespan 起来了，开始逐项验：", flush=True)

        await _check_tables()
        await _check_prompts()
        _check_handler()
        _check_model_roles()
        await _check_service()

    print("lifespan 已干净退出。", flush=True)

    if _FAILS:
        print(f"\n失败 {len(_FAILS)} 项：")
        for item in _FAILS:
            print(f"  - {item}")
        return 1

    print("\n全部通过。")
    return 0


if __name__ == "__main__":

    # ⚠️ **必须传 `loop_factory`**，见另一个冒烟脚本里同一处的长说明：
    # Windows 上 `asyncio.run` 默认建 ProactorEventLoop，而 psycopg 跑不了
    # 它 —— lifespan 会在连库前显式拦下。走 app/serve.py 那份 factory，
    # 与 `python run_server.py` 同一条路径。
    from app.serve import loop_factory

    raise SystemExit(
        asyncio.run(main(), loop_factory=loop_factory())
    )
