"""
Database initialization.

Responsibilities
----------------
- Ensure all SQLAlchemy models are imported.
- Create database tables according to SQLAlchemy metadata.

Important
---------
This initializer creates TABLES inside an existing PostgreSQL database.

It does NOT create the PostgreSQL database itself.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from app.common.models import Base
from app.common.utils import import_package_modules


# ============================================================
# IMPORTANT
# ============================================================
#
# SQLAlchemy only knows about models that have already been
# imported before Base.metadata.create_all() is called.
#
# Therefore all entities must be imported here.
#
# These imports intentionally exist for model registration.
# ============================================================


def import_all_models() -> None:
    """
    自动发现并导入所有 SQLAlchemy Model。

    约定：
        所有 Model 必须位于：
            app.services.*.models.sqlalchemy

    只要符合这个目录规范，
    新增 Model 后无需修改 initializer.py。
    """

    packages = [
        "app.services.file.models.sqlalchemy",
        "app.services.document.models.sqlalchemy",
        # ⚠️ 这个列表是**硬编码**的 —— 与上面 docstring 说的
        # "符合目录规范就无需修改本文件"不符。新增模块仍要加一行。
        # 改成自动发现 app.services.*.models.sqlalchemy 是独立改动，
        # 会动到既有模块，单独做。
        "app.services.tool.models.sqlalchemy",
        # capability 与 tool 之间有一条**跨模块** relationship
        # （Capability.definition_links ↔ ToolDefinitionClassification
        # .classification）。两个模块的模型必须都 import 过，mapper
        # 才能解析那个字符串目标 —— 缺一个就是那句离原因很远的
        # "failed to locate a name"。
        "app.services.capability.models.sqlalchemy",
        "app.services.session.models.sqlalchemy",
        # 挂在 agent_message 上的两张记忆索引表（agent_message_fts /
        # agent_message_vec）。
        #
        # ⚠️ 这一行有**前置条件**：agent_message_vec 的列类型是 pgvector
        # 的 `vector(512)`，扩展没装时 create_all 会报
        # `type "vector" does not exist` —— 而那是**应用启动失败**，
        # 不是这个模块不可用。所以顺序是死的：
        #     装扩展文件 → 重启 PG → CREATE EXTENSION vector → 才加这一行
        # 详见 app/services/memory/models.sql 顶部与 models/sqlalchemy/
        # agent_message_vec.py 的模块 docstring。
        #
        # ⚠️ 它对 session 模块没有反向依赖：两张表**没有外键**（与
        # agent_message 自己同样的取舍），所以 session 先建、memory 后建
        # 都行，改动 session 那边不会连带这里。
        "app.services.memory.models.sqlalchemy",
        # 统一异步任务基础设施（tasks / task_outbox）。
        # 它没有跨模块 relationship，本来漏了也不会报"failed to locate
        # a name"—— 漏加的症状是表建不出来，且启动**不报错**，
        # 直到第一条 Task 提交时才炸 "relation tasks does not exist"。
        "app.services.task.models.sqlalchemy",
    ]

    import_package_modules(*packages)


async def initialize_database(
        engine: AsyncEngine,
) -> None:
    """
    Create all missing database tables.

    Existing tables are NOT modified or dropped.

    create_all() is idempotent:
        - table exists    -> do nothing
        - table missing   -> create it
    """
    import_all_models()

    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all
        )
