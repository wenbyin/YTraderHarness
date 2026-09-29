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

import importlib
import pkgutil

from sqlalchemy.ext.asyncio import AsyncEngine

from app.common.models import Base


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
    ]

    for package_name in packages:
        package = importlib.import_module(
            package_name
        )

        if not hasattr(package, "__path__"):
            continue

        for module_info in pkgutil.walk_packages(
                package.__path__,
                package.__name__ + ".",
        ):
            importlib.import_module(
                module_info.name
            )


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
