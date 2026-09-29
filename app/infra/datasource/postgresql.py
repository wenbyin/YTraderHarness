from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import DataSource
from .models import (
    PostgreSQLDataSourceDefinition,
)


class PostgreSQLDataSource(DataSource):
    """
    PostgreSQL DataSource。

    提供：
        - AsyncEngine
        - AsyncSession factory
    """

    def __init__(
            self,
            definition: PostgreSQLDataSourceDefinition,
    ) -> None:
        super().__init__(
            definition.name
        )

        config = definition.config

        self._engine: AsyncEngine = (
            create_async_engine(
                config.dsn,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.pool_timeout,
                pool_recycle=config.pool_recycle,
                echo=config.echo,
                pool_pre_ping=True,
            )
        )

        self._session_factory = (
            async_sessionmaker[
                AsyncSession
            ](
                bind=self._engine,
                expire_on_commit=False,
            )
        )

        self._config = config

    @property
    def resolved_dsn_psycopg(self) -> str:
        """
        裸 psycopg DSN，供 LangGraph checkpointer 使用。

        暴露在这里是为了让 checkpointer 数据源能通过
        `derive_from: postgresql` 复用本数据源的配置 ——
        密码只写一处，也就不会出现两处不一致。

        见 app/infra/datasource/factory.py 的
        _resolve_checkpointer_dsn。
        """
        return self._config.resolved_dsn_psycopg

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(
            self,
    ) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def health_check(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(
                    text("SELECT 1")
                )

            return True

        except Exception as err:
            print(err)
            return False

    async def close(self) -> None:
        await self._engine.dispose()

    def resource(self) -> AsyncEngine:
        return self._engine
