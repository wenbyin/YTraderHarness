from __future__ import annotations

from neo4j import AsyncDriver, AsyncGraphDatabase

from .base import DataSource
from .models import (
    Neo4jDataSourceDefinition,
)


class Neo4jDataSource(DataSource):
    """
    Neo4j DataSource。

    只负责 AsyncDriver 的生命周期。

    不负责：
        - Cypher 语句
        - 约束 / 索引初始化
        - GraphRAG 检索逻辑
    """

    def __init__(
            self,
            definition: Neo4jDataSourceDefinition,
    ) -> None:
        super().__init__(
            definition.name
        )

        config = definition.config

        # 不传 username 就当匿名访问；传了但没密码则按空密码走，
        # 让服务端去拒绝，而不是在这里静默降级成匿名连接。
        auth = (
            (
                config.username,
                config.password or "",
            )
            if config.username
            else None
        )

        # 驱动是惰性的：这里不建连，配置写错要到 health_check
        # 才会暴露。这正是 health_check 存在的意义。
        self._driver: AsyncDriver = (
            AsyncGraphDatabase.driver(
                config.uri,
                auth=auth,
                max_connection_pool_size=(
                    config.max_connection_pool_size
                ),
                connection_timeout=(
                    config.connection_timeout
                ),
                max_connection_lifetime=(
                    config.max_connection_lifetime
                ),
            )
        )

        self._database = config.database

    async def health_check(self) -> bool:
        try:
            await self._driver.verify_connectivity()

            return True

        except Exception:
            return False

    async def close(self) -> None:
        await self._driver.close()

    def resource(
            self,
    ) -> AsyncDriver:
        return self._driver

    @property
    def driver(
            self,
    ) -> AsyncDriver:
        return self._driver

    @property
    def database(
            self,
    ) -> str | None:
        """
        目标数据库名，None 表示用服务端默认库。

        调用方每次开 session 都要带上它，否则会落到服务端
        默认库上 —— 在配了多库的实例上就是查错库。
        """
        return self._database
