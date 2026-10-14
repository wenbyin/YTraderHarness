from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .base import DataSource
from .factory import DataSourceFactory
from .loader import DataSourceConfigLoader
from .manager import DataSourceManager
from .models import (
    DataSourceDefinition,
    DataSourceType,
)

logger = logging.getLogger(
    __name__
)


class DataSourceStartup:
    """
    DataSource 启动器。

    生命周期：

        discover config
            ↓
        validate
            ↓
        create client
            ↓
        health check
            ↓
        register
    """

    def __init__(
            self,
            config_directory: str | Path,
    ) -> None:
        self._loader = (
            DataSourceConfigLoader(
                config_directory
            )
        )

        self._manager = (
            DataSourceManager()
        )

    @property
    def manager(
            self,
    ) -> DataSourceManager:
        return self._manager

    async def initialize(
            self,
    ) -> DataSourceManager:

        definitions = (
            self._loader.load_all()
        )

        if not definitions:
            raise RuntimeError(
                "No datasource configuration found"
            )

        # 依赖别的数据源的定义必须排在后面 —— 目前只有
        # checkpointer（它的 DSN 可以从 postgresql 推导，
        # 因此 postgresql 必须先注册）。
        ordered = sorted(
            definitions,
            key=lambda d: (
                d.type
                == DataSourceType.POSTGRESQL_CHECKPOINTER
            ),
        )

        for definition in ordered:

            if not definition.enabled:
                logger.info(
                    "Datasource disabled: %s",
                    definition.name,
                )
                continue

            await self._initialize_one(
                definition
            )

        return self._manager

    async def _initialize_one(
            self,
            definition: DataSourceDefinition,
    ) -> None:

        source: DataSource | None = None

        try:

            source = (
                DataSourceFactory.create(
                    definition,
                    manager=self._manager,
                )
            )

            # 构造是同步的，所以那些"构造即建连"的客户端（Milvus 的
            # gRPC）需要一个 await 得着的地方把阻塞挪出事件循环 ——
            # 见 DataSource.initialize 的说明。默认空实现，绝大多数
            # 数据源一行都不用写。
            await source.initialize()

            healthy = await asyncio.wait_for(
                source.health_check(),
                timeout=(
                    definition
                    .startup_timeout_seconds
                ),
            )

            if not healthy:
                raise RuntimeError(
                    "Health check failed"
                )

            self._manager.register(
                source
            )

            logger.info(
                "Datasource ready: %s",
                definition.name,
            )

        except Exception as exc:

            # 构造成功后失败的数据源已经持有连接池 / HTTP 会话，
            # 但它没进 manager，close_all() 收不到它 —— 必须在这里
            # 关掉。否则每次启动都泄漏一个 session，asyncio 会在
            # 退出时报 "Unclosed client session"（实测见过）。
            #
            # 探活超时也走这条路径：wait_for 取消的是 health_check
            # 协程，客户端本身仍然完好，close() 正常生效。
            if source is not None:
                try:
                    await source.close()

                except Exception:
                    logger.warning(
                        "Failed to close datasource "
                        "after init failure: %s",
                        definition.name,
                        exc_info=True,
                    )

            if definition.required:
                logger.error(
                    "Required datasource "
                    "failed: %s",
                    definition.name,
                    exc_info=True,
                )

                raise RuntimeError(
                    f"Required datasource "
                    f"{definition.name} "
                    f"failed to initialize"
                ) from exc

            logger.warning(
                "Optional datasource "
                "unavailable: %s",
                definition.name,
            )
