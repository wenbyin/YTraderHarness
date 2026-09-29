from __future__ import annotations

import asyncio

from pymilvus import MilvusClient

from .base import DataSource
from .models import (
    MilvusDataSourceDefinition,
)


class MilvusDataSource(DataSource):
    """
    Milvus DataSource。

    只负责 Milvus Client 生命周期。

    MilvusClient 当前主要提供同步接口，
    因此 async health check 通过 asyncio.to_thread()
    避免阻塞 FastAPI event loop。
    """

    def __init__(
            self,
            definition: MilvusDataSourceDefinition,
    ) -> None:
        super().__init__(
            definition.name
        )

        config = definition.config

        kwargs = {
            "uri": config.uri,
        }

        if config.token:
            kwargs["token"] = config.token

        if config.db_name:
            kwargs["db_name"] = config.db_name

        self._client = MilvusClient(
            **kwargs
        )

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(
                self._client.list_collections
            )

            return True

        except Exception:
            return False

    async def close(self) -> None:
        close_method = getattr(
            self._client,
            "close",
            None,
        )

        if close_method is None:
            return

        await asyncio.to_thread(
            close_method
        )

    def resource(
            self,
    ) -> MilvusClient:
        return self._client

    @property
    def client(
            self,
    ) -> MilvusClient:
        return self._client
