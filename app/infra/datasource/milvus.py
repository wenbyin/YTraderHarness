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

    ⚠️ 客户端的**构造**同样阻塞，而且是把整个事件循环按住
    ---------------------------------------------------
    `MilvusClient(...)` 在构造时就去 gRPC 建连（连不上会重试着等）。
    它原先写在 `__init__` 里，而 `__init__` 是同步的 —— 实测本地 Milvus
    没起时按住事件循环 **3.3 秒**：那几秒里所有请求、SSE 心跳、以及 Redis
    的读超时定时器一起停摆，最后以一条"Timeout reading from
    localhost:6379"收场，看起来完全是网络故障。已由卡顿哨兵抓到
    （app/common/loop_watchdog.py）。

    所以构造挪进了 `initialize()` —— 那里能 await，因此能包 to_thread。
    启动顺序是 create -> initialize -> health_check -> register。
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

        self._kwargs = kwargs

        #: 在 initialize() 里建（见类 docstring）。None = 还没建成功，
        #: 此时任何取用都是编程错误 —— 由 _require_client 明确报出来，
        #: 而不是让 None 传到下游变成一个看不懂的 AttributeError。
        self._client: MilvusClient | None = None

    async def initialize(self) -> None:
        """
        建客户端 —— **必须在这里，不能搬回 `__init__`**。

        理由见类 docstring：构造即建连，而循环上等不起。用 to_thread 把
        它挪到线程里，事件循环在那几秒里照常服务（心跳照发、请求照回）。
        """

        self._client = await asyncio.to_thread(
            MilvusClient,
            **self._kwargs,
        )

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(
                self._require_client().list_collections
            )

            return True

        except Exception:
            return False

    async def close(self) -> None:
        if self._client is None:
            # 没建成功（或已被关过一次）—— 没什么可关的。
            return

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
        return self._require_client()

    @property
    def client(
            self,
    ) -> MilvusClient:
        return self._require_client()

    # ========================================================

    def _require_client(self) -> MilvusClient:
        """取客户端；还没初始化就明确报错。"""

        if self._client is None:
            raise RuntimeError(
                "Milvus 客户端尚未初始化：initialize() 没跑或没成功。"
                "数据源只有在 initialize + health_check 都通过后才会被"
                "注册进 manager，因此走到这里说明初始化被绕过了。"
            )

        return self._client
