from __future__ import annotations

from elasticsearch import AsyncElasticsearch

from .base import DataSource
from .models import (
    ElasticsearchDataSourceDefinition,
)


class ElasticsearchDataSource(DataSource):
    """
    Elasticsearch DataSource。

    只负责 ES client 生命周期。

    不负责：
        - Index mapping
        - Query DSL
        - Document search
    """

    def __init__(
            self,
            definition: ElasticsearchDataSourceDefinition,
    ) -> None:
        super().__init__(
            definition.name
        )

        config = definition.config

        kwargs = {
            "hosts": config.hosts,
            "verify_certs": config.verify_certs,
            "request_timeout": config.request_timeout,
        }

        # 刻意不传 node_class，用 elastic_transport 的默认值
        # （AiohttpHttpNode）。这里记下为什么**不**改用 httpx：
        #
        # httpx 后端（HttpxAsyncHttpNode）内部是
        #     httpx.AsyncClient(..., )   # 没给 trust_env
        # trust_env 默认为 True，于是它会去读环境变量**和 Windows
        # 注册表**里的系统代理。装了 Clash 之类工具的机器上注册表里
        # 有 http://127.0.0.1:7890，结果连 localhost:9200 都会被塞进
        # 代理，返回 502 Bad Gateway —— 本机实测复现。
        #
        # 而 aiohttp 从不读系统代理（trust_env 默认 False），
        # 这正是数据库客户端该有的行为：把 ES 查询悄悄转发给一个
        # 本机代理既连不通、也等于把内网请求泄给了它。
        #
        # 这个坑发生在上游 __init__ 里，没有配置项可以关掉，所以只能
        # 靠选对后端。aiohttp 不在依赖里时它会抛 ValueError，
        # 被 startup 降级成一条 "Optional datasource unavailable"
        # 警告，ES 从此静默不可用 —— 因此 requirements.txt 里
        # aiohttp 是显式 pin 的。

        if config.username:
            kwargs["basic_auth"] = (
                config.username,
                config.password or "",
            )

        self._client = AsyncElasticsearch(
            **kwargs
        )

    async def health_check(self) -> bool:
        try:
            return bool(
                await self._client.ping()
            )

        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()

    def resource(
            self,
    ) -> AsyncElasticsearch:
        return self._client

    @property
    def client(
            self,
    ) -> AsyncElasticsearch:
        return self._client
