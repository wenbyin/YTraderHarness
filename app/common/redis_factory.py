"""
应用级 Redis 客户端工厂 —— 与 `RepositoryFactory.session_factory` 同形。

为什么需要它
------------
在它之前，Redis 客户端只有两个获取途径：

    组合根（lifespan）里 `datasource_manager.get_optional("redis").client`
    启动时把 client 塞进某个模块级单例（工具缓存就是这么做的）

后者的问题是"必须先初始化"：任何新消费者都要在 lifespan 里再加一行
初始化，而且取用方只能相信自己拿到的那个全局变量。

这里改成与数据库会话工厂一样的形状：

    lifespan:  build_redis_factory(redis_source)     # 装配一次
    任何地方:  get_redis_factory().client()          # 用时再取

`build_redis_factory` 接受 DataSource 的鸭子类型（有 `client` /
`sync_client` / `stream_client`），本模块**不认识** app.infra，符合
app.common 不反向依赖的纪律。

⚠️ Redis 是 `required: false`：source 可以是 None。这时
`available` 为 False，`optional_client()` 返回 None；需要强制存在的
调用方用 `client()` —— 它会抛一条说清原因的 RuntimeError。
"""

from __future__ import annotations

from typing import Any, Protocol


__all__ = ["RedisClientFactory", "RedisClients"]


class RedisClients(Protocol):
    """
    DataSource 的鸭子类型。

    三份客户端各有用途（见 infra/datasource/redis.py）：
        client         业务短查询（asyncio）
        sync_client    事件发布端写线程（同步）
        stream_client  事件流长连接订阅（asyncio）
    """

    @property
    def client(self) -> Any: ...

    @property
    def sync_client(self) -> Any: ...

    @property
    def stream_client(self) -> Any: ...


class RedisClientFactory:
    """
    Redis 客户端的取用入口。

    与 `RepositoryFactory` 一样是**holder**：内容由组合根注入，本类
    只负责"保存 + 按需交出"，不建连、不关池（生命周期归数据源）。
    """

    def __init__(self, source: RedisClients | None = None) -> None:
        self._source = source

    @property
    def available(self) -> bool:
        return self._source is not None

    def optional_client(self) -> Any | None:
        """asyncio 业务短查询客户端；Redis 未装配时返回 None。"""

        return None if self._source is None else self._source.client

    def client(self) -> Any:
        """asyncio 业务短查询客户端；Redis 未装配时抛。"""

        if self._source is None:
            raise RuntimeError(
                "Redis 数据源没有装配（redis.yaml 的 required=false，"
                "本地没起 Redis 时这是正常状态）。"
                "需要 Redis 的调用方应当先用 optional_client() 判断，"
                "或退回进程内实现。"
            )

        return self._source.client

    def optional_sync_client(self) -> Any | None:
        return None if self._source is None else self._source.sync_client

    def optional_stream_client(self) -> Any | None:
        return None if self._source is None else self._source.stream_client
