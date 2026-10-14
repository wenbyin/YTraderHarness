from __future__ import annotations

from typing import Any

import redis as redis_sync
from redis.asyncio import Redis

from .base import DataSource
from .models import (
    RedisConfig,
    RedisDataSourceDefinition,
)


def _client_options(
        config: RedisConfig,
) -> dict[str, Any]:
    """
    `from_url` 的共用选项 —— async 与 sync 两个客户端**只在这里写一次**。

    两处各抄一遍的代价是漂移：将来加一个参数必然漏掉一处，而漏掉的那份
    会拿 redis-py 的默认值静默跑起来（例如漏了 socket_timeout，卡住的
    连接就永久挂住，正是 redis.yaml 里那段注释要避免的事）。
    """

    return dict(
        max_connections=config.max_connections,
        socket_timeout=config.socket_timeout,
        socket_connect_timeout=config.socket_connect_timeout,
        health_check_interval=config.health_check_interval,
        decode_responses=config.decode_responses,
        client_name=config.client_name,
    )


class RedisDataSource(DataSource):
    """
    Redis DataSource。

    只负责 Redis 连接池的生命周期。

    持**三份**客户端，生命周期都归这里
    ----------------------------------
        client          asyncio 客户端，业务短查询用（`resource()` / `client`）
        sync_client     同步客户端，事件发布端的写线程专用
        stream_client   asyncio 客户端，事件流的**长连接**订阅专用

    三份不是"同一个池的三次引用"，而是三个独立的池（参数由同一份
    RedisConfig 展开，见 _client_options）—— 所以"多份引用 = 多个释放
    时机"那条纪律并没有被违反：这里始终是唯一的所有者，close() 一次
    把三份都关掉。

    ⚠️ 为什么订阅端必须单独一个池
    ----------------------------
    前两个的占用形态都是"借一下还回去"：一条命令发完就还池。事件流订阅
    不是 —— 订阅者在整个订阅期间**一直占着**它那条连接（一个 XREAD BLOCK
    接一个，中间几乎不还给池）。

    共用一个池的后果实测过：池上限设成 10 时，并发订阅数一到 10，整个
    进程的 Redis 访问就一起失败，报 redis.exceptions.MaxConnectionsError。
    而且那条异常继承自 ConnectionError，会被读取端的"网络抖动重试"逻辑
    接住、无退避地空转三次再抛 500 —— 浏览器 EventSource 收到 500 就按
    规范自动重连，于是变成"池满 → 500 → 重连 → 池还是满"的死循环。

    拆开之后，"订阅者把池占满"最多只影响订阅者自己，短命令与发布端不受
    牵连。这也是 redis.py 里 sync_client 当初独立出去的同一条理由。

    不负责：
        - 键的命名与前缀策略
        - 值的序列化格式
        - 缓存失效 / 淘汰策略

    这些刻意不在这里做。同一个 Redis 上，缓存要的序列化格式
    和分布式锁要的未必一样，把任何一种定在这里，等于替所有
    调用方做了选择，而选错的表现是"存进去的东西取出来不认得"。
    """

    def __init__(
            self,
            definition: RedisDataSourceDefinition,
    ) -> None:
        super().__init__(
            definition.name
        )

        config = definition.config

        # from_url 是惰性的：只解析 URL、建连接池对象，**不建连**。
        # 配置写错要到 health_check 才会暴露 —— 这正是它存在的意义，
        # 与 neo4j / elasticsearch 那两个数据源一致。
        #
        # 连接池由 client 自己持有，close() 时一并释放；外面不需要
        # （也不该）再单独存一个 pool 引用 —— 两份引用意味着两个
        # 释放时机，而释放两次是未定义行为。
        self._client: Redis = Redis.from_url(
            config.url,
            **_client_options(config),
        )

        # 事件发布端的**写线程**专用。同样惰性：from_url 只解析 URL、
        # 建连接池对象，不建连。
        #
        # 为什么不复用上面那个
        # --------------------
        # 上面那个是 asyncio 客户端，只在"某个事件循环里被 await"时才有
        # 意义。写线程里没有循环可 await，而把协程丢回循环
        # （run_coroutine_threadsafe）又会被"循环卡住"连带拖死 —— 那种
        # 卡顿恰恰是写线程存在的理由。所以它必须是一份**同步**客户端。
        #
        # max_connections=1：只有一条写线程用它。redis 的命令执行本身是
        # 单线程的，开大只是白占服务端的连接额度（见 models.py 里
        # RedisConfig.max_connections 的说明）。
        self._sync_client: redis_sync.Redis = redis_sync.Redis.from_url(
            config.url,
            **{
                **_client_options(config),
                "max_connections": 1,
            },
        )

        # 事件流订阅端的**长连接**专用。同样惰性：from_url 只解析 URL、
        # 建连接池对象，不建连。
        #
        # 与上面那个 asyncio 客户端的区别**只有池上限**：占用的形态不同，
        # 所以额度必须分开算（理由见类 docstring）。socket_timeout 等参数
        # 仍由 _client_options 统一展开 —— 那一条是"读超时/阻塞时长"的
        # 约束，两边一致才对。
        self._stream_client: Redis = Redis.from_url(
            config.url,
            **{
                **_client_options(config),
                "max_connections": config.stream_max_connections,
            },
        )

    async def health_check(self) -> bool:
        try:
            await self._client.ping()

            return True

        except Exception:
            return False

    async def close(self) -> None:
        # 用 aclose() 而不是 close()：后者从 redis-py 5.0.1 起是
        # 已废弃的别名，走它会刷 DeprecationWarning。
        await self._client.aclose()

        # 长连接那份。同样必须**先**停掉订阅端（build/close 那一对，
        # 见 app/agent/event/run_stream.py 的注册表 aclose），再走到这里
        # —— 有任务还在它脚下的连接上阻塞读取时关池，那些任务接下来每条
        # 都是 ConnectionError，收尾日志里看起来像"Redis 挂了"。
        await self._stream_client.aclose()

        # 同步客户端只有 close()，没有 aclose。
        #
        # ⚠️ 调用顺序有要求：必须**先**把事件发布端的写线程停掉（
        #    close_event_publisher），再走到这里。线程还在用它脚下的
        #    连接时关池，写线程接下来每条都是 ConnectionError —— 那会
        #    在收尾日志里刷出一串看起来像"Redis 挂了"的噪音。
        self._sync_client.close()

    def resource(
            self,
    ) -> Redis:
        return self._client

    @property
    def client(
            self,
    ) -> Redis:
        return self._client

    @property
    def stream_client(
            self,
    ) -> Redis:
        """
        asyncio 客户端，事件流的**长连接**订阅专用（独立连接池）。

        与 `client` 一样，生命周期归本数据源 —— 取用方只借不关。**更**不
        该在订阅结束时关掉它：那会连带关掉池里别的订阅者脚下的连接。

        取用方只有组合根（app/main.py 装配事件流读取端时），以及
        app/agent/event/run_stream.py 里的订阅实现。
        """

        return self._stream_client

    @property
    def sync_client(
            self,
    ) -> redis_sync.Redis:
        """
        同步客户端，给事件发布端的**写线程**用。

        与 `client` 一样，生命周期归本数据源 —— 取用方只借不关
        （两份引用 = 两个释放时机，释放两次是未定义行为）。
        """

        return self._sync_client
