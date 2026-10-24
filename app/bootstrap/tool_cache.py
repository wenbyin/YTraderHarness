"""
工具结果缓存的装配。

生产用 Redis
------------
`RedisDataSource.client` 是 asyncio 业务短查询池（不是事件流那个长连接
池，也不是发布端的同步客户端）。缓存命令都是短命令，借一下还回去，
用这个池正好。

拿不到 Redis 时**不报错**
------------------------
`redis.yaml` 是 required: false，本地没起 Redis 是常态。这时退回进程内
实现并打 warning：缓存是优化，不该拦住应用启动；代价是多实例不共享、
重启清零。

client 归数据源所有
------------------
这里只把 client 交给 `RedisToolCache`，**不**负责关闭。关闭是
`datasource_manager.close_all()` 的事（与事件流同一条纪律：两份引用 =
两个释放时机，释放两次是未定义行为）。
"""

from __future__ import annotations

import logging

from app.agent.tools.processor.runtime.cache import (
    InProcessToolCache,
    RedisToolCache,
    configure_tool_cache,
)
from app.common.factories import get_redis_factory

__all__ = ["initialize_tool_cache"]

logger = logging.getLogger(__name__)


def initialize_tool_cache() -> None:
    """
    选择并安装进程级工具缓存实现。

    必须在图装配之前调用 —— 节点第一次跑工具时就要拿到最终实现。

    Redis 客户端从应用级 `RedisClientFactory` 取（与 session_factory
    同形），不再直接摸 `DataSourceManager`。
    """

    client = get_redis_factory().optional_client()

    if client is None:
        logger.warning(
            "工具缓存：Redis 数据源不可用，退回进程内实现。"
            "多实例之间不共享、重启清零 —— 缓存仍然可用，只是命中率"
            "低于理论值。"
        )

        configure_tool_cache(InProcessToolCache())

        return

    configure_tool_cache(
        RedisToolCache(client)
    )

    logger.info("工具缓存：使用 Redis。")
