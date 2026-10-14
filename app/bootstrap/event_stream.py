"""
运行事件流（Redis Stream）：发布端、广播端、以及它们的停摆。

事件是**旁路**：Redis 连不上也要能启动、也能跑完，只是订阅者看不到中间
过程。因此下面用 get_optional 而不是 get —— redis.yaml 是 required: false，
拿不到就装一个空发布器（NullRunEventPublisher），节点侧照常调用、
只是什么都不发。

⚠️ 位置：必须在业务图装配**之前** —— 节点第一次运行就要能拿到发布器。

连接不在这里建、也不在这里关
--------------------------
client 归 RedisDataSource 所有，我们只是接过来用；关闭是
datasource_manager.close_all() 的事。两份引用 = 两个释放时机，而释放
两次是未定义行为。

发布端与读取端用的**不是同一个** client，这一点是刻意的：

    读取端  datasource.stream_client —— asyncio，**长连接专用池**
    发布端  datasource.sync_client   —— 同步客户端，跑在写线程里

写线程存在的理由就是"不依赖事件循环"，而 asyncio 客户端离了循环什么都
不是；详见 app/agent/event/publisher.py 的模块 docstring。

⚠️ 读取端刻意**不用** datasource.resource()
----------------------------------------
那个是业务短查询用的池。订阅者在整个订阅期间一直占着它那条连接，
共用一个池的后果实测过：并发订阅数一到池上限，全进程的 Redis 访问一起
失败（包括与事件流毫无关系的调用）。所以长连接单独一个池，见
app/infra/datasource/redis.py 的类 docstring。

停摆顺序由组合根决定
------------------
本模块只提供"关自己"的那一步（aclose_event_stream）。它与图调度器、
数据源之间的**先后**约束跨模块，因此留在 main.py 的 finally 里显式排 ——
理由写在那边。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.event import (
    RedisStreamSink,
    build_event_publisher,
    build_run_stream_registry,
    close_event_publisher,
    reset_run_stream_registry,
    start_event_publisher,
)
from app.services.session.application.run_liveness import (
    probe_run_liveness,
)

__all__ = ["aclose_event_stream", "initialize_event_stream"]

logger = logging.getLogger(__name__)


async def initialize_event_stream(
        datasource_manager: Any,
) -> None:
    """
    装配并启动事件流的发布端与广播端。

    拿不到 Redis 数据源时不报错：装空发布器，运行照常。
    """

    redis_source = datasource_manager.get_optional("redis")

    sink = (
        RedisStreamSink(redis_source.sync_client)
        if redis_source is not None
        else None
    )

    # build_* 是同步纯装配，起线程与探活放在 start_event_publisher 里 ——
    # 它是 IO，await 的时机要在代码里一眼看得见。探活失败会自己退回空
    # 对象，运行照常。
    build_event_publisher(sink)
    await start_event_publisher()

    # 广播端的探针：问"这个 run 还会产出事件吗"。
    #
    # ⚠️ 这里给的是**函数本身**，不是调用结果 —— 它要等事件流空闲时才被
    #    调用，而那时 lifespan 早就装配完了。函数内部靠
    #    get_graph_dispatcher() / get_repository_factory() 取依赖，因此
    #    不受"读取端先于 dispatcher 装配"这个顺序的影响。
    build_run_stream_registry(
        redis_source.stream_client if redis_source is not None else None,
        probe=probe_run_liveness,
    )


async def aclose_event_stream() -> None:
    """
    停掉发布端的写线程与广播端。

    **不关 client** —— 它归数据源管理器所有（由
    datasource_manager.close_all() 关）。这里只是别让模块级单例继续指着
    一个已关闭的连接池 —— 那个症状要等到下次有人 import 并调用时才出现。

    ⚠️ 必须在 datasource_manager.close_all() **之前**调用：反过来是
    "连接池先被关、任务还在它上面阻塞读"，收尾日志里会刷出一串看起来像
    "Redis 挂了"的噪音。

    先停写线程（close_event_publisher），再停广播端：写线程脚下的同步
    客户端归数据源管，线程还活着时关池，它后面每条都是 ConnectionError。
    """

    try:
        await close_event_publisher()
    except Exception:
        logger.exception("Failed to stop event publisher.")

    # reset_* 会把单例取出来并清掉，因此不必再单独 reset_event_publisher。
    registry = reset_run_stream_registry()

    if registry is not None:
        try:
            await registry.aclose()
        except Exception:
            logger.exception("Failed to stop event stream registry.")
