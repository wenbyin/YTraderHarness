"""
运行事件发布端的 Redis Stream 实现（sink + 进程级发布器）。

契约在 `app/common/event/publisher.py`（EventPublisher）；这里只放
"怎么写进 Redis"：

    EventSink              写线程真正落地的地方（同步接口）
    RedisStreamSink        XADD + EXPIRE，一次往返
    build_event_publisher  有 sink -> RouterEventPublisher；没有 -> 空对象
    emit_event             唯一入口：一条规范事件，由 EventRouter 按 sink 分发

排队、写线程、丢弃策略在 `sink.py`（每个 sink 一个 `SinkRuntime`）。

事件是**旁路**：`emit_event` 与所有 sink 操作都**从不外抛**。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from app.common.event import EventPublisher
from app.common.event.spec import Event

from .config import StreamConfig, load_stream_config

if TYPE_CHECKING:
    import redis as redis_sync

__all__ = [
    "EventSink",
    "NullEventPublisher",
    "RedisStreamSink",
    "build_event_publisher",
    "close_event_publisher",
    "emit_event",
    "get_event_publisher",
    "reset_event_publisher",
    "start_event_publisher",
]


logger = logging.getLogger(__name__)


@runtime_checkable
class EventSink(Protocol):
    """
    写线程真正落地的地方。

    **同步接口** —— 它跑在没有事件循环的线程里，因此刻意不认识 asyncio
    （也正因为不认识，事件循环被卡住时它照常干活）。

    单独抽出这一层是为了让"怎么写进 Redis"与"怎么排队、怎么停"分开：
    测试只换前者（FakeSink）就能把线程逻辑跑完，不必碰真 Redis。
    """

    def ping(self) -> bool:
        """探活。**绝不抛** —— 失败就是 False。"""

        ...

    def write(
            self,
            key: str,
            fields: Mapping[str, Any],
            *,
            maxlen: int | None,
            ttl_seconds: int | None,
    ) -> None:
        """
        写一条记录。**失败必须抛** —— 由写线程接住并留痕（见 _consume）。

        刻意不给它"返回 bool"的选项：这里只有两种结局（写成功、抛异常），
        多一个返回值就多一处"到底该信谁"的歧义。
        """

        ...


class RedisStreamSink:
    """
    往 `<key_prefix>:<topic_id>` 写一条 Stream。

    收的是 infra/datasource/redis.py 的 `sync_client`（同步客户端）。
    与别处一样：**借用**，不负责关闭 —— 连接池生命周期归数据源管理器。
    """

    def __init__(
            self,
            client: "redis_sync.Redis",
    ) -> None:

        self._client = client

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())

        except Exception:       # noqa: BLE001 —— 探活失败就是"没通道"
            return False

    def write(
            self,
            key: str,
            fields: Mapping[str, Any],
            *,
            maxlen: int | None,
            ttl_seconds: int | None,
    ) -> None:

        # ⚠️ `with` 只负责收尾时 reset()，**不会**替你 execute()。
        # 少了下面那句 pipe.execute()，命令会静静地躺在缓冲里被丢掉 ——
        # 症状是"一条事件都没发出去，但也不报错"，是本模块最难查的一种。
        with self._client.pipeline(
                transaction=False,
        ) as pipe:

            # MAXLEN ~ 近似裁剪，理由见 event_stream.yaml。
            pipe.xadd(
                key,
                fields,
                maxlen=maxlen,
                approximate=True,
            )

            # TTL 每次写入都刷新（滑动）。maxlen 限制**一条** stream
            # 有多长，TTL 保证**键**最终会消失 —— 两者管的东西不同。
            if ttl_seconds:
                pipe.expire(key, ttl_seconds)

            # XADD 与 EXPIRE 之间没有原子性要求，所以不开 MULTI
            # （transaction=False）：省一层 MULTI/EXEC。
            pipe.execute()


class NullEventPublisher:
    """
    什么都不做的发布器。

    **不是**"随便塞一个空实现"的反模式 —— 它有明确的职责：Redis 是
    `required: false`（app/config/datasource/redis.yaml），本地连不上时应用要能
    照常启动，而节点侧不该因为"没有事件通道"就跑不起来。用一个空对象
    收下调用，好过在每个调用点写 `if publisher is not None` —— 那样
    每处都要记得判一次，漏一处就是一次 NoneType 崩溃。

    构造它**不读配置文件**：空对象的全部意义就是"在任何环境下都能
    构造出来"，配置缺失时也必须能。
    """

    @property
    def enabled(self) -> bool:
        """永远是 False —— 这正是它存在的意义。"""

        return False

    async def emit(self, event: Event) -> dict[str, bool]:
        """没有 sink，自然没有分发结果。"""

        return {}


# ============================================================
# 进程级单例
# ============================================================
#
# 事件本来就要汇到一个地方再往外发，因此是单例而不是每次构造 ——
# 形状与 get_model_factory / get_tool_event_bus 一致。

#: 进程级单例。None = 还没装配（见 get_event_publisher）。
_publisher: EventPublisher | None = None


def build_event_publisher(
        sink: EventSink | None,
) -> EventPublisher:
    """
    装配进程级发布器。**同步、不碰 IO**，由组合根调用一次。

    sink 为 None（Redis 数据源没起来 —— `required: false` 时这是常态）
    就装一个空对象，于是调用方永远不需要判断"有没有事件通道"。

    只装配**不启动**：探活与起线程都是 IO，放在 `start_event_publisher`
    里 —— 于是 main.py 的 lifespan 里"await 了什么"一眼可见，与
    build_run_stream_registry 的同步装配同一形状。
    """

    global _publisher

    if sink is None:
        _publisher = NullEventPublisher()
        return _publisher

    # 延迟 import：routing 依赖本模块的 EventSink / NullEventPublisher，
    # 模块级互相 import 会成环。装配发生在启动期，延迟成本可以忽略。
    from .routing import RouterEventPublisher

    _publisher = RouterEventPublisher(sink)

    return _publisher


async def start_event_publisher(
        *,
        ping_timeout: float = 2.0,
) -> bool:
    """
    探活 + 起写线程。**由组合根 await 一次**（app/main.py 的 lifespan）。

    探活失败（Redis 连不上）就退回空对象并返回 False —— 与"数据源没起来"
    是同一种结局。为什么要探这一次：`enabled` 是**语义开关**，谎报 True
    会让模型层的 commit 规则记上"已经推给消费端内容了"、失败不再重试，
    而实际上没人在收。

    ⚠️ 这是**一次性快照**，不是实时真相。探活通过之后 Redis 掉线，
    `enabled` 仍然是 True。这与改造前一致（那时它只反映"装没装 client"），
    不是本次引入的；真要实时探活，得在探针上接告警，而不是让发布路径
    每次去 ping。

    探活超时用 wait_for 卡住：redis.yaml 的 socket_timeout 是 5 秒，
    不卡就会白等 5 秒才启动完。
    """

    global _publisher

    publisher = _publisher

    if publisher is None or isinstance(publisher, NullEventPublisher):
        return False

    ping = getattr(publisher, "ping", None)
    start = getattr(publisher, "start", None)

    if ping is None or start is None:
        return False

    try:
        alive = await asyncio.wait_for(
            asyncio.to_thread(ping),
            ping_timeout,
        )

    except Exception:       # noqa: BLE001 —— 探活失败不是异常路径
        alive = False

    if not alive:
        logger.warning(
            "Redis 探活失败，事件通道退回空对象（enabled=False）。"
            "运行照常，订阅端看不到中间过程。"
        )
        _publisher = NullEventPublisher()

        return False

    start()

    return True


async def close_event_publisher(
        timeout: float = 10.0,
) -> None:
    """
    停写线程并清单例。**由组合根在收尾时 await 一次**。

    ⚠️ 位置有要求（app/main.py 的 lifespan）：
        排在 dispatcher.aclose() **之后** —— 还在收尾的图任务要把
        run.completed 这类终止事件发出去，先停线程会让订阅端永远等不到
        终止事件、SSE 连接挂死；
        排在 datasource_manager.close_all() **之前** —— 写线程用的同步
        客户端归数据源管，线程还活着时关池，它后面每条都是 ConnectionError。

    **不关** sink 背后的客户端 —— 那归数据源管理器（见 redis.py）。
    """

    global _publisher

    publisher = _publisher

    aclose = getattr(publisher, "aclose", None) if publisher is not None else None

    if aclose is not None:
        await aclose(timeout)

    _publisher = None


def get_event_publisher() -> EventPublisher:
    """
    取进程级发布器。**未装配时返回空对象，而不是抛异常。**

    ⚠️ 这里刻意**偏离** get_service_factory() 的做法
    ----------------------------------------------
    那个 getter 在未构建时抛，因为它拿到 None 就没法工作。这个不同：
    Redis 是 required: false，**没有事件通道是一个合法的运行状态**，
    而不是"装配漏了一步"。节点因为取不到事件通道就崩，等于让一条旁路
    把主流程拖死 —— 与"事件是旁路"的整个立场矛盾。

    所以这里的语义是"总有一个能调的对象"，最坏情况是它什么都不做。
    """

    global _publisher

    if _publisher is None:
        _publisher = NullEventPublisher()

    return _publisher


def reset_event_publisher() -> None:
    """
    清掉单例（测试用）。

    与 get_thread_run_lock 的 reset_thread_run_lock 同一个用途：让下一条
    用例从"未装配"这个真实初态开始，而不是继承上一条留下的对象。

    ⚠️ 只清单例，**不停写线程**。若那个发布器已经 start() 过，请改用
    `await close_event_publisher()` —— 否则线程会一直挂着（它是 daemon，
    不会拦住进程退出，但会继续持有队列与 sink）。
    """

    global _publisher

    _publisher = None


# ============================================================
# 最常用的入口
# ============================================================


async def emit_event(event: Event) -> dict[str, bool]:
    """
    v2 入口：投递一条**规范事件**，由 `EventRouter` 按放行集合分发。

    返回 `{sink 名: 是否已接收}`。没有路由能力时（空发布器 / 旧实现）
    返回空字典 —— 事件是旁路，绝不外抛。
    """

    publisher = get_event_publisher()

    emit = getattr(publisher, "emit", None)

    if emit is None:
        return {}

    try:
        return await emit(event)

    except Exception:  # noqa: BLE001 —— 旁路绝不外抛
        logger.exception("规范事件投递失败；事件是旁路，不影响运行。")
        return {}
