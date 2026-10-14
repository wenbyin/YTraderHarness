"""
运行事件发布端的 **Redis Stream 实现**。

契约在 `app/common/event/publisher.py`（EventPublisher）；这里只放
"怎么写进 Redis"。

投递模型
--------
    publish()  **从不抛异常**，返回一个 bool。

这是契约要求，不是防御性编程。事件是**旁路**：一次 Redis 抖动不该让
整个 agent 运行失败，更不该让用户看到一次"模型不可用"。同一条立场见
tools/processor/runtime/events.py 的模块 docstring。

失败按 **2 的幂**节流打 warning（第 1、2、4、8… 次各一条）：一次都不报
会让"流里少了一段"无从解释，每次都报会把一次长时间的 Redis 不可用变成
日志洪水。

为什么写入搬进了一条自己的线程
------------------------------
在此之前，XADD 是跑在**事件循环**上的 `await`。循环一旦被别的同步代码
卡住（本仓库真实发生过：启动后第一次请求卡 ≥5 秒），这条流上的写入就
跟着一起停摆；更糟的是读取端那边的读超时定时器会在循环恢复的**瞬间**
先到期，把"回复早就到了"谎报成网络故障（见 reader.py 的说明）。

所以写入离开循环，落进一条常驻线程：

    事件循环 ── put_nowait ──▶ 有界队列 ──▶ 写线程（同步客户端）──▶ Redis

⚠️ `publish()` 体内**不许出现 `await`**
--------------------------------------
这是硬约束，不是风格偏好。没有 await，它相对事件循环就是一个不可分割
的回合，于是与 `aclose()` 的"置停止位"天然串行 —— **不需要锁**就能保证
"要么入队成功、要么被明确拒绝"。谁往里面塞一个 await，这个证明就塌了：
会出现"入队成功但没人消费"，也就是最难看的那种静默丢失。

⚠️ 为什么队列**有界**，且一旦丢就**丢到排空为止**
----------------------------------------------
先看为什么必须有界。写线程的吞吐有上界 —— Redis 停摆时每条写入都要
吃掉一个 socket_timeout（app/config/datasource/redis.yaml: 5 秒），即 1 条/5 秒；
而 delta 是**逐 token** 的，突发可以上百条/秒。无界队列在这里不是
"缓冲"，是"内存泄漏"：进得远快于出，几十分钟就能把进程吃光。

再看丢了之后怎么办。只"拒绝当下这一条、后面接着收"是不够的：队列在
第 4 条满、第 5 条时恰好被消费端腾出了一个位置，于是第 4 条丢了、第 5
条收了 —— 写出去的序列是 1,2,3,5，**正文中间豁了一个口子**。这正是
"后台队列会撕开 delta"那句话真正指向的东西，只是原因不在"丢最旧"，
而在"丢了之后还接着收"。

所以这里的策略是：**一旦丢，就丢到队列排空为止**。于是写出去的那段
永远是一个**连续前缀** —— 要么都在，要么从某一处起就没有了，中间不会
有空洞。恢复的判据是"队列空了"（由事件循环线程观察，写线程不参与这个
判断，因此没有跨线程的写）。

代价：一次长停摆会丢掉"从队列满那一刻起、到队列排空为止"的全部事件，
比"接着收"丢得更多。换来的是订阅端拿到的正文永远读得通；真的丢到了，
`message.completed` 里带全文，可以据此对账。

代价（契约里也记了一笔，这里再列一次）：
    - 单线程 = 队头阻塞：某一条 XADD 卡住，所有 run 的可见延迟一起变差。
    - 进程被 SIGKILL / 断电时，队列里还没写完的事件必然丢。
    - `publish()` 返回 True 只表示"**已接收**（入队）"，写入结果不回报。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from app.common.event import (
    EventPublisher,
    RunEvent,
    RunEventType,
    stream_key,
)

from .config import StreamConfig, load_stream_config

if TYPE_CHECKING:
    import redis as redis_sync

__all__ = [
    "EventSink",
    "NullEventPublisher",
    "RedisEventPublisher",
    "RedisStreamSink",
    "build_event_publisher",
    "close_event_publisher",
    "get_event_publisher",
    "publish_run_event",
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


@dataclass(frozen=True, slots=True)
class _Write:
    """
    队列里的一条待写记录。

    内容在**入队那一刻**就冻结了（`to_fields()` 已经跑完），写线程只负责
    搬运 —— 于是调用方之后改自己的 data 字典，也不会影响已经收下的事件。
    """

    key: str

    fields: dict[str, Any]

    maxlen: int | None

    ttl_seconds: int | None


class RedisEventPublisher:
    """
    把事件交给写线程，最终落到 `<key_prefix>:<topic_id>` 这条 Stream。

    客户端是**注入**的，本类不建连接
    ----------------------------------
    sink 背后的连接池归 infra/datasource/redis.py。组合根（app/main.py
    的 lifespan）从数据源管理器取了传进来，与 build_factories 的池子
    同一条纪律。因此本类**不关**那个客户端：两份引用 = 两个释放时机，
    而释放两次是未定义行为。
    """

    def __init__(
            self,
            sink: EventSink,
            config: StreamConfig | None = None,
            *,
            queue_size: int | None = None,
    ) -> None:

        self._sink = sink

        self._config = config or load_stream_config()

        #: 有界 —— 理由见模块 docstring。queue_size 只给测试用
        #: （要构造"队列满"这个状态，真按配置的几千条灌太慢）。
        self._queue: queue.Queue[_Write] = queue.Queue(
            maxsize=queue_size or self._config.write_queue_size,
        )

        #: 写线程。None = 还没 start()（或已经 aclose()）。
        self._thread: threading.Thread | None = None

        #: 停止位。**不是**往队列里塞哨兵 —— 队列满时 put 会阻塞，而
        #: 调用 aclose() 的是事件循环线程，堵在那里就是同步阻塞循环。
        #: 停止信号本来也不该排队（它必须能插队）。
        self._stopping = threading.Event()

        #: 计数器。`_failures` 由写线程写、`_rejected` 由事件循环线程写，
        #: 读取则两边都有（failures/rejected 探针、节流判断），所以统一
        #: 用一把锁护住 —— `+= 1` 不是原子操作，在 free-threaded 解释器
        #: 下更是如此（与 runtime/metrics.py 的做法一致）。
        self._lock = threading.Lock()
        self._failures = 0
        self._rejected = 0

        #: 是否处于"丢弃期"。只由事件循环线程读写（写线程不碰它），
        #: 因此不需要锁 —— 这正是 publish() 里不许有 await 换来的。
        #: 见模块 docstring：一旦开始丢，丢到队列排空为止，好让写出去的
        #: 那段始终是连续前缀。
        self._dropping = False

    # ========================================================
    # 生命周期
    # ========================================================

    def ping(self) -> bool:
        """
        探活。**同步** —— 由 start_event_publisher 包在 to_thread 里调。

        不在事件循环上直接跑：redis.yaml 的 socket_timeout 是 5 秒，
        Redis 连不上时它会老老实实等满，而那 5 秒会原样卡住整个启动。
        """

        return self._sink.ping()

    def start(self) -> None:
        """
        起写线程（幂等）。由 start_event_publisher 调用。

        daemon=True 是兜底：收尾时 join 有上界，真的排不空（Redis 一直
        不可用、队列还很长）也不能让进程退不出去 —— 代价是那种情况下
        队列里剩下的事件随进程消失，这一点写在模块 docstring 的代价里。
        """

        if self._thread is not None:
            return

        self._stopping.clear()

        self._thread = threading.Thread(
            target=self._consume,
            name="event-publisher",
            daemon=True,
        )

        self._thread.start()

    async def aclose(
            self,
            timeout: float = 10.0,
    ) -> None:
        """
        停写线程：**先排空**已接收的，再退出。

        timeout 要大于 socket_timeout（redis.yaml: 5 秒），否则正常收尾
        也会被误判成"排不空"；但也不能按"队列长度 × 5 秒"去算，那会让
        关停假死几分钟。取一个固定上界，超了就认账。
        """

        thread, self._thread = self._thread, None

        if thread is None:
            return

        # 置停止位 → 写线程把手上的和队列里剩下的写完就退出。
        self._stopping.set()

        # join 是阻塞调用，包进 to_thread —— 它不该占住事件循环。
        # wait_for 只兜底：to_thread 里的线程没法被取消，超时了我们只是
        # 不再等它（daemon 线程会随进程结束）。
        try:
            await asyncio.wait_for(
                asyncio.to_thread(thread.join, timeout),
                timeout + 1,
            )

        except Exception:       # noqa: BLE001 —— 排不空不是异常路径
            logger.warning(
                "事件写线程未能在 %.0f 秒内退出，剩余 %d 条未写出。"
                "它是 daemon 线程，会随进程结束 —— 那部分事件就丢了。",
                timeout,
                self.pending,
            )
            return

        if thread.is_alive():       # pragma: no cover - 竞态兜底
            logger.warning(
                "事件写线程仍在运行，剩余 %d 条未写出。",
                self.pending,
            )

    # ========================================================
    # 对外：契约的两个成员
    # ========================================================

    @property
    def enabled(self) -> bool:
        """
        是否真的有通道（拿到了 sink，且配置里开着）。

        ⚠️ 这不是探活，也不是实时真相。调用方（agent 节点）用它决定
        **要不要走流式**：没有消费者还走流式是纯亏，而且每发一个增量都会
        让 Model Runtime 的 commit 规则记上"已经推给消费端内容了"，于是
        失败不再重试 —— 可实际上没人在收。所以它是一个**语义开关**。

        启动时的那次探活（start_event_publisher）是**一次性快照**：
        探活通过之后 Redis 掉线，这里仍然是 True。
        """

        return self._config.enabled

    async def publish(self, event: RunEvent) -> bool:
        """
        把事件交给写线程，返回**是否已接收**（入队成功）。**从不抛异常。**

        返回值语义与改造前不同：以前是"写进 Redis 了吗"，现在是"收下了
        吗"。交付真实写入结果就必须 await，而 await 一上，线程带来的
        收益（不让写入挂在循环上）就全没了 —— 二者只能取一个。契约
        （app/common/event/publisher.py）本就允许调用方完全忽略它。

        `enabled: false` 时直接返回 False，一条 Redis 命令都不发。
        """

        config = self._config

        if not config.enabled:
            return False

        # 没起线程 = 装配没走完；已置停止位 = 正在收尾。
        # 两种情况入队都等于排进一个没人消费的队列 —— 那正是最难查的
        # 静默丢失，所以宁可直接拒收（调用方拿到 False）。
        if self._thread is None or self._stopping.is_set():
            return False

        key = stream_key(event.topic, event.topic_id)

        # 丢弃期：一直丢到队列排空为止（见模块 docstring 关于"连续前缀"）。
        # 队列空了这个判断自然翻转，不需要谁去重置一个状态位。
        if self._dropping:
            if not self._queue.empty():
                self._record_rejection(key)
                return False

            self._dropping = False

        try:
            # 序列化在入队这一刻完成（见 _Write）。data 不可序列化之类
            # 的问题在这里就要拦下 —— 契约要求绝不抛。
            write = _Write(
                key=key,
                fields=event.to_fields(),
                maxlen=config.maxlen,
                ttl_seconds=config.ttl_seconds,
            )

        except Exception as exc:        # noqa: BLE001
            self._record_failure(exc, key)
            return False

        try:
            self._queue.put_nowait(write)

        except queue.Full:
            self._dropping = True
            self._record_rejection(key)
            return False

        return True

    # ========================================================
    # 写线程
    # ========================================================

    def _consume(self) -> None:
        """
        写线程主循环：取一条、写一条，直到"停止位已置**且**队列已空"。

        ⚠️ 单条失败**不能**让循环退出（同 dispatcher_impl.py 的立场）：
        线程一退，队列就成了只进不出的黑洞，症状又是静默丢失。
        """

        while True:

            try:
                item = self._queue.get(timeout=0.2)

            except queue.Empty:
                # 空了才看停止位：这样"会退"的标准是"排空了"，而不是
                # "收到停止信号了" —— 后者会丢掉已经收下的事件。
                if self._stopping.is_set():
                    return

                continue

            try:
                self._sink.write(
                    item.key,
                    item.fields,
                    maxlen=item.maxlen,
                    ttl_seconds=item.ttl_seconds,
                )

            except Exception as exc:    # noqa: BLE001 —— 见模块 docstring
                self._record_failure(exc, item.key)

            finally:
                # task_done 是 drain() 能等到人的前提。
                self._queue.task_done()

    async def drain(
            self,
            timeout: float = 5.0,
    ) -> None:
        """
        等队列排空。**给测试用** —— 生产路径上没人需要等写入完成（那正是
        这次改造要避免的事）。

        join() 是阻塞调用，因此包在 to_thread 里。超时后线程还在跑，
        这里只是不再等（测试会看到断言失败，正是它该有的样子）。
        """

        await asyncio.wait_for(
            asyncio.to_thread(self._queue.join),
            timeout,
        )

    # ========================================================
    # 留痕与探针
    # ========================================================

    def _record_failure(
            self,
            exc: BaseException,
            key: str,
    ) -> None:
        """
        记一次**写入**失败（写线程抛异常，或入队前序列化失败）。

        按 2 的幂节流（第 1、2、4、8… 次各报一条）。永远只打 warning。
        """

        with self._lock:
            self._failures += 1
            count = self._failures

        # 非 2 的幂就跳过：x & (x - 1) == 0 表示 x 是 2 的幂。
        if count & (count - 1):
            return

        logger.warning(
            "运行事件投递失败 %d 次（最近一次 key=%s）：%s: %s。"
            "事件是旁路，不影响运行本身；但订阅者会少看到这段内容。",
            count,
            key,
            type(exc).__name__,
            exc,
        )

    def _record_rejection(self, key: str) -> None:
        """
        记一次**拒收**（队列满，或正处于丢弃期）。

        与写入失败分开计数：拒收意味着"消费端跟不上"，写入失败意味着
        "Redis 有问题"，混在一个数字里会把两种故障的排查方向搅在一起。
        """

        with self._lock:
            self._rejected += 1
            count = self._rejected

        if count & (count - 1):
            return

        logger.warning(
            "事件队列积压，已丢弃事件 %d 条（最近一次 key=%s，队列上限 %d）。"
            "写线程跟不上，通常是 Redis 变慢或不可用 —— 会一直丢到队列"
            "排空为止，所以订阅端拿到的是连续的一段前缀，正文读得通。"
            "事件是旁路，不影响运行本身。",
            count,
            key,
            self._queue.maxsize,
        )

    @property
    def failures(self) -> int:
        """累计写入失败数（探活用）。"""

        with self._lock:
            return self._failures

    @property
    def rejected(self) -> int:
        """累计因队列满被拒收的事件数（探活用）。"""

        with self._lock:
            return self._rejected

    @property
    def pending(self) -> int:
        """队列里还没写出去的条数（探活用）。"""

        return self._queue.qsize()


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

    async def publish(self, event: RunEvent) -> bool:
        """永远 False（没有通道）。绝不抛。"""

        return False


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

    _publisher = (
        RedisEventPublisher(sink)
        if sink is not None
        else NullEventPublisher()
    )

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

    if not isinstance(publisher, RedisEventPublisher):
        return False

    try:
        alive = await asyncio.wait_for(
            asyncio.to_thread(publisher.ping),
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

    publisher.start()

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

    if isinstance(publisher, RedisEventPublisher):
        await publisher.aclose(timeout)

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


async def publish_run_event(
        type: RunEventType,
        topic_id: str,
        *,
        data: Mapping[str, Any] | None = None,
        node: str | None = None,
) -> bool:
    """
    构造 + 投递一条事件。同样**从不抛异常**。

    topic 从配置取，不由调用方传 —— "这条流属于哪个类别"是部署级的
    配置，不是每次调用都要重新决定的事。调用方只关心：哪个 run、
    什么事件、带什么数据。

    `topic_id` 就是 run_id（见 app/common/event/event.py 的三个词）。
    """

    publisher = get_event_publisher()

    event = RunEvent(
        type=type,
        topic=load_stream_config().key_prefix,
        topic_id=topic_id,
        data=data or {},
        node=node,
    )

    return await publisher.publish(event)
