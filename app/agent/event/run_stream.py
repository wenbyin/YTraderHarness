"""
按 run 复用的事件流**广播端** —— 一个 run 一条连接，服务所有订阅者。

它替代的是什么
--------------
改造前每个 SSE 订阅者各跑一个 `XREAD BLOCK` 无限循环（见 reader.py 的
历史），于是**每个订阅者常驻占用一条 Redis 连接**。池上限 10，所以第 11
个并发订阅就会让整个进程的 Redis 访问一起失败；更糟的是那条异常继承自
ConnectionError，被读取端的"网络抖动重试"逻辑接住、无退避空转三次再抛
500 —— 浏览器 EventSource 收到 500 按规范自动重连，撞上仍然满的池，
于是"池满 → 500 → 重连 → 池还是满"地转下去。

现在：一个 run 一条流，**恰好一个 asyncio task + 恰好一条连接**，订阅者
只是挂在这个 task 上的队列。

一条纪律：循环 task 是**所有**订阅者队列的唯一写入者
-------------------------------------------------
阶段一 补发：新挂上来的订阅者，用它的游标 `XRANGE (start, cursor]` 把
          缺的那段按序推进它的队列，然后标记它 live。
阶段二 尾随：`XREAD {key: cursor} BLOCK block_ms`，每条推进 cursor 并投给
          所有 live 订阅者。

因为读**和**投递都在同一个 task 里，而且 cursor 只被它推进，"快照读"与
"挂上广播"之间不存在竞态：既不会漏一条，也不会重复投一条。这正是**不**
让各订阅者自己发 XRANGE 的理由 —— 那需要在下游做去重，而去重写错的症状
是"偶尔少一条 delta"，很难发现在。

⚠️ 有界阻塞：补发期间尾随是停的
----------------------------
补发由循环自己做，所以一次大补发会把该 run 的尾随读推迟一小会儿。量级是
`流的长度 / 批大小` 次往返，有界。换来的是一条连接服务全部订阅者 ——
这个取舍是刻意的。

⚠️ 为什么用协作式停机，而不是 task.cancel()
--------------------------------------
`CancelledError` 打在 `XREAD BLOCK` 中间时，redis-py 那条连接上会留下一个
**没被读走的回复**，然后它被还回池里 —— 下一个拿到它的 run 会读到上一条
命令的回复，于是事件串流、甚至永远卡住（协议错位）。超时那条路径 redis-py
自己会 `disconnect`（见 reader.py 的说明），取消这条不会。

所以停机走 `_stop` 事件：循环在每轮开头看一眼，正在飞的 `XREAD BLOCK` 自己
在 `block_ms` 内返回，然后干净退出。代价是拆除最多等一个 `block_ms`。

关流规则（三种"没有下文了"）
--------------------------
1. **终止事件**（run.completed / run.failed）—— 流上不会再有事件了。补发与
   尾随里发现都算。
2. **暂停等审批** —— 空闲达阈值后探活，DB 状态是 `waiting_input` **且**本流
   已经投递过至少一条 `hitl.requested` 时才关。
   ⚠️ "至少一条"这个前提不是多余的：`hitl.requested` 在扇出下是**逐个发**
   的（agent_run_dispatch.py:756-771），而权威的"全发完了"信号是 **DB 状态**
   —— 派发路径先发完所有事件、**之后**才把状态写成 waiting_input（:540 发
   事件、:585 才 _finish）。所以"状态已是 waiting_input"蕴含"所有 interrupt
   都已发布"；再加上空闲阈值（默认 5 次空读 ≈ 5 秒）作为写队列落地的余量，
   客户端拿全了 interrupt_id 才关流。见 app/common/event/event.py 的
   STREAM_CLOSING_EVENT_TYPES。
3. **执行器没了** —— 空闲探活发现"调度器手里没有它"**且** DB 还停在
   running/queued：说明跑它的那个东西已经不在了（崩溃 / 重启过），这条流
   再也不会有人来关。这正是"已完成的 run 被重连后永久挂着"之外的另一半泄漏。

⚠️ 探活只在**空闲**时跑
---------------------
有事件在流动时绝不查库 —— 一个正常跑着的长回答不该因为探活去打搅
Postgres。探活也确实占着这条流的循环（它 `await` 期间尾随是停的），所以
它是**兜底**，不是主路径：正常情况下流是被终止事件关掉的。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING
from uuid import UUID

from redis import exceptions as redis_exceptions

from app.common.event import RunEventType, stream_key

from .config import StreamConfig, load_stream_config
from .reader import TERMINAL_VALUES, StreamEntry, decode as _decode

if TYPE_CHECKING:
    from redis.asyncio import Redis


__all__ = [
    "RunLiveness",
    "RunLivenessProbe",
    "RunStreamAttachment",
    "RunStreamRegistry",
    "StreamClosed",
    "StreamFailed",
    "TooManyRunStreamsError",
    "build_run_stream_registry",
    "get_run_stream_registry",
    "reset_run_stream_registry",
]


logger = logging.getLogger(__name__)


#: 一次 XREAD / XRANGE 最多取多少条的**上限**。
#:
#: 有界是为了让每个订阅者队列的**单轮**增量有界 —— 一轮投一万条进去，
#: 再大的队列也扛不住。分块之后，慢订阅者会在若干轮之后被判为"跟不上"，
#: 而不是在某一轮里被瞬间冲垮。
#:
#: ⚠️ 真正用的批大小还要与队列容量取小，见 `_chunk_for`。
_CHUNK = 500


def _chunk_for(config: StreamConfig) -> int:
    """
    本配置下单轮取多少条：`min(_CHUNK, 队列容量)`。

    ⚠️ 为什么必须与队列容量绑在一起
    ----------------------------
    一轮 XREAD 读回来的那批记录是**同步**投给所有订阅者的 —— `_fanout`
    到 `_offer` 全程没有 await，消费方拿不到执行权。所以批比队列大的话，
    一个**完全正常**的订阅者只要没来得及被调度，就会在那一轮里被自己的
    队列撑满、判为跟不上并断开。那是误杀，而且它看起来像"客户端太慢"，
    排查会往前端找。

    绑在一起之后两者不可能打架，`subscriber_queue_size` 也就能自由调小
    （测试要小队列来验踢人那条路径）。
    """

    return max(1, min(_CHUNK, config.subscriber_queue_size - 1))


#: 探活的超时（秒）。
#:
#: 探活占着这条流的循环，一次卡住的查库会把该 run 所有订阅者的心跳一起
#: 停掉（连"客户端已断开"都发现不了）。超时就当"不知道"，见 `_probe`。
_PROBE_TIMEOUT_SECONDS = 5.0


#: 可重试的读失败。与改造前一致：刻意只收这两个。
#: ResponseError / DataError 是**我们自己**发错了命令，属于真 bug，
#: 重试只是把 bug 藏起来。
_RETRYABLE = (
    redis_exceptions.TimeoutError,
    redis_exceptions.ConnectionError,
)


def _consume_task_exception(task: asyncio.Task) -> None:
    """
    收下循环 task 的异常回执（异常本身已经扇出给订阅者了）。

    不取的话，asyncio 会在 task 被回收时打 "Task exception was never
    retrieved" —— 那条日志看起来像"有个错误没人管"，会把人引到错误的方向。
    """

    if not task.cancelled():
        task.exception()


def _fields(raw_fields: Any) -> dict[str, str]:
    """
    Redis 的一条记录（字段名与值都是 bytes）-> 全 str 的扁平映射。

    ⚠️ **字段名也要解**，不只是值
    -------------------------
    `decode_responses: false` 下 `b"event"` 与 `"event"` 是两个不同的键。
    只解值的话 `entry.fields.get("event")` 拿到 None —— 而
    `StreamEntry.event_type` 正是它，于是**每条记录的事件名都是 None**：
    终止事件判不出来、流永远不关、SSE 那边也分派不了事件名。症状是
    "连接一直在、什么事件都收不到"，离原因很远。
    """

    return {
        _decode(name): _decode(value)
        for name, value in raw_fields.items()
    }


class RunLiveness(str, Enum):
    """
    探活的结果。"这个 run 还会产出事件吗"。

    ⚠️ 它**不是**数据库那份状态的枚举
    ------------------------------
    数据库的 RunStatus 回答"这个 run 最后被记成了什么"，这里回答"现在还有
    必要为它挂着一条流吗"。两者的映射（含"调度器手里还有没有它"）在
    `build_run_liveness_probe` 那一处，只写一遍。
    """

    #: 还会。继续等。
    ALIVE = "alive"

    #: 运行已终止（completed / failed / cancelled）。流上不会再有事件。
    FINISHED = "finished"

    #: 停在 waiting_input 等审批。事件已发全，连接留着只是白占。
    PAUSED = "paused"

    #: 调度器手里没有它，而数据库还停在 running/queued ——
    #: 跑它的进程已经没了（崩溃或重启）。
    EXECUTOR_GONE = "executor_gone"

    #: agent_run 行不存在。无从谈起。
    MISSING = "missing"


class RunLivenessProbe(Protocol):
    """
    探活探针：问一句"这个 run 现在什么情况"。

    由**组合根**注入（app/main.py），实现在
    app/services/session/application/run_liveness.py —— 判据要用到调度器与
    agent_run 表，两者都不归事件层管，所以这一层只认这个协议。
    """

    async def __call__(self, run_id: UUID) -> RunLiveness: ...


# ============================================================
# 投递给订阅者的信号
# ============================================================


@dataclass(frozen=True, slots=True)
class StreamClosed:
    """
    这条流**正常**收尾了 —— 订阅者据此结束迭代（不是错误）。

    `reason` 只用于日志与排查，不是给客户端判断用的契约：客户端该做的是
    "结束了就收拾"（收到终止事件时它本来就已经知道了）。
    """

    reason: str


@dataclass(frozen=True, slots=True)
class StreamFailed:
    """
    这条流**读不下去了**（Redis 掉了、池满了这类）。

    它被投给每一个订阅者并**抛出**，而不是变成一次安静的收尾 —— 与 reader.py
    开头那条立场一致：订阅方拿到一个"卡住的空流"比拿到一个明确的错误糟得多。
    """

    error: BaseException


#: 队列里能出现的东西。None 是**心跳节拍**（该发一个 SSE 注释行了），
#: 不是数据 —— 见 `_iterate`。
_Item = StreamEntry | None | StreamClosed | StreamFailed


class TooManyRunStreamsError(RuntimeError):
    """
    同时在订阅的 run 数到上限了。

    刻意是个明确的异常而不是让连接池到 redis-py 内部才炸：池满抛出的
    MaxConnectionsError 看不出所以然，还容易被误当成网络抖动。路由据此
    给一个说明原因的 503。
    """


@dataclass(slots=True)
class _Subscriber:
    """一个订阅者：一条待发队列 + 它的状态。"""

    queue: asyncio.Queue[_Item]

    #: 有效容量。**刻意比队列的 maxsize 小 1**：那一个空位是留给收尾信号的。
    #:
    #: 没有这个保留位的话，队列满时收尾信号就塞不进去，订阅者会永远等在
    #: `queue.get()` 上 —— 正是"永远卡住的空流"那个最糟的失败形态。
    capacity: int

    #: 补发做完了吗。没做完之前不收尾随事件（它那段历史正在被写入）。
    live: bool = False

    #: 被踢掉的原因（消费太慢）。非 None 表示这个订阅者已经出局。
    dropped: str | None = None

    #: 已投递的条目数（诊断用）。
    delivered: int = 0


class RunStreamAttachment:
    """
    一个订阅者挂上某条流之后的句柄：可迭代、可关闭。

    由 `RunStreamRegistry.attach` **同步**返回（准入要在响应开始前完成），
    调用方负责在收尾时 `aclose()`：

        attachment = registry.attach(run_id, start=start)

        try:
            async for item in attachment:
                ...
        finally:
            attachment.aclose()

    ⚠️ `aclose()` 必须幂等且同步
    ---------------------------
    它可能被调两次（正常收尾 + 兜底清理），也可能和循环踢人撞上（慢订阅者
    是被 `RunStream._drop` 摘掉的）。两种情况下都不该出问题。
    """

    __slots__ = ("_stream", "_sub", "_closed", "_iterator")

    def __init__(self, *, stream: "RunStream", sub: _Subscriber) -> None:

        self._stream = stream
        self._sub = sub
        self._closed = False

        #: ⚠️ 只造一次。
        #:
        #: 直接 `return self._stream._iterate(self._sub)` 的话，每 `__aiter__`
        #: 一次就多一个消费者读**同一条队列** —— 两条 `async for` 会各拿到
        #: 一半事件，而且都不报错。症状是"事件偶尔少几条"，很难往这里想。
        self._iterator: AsyncIterator[_Item] | None = None

    def __aiter__(self) -> AsyncIterator[_Item]:

        if self._iterator is None:
            self._iterator = self._stream._iterate(self._sub)

        return self._iterator

    @property
    def subscriber_count(self) -> int:
        return self._stream.subscriber_count

    @property
    def closed_reason(self) -> str | None:
        """
        这条订阅为什么收尾（None = 还没收尾）。

        ⚠️ 收尾信号**不会**作为一条内容被产出
        -----------------------------------
        `_iterate` 读到它就让异步迭代**正常结束**（错误那种是抛出去），
        所以 `async for` 里看不到它 —— 想知道原因要读这里。这样
        `_frames` 那侧只需要一个普通的 `async for`，不必在循环体里判别
        "这一条是数据还是控制信号"（那种判别写错的症状是客户端收到一帧
        内容为 null 的事件）。

        被踢掉的订阅者优先报**它自己**的原因（"你太慢了"），而不是整条流
        收尾的原因 —— 对它来说那才是真正发生了什么。
        """

        if self._sub.dropped is not None:
            return self._sub.dropped

        return self._stream.closed_reason

    def aclose(self) -> None:
        """摘掉这个订阅者。幂等。"""

        if self._closed:
            return

        self._closed = True

        self._stream.detach(self._sub)


# ============================================================
# 一个 run 的广播端
# ============================================================


class RunStream:
    """
    一个 run 的事件流：一个循环 task，服务任意多个订阅者。

    生命周期归 `RunStreamRegistry` 管 —— 别处不要直接构造它。
    """

    def __init__(
            self,
            *,
            key: str,
            run_id: UUID,
            client: "Redis",
            probe: RunLivenessProbe | None = None,
            config: StreamConfig | None = None,
            on_detach: Callable[["RunStream"], None] | None = None,
    ) -> None:

        self._key = key

        self._run_id = run_id

        self._client = client

        self._probe = probe

        self._config = config

        #: 循环结束时回调，供注册表做身份校验的摘除。
        self._on_detach = on_detach

        self._subscribers: list[_Subscriber] = []

        #: 等待补发的订阅者（`(订阅者, 起点游标)`）。循环每轮取空为止。
        self._pending: list[tuple[_Subscriber, str]] = []

        self._task: asyncio.Task | None = None

        #: 协作式停机（**不是** cancel，理由见模块 docstring）。
        self._stop = asyncio.Event()

        #: 最后一个订阅者离开时**同步**置起。注册表据此判定"这条已经在
        #: 拆了，别再把新订阅者挂上来"，并且只有它能让注册表换一个新的。
        self._closing = False

        #: 尾随游标。`None` = 还没确立（循环启动时确立一次）。
        self._cursor: str | None = None

        #: 这条流上出现过 hitl.requested 吗。关流规则 2 的前提条件。
        self._hitl_seen = False

        #: 连续空读次数，用于触发探活。
        self._empty_reads = 0

        #: 正常收尾的原因（None = 还没收尾）。
        self._closed: StreamClosed | None = None

        #: 读不下去时的异常。
        self._failure: BaseException | None = None

    # ========================================================
    # 状态（注册表与路由读）
    # ========================================================

    @property
    def is_closing(self) -> bool:
        """已经在拆了 —— 注册表不该再把新订阅者挂上来。"""

        return self._closing

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def closed_reason(self) -> str | None:
        return self._closed.reason if self._closed else None

    # ========================================================
    # 生命周期
    # ========================================================

    def start(self) -> None:
        """起循环 task。注册表在首次订阅时调用。"""

        if self._task is not None:
            return

        self._task = asyncio.create_task(
            self._run(),
            name=f"run-stream:{self._run_id}",
        )

        # 取走异常回执。
        #
        # 循环抛出去的东西**已经**扇出给每个订阅者了（见 `_teardown`），
        # 但 task 自己那份异常没人 await —— 于是 asyncio 会在它被回收时打一条
        # "Task exception was never retrieved"。那条日志在运维眼里像"有个
        # 错误没人处理"，实际上已经处理过了，只是回执没收。
        # 与 infra/agent/tool_executor.py 的 `_spawn` 同一个做法。
        self._task.add_done_callback(_consume_task_exception)

    async def aclose(self) -> None:
        """
        停机：置 `_stop` 并等循环自己退出。

        ⚠️ 刻意**不** cancel（理由见模块 docstring）。因此调用方要能接受
        最多等一个 `block_ms` 多一点。等不到就放弃等待 —— 任务仍在后台
        收尾，但至少不再阻塞停机流程；那种情况下连接由数据源关闭兜底。
        """

        self._stop.set()

        task = self._task

        if task is None:
            return

        self._task = None

        config = self._config or load_stream_config()

        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=(config.block_ms / 1000.0) * 2 + _PROBE_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "事件流 %s 的循环在停机时限内没有退出；不再等它。",
                self._key,
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            # 循环自己抛过的东西已经扇出给订阅者了（见 _teardown），
            # 这里再记一次只会重复。
            pass

    # ========================================================
    # 订阅
    # ========================================================

    def attach(self, start: str = "0") -> _Subscriber:
        """
        挂上一个订阅者。**同步**，因此准入与登记是原子的一个回合。

        为什么是同步而不是 `async with stream.subscribe(...)`：

            准入（"这个 run 的流数到上限了吗"）必须在**响应开始之前**做完
            —— 一旦 `StreamingResponse` 已经吐出了状态行，就再也改不成 503
            了。把 attach 做成同步的，路由就能先挂载、拿到明确的异常、
            再决定返回什么。

        读它产出内容用 `_iterate(sub)`，摘掉用 `detach(sub)`。

        ⚠️ 从准入到登记之间**没有 await** —— 单线程事件循环里这就是原子的
        一个回合（同 infra/graph/thread_lock.py:62-79 的论证）。注册表那边
        靠 `is_closing` 保证不会把订阅者挂到一条正在拆的流上。
        """

        config = self._config or load_stream_config()

        # maxsize 比容量大 1：多出来的那一格是收尾信号的保留位。
        sub = _Subscriber(
            queue=asyncio.Queue(maxsize=config.subscriber_queue_size),
            capacity=config.subscriber_queue_size - 1,
        )

        self._subscribers.append(sub)
        self._pending.append((sub, start))

        return sub

    async def _iterate(
            self,
            sub: _Subscriber,
    ) -> AsyncIterator[_Item]:
        """
        把该订阅者的队列转成异步迭代器。

        ⚠️ 消费方的释放**不能只靠这个生成器的终结器**
        ------------------------------------------
        Starlette 的 StreamingResponse 是直接 `async for chunk in
        body_iterator` 的（responses.py:250），**没有**用 `aclosing` 包一层。
        客户端断开时生成器被挂起，它的 `finally` 只有等到被垃圾回收时才跑
        —— 不确定，可能很久。

        所以真正的保证是**两条一起**：
          1. `_frames` 每次心跳都问 `request.is_disconnected()`，发现断开就
             `return` —— 那是一条**正常返回**路径，`finally` 当场就跑；
          2. 队列有界：真正卡死、连心跳都没人取的订阅者，攒满心跳后会被
             `_offer` 判为跟不上并踢掉（这是兜底，不是主路径）。
        """

        while True:
            item = await sub.queue.get()

            if isinstance(item, StreamClosed):
                return

            if isinstance(item, StreamFailed):
                raise item.error

            yield item

    def detach(self, sub: _Subscriber) -> None:
        """
        摘掉一个订阅者。**幂等** —— 循环踢人和订阅方自己退出都会走到这里。

        同步、无 await，因此与 `subscribe` 的登记是原子的一个回合。

        最后一个订阅者离开时置 `_closing`：注册表据此**立刻**改用一个新
        实例，而不是把新订阅者挂到这条正在拆的流上（那会让它等一个已经
        停下来的循环，永远收不到东西）。
        """

        if sub in self._subscribers:
            self._subscribers.remove(sub)

        # 还没轮到补发的，一并撤掉。
        self._pending[:] = [
            item for item in self._pending if item[0] is not sub
        ]

        if not self._subscribers and not self._closing:
            self._closing = True
            self._stop.set()

    # ========================================================
    # 投递（只被循环 task 调用）
    # ========================================================

    def _offer(self, sub: _Subscriber, item: _Item) -> bool:
        """
        往一个订阅者的队列投一条。返回 False 表示**它被判为跟不上、已出局**。

        满了就踢掉**这一个**订阅者，而不是：
          - 阻塞共享循环 —— 那会让这个 run 的所有订阅者跟着最慢的那个卡住；
          - 丢最旧的一条 —— 那会让正文静默缺洞（message.delta 是逐 token
            的），而前端无从知道缺了哪段。

        踢掉时把积压**清空**再放收尾信号：反正这个订阅者要出局了，它那份
        积压已经不可能被读完；不清的话信号塞不进满队列，它会永远等在
        `queue.get()` 上。
        """

        if sub.dropped is not None:
            return False

        if sub.queue.qsize() >= sub.capacity:
            self._drop(sub, "订阅者消费太慢，待发队列已满")
            return False

        try:
            sub.queue.put_nowait(item)

        except asyncio.QueueFull:
            # 容量算错才会走到这里；按同一个策略兜底，不静默丢数据。
            self._drop(sub, "订阅者消费太慢，待发队列已满")
            return False

        sub.delivered += 1

        return True

    def _drop(self, sub: _Subscriber, reason: str) -> None:
        """把一个订阅者判为跟不上并收尾它（不动别人）。"""

        sub.dropped = reason

        logger.warning(
            "事件流 %s：踢掉一个订阅者（已投递 %d 条）—— %s。"
            "其余订阅者不受影响；被踢的那一方按 Last-Event-ID 重连即可"
            "续上，只是要快一点消费。",
            self._key,
            sub.delivered,
            reason,
        )

        self._drain(sub)

        try:
            sub.queue.put_nowait(StreamClosed(reason=reason))

        except asyncio.QueueFull:       # pragma: no cover - 已清空，理论不可达
            pass

        self.detach(sub)

    @staticmethod
    def _drain(sub: _Subscriber) -> None:
        """清空一个订阅者的积压。"""

        while True:
            try:
                sub.queue.get_nowait()

            except asyncio.QueueEmpty:
                return

    # ========================================================
    # 循环
    # ========================================================

    async def _run(self) -> None:
        """
        唯一的循环。见模块 docstring 的两个阶段。

        任何逃出去的东西都会被 `_teardown` 扇出给每个订阅者 —— 不允许出现
        "循环死了而订阅者还在队列上等"的形态。
        """

        config = self._config or load_stream_config()

        try:
            await self._establish_cursor()

            while True:

                # ⚠️ 保证每一轮至少让出一次控制权。
                #
                # 这**不是**可有可无的礼貌 —— 少了它，只要某一轮的
                # xread / xrange 没有真的挂起（客户端实现返回太快、或者
                # 将来换了别的实现），这个 `while True` 就会在**同一个**
                # 事件循环回调里一直转下去，整个进程的其它协程（包括所有
                # 别的 run 的流、以及 HTTP 请求）全部饿死，而且**没有任何
                # 报错**。真 redis-py 会 await 套接字，所以平时看不见；
                # 这种"平时看不见、出问题就是全站静默卡死"的形态不值得赌。
                await asyncio.sleep(0)

                if self._stop.is_set() or not self._subscribers:
                    self._closed = StreamClosed(reason="没有订阅者了")
                    return

                # ---- 阶段一：把新挂上来的补齐 ----
                # 反复取到空为止：给一个订阅者补发期间可能又来了新的。
                while self._pending:

                    sub, start = self._pending.pop(0)

                    # 排队期间它可能已经走了（客户端断开）。
                    if sub not in self._subscribers:
                        continue

                    if await self._catch_up(sub, start, config):
                        self._closed = StreamClosed(
                            reason="补发时读到终止事件"
                        )
                        return

                # ---- 阶段二：尾随 ----
                response = await self._read_tail(config, self._heartbeat)

                if response:
                    self._empty_reads = 0

                    if self._deliver(response):
                        self._closed = StreamClosed(
                            reason="读到终止事件"
                        )
                        return

                    continue

                # ---- 阶段三：空读 —— 记账，必要时探活 ----
                self._empty_reads += 1

                if self._empty_reads < config.idle_probe_after_reads:
                    self._heartbeat()
                    continue

                self._empty_reads = 0

                verdict = await self._ask_liveness()

                if verdict is RunLiveness.ALIVE:
                    self._heartbeat()
                    continue

                if (
                        verdict is RunLiveness.PAUSED
                        and not self._hitl_seen
                ):
                    # 状态说在等审批，但这条流一条 hitl.requested 都没投过
                    # —— 关掉的话客户端手里就没有 interrupt_id，也就无法
                    # resume。等它到齐（见模块 docstring 的关流规则 2）。
                    logger.info(
                        "事件流 %s：DB 停在 waiting_input，但本流还没投递过"
                        "任何 hitl.requested，继续等审批事件写进流里。",
                        self._key,
                    )
                    self._heartbeat()
                    continue

                self._closed = StreamClosed(reason=verdict.value)
                return

        except BaseException as exc:  # noqa: BLE001

            # 与派发路径同一条理由：CancelledError 也要走收尾，否则订阅者
            # 会永远等在队列上。Asyncio 层取消时我们**原样抛回**，但收尾
            # 已经做完了。
            if not isinstance(exc, asyncio.CancelledError):
                self._failure = exc

                logger.warning(
                    "事件流 %s 的循环终止：%s: %s。已把错误扇出给 %d 个订阅者。",
                    self._key,
                    type(exc).__name__,
                    exc,
                    len(self._subscribers),
                )

            raise

        finally:
            self._teardown()

    async def _establish_cursor(self) -> None:
        """
        确立尾随游标的起点 = **此刻**这条流的末尾。

        为什么是末尾而不是 0：从 0 开始的话，第一次 `XREAD` 会把整条流
        （上限 maxlen，默认 10000）一次性拉回来。历史由各订阅者的补发负责，
        尾随只管"从现在起的新事件"。

        流的键还不存在时（运行刚建、第一个事件还没发）返回空，游标取 "0"
        —— 那时流本来就空，`XREAD` 会阻塞着等第一条写进来。
        """

        tail = await self._client.xrevrange(
            self._key, "+", "-", count=1
        )

        if tail:
            self._cursor = _decode(tail[0][0])

        else:
            self._cursor = "0"

    async def _catch_up(
            self,
            sub: _Subscriber,
            start: str,
            config: StreamConfig,
    ) -> bool:
        """
        把 `sub` 从它的起点补到当前游标。返回 True 表示补发里出现了终止事件。

        ⚠️ 游标语义的转换在这一处收口
        ---------------------------
        `start` 是按 **XREAD** 的语义给的（排他：该 id 本身不返回），而
        XRANGE 的起点默认**包含**。所以这里要显式加 `(`。
        三个取值各自的处理：

            "0"        从最开头 —— 用 "-"（最小值，包含）最稳，
                       不必赌 "(0" 能不能被解析
            "$"        只要连接之后的新事件 —— XRANGE 不认这个记号，
                       直接跳过补发，标记 live
            具体 id     "(<id>"，即该 id 之后

        终点是 `self._cursor`（**包含**），因此补发与尾随的边界正好接上：
        cursor 是"尾随已经投出去的最后一条"，补发把它也带给新订阅者，
        而尾随从这里往后读 —— 不重不漏。
        """

        if start == "$":
            sub.live = True
            return False

        cursor = self._cursor or "0"

        lower = "-" if start == "0" else f"({start}"

        chunk = _chunk_for(config)

        while True:

            batch = await self._client.xrange(
                self._key,
                lower,
                cursor,
                count=chunk,
            )

            if not batch:
                break

            for entry_id, raw_fields in batch:

                entry = StreamEntry(
                    id=_decode(entry_id),
                    fields=_fields(raw_fields),
                )

                if not self._offer(sub, entry):
                    return False

                if self._note(entry):
                    return True

            if len(batch) < chunk:
                break

            # 下一块从这一块的末尾之后开始。
            lower = f"({_decode(batch[-1][0])}"

            # 给别的任务一个回合的机会（包括订阅方读走队列）。
            await asyncio.sleep(0)

        sub.live = True

        return False

    async def _read_tail(
            self,
            config: StreamConfig,
            on_retry: Callable[[], None] | None = None,
    ) -> Any:
        """
        一次阻塞尾随读取，带**有上限的重试**。

        重试的理由与改造前一样（见 reader.py 的模块 docstring）：事件循环
        被别处的同步代码卡住 ≥ socket_timeout 时，asyncio 的读超时定时器
        会在循环恢复的瞬间先到期，把"回复其实早就到了"谎报成超时。那种
        情况重试一次基本必成。

        ⚠️ MaxConnectionsError 必须**排除**在重试之外
        -------------------------------------------
        它继承自 ConnectionError，会被 _RETRYABLE 接住 —— 但它压根不是网络
        问题：那是**本地**的池记账失败，在任何 I/O 之前就决定了，重试同样
        立刻失败。而且"池满"的成因恰恰包括"我们自己就是占着池的那些连接"，
        所以无退避地连试三次只是把 3 次失败烧成 1 个 500，让调用方以为是
        网络故障。直接抛，让路由把它变成一个说得清的错误。
        """

        consecutive = 0

        while True:

            try:
                return await self._client.xread(
                    {self._key: self._cursor},
                    block=config.block_ms,
                    count=_chunk_for(config),
                )

            except redis_exceptions.MaxConnectionsError:
                raise

            except _RETRYABLE as exc:

                consecutive += 1

                if consecutive >= config.read_max_consecutive_errors:
                    logger.warning(
                        "事件流 %s 连续读取失败 %d 次，不再重试：%s: %s。",
                        self._key,
                        consecutive,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                # 先给订阅方一个节拍：改造前每次重试都会产出 None，那不只是
                # 节流 —— SSE 那一层正是借着它去问 `is_disconnected()`。
                # 重试期间一次都不发的话，客户端在这段时间里断开了也不会被
                # 发现（最多三次重试都白等）。有界，所以照旧。
                if on_retry is not None:
                    on_retry()

                # 退一步再试，别把 3 次预算烧在微秒级里。
                await asyncio.sleep(0.05)

    def _deliver(self, response: Any) -> bool:
        """
        把一轮尾随读到的记录投给所有 live 订阅者。返回 True 表示该关流了。

        `response` 是 XREAD 的原始形状（key 与字段都是 bytes ——
        `decode_responses: false` 是刻意的默认）。
        """

        for _stream_key, raw_entries in response:

            for entry_id, raw_fields in raw_entries:

                self._cursor = _decode(entry_id)

                entry = StreamEntry(
                    id=self._cursor,
                    fields=_fields(raw_fields),
                )

                if self._note(entry):
                    # 终止事件也要投出去 —— 订阅方正是靠它收尾的。
                    self._fanout(entry)
                    return True

                self._fanout(entry)

        return False

    def _fanout(self, entry: StreamEntry) -> None:
        """投给所有 live 订阅者。补发还没做完的不收 —— 那段历史正在写。"""

        for sub in list(self._subscribers):

            if sub.live:
                self._offer(sub, entry)

    def _heartbeat(self) -> None:
        """
        发一个节拍（`None`），让订阅方有机会发 SSE 注释行并发现断线。

        ⚠️ 只在队列**空着**的时候发
        ------------------------
        心跳是节拍，不是数据。堆在队列里不但没意义，而且**有害**：消费慢的
        订阅者会因为攒了一整队心跳而撑满队列、被判为"跟不上"并断开 ——
        挤掉它的恰恰是心跳，而它真正还没读到的那些事件一起被清了。

        队列里还有东西就说明消费方有活可干（每一次读都会顺带发现断线），
        这时不发。

        ⚠️ 因此心跳**不参与**丢最旧/踢人那套策略
        ------------------------------------
        它没有信息量，丢了不影响任何东西；而真事件少一条就是正文缺一段。
        两者共用一条投递路径是错的。
        """

        for sub in list(self._subscribers):

            if not sub.live or sub.dropped is not None:
                continue

            if not sub.queue.empty():
                continue

            try:
                sub.queue.put_nowait(None)

            except asyncio.QueueFull:   # pragma: no cover - 刚判过空
                pass

    def _note(self, entry: StreamEntry) -> bool:
        """
        记一笔这条记录对"关流"意味着什么。返回 True = 立刻关流。

        ⚠️ hitl.requested 在这里**不**关流，只做记号
        -----------------------------------------
        它会不会关，取决于探活看到 DB 是否已转 waiting_input —— 那才是
        "所有 interrupt 都发完了"的权威信号（见模块 docstring 的规则 2）。
        凭"批次里最后一条"去关是错的：扇出下的第二条可能只晚几毫秒、
        落在下一轮里，按批次关就会把它关在门外，而客户端永远凑不齐
        resume 需要的 interrupt_id。
        """

        event_type = entry.event_type

        if event_type is None:
            return False

        if event_type == RunEventType.HITL_REQUESTED.value:
            self._hitl_seen = True
            return False

        return event_type in TERMINAL_VALUES

    async def _ask_liveness(self) -> RunLiveness:
        """
        探活一次。**没有探针就永远说 ALIVE**。

        ⚠️ 方法名刻意不叫 `_probe` —— 那与 `self._probe`（注入进来的探针
        本身）撞名，而实例属性会把同名方法**覆盖掉**，于是 `self._probe()`
        要么调到探针（参数对不上）、要么在没配探针时炸成
        "'NoneType' object is not callable"。这类错误在静态检查里看不出来。


        ⚠️ 超时或报错一律当"还活着"
        ------------------------
        探活只是**兜底**（正常路径是终止事件关流），而一次数据库抖动不该
        把所有正在看的流一起杀掉 —— 那会把一个短暂的故障放大成一片断线
        重连。所以这里宁可少关，不冒误杀：查不动就继续等，下一轮再问。
        日志用 warning 明说这是"探不出来"，不是"判定它死了"。
        """

        if self._probe is None:
            return RunLiveness.ALIVE

        try:
            return await asyncio.wait_for(
                self._probe(self._run_id),
                timeout=_PROBE_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "事件流 %s：探活超时（%.1fs），探不出结果，"
                "当作「还活着」继续等。",
                self._key,
                _PROBE_TIMEOUT_SECONDS,
            )
            return RunLiveness.ALIVE

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "事件流 %s：探活失败（%s: %s），探不出结果，"
                "当作「还活着」继续等。",
                self._key,
                type(exc).__name__,
                exc,
            )
            return RunLiveness.ALIVE

    def _teardown(self) -> None:
        """
        循环的收尾：给每个还挂着的订阅者一个**确定**的结局。

        ⚠️ 这一步不能省。少了它，循环因为异常退出时订阅者会永远等在
        `queue.get()` 上 —— 那正是 reader.py 开头警告的"卡住的空流"，
        比报错难查得多。

        保留位保证了收尾信号一定塞得进去（`capacity` 比 maxsize 小 1）。
        """

        signal: _Item = (
            StreamFailed(error=self._failure)
            if self._failure is not None
            else StreamClosed(
                reason=self._closed.reason if self._closed else "流已结束"
            )
        )

        for sub in list(self._subscribers):

            if sub.dropped is not None:
                continue

            try:
                sub.queue.put_nowait(signal)

            except asyncio.QueueFull:   # pragma: no cover - 保留位保证了可达性
                self._drain(sub)
                sub.queue.put_nowait(signal)

        self._subscribers.clear()
        self._pending.clear()

        if self._on_detach is not None:
            self._on_detach(self)


# ============================================================
# 注册表
# ============================================================


class RunStreamRegistry:
    """
    进程级的 `run_id -> RunStream`。

    创建与拆除的竞态靠两条保证，都不是靠"读改写之间没有 await"那一条
    （拆除必然有 await，所以那条**不适用**）：

        1. `RunStream.is_closing` 在最后一个订阅者离开时**同步**置起。
           注册表看到它就换一个新实例，绝不把新订阅者挂到正在拆的流上。
        2. 摘除带**身份校验**（`self._streams.get(key) is stream`）。
           于是"旧流收尾时把新流从字典里删掉"这种覆盖不可能发生。
    """

    def __init__(
            self,
            client: "Redis",
            *,
            probe: RunLivenessProbe | None = None,
            config: StreamConfig | None = None,
    ) -> None:

        self._client = client

        self._probe = probe

        self._config = config

        self._streams: dict[str, RunStream] = {}

    # ========================================================

    def attach(
            self,
            run_id: UUID,
            *,
            start: str = "0",
    ) -> "RunStreamAttachment":
        """
        挂上一个订阅者，返回可迭代、可关闭的句柄。**同步**。

        同步是为了让准入（上限检查）发生在路由**返回响应之前** —— 一旦
        StreamingResponse 开始吐状态行，就再也改不成 503 了。

        Raises:
            TooManyRunStreamsError: 同时在订阅的 run 数已达上限。调用方
                （路由）据此给 503，而不是让连接池在 redis-py 内部炸掉。
        """

        config = self._config or load_stream_config()

        key = stream_key(config.key_prefix, str(run_id))

        stream = self._streams.get(key)

        # 没有、或者那条已经在拆了 —— 都要新建一条。
        if stream is None or stream.is_closing:

            if stream is None and len(self._streams) >= (
                    config.max_active_run_streams
            ):
                raise TooManyRunStreamsError(
                    f"同时在订阅的事件流已达上限 "
                    f"（max_active_run_streams="
                    f"{config.max_active_run_streams}）。"
                    "这**不是** Redis 挂了 —— 是并发订阅的 run 数到了"
                    "上限。等别的流结束，或调大这个配置"
                    "（并同步调大 redis.yaml 的 stream_max_connections，"
                    "后者必须严格更大）。"
                )

            stream = RunStream(
                key=key,
                run_id=run_id,
                client=self._client,
                probe=self._probe,
                config=config,
                on_detach=self._forget,
            )

            self._streams[key] = stream

            stream.start()

        return RunStreamAttachment(stream=stream, sub=stream.attach(start))

    # ========================================================

    def _forget(self, stream: RunStream) -> None:
        """
        循环收尾时摘掉自己。**带身份校验** —— 只删"还是我"的那一条。

        没有这个校验就会出现：旧流收尾的瞬间，一个新订阅者已经建了新流并
        写进同一个键，旧流的收尾再把新流删掉 —— 那个新订阅者从此没人管。
        """

        if self._streams.get(stream._key) is stream:
            del self._streams[stream._key]

    # ========================================================

    @property
    def active_keys(self) -> list[str]:
        """当前在订阅的流键（诊断用）。"""

        return sorted(self._streams)

    @property
    def active_count(self) -> int:
        return len(self._streams)

    async def aclose(self) -> None:
        """
        停掉所有流。**由组合根在停机流程里调用**，且必须排在
        `datasource_manager.close_all()` **之前** —— 反过来是"连接先被关、
        任务还在它上面阻塞读"，收尾日志里会刷出一串看起来像"Redis 挂了"
        的噪音。
        """

        streams = list(self._streams.values())

        self._streams.clear()

        for stream in streams:
            await stream.aclose()


# ============================================================
# 进程级单例
# ============================================================

#: 进程级单例。None = 还没装配，或**根本没有 Redis**。
#: 与改造前一样，没有"空对象"：一条空流会让订阅方永远挂着，那是比报错
#: 糟得多的失败方式。Redis 没起来时路由给 503。
_registry: RunStreamRegistry | None = None


def build_run_stream_registry(
        client: "Redis | None",
        *,
        probe: RunLivenessProbe | None = None,
) -> RunStreamRegistry | None:
    """
    装配进程级广播端。**由组合根调用一次**（app/main.py 的 lifespan）。

    client 为 None（Redis 没起来）时返回 None —— 路由据此给 503。
    """

    global _registry

    _registry = (
        RunStreamRegistry(client, probe=probe)
        if client is not None
        else None
    )

    return _registry


def get_run_stream_registry() -> RunStreamRegistry | None:
    """取进程级广播端。未装配 / 没有 Redis 时返回 None。"""

    return _registry


def reset_run_stream_registry() -> RunStreamRegistry | None:
    """
    清掉单例，返回**原来那个**（没有就返回 None）。

    ⚠️ 返回值是停机的关键，不是顺手给的
    ----------------------------------
    停机流程要的是"清掉引用"**和**"把已有的流停掉"，两件事必须都做：
    先取旧的、再清引用，调用方拿到旧的那个去 `aclose()`。只清引用的话，
    每个还在跑的循环 task 会继续占着它的连接，而数据源马上就要关池 ——
    那正是收尾日志里那串"看起来像 Redis 挂了"的噪音。

    单单清引用（不 aclose）只有测试用得上：那种场景下没有真的循环。

    注意它**不 await**（因此是同步的）：停机流程里"取出来"和"清掉"之间
    不能有挂起点，否则一个正在订阅的请求可能在这中间拿到一个已经清空的
    注册表。
    """

    global _registry

    previous = _registry

    _registry = None

    return previous
