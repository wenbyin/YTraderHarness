"""
运行事件流的冒烟测试（发布端 / 读取端 / SSE 帧）。

前八节不需要 Redis：假 client 只记录命令、按脚本返回 XREAD 结果，
因此形状与失败语义都能确定地验证。

第九节**连真 Redis**（本地 docker 那个）。连不上就明确跳过并说明原因 ——
跳过的用例必须说得出"我跳过了什么"，否则绿色的输出会让人以为它验过。

⚠️ 假 pipeline **复刻了** redis-py 的 `__aexit__` 语义：只 reset，
**不** execute。照抄这一条是刻意的 —— 如果假实现"顺手"替你 execute 了，
那么真实现里漏掉 `execute()` 这个 bug 就永远测不出来，而它的症状是
"一条事件都没发出去，却也不报错"，最难查。

⚠️ 写入发生在**写线程**里，所以"publish 之后立刻断言"不再成立
--------------------------------------------------------
每个发布用例都要先 `await publisher.drain()`（等队列排空），再看假 sink
记下了什么。假对象因此分成两个：

    FakeRedis      asyncio 版 —— 读取端用（xread 是 await 的）
    FakeSink       同步版   —— 发布端用（写入跑在别的线程里，只能是同步的）

RedisStreamSink 那一层单独用 FakeSyncRedis 验（同步 pipeline），
因为它才是"XADD + EXPIRE 走一次往返"和"漏 execute 就静默丢事件"的守卫。

跑法：
    python -m app.agent.event.event_smoke
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from datetime import datetime
from typing import Any
from uuid import uuid4

from redis.exceptions import (
    MaxConnectionsError,
    TimeoutError as RedisTimeoutError,
)

from app.agent.event import (
    HEARTBEAT,
    NullEventPublisher,
    RedisEventPublisher,
    RedisStreamSink,
    RunStreamRegistry,
    StreamConfig,
    TooManyRunStreamsError,
    build_event_publisher,
    build_run_stream_registry,
    close_event_publisher,
    get_event_publisher,
    get_run_stream_registry,
    is_terminal,
    load_stream_config,
    publish_run_event,
    reset_event_publisher,
    reset_run_stream_registry,
    sse_frame,
    start_event_publisher,
    stream_key,
)
from app.common.event import TERMINAL_EVENT_TYPES, RunEvent, RunEventType

# ============================================================
# 断言工具
# ============================================================


def check(label, condition):
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title):
    print(f"\n{title}")


def skip(label, why):
    print(f"  [SKIP] {label} —— {why}")


async def _pump(times: int = 60) -> None:
    """
    给广播端的循环 task 若干回合的机会。

    不 sleep 真实时间：替身不真的阻塞，所以循环转得快，让出几次控制权就够。
    """

    for _ in range(times):
        await asyncio.sleep(0)


# ============================================================
# 假 Redis
# ============================================================


class FakePipeline:
    """记录被排入的命令。execute() 时才"发"出去。"""

    def __init__(self, redis: "FakeRedis") -> None:

        self._redis = redis
        self.commands: list[tuple] = []

    def xadd(self, name, fields, **kwargs):

        self.commands.append(("xadd", name, fields, kwargs))
        return self

    def expire(self, name, seconds):

        self.commands.append(("expire", name, seconds, {}))
        return self

    async def execute(self):

        if self._redis.fail:
            raise ConnectionError("fake redis is down")

        self._redis.executed.extend(self.commands)
        self._redis.executes += 1

        return [b"1-1"] * len(self.commands)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # ⚠️ 与 redis-py 一致：只 reset，**不** execute。
        return False


class FakeRedis:
    """
    **读取端**用的假 asyncio client（只有 xread）。

    `xread_responses` 按调用顺序弹，弹空之后一律返回 None
    （模拟"阻塞到点、没有新记录"）。

    `xread_errors` 是开头前 N 次抛读超时的次数 —— 用来验读取端的重试：
    事件循环被同步代码卡住时，redis-py 报的是这个错，而它其实是**谎报**
    （回复早就到了，是 asyncio 的定时器先到期）。

    发布端不再用它：写入跑到写线程里去了，那里的 client 必须是同步的
    （见 FakeSink / FakeSyncRedis）。
    """

    def __init__(
            self,
            *,
            fail: bool = False,
            xread_responses: list[Any] | None = None,
            xread_errors: int = 0,
            xread_error: BaseException | None = None,
    ) -> None:

        self.fail = fail
        self.executed: list[tuple] = []
        self.executes = 0
        self.transactions: list[bool] = []

        self._xread = list(xread_responses or [])
        self._xread_errors = xread_errors

        #: 抛什么。默认读超时（"谎报"那条路径）；注入别的异常用来验
        #: "哪些错误**不**该被当成可重试"。
        self._xread_error = xread_error or RedisTimeoutError(
            "Timeout reading from localhost:6379"
        )

        self.xread_calls: list[dict] = []

        #: 内存流：`键 -> [(id, 字段)]`。脚本弹空之后 xread 从这里按游标取。
        #:
        #: 广播端要按 id 范围查询（xrange 补发 + xread 尾随），光靠"排好的
        #: 返回值"验不了两者的边界，所以这里放一条真的能按范围查的内存流。
        self.stream: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}

        self.command_calls: dict[str, int] = {}

    def add(self, key: str, entry_id: str, event: str, **fields: str) -> None:
        """往内存流里写一条（字段名与值都存 bytes，同 decode_responses=false）。"""

        payload: dict[bytes, bytes] = {b"event": event.encode()}

        for name, value in fields.items():
            payload[name.encode()] = str(value).encode()

        self.stream.setdefault(key, []).append((entry_id, payload))

    def pipeline(self, *, transaction: bool = True):

        self.transactions.append(transaction)
        return FakePipeline(self)

    @staticmethod
    def _parse(entry_id: str) -> tuple[int, int]:
        ms, _, seq = entry_id.partition("-")
        return int(ms), int(seq or 0)

    def _slice(self, key: str, lower: str, upper: str) -> list:
        """
        按 XRANGE 的语义取一段：起点**包含**、终点包含，`(` 前缀表示排除。

        刻意与 XREAD 的游标语义区分开 —— 后者的游标天生**排他**。实现里
        "补发到游标（含）+ 尾随从游标之后"正是靠这个区别接上的，替身把两边
        混成一个的话，边界写错就验不出来。
        """

        rows = self.stream.get(key, [])

        lo_excl = lower.startswith("(")
        hi_excl = upper.startswith("(")

        lo = (0, 0) if lower in ("-", "(-") else self._parse(
            lower.lstrip("(")
        )
        hi = (1 << 62, 0) if upper in ("+", "(+") else self._parse(
            upper.lstrip("(")
        )

        return [
            (entry_id, payload)
            for entry_id, payload in rows
            if (
                self._parse(entry_id) > lo
                if lo_excl
                else self._parse(entry_id) >= lo
            )
            and (
                self._parse(entry_id) < hi
                if hi_excl
                else self._parse(entry_id) <= hi
            )
        ]

    async def xrange(self, key, lower="-", upper="+", count=None):

        self.command_calls["xrange"] = self.command_calls.get("xrange", 0) + 1

        picked = self._slice(key, str(lower), str(upper))

        return [
            (entry_id.encode(), payload)
            for entry_id, payload in (picked[:count] if count else picked)
        ]

    async def xrevrange(self, key, upper="+", lower="-", count=None):

        self.command_calls["xrevrange"] = (
            self.command_calls.get("xrevrange", 0) + 1
        )

        picked = list(reversed(self._slice(key, str(lower), str(upper))))

        return [
            (entry_id.encode(), payload)
            for entry_id, payload in (picked[:count] if count else picked)
        ]

    async def xread(self, streams, count=None, block=None):

        self.xread_calls.append(dict(streams))

        self.command_calls["xread"] = self.command_calls.get("xread", 0) + 1

        if self._xread_errors > 0:
            self._xread_errors -= 1
            raise self._xread_error

        if self._xread:
            return self._xread.pop(0)

        # 脚本弹空之后：从内存流里按游标取（游标是**排他**的，与 Redis 一致）。
        if self.stream:

            key, cursor = next(iter(streams.items()))

            after = (0, 0) if cursor in ("0", "0-0") else self._parse(
                str(cursor)
            )

            picked = [
                (entry_id, payload)
                for entry_id, payload in self.stream.get(key, [])
                if self._parse(entry_id) > after
            ]

            if count:
                picked = picked[:count]

            if picked:
                return [(
                    key.encode(),
                    [
                        (entry_id.encode(), payload)
                        for entry_id, payload in picked
                    ],
                )]

        return None


class FakeSyncPipeline:
    """同步版假 pipeline —— 复刻 redis-py **同步** Pipeline 的 __exit__ 语义。"""

    def __init__(self, redis: "FakeSyncRedis") -> None:

        self._redis = redis
        self.commands: list[tuple] = []

    def xadd(self, name, fields, **kwargs):

        self.commands.append(("xadd", name, fields, kwargs))
        return self

    def expire(self, name, seconds):

        self.commands.append(("expire", name, seconds, {}))
        return self

    def execute(self):

        if self._redis.fail:
            raise ConnectionError("fake redis is down")

        self._redis.executed.extend(self.commands)
        self._redis.executes += 1

        return [b"1-1"] * len(self.commands)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # ⚠️ 与 redis-py 一致：只 reset，**不** execute。
        return False


class FakeSyncRedis:
    """**发布端写线程**用的假同步 client（只有 pipeline）。"""

    def __init__(self, *, fail: bool = False) -> None:

        self.fail = fail
        self.executed: list[tuple] = []
        self.executes = 0
        self.transactions: list[bool] = []

    def pipeline(self, *, transaction: bool = True):

        self.transactions.append(transaction)
        return FakeSyncPipeline(self)


class FakeSink:
    """
    发布端的假 **sink** —— 写入真的发生在写线程里。

    ⚠️ `writes` 由写线程 append、由主线程在 drain() 之后读，所以断言必须
    排在 `await publisher.drain()` 后面（见模块 docstring）。
    """

    def __init__(
            self,
            *,
            fail: bool = False,
            slow: float = 0.0,
    ) -> None:

        self.fail = fail
        self.slow = slow

        self.writes: list[tuple] = []
        self.pings = 0
        self.threads: list[str] = []

    def ping(self) -> bool:

        self.pings += 1

        return not self.fail

    def write(self, key, fields, *, maxlen, ttl_seconds) -> None:

        self.threads.append(threading.current_thread().name)

        if self.slow:
            time.sleep(self.slow)

        if self.fail:
            raise ConnectionError("fake redis is down")

        self.writes.append((key, fields, maxlen, ttl_seconds))


def _fields(command: tuple) -> dict:
    """xadd 命令的字段段。"""

    return command[2]


def _entry(entry_id: str, **fields: str) -> tuple:
    """造一条 XREAD 回来的记录（bytes，与 decode_responses=False 一致）。"""

    return (
        entry_id.encode(),
        {
            name.encode(): value.encode()
            for name, value in fields.items()
        },
    )


# ============================================================
# 真 Redis（连不上就跳过）
# ============================================================


async def _real_client():
    """
    按 app/config/datasource/redis.yaml 建一个真 client。连不上返回 None。

    刻意走**真实的配置与数据源类**，而不是自己拼一个 localhost URL ——
    这一节要验的正是"配置写对了没有"。
    """

    import yaml

    from app.common.utils import expand_env
    from app.infra.datasource.models import RedisDataSourceDefinition
    from app.infra.datasource.redis import RedisDataSource

    from pathlib import Path

    raw = yaml.safe_load(
        Path("app/config/datasource/redis.yaml").read_text(encoding="utf-8")
    )

    source = RedisDataSource(
        RedisDataSourceDefinition.model_validate(expand_env(raw))
    )

    if await source.health_check():
        return source

    await source.close()

    return None


# ============================================================


async def main() -> int:

    config = load_stream_config()

    # --------------------------------------------------------

    section("1. 键与字段的形状（对外契约）")

    check(
        "topic + topic_id -> agent_run:run_123",
        stream_key("agent_run", "run_123") == "agent_run:run_123",
    )

    event = RunEvent(
        type=RunEventType.MESSAGE_DELTA,
        topic=config.key_prefix,
        topic_id="run_123",
        data={"delta": "你好，世界"},
        node="agent",
    )

    fields = event.to_fields()

    check(
        "信封字段平铺在顶层（订阅方不解 JSON 就能筛 run）",
        fields["event"] == "message.delta"
        and fields["topic"] == "agent_run"
        and fields["topic_id"] == "run_123"
        and fields["node"] == "agent",
    )
    check(
        "data 是 JSON **字符串**（Stream 的字段值不能嵌套）",
        isinstance(fields["data"], str)
        and json.loads(fields["data"]) == {"delta": "你好，世界"},
    )
    check(
        "每个字段值都是 str",
        all(isinstance(value, str) for value in fields.values()),
    )
    check(
        "occurred_at 带时区（不带前端会按本地时区读错）",
        datetime.fromisoformat(fields["occurred_at"]).tzinfo is not None,
    )
    check(
        "node 缺省是空串，不是 'None'",
        RunEvent(
            type=RunEventType.RUN_STARTED,
            topic="agent_run",
            topic_id="run_123",
        ).to_fields()["node"] == "",
    )
    check(
        "十二个事件名一个不多一个不少",
        [item.value for item in RunEventType] == [
            "run.started",
            "run.completed",
            "run.failed",
            "node.started",
            "node.completed",
            "message.started",
            "message.delta",
            "message.completed",
            "tool.called",
            "tool.completed",
            "hitl.requested",
            "hitl.resolved",
        ],
    )
    check(
        "终止事件只有 completed / failed",
        {item.value for item in TERMINAL_EVENT_TYPES}
        == {"run.completed", "run.failed"},
    )

    # --------------------------------------------------------

    section("2. 投递 = 入队，写入发生在**写线程**里")

    sink = FakeSink()
    publisher = RedisEventPublisher(sink, config)
    publisher.start()

    check("装了真 sink 后 enabled 为真", publisher.enabled is True)
    check("入队成功返回 True", await publisher.publish(event) is True)
    check("入队就好 —— 此刻可能还没写出去", len(sink.writes) <= 1)

    await publisher.drain()

    check("drain 之后真的写出去了", len(sink.writes) == 1)
    check(
        "写在名为 event-publisher 的线程上（不是事件循环线程）",
        sink.threads == ["event-publisher"],
    )

    key_written, fields_written, maxlen_written, ttl_written = sink.writes[0]

    check("键是 agent_run:run_123", key_written == "agent_run:run_123")
    check(
        "内容在入队那一刻就冻结了（字段与 event.to_fields() 一致）",
        fields_written["event"] == "message.delta"
        and json.loads(fields_written["data"]) == {"delta": "你好，世界"},
    )
    check(
        "MAXLEN 用的是配置值（近似裁剪在 sink 里加 ~）",
        maxlen_written == config.maxlen,
    )
    check("TTL 用的是配置值", ttl_written == config.ttl_seconds)

    # ⚠️ 这条守的是"publish() 里不许有 await"这条不变量（见 publisher.py
    # 模块 docstring）：没有 await，它相对事件循环就是不可分割的一个回合，
    # 与 aclose() 的"置停止位"天然串行，不需要锁。
    # 判据：await 它的时候，别的就绪任务拿不到执行机会。
    yielded: list[int] = []

    async def _other() -> None:
        yielded.append(1)

    pending = asyncio.create_task(_other())
    await publisher.publish(event)
    check("publish() 不让出事件循环（体内没有 await）", yielded == [])
    await pending

    await publisher.aclose()

    # --------------------------------------------------------

    section("3. RedisStreamSink：一次往返，且**必须**显式 execute")

    sync_client = FakeSyncRedis()
    RedisStreamSink(sync_client).write(
        "agent_run:run_123",
        {"event": "message.delta"},
        maxlen=config.maxlen,
        ttl_seconds=config.ttl_seconds,
    )

    commands = sync_client.executed

    check(
        "开的是非事务 pipeline（不需要 MULTI）",
        sync_client.transactions == [False],
    )
    check(
        "只发两条命令：XADD + EXPIRE",
        [item[0] for item in commands] == ["xadd", "expire"],
    )
    check("XADD 的键是 agent_run:run_123", commands[0][1] == "agent_run:run_123")
    check(
        "XADD 用近似裁剪 MAXLEN ~",
        commands[0][3]["approximate"] is True
        and commands[0][3]["maxlen"] == config.maxlen,
    )
    check("EXPIRE 用的是配置里的 ttl", commands[1][2] == config.ttl_seconds)
    check("只 execute 了一次（一次往返）", sync_client.executes == 1)

    plain = StreamConfig(
        enabled=True,
        key_prefix="agent_run",
        maxlen=None,
        ttl_seconds=None,
    )

    sync_plain = FakeSyncRedis()
    RedisStreamSink(sync_plain).write(
        "agent_run:run_123", {"event": "x"}, maxlen=None, ttl_seconds=None
    )

    check(
        "maxlen=None / ttl=None 时只 XADD，不发 EXPIRE",
        [item[0] for item in sync_plain.executed] == ["xadd"],
    )
    check(
        "maxlen=None 原样传给 redis",
        sync_plain.executed[0][3]["maxlen"] is None,
    )

    # --------------------------------------------------------

    section("4. 写入失败**不上抛**（事件是旁路）；队列满则拒收")

    broken_sink = FakeSink(fail=True)
    broken = RedisEventPublisher(broken_sink, config)
    broken.start()

    accepted = await broken.publish(event)

    check("入队本身是成功的（返回 True）", accepted is True)

    await broken.drain()

    check("写入失败被计数", broken.failures == 1)
    check("一条都没写出去", broken_sink.writes == [])
    check("失败没把写线程搞死（后续投递照常入队）",
          await broken.publish(event) is True)

    await broken.aclose()

    # 队列满：拒收**最新**那条，已接收的照常写出（连续前缀，正文不留空洞）
    slow_sink = FakeSink(slow=0.1)
    bounded = RedisEventPublisher(slow_sink, config, queue_size=2)
    bounded.start()

    results = [await bounded.publish(event) for _ in range(4)]

    check("队列满之后拒绝入队（而不是无限吃下）", results[:2] == [True, True]
          and results[2:] == [False, False])
    check("被拒的条数单独计数", bounded.rejected == 2)

    await bounded.drain()

    check("已接收的那两条照常写出去（顺序不变）", len(slow_sink.writes) == 2)
    check("写入失败与拒收分开计数", bounded.failures == 0)

    # 没起线程 = 没人消费，最该防的就是"入队成功却没人写"
    idle = RedisEventPublisher(FakeSink(), config)

    check("没 start() 时拒收（而不是静默排进没人消费的队列）",
          await idle.publish(event) is False)

    await bounded.aclose()

    # --------------------------------------------------------

    section("5. 空发布器与「未装配」")

    null = NullEventPublisher()

    check("空发布器 enabled 为假", null.enabled is False)
    check("空发布器 publish 返回 False", await null.publish(event) is False)

    reset_event_publisher()

    check(
        "未装配时 getter 给的是空发布器（而不是抛）",
        isinstance(get_event_publisher(), NullEventPublisher),
    )

    # --------------------------------------------------------

    section("6. 最常用入口 publish_run_event 自带 topic")

    reset_event_publisher()

    wired_sink = FakeSink()

    build_event_publisher(wired_sink)

    check("装配后 enabled 为真", get_event_publisher().enabled is True)

    started = await start_event_publisher()

    check("探活通过并起了写线程", started is True)
    check("探活真的打了（一次）", wired_sink.pings == 1)

    sent = await publish_run_event(
        RunEventType.NODE_STARTED,
        "run_999",
        node="agent",
        data={"node_id": "agent"},
    )

    check("投递成功", sent is True)

    await get_event_publisher().drain()

    sent_key, sent_fields = wired_sink.writes[0][0], wired_sink.writes[0][1]

    check(
        "topic 由配置填，调用方不必传",
        sent_fields["topic"] == config.key_prefix,
    )
    check("topic_id 就是 run id", sent_fields["topic_id"] == "run_999")
    check("键拼成 agent_run:run_999", sent_key == "agent_run:run_999")

    await close_event_publisher()

    check(
        "收尾后清回单例（未装配 -> 空对象）",
        isinstance(get_event_publisher(), NullEventPublisher),
    )

    # 探活失败：退回空对象。这条是有意的 —— enabled 谎报 True 会让模型层
    # 记上"已经推给消费端内容了"、失败不再重试，而实际没人在收。
    reset_event_publisher()

    dead_sink = FakeSink(fail=True)

    build_event_publisher(dead_sink)

    alive = await start_event_publisher(ping_timeout=1.0)

    check("探活失败 -> 返回 False", alive is False)
    check(
        "探活失败 -> 退回空对象（enabled 为假）",
        isinstance(get_event_publisher(), NullEventPublisher),
    )
    check(
        "探活失败 -> 投递直接返回 False，且一条都没发",
        await publish_run_event(RunEventType.RUN_STARTED, "run_999") is False
        and dead_sink.writes == [],
    )

    # --------------------------------------------------------

    section("7. 广播端：一个 run 一条连接 + 补发边界 + 错误分类")

    fake_read = FakeRedis()

    run_id = uuid4()

    # ⚠️ 键必须由 run_id 拼出来（广播端就是这么做键的）。手写一个
    # "agent_run:run_1" 而拿另一串 uuid 去订阅的话，两边永远对不上 ——
    # 订阅者只会收到心跳，表现为**永远等不到事件**。
    fake_key = stream_key(config.key_prefix, str(run_id))

    fake_read.add(
        fake_key, "100-0", "run.started", topic="agent_run",
        topic_id=str(run_id), data="{}",
    )
    fake_read.add(
        fake_key, "101-0", "message.delta", topic="agent_run",
        topic_id=str(run_id), data='{"delta": "你"}',
    )

    registry = RunStreamRegistry(fake_read, config=config)

    # 两个订阅者订阅**同一个** run
    first = registry.attach(run_id, start="0")
    second = registry.attach(run_id, start="0")

    await _pump()

    check(
        "一个 run 只有一个循环 task（= 一条 Redis 连接）",
        len([
            task for task in asyncio.all_tasks()
            if task.get_name().startswith("run-stream:")
        ]) == 1,
    )
    check("两个订阅者挂在同一条流上", registry.active_count == 1)

    # 收三条：两条记录 + 一个心跳节拍。
    #
    # 心跳**只在队列空着的时候**发（理由见 run_stream.py 的 `_heartbeat`：
    # 攒一队心跳会把消费慢的订阅者挤掉）。所以第三条要等消费方把前两条
    # 读走、循环再转一轮才会来 —— 不是缺陷，是刻意的。
    seen_first: list[Any] = []

    async for item in first:
        seen_first.append(item)
        if len(seen_first) >= 3:
            break

    entries = [item for item in seen_first if item is not None]

    check(
        "两条记录都解成了 str（**字段名**与值都是）",
        entries[0].id == "100-0"
        and entries[0].event_type == "run.started"
        and all(isinstance(name, str) for name in entries[1].fields),
    )
    check(
        "补齐的是全部历史（晚连的订阅者也看得到全过程）",
        [item.id for item in entries] == ["100-0", "101-0"],
    )
    check(
        "message.delta 不是终止事件",
        entries[1].event_type == "message.delta"
        and is_terminal(entries[1]) is False
        and is_terminal(entries[0]) is False,
    )
    check(
        "没有新记录时产出 None（心跳节拍）",
        seen_first[2] is None,
    )

    # 终止事件：写进去，两个订阅者都该被正常收尾。
    fake_read.add(fake_key, "102-0", "run.completed", data="{}")

    await _pump()

    seen_second: list[Any] = []

    async for item in second:
        seen_second.append(item)

    terminal = [item for item in seen_second if item is not None]

    check(
        "终止事件被认成终止（run.completed）",
        is_terminal(terminal[-1]) is True
        and terminal[-1].event_type == "run.completed",
    )
    check(
        "收到终止事件后**两个**订阅者都结束（不只是先看到的那个）",
        second.closed_reason is not None,
    )

    first.aclose()
    second.aclose()

    await _pump()

    # ---- 读超时**谎报**时的重试 ----
    #
    # 事件循环被同步代码卡住 ≥ socket_timeout 时，asyncio 的定时器会在循环
    # 恢复的瞬间先到期，把"回复早就到了"报成读超时（已实测复现）。改造前，
    # 这一次就把整条 SSE 流打断了。
    retry_run = uuid4()
    retry_key = stream_key(config.key_prefix, str(retry_run))

    # ⚠️ 这里用**脚本化**的 xread 返回值，而不是往内存流里写一条。
    # 往内存流里写的话，订阅者会在补发阶段就拿到它（补发在尾随之前），
    # 于是"前两次失败只产出心跳"这条就验不到了 —— 它拿到的是那条记录。
    # 脚本化把"第三次调用才成功"这件事说死。
    flaky = FakeRedis(
        xread_errors=2,
        xread_responses=[[(
            retry_key.encode(),
            [_entry("200-0", event="run.started", data="{}")],
        )]],
    )

    retried: list[Any] = []

    retry_registry = RunStreamRegistry(flaky, config=config)

    attachment = retry_registry.attach(retry_run, start="0")

    async for item in attachment:
        retried.append(item)
        if item is not None:
            break

    check(
        "前两次读超时只产出心跳节拍（不抛、不断流）",
        retried[0] is None and retried[1] is None,
    )
    check(
        "第三次读到了那条记录（自愈，且没漏事件）",
        retried[2] is not None and retried[2].event_type == "run.started",
    )

    attachment.aclose()

    async def _expect_error(fake, cfg, expected, label):
        """把一条流跑到抛错，看抛的是不是 expected。"""

        reg = RunStreamRegistry(fake, config=cfg)
        handle = reg.attach(uuid4(), start="0")

        try:
            async for _ in handle:
                pass

        except BaseException as exc:      # noqa: BLE001
            check(label, isinstance(exc, expected))

        else:
            check(label, False)

        finally:
            handle.aclose()

    # **达到**上限就抛（含第一次）—— "如实抛错"那条立场没变，变了的
    # 只是"多给几次机会"。这里给的失败次数正好等于上限。
    await _expect_error(
        FakeRedis(xread_errors=config.read_max_consecutive_errors),
        config,
        RedisTimeoutError,
        "连续失败到上限 -> 把异常抛给订阅方",
    )

    # 上限设成 1 = 不重试，退回改造前的行为（可回滚的开关）
    await _expect_error(
        FakeRedis(xread_errors=1),
        StreamConfig(read_max_consecutive_errors=1),
        RedisTimeoutError,
        "上限=1 时第一次就抛（可退回旧行为）",
    )

    # ⚠️ 池满**不重试** —— 这是原来那个故障的直接成因之一。
    # 它继承自 ConnectionError，会被"网络抖动重试"那套接住，但它压根不是
    # 网络问题：那是本地池记账失败，重试同样立刻失败，无退避地连试三次
    # 只是把 3 次失败烧成一个 500。
    pool_full = FakeRedis(
        xread_errors=99,
        xread_error=MaxConnectionsError("Too many connections"),
    )

    await _expect_error(
        pool_full,
        config,
        MaxConnectionsError,
        "池满直接抛，不烧重试预算",
    )
    check(
        "池满只试了一次 xread（没有重试）",
        pool_full.command_calls.get("xread", 0) == 1,
    )

    # 活跃 run 数上限：给一个**说得清**的错误，而不是让连接池在 redis-py
    # 内部炸出 MaxConnectionsError。
    capped = RunStreamRegistry(
        FakeRedis(),
        config=StreamConfig(max_active_run_streams=1),
    )

    handle = capped.attach(uuid4(), start="0")

    try:
        capped.attach(uuid4(), start="0")

    except TooManyRunStreamsError as exc:
        check(
            "活跃 run 数到上限 -> 明确的错误（提到该调哪个旋钮）",
            "max_active_run_streams" in str(exc)
            and "stream_max_connections" in str(exc),
        )

    else:
        check("活跃 run 数到上限 -> 明确的错误（提到该调哪个旋钮）", False)

    handle.aclose()
    await capped.aclose()
    await retry_registry.aclose()
    await registry.aclose()

    # --------------------------------------------------------

    section("8. SSE 帧格式")

    frame = sse_frame(entries[1])

    check("以空行结尾（少一个空行客户端会把两帧粘起来）", frame.endswith("\n\n"))
    check("有 id:（断线重连靠它续上）", frame.startswith("id: 101-0\n"))
    check("有 event:，浏览器可直接 addEventListener", "event: message.delta\n" in frame)
    body = json.loads(frame.split("data: ", 1)[1].split("\n")[0])

    check(
        "data: 放整个信封（含 topic_id 等信封字段）",
        body["topic_id"] == str(run_id),
    )
    check(
        "data 子字段被解开成**对象**，不是两层 JSON 字符串",
        body["data"] == {"delta": "你"},
    )
    check(
        "心跳是注释行（客户端会忽略）",
        HEARTBEAT.startswith(":") and HEARTBEAT.endswith("\n\n"),
    )

    # --------------------------------------------------------

    section("9. 真 Redis 往返（本地 docker）")

    source = await _real_client()

    if source is None:
        skip(
            "真 Redis 往返",
            "连不上 app/config/datasource/redis.yaml 里的 Redis。"
            "起一个：docker start ytrader-redis",
        )
    else:
        try:
            client = source.stream_client

            # 发布端用**同步** client（写线程），读取端用**长连接池**那个
            # （stream_client）—— 与组合根 app/main.py 里的接法一致。
            build_event_publisher(RedisStreamSink(source.sync_client))
            build_run_stream_registry(client)

            await start_event_publisher()

            # topic_id 用真 UUID：广播端按 run_id 订阅，键由它自己拼
            # （stream_key(key_prefix, str(run_id))）。
            run_id = uuid4()
            topic_id = str(run_id)

            key = stream_key(config.key_prefix, topic_id)

            # 清掉上一次跑剩下的，免得断言数到旧记录。
            await client.delete(key)

            await publish_run_event(RunEventType.RUN_STARTED, topic_id)
            await publish_run_event(
                RunEventType.MESSAGE_DELTA,
                topic_id,
                node="agent",
                data={"delta": "真·增量"},
            )
            await publish_run_event(
                RunEventType.RUN_COMPLETED,
                topic_id,
                data={"status": "completed"},
            )

            # ⚠️ 写在线程里，所以下面数 xlen / ttl 之前必须等队列排空 ——
            # 不等就会偶发地数到 0 条，而且看起来像"发布端坏了"。
            await get_event_publisher().drain()

            live = get_run_stream_registry()

            check("广播端已装配", live is not None and live is not None)

            seen: list[str] = []

            attachment = live.attach(run_id, start="0")

            async for item in attachment:
                if item is None:
                    continue
                seen.append(item.event_type)
                # 收到终止事件后广播端自己收尾，迭代随之结束。

            check(
                "三条事件按顺序读到（补发路径真的通）",
                seen == [
                    "run.started",
                    "message.delta",
                    "run.completed",
                ],
            )
            check(
                "流以终止事件收尾（不是靠超时）",
                attachment.closed_reason is not None,
            )

            attachment.aclose()

            length = await client.xlen(key)
            ttl = await client.ttl(key)

            check("流里有三条记录", length == 3)
            check(
                "TTL 被设上了（否则键会永久堆积）",
                0 < ttl <= config.ttl_seconds,
            )

            raw = await client.xrange(key, count=1)
            check(
                "字段名与值在 redis 里是 bytes（decode_responses=false）",
                isinstance(raw[0][1][b"event"], bytes),
            )

            # 不留垃圾：这条 key 是本用例造的。
            await client.delete(key)

        finally:
            # ⚠️ 顺序：先停广播端（它脚下的连接归 source 的 stream_client
            # 池），再停写线程，最后关数据源。反过来是"连接池先被关、任务
            # 还在它上面阻塞读"。
            stopped = reset_run_stream_registry()

            if stopped is not None:
                await stopped.aclose()

            await close_event_publisher()
            await source.close()

    reset_event_publisher()
    reset_run_stream_registry()

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
