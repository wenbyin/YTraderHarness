"""
`RunStreamRegistry` / `RunStream`：按 run 复用读取端 + 探活关流。

验的是改造的三条主张，每条都对应一个**已经实测发生过**的故障：

    1. 一个 run 的订阅者共用**一条**连接
       （原故障：并发订阅数打满池 → MaxConnectionsError → 500 → 重连死循环）
    2. 补发与尾随的边界严丝合缝 —— 不重不漏
       （原故障：每个订阅者各自 XREAD，边界靠"最后一条 id"传递）
    3. 没有下文时流会**自己关**
       （原故障：已结束的 run 被浏览器重连后永久挂着，一条连接一个 run）

以及本次实现里最容易写错的几处：慢订阅者只踢它自己、最后一个订阅者走了
之后流要拆掉、活跃 run 数到上限要给明确的错误。

全部用内存替身（tests/agent/event/stream_fakes.py），**不 sleep** ——
漏跑通过 `idle_probe_after_reads` 计数触发，因此测试跑得快且不飘。
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from app.agent.event import (
    RunLiveness,
    RunStreamRegistry,
    StreamClosed,
    StreamFailed,
    TooManyRunStreamsError,
)
from app.agent.event.config import StreamConfig
from app.agent.event.run_stream import RunStream
from app.agent.event.sse import HEARTBEAT, sse_frame

from stream_fakes import FakeStreamRedis


# ============================================================
# 脚手架
# ============================================================


def make_config(**overrides: Any) -> StreamConfig:
    """一份适合测试的配置：探活阈值很小，队列很短。"""

    base: dict[str, Any] = {
        # 替身不真的阻塞，所以 0 不会变成忙轮询（真 Redis 下会）。
        "block_ms": 0,
        "idle_probe_after_reads": 2,
        "subscriber_queue_size": 8,
        "max_active_run_streams": 3,
    }

    base.update(overrides)

    return StreamConfig(**base)


class FakeProbe:
    """按脚本回答的探针。"""

    def __init__(self, verdict: RunLiveness = RunLiveness.ALIVE) -> None:

        self.verdict = verdict
        self.calls = 0

    async def __call__(self, run_id) -> RunLiveness:

        self.calls += 1

        return self.verdict


class BoomProbe:
    """探活时炸掉 —— 用来验"探不出来 ≠ 判定它死了"。"""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, run_id) -> RunLiveness:

        self.calls += 1

        raise RuntimeError("数据库抖了一下")


async def pump(times: int = 60) -> None:
    """给循环 task 若干回合的机会。"""

    for _ in range(times):
        await asyncio.sleep(0)


async def collect(attachment, limit: int = 200) -> list[Any]:
    """把附件读到收尾（或读满 limit 条）。"""

    items: list[Any] = []

    async for item in attachment:
        items.append(item)

        if len(items) >= limit:
            break

    return items


def events(items: list[Any]) -> list[str]:
    """只要真事件（滤掉心跳节拍 None 与收尾信号）。"""

    return [
        item.event_type
        for item in items
        if not isinstance(item, (StreamClosed, StreamFailed))
        and item is not None
    ]


def entries(items: list[Any]) -> list[str]:
    """事件的 entry id。"""

    return [
        item.id
        for item in items
        if not isinstance(item, (StreamClosed, StreamFailed))
        and item is not None
    ]


def stream_tasks() -> list[asyncio.Task]:
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("run-stream:")
    ]


@pytest.fixture
def redis() -> FakeStreamRedis:
    return FakeStreamRedis()


# ============================================================
# 主张 1：一个 run 一条连接
# ============================================================


async def test_two_subscribers_share_one_loop(redis: FakeStreamRedis) -> None:

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    a = registry.attach(run_id, start="0")
    b = registry.attach(run_id, start="0")

    await pump()

    # 一个 run 一个循环 task —— 这就是"一条连接"在测试里的可见形态。
    assert len(stream_tasks()) == 1

    # 两个订阅者各自都收到了那条事件（各有一份，不是分着拿）。
    items_a = await asyncio.wait_for(collect(a), timeout=2)
    items_b = await asyncio.wait_for(collect(b), timeout=2)

    assert events(items_a) == ["run.started"]
    assert events(items_b) == ["run.started"]

    a.aclose()
    b.aclose()


async def test_second_subscriber_does_not_start_a_second_loop(
        redis: FakeStreamRedis,
) -> None:
    """同一个 run 的第 N 个订阅者不再新增循环。"""

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    handles = [registry.attach(run_id, start="0") for _ in range(3)]

    await pump()

    assert len(stream_tasks()) == 1
    assert registry.active_count == 1

    for handle in handles:
        handle.aclose()


# ============================================================
# 主张 2：补发 / 尾随的边界
# ============================================================


async def test_late_subscriber_gets_history_then_live(
        redis: FakeStreamRedis,
) -> None:
    """
    晚连的订阅者：先把已有的补齐，再跟着实时走 —— 一条不漏、一条不重。

    这是最容易写错的地方：补发的终点（含）与尾随的起点（不含）必须正好
    接上。多一格就是重复投递，少一格就是静默丢事件。
    """

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")
    redis.add("2-0", "node.started")

    early = registry.attach(run_id, start="0")

    await pump()

    # 早连的那个已经拿到前两条了
    assert events(await asyncio.wait_for(collect(early), timeout=2)) == [
        "run.started",
        "node.started",
    ]

    # 现在晚连一个 —— 它要把前两条**补**回来
    late = registry.attach(run_id, start="0")

    await pump()

    # 再写一条，它也应该收到（走尾随那条路）
    redis.add("3-0", "message.delta")

    await pump()

    late_items = await asyncio.wait_for(collect(late), timeout=2)

    assert entries(late_items) == ["1-0", "2-0", "3-0"]

    early.aclose()
    late.aclose()


async def test_reconnect_with_cursor_only_gets_the_remainder(
        redis: FakeStreamRedis,
) -> None:
    """带 Last-Event-ID 重连：只要那条**之后**的，不要重发已看过的。"""

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    for index in range(1, 6):
        redis.add(f"{index}-0", "message.delta")

    attachment = registry.attach(run_id, start="3-0")

    await pump()

    items = await asyncio.wait_for(collect(attachment), timeout=2)

    assert entries(items) == ["4-0", "5-0"]

    attachment.aclose()


async def test_start_zero_includes_the_very_first_entry(
        redis: FakeStreamRedis,
) -> None:
    """`start="0"` 要从**第一条**开始（XRANGE 的下界是包含的）。"""

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump()

    assert entries(
        await asyncio.wait_for(collect(attachment), timeout=2)
    ) == ["1-0"]

    attachment.aclose()


async def test_attach_before_key_exists_still_works(
        redis: FakeStreamRedis,
) -> None:
    """
    键还不存在时就挂上来（运行刚建、第一个事件还没发）—— 不能报错，
    而且要能等到第一个事件。
    """

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    attachment = registry.attach(run_id, start="0")

    await pump()

    redis.add("1-0", "run.started")

    await pump()

    assert entries(
        await asyncio.wait_for(collect(attachment), timeout=2)
    ) == ["1-0"]

    attachment.aclose()


# ============================================================
# 主张 3：什么时候关
# ============================================================


async def test_terminal_event_closes_every_subscriber(
        redis: FakeStreamRedis,
) -> None:
    """终止事件把**所有**订阅者一起收尾（不只是先看到的那个）。"""

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    a = registry.attach(run_id, start="0")
    b = registry.attach(run_id, start="0")

    await pump()

    redis.add("2-0", "run.completed")

    await pump()

    items_a = await asyncio.wait_for(collect(a), timeout=2)
    items_b = await asyncio.wait_for(collect(b), timeout=2)

    assert events(items_a) == ["run.started", "run.completed"]
    assert events(items_b) == ["run.started", "run.completed"]

    # 收尾原因读 `closed_reason`，**不是**从内容里找 —— 收尾信号被
    # `_iterate` 消费掉了（让 `async for` 正常结束），不会作为一条内容产出。
    assert a.closed_reason is not None
    assert "终止" in a.closed_reason
    assert b.closed_reason is not None

    a.aclose()
    b.aclose()


async def test_paused_closes_only_after_the_interrupt_was_delivered(
        redis: FakeStreamRedis,
) -> None:
    """
    ⚠️ 这条是"**所有** interrupt 到齐才关"的核心断言。

    光看见 DB 是 waiting_input 不够 —— 队列里那条 hitl.requested 还没投
    出去的话，关掉就等于把 interrupt_id 关在门外，客户端再也凑不齐 resume
    需要的参数。所以实现要求两个条件同时成立：状态是 waiting_input **且**
    本流已经投递过至少一条 hitl.requested。

    这里分两段验：先只有状态（不该关），再补上事件（该关）。
    """

    registry = RunStreamRegistry(
        redis,
        probe=FakeProbe(RunLiveness.PAUSED),
        config=make_config(),
    )

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump(200)

    # 探活已经在说 waiting_input 了，但一条 hitl.requested 都还没有 ——
    # 流必须还开着（否则客户端拿不到 interrupt_id）。
    assert events(await asyncio.wait_for(collect(attachment, limit=1), timeout=2))

    redis.add("2-0", "hitl.requested", interrupt_id="abc")

    await pump(200)

    items = await asyncio.wait_for(collect(attachment), timeout=2)

    assert events(items)[-1] == "hitl.requested"
    assert attachment.closed_reason is not None
    assert "paused" in attachment.closed_reason

    attachment.aclose()


async def test_executor_gone_closes_the_stream(
        redis: FakeStreamRedis,
) -> None:
    """执行器没了（崩溃/重启后的残留）—— 也要主动关，否则永远挂着。"""

    registry = RunStreamRegistry(
        redis,
        probe=FakeProbe(RunLiveness.EXECUTOR_GONE),
        config=make_config(),
    )

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump(200)

    items = await asyncio.wait_for(collect(attachment), timeout=2)

    # 事件照常收到（run.started 是补发来的），然后流自己关了。
    assert events(items) == ["run.started"]
    assert attachment.closed_reason is not None
    assert "executor_gone" in attachment.closed_reason

    attachment.aclose()


async def test_probe_failure_keeps_the_stream_open(
        redis: FakeStreamRedis,
) -> None:
    """
    探活报错 ≠ 判定它死了。

    一次数据库抖动不该把所有正在看的流一起杀掉 —— 那会把一个短暂故障
    放大成一片断线重连。所以探不出来就继续等。
    """

    probe = BoomProbe()

    registry = RunStreamRegistry(
        redis,
        probe=probe,
        config=make_config(),
    )

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump(200)

    # 探针确实被问过（不是压根没跑），但流没有被关掉。
    assert probe.calls > 0
    assert registry.active_count == 1

    attachment.aclose()


async def test_no_probe_means_never_closed_by_probing(
        redis: FakeStreamRedis,
) -> None:
    """没配探针时不能自己乱关 —— 只能靠终止事件收尾。"""

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump(200)

    assert registry.active_count == 1

    attachment.aclose()


# ============================================================
# 慢订阅者：只踢它自己
# ============================================================


async def test_slow_subscriber_is_dropped_without_affecting_others(
        redis: FakeStreamRedis,
) -> None:
    """
    一个不消费的订阅者把队列撑满之后，**只有它**被踢掉。

    共享循环不能被它堵住（那会拖住同一个 run 的所有订阅者），也不能丢
    最旧的事件（那会让正文静默缺洞）。所以策略是断开它一个。
    """

    # 小队列：单轮批大小跟着队列走（min(500, 容量)），所以调小不会误杀 ——
    # 见 run_stream.py 的 `_chunk_for`。
    registry = RunStreamRegistry(
        redis,
        config=make_config(subscriber_queue_size=6),
    )

    run_id = uuid4()

    redis.add("1-0", "run.started")

    slow = registry.attach(run_id, start="0")
    fast = registry.attach(run_id, start="0")

    # 快的那个**并发地**消费 —— 关键：慢的之所以慢，是因为没人读它的队列。
    # 要是两个都不读，两个都会被踢（那验不出"只踢它自己"）。
    fast_items: list[Any] = []

    async def consume_fast() -> None:
        async for item in fast:
            fast_items.append(item)

    reader = asyncio.create_task(consume_fast())

    # 一条一条地灌，中间让消费方跟上 —— 真实的事件流就是这样一批批来的。
    # 慢的那个攒着不读，攒到超过容量（6 格留 1 格 -> 容量 5）才被踢。
    deltas = 0

    for index in range(2, 14):
        redis.add(f"{index}-0", "message.delta")
        deltas += 1
        await pump(6)

    # 收尾得靠一个终止事件：没有它这条流本来就该一直开着，快的那个也就
    # 永远不会结束（那正是设计意图，不是 bug）。
    redis.add("99-0", "run.completed")

    await pump(30)

    await asyncio.wait_for(reader, timeout=3)

    # 快的那个照常拿到全部增量，并且正常收尾。
    fast_deltas = [name for name in events(fast_items) if name == "message.delta"]

    assert len(fast_deltas) == deltas
    assert "run.completed" in events(fast_items)
    assert fast.closed_reason is not None

    # 慢的那个拿到的是"你太慢了"的收尾 —— 只影响它自己。
    slow_items = await asyncio.wait_for(collect(slow), timeout=2)

    slow_deltas = [name for name in events(slow_items) if name == "message.delta"]

    assert "太慢" in (slow.closed_reason or "")
    assert len(slow_deltas) < deltas

    fast.aclose()
    slow.aclose()


# ============================================================
# 生命周期：拆除与上限
# ============================================================


async def test_stream_is_torn_down_when_the_last_subscriber_leaves(
        redis: FakeStreamRedis,
) -> None:
    """
    最后一个订阅者走了就拆掉这条流（取消循环、从注册表摘掉）。

    拆不掉就是本次要修的那个泄漏原样搬回来 —— 连接留着、任务留着。
    """

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    a = registry.attach(run_id, start="0")
    b = registry.attach(run_id, start="0")

    await pump()

    assert registry.active_count == 1

    a.aclose()

    await pump()

    # 还有一个人在，流不能拆。
    assert registry.active_count == 1

    b.aclose()

    await pump(200)

    assert registry.active_count == 0
    assert stream_tasks() == []


async def test_new_subscriber_after_teardown_gets_a_fresh_stream(
        redis: FakeStreamRedis,
) -> None:
    """
    拆掉之后又有人来订阅（比如审批完 resume、前端重新开流）——
    必须起一条**新的**流，而不是挂到那条已经停了的上面。

    挂上去的症状是：订阅者永远收不到东西（循环已经退了），而且不报错。
    """

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    first = registry.attach(run_id, start="0")

    await pump()

    first.aclose()

    await pump(200)

    assert registry.active_count == 0

    # resume 之后重新开流
    redis.add("2-0", "run.completed")

    second = registry.attach(run_id, start="1-0")

    await pump(200)

    items = await asyncio.wait_for(collect(second), timeout=2)

    assert entries(items) == ["2-0"]

    second.aclose()


async def test_capacity_limit_raises_an_explicit_error(
        redis: FakeStreamRedis,
) -> None:
    """
    同时在看的 run 数到上限 —— 给一个**说得清**的错误。

    这正是原来那个故障的根治处：以前是连接池到 redis-py 内部才炸出
    MaxConnectionsError（还容易被误当成网络抖动），现在上限是显式的、
    可配置的，而且错误文本告诉运维该去动哪个旋钮。
    """

    registry = RunStreamRegistry(
        redis,
        config=make_config(max_active_run_streams=2),
    )

    handles = [
        registry.attach(uuid4(), start="0")
        for _ in range(2)
    ]

    await pump()

    with pytest.raises(TooManyRunStreamsError) as caught:
        registry.attach(uuid4(), start="0")

    message = str(caught.value)

    assert "max_active_run_streams" in message
    assert "stream_max_connections" in message

    for handle in handles:
        handle.aclose()


async def test_a_run_at_capacity_does_not_count_against_itself(
        redis: FakeStreamRedis,
) -> None:
    """上限算的是 **run 数**，不是订阅者数 —— 同一个 run 多开几个不占额度。"""

    registry = RunStreamRegistry(
        redis,
        config=make_config(max_active_run_streams=2),
    )

    run_id = uuid4()

    handles = [
        registry.attach(run_id, start="0")
        for _ in range(5)
    ]

    await pump()

    assert registry.active_count == 1

    for handle in handles:
        handle.aclose()


# ============================================================
# 与 SSE 那一层的接缝
# ============================================================


async def test_attachment_yields_frames_ready_entries(
        redis: FakeStreamRedis,
) -> None:
    """
    附件产出的是**成帧前**的记录，`sse_frame` 能把它们直接编成帧。

    这一条把两侧钉在一起：改 run_stream 的产出形状而没改 sse，这里会红。
    """

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "message.delta", data='{"delta": "你"}')
    redis.add("2-0", "run.completed")

    attachment = registry.attach(run_id, start="0")

    await pump()

    items = await asyncio.wait_for(collect(attachment), timeout=2)

    frames = [
        sse_frame(item)
        for item in items
        if item is not None and not isinstance(item, (StreamClosed, StreamFailed))
    ]

    assert "id: 1-0" in frames[0]
    assert "event: message.delta" in frames[0]
    assert '"delta": "你"' in frames[0]
    assert "event: run.completed" in frames[1]

    attachment.aclose()


async def test_heartbeats_are_emitted_while_idle(redis: FakeStreamRedis) -> None:
    """空闲时产出心跳节拍（`None`），SSE 那一层据此发注释行并发现断线。"""

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump(200)

    items = await asyncio.wait_for(collect(attachment, limit=3), timeout=2)

    assert items[0].event_type == "run.started"
    assert items[1] is None
    assert HEARTBEAT.startswith(":")

    attachment.aclose()


# ============================================================
# 读失败：如实抛出，不做"卡住的空流"
# ============================================================


async def test_read_failure_is_fanned_out_as_an_error(
        redis: FakeStreamRedis,
) -> None:
    """
    Redis 读不动了要**抛给每一个订阅者**，而不是让他们在队列上永远等。

    与 reader.py 开头那条立场一致：卡住的空流比明确的错误难查得多。
    """

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump()

    from redis.exceptions import ConnectionError as RedisConnectionError

    redis.xread_error = RedisConnectionError("连接被掐了")
    redis.xread_errors_left = 99
    redis.entries.clear()

    with pytest.raises(RedisConnectionError):
        await asyncio.wait_for(collect(attachment), timeout=3)

    attachment.aclose()


async def test_pool_exhaustion_is_not_retried(
        redis: FakeStreamRedis,
) -> None:
    """
    ⚠️ MaxConnectionsError **不重试**（原故障的直接成因之一）。

    它继承自 ConnectionError，会被"网络抖动重试"那套接住 —— 但它压根
    不是网络问题：那是本地的池记账失败，在任何 I/O 之前就决定了，重试
    同样立刻失败。无退避地连试三次只是把 3 次失败烧成一个 500。

    这里验的是"第一次就抛出去"，而不是"试满 N 次再抛"。
    """

    from redis.exceptions import MaxConnectionsError

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump()

    redis.calls["xread"] = 0
    redis.xread_error = MaxConnectionsError("Too many connections")
    redis.xread_errors_left = 99
    redis.entries.clear()

    with pytest.raises(MaxConnectionsError):
        await asyncio.wait_for(collect(attachment), timeout=3)

    # 只试了一次 —— 没有把重试预算烧在网络异常的判断上。
    assert redis.calls["xread"] == 1

    attachment.aclose()


# ============================================================
# 停机
# ============================================================


async def test_aclose_stops_the_loop_cooperatively(
        redis: FakeStreamRedis,
) -> None:
    """
    停机走协作式 `_stop`，**不是** cancel。

    取消打在 XREAD BLOCK 中间会让那条连接上留下一个没读走的回复，然后它
    被还回池里 —— 下一个拿到它的 run 会读到上一条命令的回复。所以停机是
    "置个标志、等这一轮阻塞自己返回"。
    """

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump()

    assert stream_tasks()

    await asyncio.wait_for(registry.aclose(), timeout=3)

    await pump()

    assert stream_tasks() == []
    assert registry.active_count == 0

    attachment.aclose()


async def test_registry_forgets_a_stream_whose_loop_died(
        redis: FakeStreamRedis,
) -> None:
    """循环因为异常退出时，注册表里不能留一条死条目。"""

    from redis.exceptions import ConnectionError as RedisConnectionError

    registry = RunStreamRegistry(redis, config=make_config())

    run_id = uuid4()

    redis.add("1-0", "run.started")

    attachment = registry.attach(run_id, start="0")

    await pump()

    redis.xread_error = RedisConnectionError("崩了")
    redis.xread_errors_left = 99
    redis.entries.clear()

    with pytest.raises(RedisConnectionError):
        await asyncio.wait_for(collect(attachment), timeout=3)

    await pump(100)

    assert registry.active_count == 0
    assert stream_tasks() == []

    attachment.aclose()
