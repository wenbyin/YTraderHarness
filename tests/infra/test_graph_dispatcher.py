"""
图调度器的两个闸门：**在等**的（queue_size）与**在跑**的（max_concurrency）。

为什么要有这些用例
------------------
这两个数字管的是两件不同的事，混起来看代码是看不出来的：

    队列满       -> dispatch 返回 accepted=False，**明确拒绝**
    并发位满     -> 信封留在队列里**等**，调用方那边只是慢

一旦有人把"并发位满"也做成拒绝，批量提交就会莫名其妙地报"满了" ——
而那本该是"排队"。反过来，把队列做成无界，内存会被无限堆积吃光。

用例把它们焊住。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.infra.graph.dispatcher import DispatchEnvelope
from app.infra.graph.dispatcher_impl import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_QUEUE_SIZE,
    InProcessGraphDispatcher,
)

GRAPH = "test-graph"


def _envelope() -> DispatchEnvelope:
    run_id = uuid4()

    return DispatchEnvelope(
        graph_name=GRAPH,
        pipeline_run_id=run_id,
        thread_id=f"test:{run_id}",
    )


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    """
    等到条件成立。轮询而不是事件 —— 要断言的是"并发峰值"这种
    只能事后统计的量，没有单个事件可等。
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("等待超时：条件始终没有成立。")

        await asyncio.sleep(0.01)


# ============================================================
# 默认值
# ============================================================


def test_defaults_match_the_agreed_limits() -> None:
    """10 个并行、256 个排队 —— 这两个数字是使用方定的，别被悄悄改小。"""

    assert DEFAULT_MAX_CONCURRENCY == 10
    assert DEFAULT_QUEUE_SIZE == 256


def test_zero_concurrency_is_rejected() -> None:
    """
    0 是个隐蔽值：队列照收（dispatch 照样 accepted），但一个都不跑。

    那种状态下没有任何报错，只表现为"点了处理但没反应" —— 与其让
    使用方去查，不如在构造时就炸。
    """

    with pytest.raises(ValueError):
        InProcessGraphDispatcher(max_concurrency=0)


# ============================================================
# 在跑的：并发上限
# ============================================================


async def test_concurrency_is_capped_but_actually_parallel() -> None:
    """
    峰值**恰好**等于上限。

    两个都要断言：只断言 `<= 上限` 的话，退回成串行执行也会过 ——
    而那正是这次改动要修的东西。
    """

    dispatcher = InProcessGraphDispatcher(
        queue_size=64, max_concurrency=3,
    )

    live = 0
    peak = 0
    finished: list = []

    async def handler(envelope: DispatchEnvelope) -> None:
        nonlocal live, peak

        live += 1
        peak = max(peak, live)

        await asyncio.sleep(0.05)

        live -= 1
        finished.append(envelope.pipeline_run_id)

    dispatcher.register(GRAPH, handler)
    await dispatcher.start()

    handles = [
        await dispatcher.dispatch(_envelope())
        for _ in range(12)
    ]

    assert all(handle.accepted for handle in handles)
    assert dispatcher.queue_size + dispatcher.running_count == 12

    await _wait_until(lambda: len(finished) == 12)

    assert peak == 3, (
        f"并发峰值应为 3，实际 {peak} —— "
        "小于 3 说明退回串行了，大于 3 说明信号量没起作用。"
    )

    await dispatcher.aclose()


# ============================================================
# 在等的：队列满才拒绝
# ============================================================


async def test_queue_full_is_the_only_reason_to_reject() -> None:
    """
    队列满 -> 拒绝，且消息说得出原因。

    刻意**不 start()**：没有 worker 来消费，队列深度就等于提交数，
    这样断言不依赖时序。
    """

    dispatcher = InProcessGraphDispatcher(
        queue_size=2, max_concurrency=1,
    )

    async def handler(envelope: DispatchEnvelope) -> None:  # pragma: no cover
        raise AssertionError("没有 worker，不该有人跑到这里。")

    dispatcher.register(GRAPH, handler)

    first = await dispatcher.dispatch(_envelope())
    second = await dispatcher.dispatch(_envelope())
    third = await dispatcher.dispatch(_envelope())

    assert first.accepted and second.accepted
    assert not third.accepted
    assert "full" in third.message


async def test_full_concurrency_queues_instead_of_rejecting() -> None:
    """
    并发位被占满时，新来的**排队**，不是被拒。

    这是本次改动的核心：批量提交应该在队列里等，只有把队列也塞满
    （256 条）才拒绝。
    """

    dispatcher = InProcessGraphDispatcher(
        queue_size=2, max_concurrency=1,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(envelope: DispatchEnvelope) -> None:
        started.set()
        await release.wait()

    dispatcher.register(GRAPH, handler)
    await dispatcher.start()

    running = await dispatcher.dispatch(_envelope())

    # 等第一条真正开工 —— 这之后它占着唯一的并发位。
    await asyncio.wait_for(started.wait(), timeout=5)

    queued_1 = await dispatcher.dispatch(_envelope())
    queued_2 = await dispatcher.dispatch(_envelope())
    overflow = await dispatcher.dispatch(_envelope())

    assert running.accepted
    assert queued_1.accepted and queued_2.accepted, (
        "并发位满了不该拒绝 —— 那是排队，队列还有空间。"
    )
    assert not overflow.accepted, "队列（2 格）也满了才该拒绝。"

    assert dispatcher.running_count == 1
    assert dispatcher.queue_size == 2

    release.set()
    await dispatcher.aclose()


async def test_queued_runs_are_reported_active() -> None:
    """
    排队中的 run 在 `is_active` 里算"活着"。

    否则"提交完立刻打开 SSE"会被探针判成"执行器没了"，把刚建好的流
    关掉 —— 排队越深这个窗口越大，并发放开的正是排队深度。
    """

    dispatcher = InProcessGraphDispatcher(
        queue_size=4, max_concurrency=1,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(envelope: DispatchEnvelope) -> None:
        started.set()
        await release.wait()

    dispatcher.register(GRAPH, handler)
    await dispatcher.start()

    await dispatcher.dispatch(_envelope())
    await asyncio.wait_for(started.wait(), timeout=5)

    queued = _envelope()
    await dispatcher.dispatch(queued)

    assert await dispatcher.is_active(queued.pipeline_run_id)

    release.set()
    await dispatcher.aclose()

    assert not await dispatcher.is_active(queued.pipeline_run_id)


# ============================================================
# 收尾
# ============================================================


async def test_aclose_waits_for_every_running_run() -> None:
    """
    关闭时等在跑的收尾，**且全部**等完。

    并发之后这件事要显式做（见 `_drain`）：worker 退出时手里可能还有
    max_concurrency 条在跑，只 await worker 会漏掉它们。
    """

    dispatcher = InProcessGraphDispatcher(
        queue_size=8, max_concurrency=3,
    )

    finished: list = []

    async def handler(envelope: DispatchEnvelope) -> None:
        await asyncio.sleep(0.1)
        finished.append(envelope.pipeline_run_id)

    dispatcher.register(GRAPH, handler)
    await dispatcher.start()

    for _ in range(3):
        await dispatcher.dispatch(_envelope())

    # 等三条都开工（占满并发位），再关。
    await _wait_until(lambda: dispatcher.running_count == 3)

    await dispatcher.aclose()

    assert len(finished) == 3, (
        "aclose 只等了 worker，没有等它起出去的那几条在跑的任务。"
    )
    assert dispatcher.running_count == 0


async def test_failure_of_one_run_does_not_stop_the_others() -> None:
    """
    单个 run 抛异常不能带走 worker —— 否则后续所有任务静默地不再执行，
    而队列看起来还是好的。
    """

    dispatcher = InProcessGraphDispatcher(
        queue_size=8, max_concurrency=2,
    )

    attempts = 0
    finished: list = []
    attempted: list = []

    async def handler(envelope: DispatchEnvelope) -> None:
        nonlocal attempts

        attempted.append(envelope.pipeline_run_id)

        # 按时序炸第一条，而不是按"还没有成功的" —— 后者在并发 2 下
        # 会把前两条一起炸掉（它们看到的是同一个 len(finished) == 0）。
        attempts += 1

        if attempts == 1:
            raise RuntimeError("这一次运行就是要炸")

        finished.append(envelope.pipeline_run_id)

    dispatcher.register(GRAPH, handler)
    await dispatcher.start()

    for _ in range(2):
        await dispatcher.dispatch(_envelope())

    # 等**两条都跑过**，而不是等 running_count == 0 —— 后者在 worker
    # 还没开工时就是真命题（此时两条都还没被取走），会立刻返回。
    await _wait_until(lambda: len(attempted) == 2)
    await _wait_until(lambda: dispatcher.running_count == 0)

    assert len(finished) == 1, "炸掉的那条连带把另一条也停了。"

    # 并发位必须被还回来：没有的话上面那条失败的会永久占掉一格（共 2 格）。
    assert dispatcher.running_count == 0

    for _ in range(4):
        await dispatcher.dispatch(_envelope())

    await _wait_until(lambda: len(finished) == 5)

    await dispatcher.aclose()
