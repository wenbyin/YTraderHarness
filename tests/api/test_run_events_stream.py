"""
`GET /runs/{run_id}/events` 的 SSE 生成逻辑（`_frames`）。

分两层：

    前面的用例用**假附件**，不碰 Redis —— 验的是成帧、心跳、断线三条规则，
    以及收尾时一定要把订阅摘掉。这些任何时候都该是绿的。
    最后一个用例连**真 Redis**（本地 docker 那个），验"补发 + 实时跟随"
    真的能跑通。连不上就 skip，并说清跳过了什么 —— 绿色的输出不该让人
    以为它验过。

⚠️ 为什么直接测 `_frames` 而不是打 HTTP 接口
------------------------------------------
打接口要起整个应用（Postgres、数据源、工厂），而这里要验的东西
（帧格式、心跳、什么时候收尾）全在那个异步生成器里，与路由装配无关。

⚠️ 为什么是"假附件"而不是"假读取端"
----------------------------------
"这个 run 还有没有下文"的判断整套搬到了 app/agent/event/run_stream.py
（那里有自己的测试：tests/agent/event/test_run_stream.py）。这一层要验的
是**翻译**：一条记录 -> 一个 SSE 帧、一个节拍 -> 一行注释。所以替身只需要
满足 `_frames` 用到的那两个方法（`__aiter__` / `aclose`），不必是一条真的流。

    真正的端到端（补发边界、关流规则）在 run_stream 那边的测试里；这里
    再摹一遍只会得到一份会漂移的副本。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from app.agent.event import (
    HEARTBEAT,
    RedisStreamSink,
    build_event_publisher,
    build_run_stream_registry,
    close_event_publisher,
    load_stream_config,
    publish_run_event,
    reset_run_stream_registry,
    sse_frame,
    start_event_publisher,
    stream_key,
)
from app.api.v1.sessions import _frames
from app.common.event import RunEventType

# ============================================================
# 替身
# ============================================================


class FakeRequest:
    """
    只要 `is_disconnected()`。

    `disconnect_after`：前 N 次问都说"还在"，第 N+1 次起说"走了"。
    """

    def __init__(self, *, disconnect_after: int | None = None) -> None:

        self.headers: dict[str, str] = {}

        self._asked = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:

        self._asked += 1

        if self._disconnect_after is None:
            return False

        return self._asked > self._disconnect_after


class FakeEntry:
    """一条记录（只要 `id` / `event_type` / `fields` 三个属性）。"""

    def __init__(self, entry_id: str, event: str, **fields: str) -> None:

        self.id = entry_id
        self.fields = {"event": event, **fields}

    @property
    def event_type(self) -> str:
        return self.fields["event"]


class FakeAttachment:
    """
    `_frames` 眼里的附件：按脚本产出条目；`None` = 一次心跳节拍。

    产出完就**正常结束**（对应真实现的"流收尾"）。`aclose()` 要能被
    观察到 —— `_frames` 的 `finally` 里一定会调它，那是订阅被摘掉的
    唯一保证。
    """

    def __init__(self, items: list[Any]) -> None:

        self._items = list(items)

        self.closed = False

        self.iterations = 0

    def __aiter__(self):

        self.iterations += 1

        return self._generate()

    async def _generate(self):

        for item in self._items:
            yield item

    def aclose(self) -> None:
        self.closed = True


async def _collect(items: list[Any], **kwargs) -> tuple[list[str], FakeAttachment]:

    attachment = FakeAttachment(items)

    request = kwargs.pop("request", None) or FakeRequest()

    frames = [
        frame
        async for frame in _frames(attachment, request=request)
    ]

    return frames, attachment


def _event_names(frames: list[str]) -> list[str]:
    """从帧里捞出 event: 那一行（跳掉心跳）。"""

    names = []

    for frame in frames:

        if frame == HEARTBEAT:
            continue

        for line in frame.splitlines():

            if line.startswith("event: "):
                names.append(line[len("event: "):])

    return names


# ============================================================
# 成帧
# ============================================================


async def test_frame_carries_id_event_and_full_envelope() -> None:
    """一条记录 -> 一个带 id / event / data 的完整帧。"""

    frames, _ = await _collect([
        FakeEntry(
            "100-0",
            "message.delta",
            topic="agent_run",
            topic_id="run_1",
            data='{"delta": "你"}',
        ),
    ])

    assert len(frames) == 1

    frame = frames[0]

    assert frame.startswith("id: 100-0\n")
    assert "event: message.delta\n" in frame
    # data 段放的是**整个信封**，与直读 Redis 看到的一致。
    assert '"topic_id": "run_1"' in frame
    # 空行结尾 —— 少了它客户端会把两帧粘起来。
    assert frame.endswith("\n\n")


async def test_frame_matches_the_shared_encoder() -> None:
    """
    这里发出去的帧与 `sse_frame` 编出来的**逐字节相同**。

    钉住这一条是因为编码规则只有一份（app/agent/event/sse.py），这一层
    只该是它的调用方 —— 哪天有人在这儿手搓字符串，这条会红。
    """

    entry = FakeEntry("100-0", "message.delta", data='{"delta": "好"}')

    frames, _ = await _collect([entry])

    assert frames[0] == sse_frame(entry)


async def test_data_is_unescaped_in_the_frame() -> None:
    """`data` 是被**解开**的对象，不是两层 JSON 字符串。"""

    frames, _ = await _collect([
        FakeEntry(
            "100-0",
            "message.delta",
            data='{"delta": "你"}',
        ),
    ])

    body = json.loads(frames[0].split("data: ", 1)[1].split("\n")[0])

    assert body["data"] == {"delta": "你"}


# ============================================================
# 心跳 / 断线
# ============================================================


async def test_idle_tick_becomes_a_heartbeat() -> None:
    """附件产出 None（一个节拍）时发一个注释行。"""

    frames, _ = await _collect([
        None,
        FakeEntry("100-0", "node.started"),
    ])

    assert frames[0] == HEARTBEAT
    assert HEARTBEAT.startswith(":")


async def test_client_disconnect_stops_the_stream() -> None:
    """客户端走了就收尾（空转时才发现 —— 但最多滞后一个节拍）。"""

    frames, attachment = await _collect(
        [None, None, None, None, None],
        request=FakeRequest(disconnect_after=2),
    )

    # 前两次空转发心跳，第三次发现已断开 -> 退出。
    assert frames == [HEARTBEAT, HEARTBEAT]

    # 而且订阅被摘掉了 —— 否则连接会一直挂在广播端那边。
    assert attachment.closed


async def test_attachment_is_closed_even_when_the_stream_ends(
) -> None:
    """流正常结束时同样要摘订阅（`finally` 那条路）。"""

    _, attachment = await _collect([
        FakeEntry("100-0", "run.completed"),
    ])

    assert attachment.closed


async def test_attachment_is_closed_when_the_client_returns_early(
) -> None:
    """
    生成器中途 `return`（这里就是客户端断开那条路）时，`finally` 也要跑。

    这一条是本次改造的**防泄漏**断言：订阅摘不掉，这个 run 的循环 task 与
    它脚下那条连接就永远不释放 —— 正是原来那个"已完成的 run 被重连后永久
    挂着"的另一半成因。
    """

    _, attachment = await _collect(
        [None, None, None],
        request=FakeRequest(disconnect_after=1),
    )

    assert attachment.closed


async def test_attachment_is_iterated_only_once() -> None:
    """
    `_frames` 只迭代附件一次。

    迭代两次会得到**两个**消费者读同一条队列，各拿一半事件，而且都不报错
    —— 症状是"事件偶尔少几条"。真实现的 `RunStreamAttachment.__aiter__`
    因此把迭代器缓存起来，这一条钉住"上层也别去迭代两遍"。
    """

    _, attachment = await _collect([FakeEntry("100-0", "run.started")])

    assert attachment.iterations == 1


# ============================================================
# 真 Redis：补发 + 实时跟随
# ============================================================


async def _real_source():
    """
    按 app/config/datasource/redis.yaml 建一个真数据源。连不上返回 None。

    走**真实的配置与数据源类**，而不是自己拼一个 localhost URL ——
    这一层要验的正是"配置写对了没有"。
    """

    from app.common.utils import expand_env
    from app.infra.datasource.models import RedisDataSourceDefinition
    from app.infra.datasource.redis import RedisDataSource

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


async def test_live_follow_against_real_redis() -> None:
    """
    订阅一条**正在写**的流：事件在写入之后才出现在帧里。

    刻意让消费者先连上、再发事件 —— 这样走的是"实时跟随"那条路径，
    而不是"一次性补发"。两条路径都会用到（晚连的客户端走补发），
    但只有这一条会暴露"阻塞读取写错了"的问题。
    """

    source = await _real_source()

    if source is None:
        pytest.skip(
            "连不上 app/config/datasource/redis.yaml 里的 Redis。"
            "起一个：docker start ytrader-redis"
        )

    key = None

    try:
        # 发布端走**同步**客户端（写入在写线程里），读取端走长连接池那个
        # （stream_client）—— 这正是组合根里的接法，见 app/main.py。
        build_event_publisher(RedisStreamSink(source.sync_client))

        registry = build_run_stream_registry(source.stream_client)

        assert registry is not None

        await start_event_publisher()

        run_id = uuid4()
        topic_id = str(run_id)
        key = stream_key(load_stream_config().key_prefix, topic_id)

        await source.stream_client.delete(key)

        frames: list[str] = []

        async def consume() -> None:

            attachment = registry.attach(run_id, start="0")

            try:
                async for frame in _frames(attachment, request=FakeRequest()):
                    frames.append(frame)
            finally:
                attachment.aclose()

        task = asyncio.create_task(consume())

        # 让消费者先挂到 XREAD 上。key 此刻**不存在**，因此这一等就是
        # 在验"阻塞读会在 key 被创建时醒来"。
        await asyncio.sleep(0.3)

        await publish_run_event(RunEventType.RUN_STARTED, topic_id)
        await publish_run_event(
            RunEventType.NODE_STARTED, topic_id, node="agent"
        )
        await publish_run_event(
            RunEventType.MESSAGE_DELTA, topic_id, node="agent",
            data={"delta": "你"},
        )
        await publish_run_event(
            RunEventType.MESSAGE_DELTA, topic_id, node="agent",
            data={"delta": "好"},
        )
        await publish_run_event(RunEventType.RUN_COMPLETED, topic_id)

        # 收到 run.completed 后广播端自己收尾；给足余量再断言超时。
        await asyncio.wait_for(task, timeout=15)

        assert _event_names(frames) == [
            "run.started",
            "node.started",
            "message.delta",
            "message.delta",
            "run.completed",
        ]

        # `data` 被解开成**对象**，不是两层 JSON 字符串。
        bodies = [
            json.loads(frame.split("data: ", 1)[1].split("\n")[0])
            for frame in frames
            if '"delta"' in frame
        ]

        assert [body["data"] for body in bodies] == [
            {"delta": "你"},
            {"delta": "好"},
        ]

    finally:
        # 清理放在 finally 里：**断言失败时也要清**。
        #
        # 这不是洁癖 —— 放在断言之后的话，一次失败的运行就会把垃圾留在
        # Redis 里（虽然有 24h TTL，但"测试挂了"不该顺带污染环境，
        # 而且排查时那条残留的流很容易被当成真实数据）。
        if key is not None:
            try:
                await source.stream_client.delete(key)
            except Exception:       # noqa: BLE001 - 清理失败不该掩盖原异常
                pass

        # ⚠️ 顺序有要求：先停广播端（它脚下的连接归 stream_client 池），
        # 再停写线程，最后关数据源。
        registry = reset_run_stream_registry()

        if registry is not None:
            await registry.aclose()

        await close_event_publisher()
        await source.close()
