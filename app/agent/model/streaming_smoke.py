"""
模型层流式的冒烟测试。

不需要真 provider、不需要密钥：注入一个假的 chat 客户端，它的
`astream()` 按脚本吐**真正的** `AIMessageChunk`。因此这里验的是真的
聚合与筛选逻辑（`_astream` / `_chunk_text` / AIMessageChunk 相加），
只有网络那一层是假的。

五个要害：

    正文增量按**产出顺序**回调 —— 错序在内容上根本看不出来
    tool_call_chunks 必须拼成真的 `tool_calls`（route_after_agent 靠它）
    思维链**不能**当正文发出去
    不能流式时退化成"一次性回调整段正文"，而不是静默不回调
    中途失败时**已经发出去的增量不重复**（那是 commit 规则的前提）

跑法：
    python -m app.agent.model.streaming_smoke
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agent.model.adapters import LangChainProviderAdapter, _chunk_text
from app.agent.model.errors import ModelError
from app.agent.model.provider import ModelClient
from app.agent.model.types import (
    ModelCapabilities,
    ModelDefinition,
    ModelRequest,
)

# ============================================================
# 断言工具
# ============================================================


def check(label, condition):
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title):
    print(f"\n{title}")


async def _async_raises(exception, coro):
    """await coro 抛出了指定异常吗。"""

    try:
        await coro
    except exception:
        return True
    except Exception:
        return False
    return False


# ============================================================
# 假的 chat 客户端
# ============================================================


class FakeRunnable:
    """按脚本吐 chunk 的假 runnable。也记下走过哪条路径。"""

    def __init__(
            self,
            chunks: list[AIMessageChunk],
            *,
            error_after: int | None = None,
    ) -> None:

        self.chunks = list(chunks)
        self.error_after = error_after

        #: astream 真的被拉了几块（0 = 根本没走流式）。
        self.streamed = 0
        #: ainvoke 被调了几次（非流式路径）。
        self.invoked = 0

    async def astream(self, messages, config=None):

        for index, chunk in enumerate(self.chunks, start=1):

            if self.error_after == index:
                raise RuntimeError("provider 中途挂了")

            self.streamed += 1

            yield chunk

    async def ainvoke(self, messages, config=None):

        self.invoked += 1

        text = "".join(
            _chunk_text(chunk) for chunk in self.chunks
        )

        return AIMessage(content=text)


def make_client(
        runnable: FakeRunnable,
        *,
        streaming: bool = True,
) -> ModelClient:
    """装一个 ModelClient，直接拿假 runnable 当 provider 客户端。"""

    return ModelClient(
        definition=ModelDefinition(
            id="fake:fake-model",
            provider="fake",
            model="fake-model",
            capabilities=ModelCapabilities(streaming=streaming),
            config={},
        ),
        client=runnable,
        adapter=LangChainProviderAdapter(),
    )


def make_request(**kwargs: Any) -> ModelRequest:

    return ModelRequest(
        messages=(HumanMessage(content="你好"),),
        **kwargs,
    )


class Collector:
    """收增量，同时记下顺序。"""

    def __init__(self) -> None:

        self.deltas: list[str] = []

    async def __call__(self, text: str) -> None:

        self.deltas.append(text)


# ============================================================


async def main() -> int:

    adapter = LangChainProviderAdapter()

    # --------------------------------------------------------

    section("1. 正文增量按产出顺序回调")

    runnable = FakeRunnable([
        AIMessageChunk(content="Hel"),
        AIMessageChunk(content="lo"),
        AIMessageChunk(content=" 世界"),
    ])

    collector = Collector()

    result = await adapter.ainvoke(
        make_client(runnable),
        make_request(),
        timeout=5.0,
        on_delta=collector,
    )

    check(
        "三块按顺序回调（不是拼好了一次性给）",
        collector.deltas == ["Hel", "lo", " 世界"],
    )
    check(
        "返回的是真正的 AIMessage，不是 chunk",
        isinstance(result.message, AIMessage)
        and not isinstance(result.message, AIMessageChunk),
    )
    check(
        "正文拼对了（含中间的空格）",
        result.message.content == "Hello 世界",
    )
    check("确实走了 provider 的流", runnable.streamed == 3)

    # --------------------------------------------------------

    section("2. tool_call_chunks 必须拼成真的 tool_calls")

    runnable_tools = FakeRunnable([
        AIMessageChunk(
            content="",
            tool_call_chunks=[{
                "name": "get_weather",
                "args": '{"city"',
                "id": "call_1",
                "index": 0,
                "type": "tool_call_chunk",
            }],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{
                "name": None,
                "args": ': "SF"}',
                "id": None,
                "index": 0,
                "type": "tool_call_chunk",
            }],
        ),
    ])

    tool_collector = Collector()

    tool_result = await adapter.ainvoke(
        make_client(runnable_tools),
        make_request(),
        timeout=5.0,
        on_delta=tool_collector,
    )

    calls = tool_result.message.tool_calls

    check(
        "参数碎片拼成了完整的 args（route_after_agent 就看这个）",
        len(calls) == 1
        and calls[0]["name"] == "get_weather"
        and calls[0]["args"] == {"city": "SF"},
    )
    check(
        "tool-call 碎片**不是**正文增量（一个字都不该发）",
        tool_collector.deltas == [],
    )

    # --------------------------------------------------------

    section("3. 思维链不能当正文发出去")

    runnable_reasoning = FakeRunnable([
        AIMessageChunk(content=[
            {"type": "reasoning", "reasoning": "让我想想…"},
        ]),
        AIMessageChunk(content=[
            {"type": "text", "text": "答案是 42"},
        ]),
    ])

    reasoning_collector = Collector()

    reasoning_result = await adapter.ainvoke(
        make_client(runnable_reasoning),
        make_request(),
        timeout=5.0,
        on_delta=reasoning_collector,
    )

    check(
        "只发了 text 块，reasoning 块被挡掉",
        reasoning_collector.deltas == ["答案是 42"],
    )
    check(
        "聚合后的消息里 text 也只剩正文",
        reasoning_result.message.text == "答案是 42",
    )

    # --------------------------------------------------------

    section("4. 不能流式时退化成一次性回调整段正文")

    runnable_plain = FakeRunnable([
        AIMessageChunk(content="Hello 世界"),
    ])

    plain_collector = Collector()

    await adapter.ainvoke(
        make_client(runnable_plain, streaming=False),
        make_request(),
        timeout=5.0,
        on_delta=plain_collector,
    )

    check(
        "模型没声明 streaming -> 走 ainvoke",
        runnable_plain.streamed == 0 and runnable_plain.invoked == 1,
    )
    check(
        "正文仍然**一次性**给到（不是静默不回调）",
        plain_collector.deltas == ["Hello 世界"],
    )

    runnable_sf = FakeRunnable([AIMessageChunk(content="{}")])

    sf_collector = Collector()

    await adapter.ainvoke(
        make_client(runnable_sf),
        make_request(response_format={"type": "json_object"}),
        timeout=5.0,
        on_delta=sf_collector,
    )

    check(
        "结构化输出一律退回非流式（各家流式实现不一致）",
        runnable_sf.streamed == 0 and runnable_sf.invoked == 1,
    )
    check(
        "结构化输出也仍然给到正文",
        sf_collector.deltas == ["{}"],
    )

    # --------------------------------------------------------

    section("5. 中途失败：已发的增量不重复")

    runnable_fail = FakeRunnable(
        [
            AIMessageChunk(content="前半段"),
            AIMessageChunk(content="后半段"),
        ],
        error_after=2,
    )

    fail_collector = Collector()

    raised = await _async_raises(
        ModelError,
        adapter.ainvoke(
            make_client(runnable_fail),
            make_request(),
            timeout=5.0,
            on_delta=fail_collector,
        ),
    )

    check("抛的是 Model 层异常（provider 异常不泄漏）", raised)
    check(
        "已经发出去的那一段**恰好一次**（commit 规则的前提）",
        fail_collector.deltas == ["前半段"],
    )

    # --------------------------------------------------------

    section("6. 流里一块都没有 -> 抛，而不是返回空消息")

    empty_collector = Collector()

    empty_raised = await _async_raises(
        ModelError,
        adapter.ainvoke(
            make_client(FakeRunnable([])),
            make_request(),
            timeout=5.0,
            on_delta=empty_collector,
        ),
    )

    check(
        "空流抛错（空消息会以 completed 收尾，最难查）",
        empty_raised,
    )
    check("一块都没发出去", empty_collector.deltas == [])

    # --------------------------------------------------------

    section("7. provider 元信息不会被拼接（langchain 的坑）")

    # langchain 的 AIMessageChunk.__add__ 对 response_metadata 里的**字符串
    # 是拼接**的，于是三块同名的 model_name 会变成
    # 'deepseek-v4deepseek-v4deepseek-v4' —— 而且不报错，坏掉的是观测里
    # 那个字段，事后完全看不出是拼出来的。
    runnable_meta = FakeRunnable([
        AIMessageChunk(
            content="a",
            response_metadata={"model_name": "deepseek-v4", "finish_reason": None},
        ),
        AIMessageChunk(
            content="b",
            response_metadata={"model_name": "deepseek-v4", "finish_reason": None},
        ),
        AIMessageChunk(
            content="c",
            response_metadata={"model_name": "deepseek-v4", "finish_reason": "stop"},
        ),
    ])

    meta_result = await adapter.ainvoke(
        make_client(runnable_meta),
        make_request(),
        timeout=5.0,
        on_delta=Collector(),
    )

    meta = meta_result.message.response_metadata

    check(
        "model_name 没有被拼成 'deepseek-v4deepseek-v4deepseek-v4'",
        meta.get("model_name") == "deepseek-v4",
    )
    check(
        "finish_reason 取的是最后那个非空值",
        meta.get("finish_reason") == "stop",
    )

    runnable_every = FakeRunnable([
        AIMessageChunk(content="a", response_metadata={"finish_reason": "stop"}),
        AIMessageChunk(content="b", response_metadata={"finish_reason": "stop"}),
    ])

    every_result = await adapter.ainvoke(
        make_client(runnable_every),
        make_request(),
        timeout=5.0,
        on_delta=Collector(),
    )

    check(
        "逐块都报 finish_reason 也不会变成 'stopstop'",
        every_result.message.response_metadata.get("finish_reason") == "stop",
    )

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
