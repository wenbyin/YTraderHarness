"""
流式调用的 **commit 规则**（规格 §9 的推论）。

一句话：

    只要有一个正文增量**发出去过**，这次调用就不再重试、不再降级。

理由不是"重试没用"，而是重试会**把正文发两遍**：消费端（浏览器 /
SSE / Redis Stream）已经渲染了半截，而我们没有任何撤回原语 ——
Redis Stream 没有撤回命令，已经画到界面上的字也擦不掉。于是第二次
尝试产出的内容会**接在**第一份残缺正文后面。

对照组同样重要：**还没吐出任何东西**的失败（连接失败、429、首字节
之前超时）必须照常重试与降级 —— 那才是瞬时故障的大多数。commit 规则
只在"已经有内容出去"时生效，不该顺手把重试整条掐掉。
"""

from __future__ import annotations

import pytest

from fakes import FakeAdapter, Outcome, make_definition, make_registry

from app.agent.model import (
    ModelProfile,
    ModelRateLimitError,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
    ModelRuntimePolicy,
)


def _runtime(adapter, *, fallbacks=(), **policy_kwargs):
    policy_kwargs.setdefault("timeout", 1.0)
    policy_kwargs.setdefault("max_retries", 2)
    policy_kwargs.setdefault("retry_backoff", 0.0)

    registry = make_registry(
        models=[
            make_definition("primary"),
            make_definition("backup"),
        ],
        profiles=[
            ModelProfile(
                id="p",
                primary="fake:primary",
                fallbacks=tuple(fallbacks),
                policy=ModelRuntimePolicy(**policy_kwargs),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.REASONING,
                profile="p",
            )
        ],
        adapter=adapter,
    )

    return registry.get_runtime(
        agent_type="react", model_role=ModelRole.REASONING
    )


def _request() -> ModelRequest:
    return ModelRequest(messages=("你好",))


class Collector:
    """收增量。顺序即断言对象。"""

    def __init__(self) -> None:

        self.deltas: list[str] = []

    async def __call__(self, text: str) -> None:

        self.deltas.append(text)


# ============================================================
# 顺路先确认增量确实是通的
# ============================================================


async def test_deltas_arrive_in_order() -> None:
    """一次成功的流式调用：增量按顺序到，返回值仍是拼好的消息。"""

    adapter = FakeAdapter(
        default=Outcome(message="ok", deltas=("你", "好", "呀"))
    )

    runtime = _runtime(adapter)
    collector = Collector()

    response = await runtime.ainvoke(_request(), on_delta=collector)

    assert collector.deltas == ["你", "好", "呀"]
    assert response.message == "ok"
    assert response.attempts == 1


# ============================================================
# commit 规则
# ============================================================


async def test_mid_stream_failure_is_terminal() -> None:
    """吐了增量之后再失败 —— 不再重试，即使错误本身可重试。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(
                    error=ModelRateLimitError("限流"),
                    deltas=("半截",),
                    raise_after=1,
                ),
                Outcome(message="本来能成功的第二次"),
            ]
        }
    )

    runtime = _runtime(adapter)
    collector = Collector()

    with pytest.raises(ModelRateLimitError):
        await runtime.ainvoke(_request(), on_delta=collector)

    assert adapter.call_count() == 1, "已经发出增量之后不该再试"
    assert collector.deltas == ["半截"], "发出去的那一段恰好一次"


async def test_mid_stream_failure_does_not_fall_back() -> None:
    """吐了增量之后再失败 —— 也不换候选：换了就是第二份正文。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(
                    error=ModelRateLimitError("限流"),
                    deltas=("半截",),
                    raise_after=1,
                ),
            ],
            "fake:backup": [Outcome(message="备胎", deltas=("备",))],
        }
    )

    runtime = _runtime(adapter, fallbacks=("fake:backup",))
    collector = Collector()

    with pytest.raises(ModelRateLimitError):
        await runtime.ainvoke(_request(), on_delta=collector)

    assert adapter.called_models() == ["fake:primary"], "不该碰备胎"
    assert collector.deltas == ["半截"]


# ============================================================
# 对照组：还没吐出东西时，重试与降级必须照旧
# ============================================================


async def test_failure_before_first_delta_still_retries() -> None:
    """一个字都还没吐出去 —— 重试照旧。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelRateLimitError("限流")),
                Outcome(message="第二次成功", deltas=("好",)),
            ]
        }
    )

    runtime = _runtime(adapter)
    collector = Collector()

    response = await runtime.ainvoke(_request(), on_delta=collector)

    assert response.attempts == 2
    assert adapter.call_count() == 2
    assert collector.deltas == ["好"]


async def test_failure_before_first_delta_still_falls_back() -> None:
    """一个字都还没吐出去 —— 降级照旧。"""

    # primary 要试满 max_retries + 1 = 3 次才会让位（见
    # test_runtime_fallback.py 的 _primary_unavailable_falls_back）。
    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelRateLimitError("限流")),
                Outcome(error=ModelRateLimitError("限流")),
                Outcome(error=ModelRateLimitError("限流")),
            ],
            "fake:backup": [Outcome(message="备胎", deltas=("备",))],
        }
    )

    runtime = _runtime(adapter, fallbacks=("fake:backup",))
    collector = Collector()

    response = await runtime.ainvoke(_request(), on_delta=collector)

    assert response.message == "备胎"
    assert adapter.called_models() == [
        "fake:primary",
        "fake:primary",
        "fake:primary",
        "fake:backup",
    ]
    assert collector.deltas == ["备"]


async def test_no_on_delta_keeps_retry_untouched() -> None:
    """不给 on_delta（没人在听）时，行为与没有流式那版完全一致。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(
                    error=ModelRateLimitError("限流"),
                    deltas=("不该发出去",),
                ),
                Outcome(message="第二次成功"),
            ]
        }
    )

    runtime = _runtime(adapter)

    response = await runtime.ainvoke(_request())

    assert response.attempts == 2
    assert adapter.deltas == [], "没有回调就不该有任何增量出去"
