"""
链驱动的冒烟测试。

不需要数据库、不需要模型 —— 链的输入是几行鸭子类型的记录加一个
"键 -> 实现"的映射，两样都能在内存里造。**链的语义是这块最容易
写错的地方**（AROUND 的两条记录、BEFORE 的短路、AFTER 的回程、
就地改参数），所以它单独有一个冒烟脚本，不混在图那一级里。

跑法：
    python -m app.infra.agent.chain_smoke
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, ClassVar

from app.agent.tools.processor.base import (
    ToolInterceptor,
    ToolObserver,
)
from app.agent.tools.processor.context import (
    AsyncToolCall,
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)
from app.infra.agent.chain import build_chain, compile_chain
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

__all__ = ["main"]


def check(label, condition):
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title):
    print(f"\n{title}")


def _raises(exception, fn) -> bool:
    try:
        fn()
    except exception:
        return True
    except Exception:
        return False
    return False


# ============================================================
# 测试替身
# ============================================================


@dataclass
class Row:
    """
    一行挂载记录的替身。

    用鸭子类型：`build_chain` 只要求这几个属性（见它的 docstring），
    因此这里不必把服务层的实体拖进来 —— 也就能用
    "AROUND + ASYNC"这种**数据库 CHECK 挡住、但语义上要兜底**的
    组合来验校验（真实实体那侧根本构造不出来）。
    """

    processor_key: str
    hook: ProcessorHook
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    config: dict[str, Any] = field(default_factory=dict)
    sequence: int = 1
    enabled: bool = True


class TraceMixin:
    """把每次被调到记进一个共享的列表，用于断言调用顺序。"""

    def __init__(self, tag: str, log: list[str], **_: Any) -> None:
        self.tag = tag
        self.log = log


class TraceAround(ToolInterceptor, TraceMixin):
    name = "smoke_around"
    supported_hooks = frozenset({ProcessorHook.AROUND})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})

    def __init__(self, tag, log, *, times: int = 1) -> None:
        super().__init__(tag, log)
        #: next_call 调几次。2 = 重试一次。
        self.times = times

    async def around(
            self, invocation, next_call, *, config, runtime=None,
    ) -> ToolOutcome:

        self.log.append(f"{self.tag}:in")
        outcome = ToolOutcome(value=None)

        for _ in range(self.times):
            outcome = await next_call()

        self.log.append(f"{self.tag}:out")

        return outcome


class TraceBefore(ToolInterceptor, TraceMixin):
    name = "smoke_before"
    supported_hooks = frozenset({ProcessorHook.BEFORE})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})

    def __init__(
            self, tag, log, *, deny: str | None = None,
            set_arg: tuple[str, Any] | None = None,
    ) -> None:
        super().__init__(tag, log)
        self.deny = deny
        self.set_arg = set_arg

    async def before(self, invocation, *, config, runtime=None):

        self.log.append(f"{self.tag}:before")

        if self.set_arg is not None:
            key, value = self.set_arg
            # 就地改 —— 契约是这么说的（见 context.py）。
            invocation.arguments[key] = value

        if self.deny is not None:
            # 短路：这次调用到此为止，工具不会被调用。
            return ToolOutcome(error=ToolCallDenied(self.deny))

        return invocation


class TraceAfter(ToolObserver, TraceMixin):
    name = "smoke_after"
    supported_hooks = frozenset({ProcessorHook.AFTER})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})

    async def after(self, invocation, outcome, *, config, runtime=None):
        self.log.append(f"{self.tag}:after")
        return outcome


class TraceObserverAsync(ToolObserver, TraceMixin):
    name = "smoke_observer_async"
    supported_hooks = frozenset({ProcessorHook.BEFORE})
    supported_execution_modes = frozenset({ExecutionMode.ASYNC})


def make_base(log: list[str]) -> AsyncToolCall:
    """最里层：假装它是真正的工具调用。"""

    async def base() -> ToolOutcome:
        log.append("base")
        return ToolOutcome(value={"ok": True}, duration_ms=1.5)

    return base


def resolve_map(mapping) -> Any:
    def resolve(key: str):
        return mapping.get(key)
    return resolve


# ============================================================
# 用例
# ============================================================


def test_nesting() -> None:
    section("嵌套顺序：AROUND 包住它之后的全部内容")

    log: list[str] = []

    chain = build_chain(
        [
            Row("a_around", ProcessorHook.AROUND, sequence=1),
            Row("b_before", ProcessorHook.BEFORE, sequence=2),
            Row("c_after", ProcessorHook.AFTER, sequence=3),
            Row("a_around", ProcessorHook.AROUND, sequence=4),
        ],
        resolve_processor=resolve_map({
            "a_around": TraceAround("A", log),
            "b_before": TraceBefore("B", log),
            "c_after": TraceAfter("C", log),
        }),
    )

    # 四条记录**都**在 sync_entries 里 —— 退出标记是在 compile 时
    # 变成 no-op 的，建链阶段不做裁剪：那样执行链才与库里的行一一
    # 对应（诊断时能直接对着 sequence 查）。
    check("四条同步记录都在", len(chain.sync_entries) == 4)

    outcome = asyncio.run(
        compile_chain(
            chain,
            ToolInvocation(tool_key="smoke.tool"),
            make_base(log),
        )()
    )

    check(
        "顺序是 A进 -> B -> base -> C -> A出",
        log == ["A:in", "B:before", "base", "C:after", "A:out"],
    )
    check("结果来自 base", outcome.value == {"ok": True})
    check("耗时不被动过", outcome.duration_ms == 1.5)


def test_nested_around() -> None:
    section("嵌套的两个 AROUND：里层在外层之内")

    log: list[str] = []

    chain = build_chain(
        [
            Row("outer", ProcessorHook.AROUND, sequence=1),
            Row("inner", ProcessorHook.AROUND, sequence=2),
            Row("inner", ProcessorHook.AROUND, sequence=3),
            Row("outer", ProcessorHook.AROUND, sequence=4),
        ],
        resolve_processor=resolve_map({
            "outer": TraceAround("OUT", log),
            "inner": TraceAround("IN", log),
        }),
    )

    asyncio.run(
        compile_chain(
            chain,
            ToolInvocation(tool_key="smoke.tool"),
            make_base(log),
        )()
    )

    check(
        "外层进 -> 内层进 -> base -> 内层出 -> 外层出",
        log == ["OUT:in", "IN:in", "base", "IN:out", "OUT:out"],
    )


def test_before_short_circuit() -> None:
    section("BEFORE 短路：工具与它下游的处理器都不跑")

    log: list[str] = []

    chain = build_chain(
        [
            Row("deny", ProcessorHook.BEFORE, sequence=1),
            Row("c_after", ProcessorHook.AFTER, sequence=2),
        ],
        resolve_processor=resolve_map({
            "deny": TraceBefore("D", log, deny="审批没过"),
            "c_after": TraceAfter("C", log),
        }),
    )

    outcome = asyncio.run(
        compile_chain(
            chain,
            ToolInvocation(tool_key="smoke.tool"),
            make_base(log),
        )()
    )

    check("base 一次都没跑", "base" not in log)
    check("下游的 AFTER 也没跑", log == ["D:before"])
    check("结果是那条拒绝", isinstance(outcome.error, ToolCallDenied))
    check("outcome.ok 为假", not outcome.ok)


def test_around_retry() -> None:
    section("AROUND 重试：next_call 调两次 = 工具跑两次")

    log: list[str] = []

    chain = build_chain(
        [
            Row("a_around", ProcessorHook.AROUND, sequence=1),
            Row("a_around", ProcessorHook.AROUND, sequence=2),
        ],
        resolve_processor=resolve_map({
            "a_around": TraceAround("A", log, times=2),
        }),
    )

    asyncio.run(
        compile_chain(
            chain,
            ToolInvocation(tool_key="smoke.tool"),
            make_base(log),
        )()
    )

    check("工具跑了两次", log.count("base") == 2)
    check("顺序对", log == ["A:in", "base", "base", "A:out"])


def test_arguments_are_live() -> None:
    section("就地改 arguments 对 base 生效（同一个 invocation 对象）")

    log: list[str] = []
    seen: list[dict[str, Any]] = []

    async def base() -> ToolOutcome:
        # 必须在这里读，不能提前读 —— 处理器改的就是这个字典。
        seen.append(dict(invocation.arguments))
        return ToolOutcome(value=None)

    invocation = ToolInvocation(
        tool_key="smoke.tool", arguments={"text": "原始值"}
    )

    chain = build_chain(
        [Row("b_before", ProcessorHook.BEFORE, sequence=1)],
        resolve_processor=resolve_map({
            "b_before": TraceBefore(
                "B", log, set_arg=("text", "被改过")
            ),
        }),
    )

    asyncio.run(compile_chain(chain, invocation, base)())

    check("base 看到改后的值", seen == [{"text": "被改过"}])
    check("调用方持有的原始 dict 也被改了（同一个对象）",
          invocation.arguments["text"] == "被改过")


def test_async_split() -> None:
    section("ASYNC 处理器被挑出链外，按 hook 分到前后两组")

    log: list[str] = []

    chain = build_chain(
        [
            Row(
                "smoke_observer_async", ProcessorHook.BEFORE,
                execution_mode=ExecutionMode.ASYNC, sequence=1,
            ),
            Row("b_before", ProcessorHook.BEFORE, sequence=2),
        ],
        resolve_processor=resolve_map({
            "smoke_observer_async": TraceObserverAsync("OBS", log),
            "b_before": TraceBefore("B", log),
        }),
    )

    check("ASYNC 不进嵌套链", len(chain.sync_entries) == 1)
    check("ASYNC BEFORE 分到 async_before",
          [e.processor_key for e in chain.async_before]
          == ["smoke_observer_async"])
    check("async_after 为空", not chain.async_after)


def test_validation() -> None:
    section("链不自洽时抛出（并指明是哪一行）")

    around = TraceAround("A", [])

    def build(rows):
        return build_chain(
            rows, resolve_processor=resolve_map({"a_around": around})
        )

    check(
        "半条 AROUND（只有先进）",
        _raises(ValueError, lambda: build([
            Row("a_around", ProcessorHook.AROUND, sequence=1),
        ])),
    )

    check(
        "半条 AROUND（后出被 enabled 关掉）",
        _raises(ValueError, lambda: build([
            Row("a_around", ProcessorHook.AROUND, sequence=1),
            Row(
                "a_around", ProcessorHook.AROUND,
                sequence=2, enabled=False,
            ),
        ])),
    )

    # ⚠️ 曾经有一条"同一个键不能有两种 hook"的校验，已删 ——
    # cache_interceptor 正是靠 BEFORE + AFTER 两条记录工作的，
    # 那条规则会误杀它。这里留一条断言防止它被重新加回来。
    check(
        "同一个键可以占多个 hook 位置（cache 就是这么配的）",
        len(build([
            Row("a_around", ProcessorHook.BEFORE, sequence=1),
            Row("a_around", ProcessorHook.AFTER, sequence=2),
        ]).sync_entries) == 2,
    )

    check(
        "AROUND + ASYNC 被兜住",
        _raises(ValueError, lambda: build([
            Row(
                "a_around", ProcessorHook.AROUND,
                execution_mode=ExecutionMode.ASYNC, sequence=1,
            ),
        ])),
    )

    check(
        "三条 AROUND 记录也不行",
        _raises(ValueError, lambda: build([
            Row("a_around", ProcessorHook.AROUND, sequence=1),
            Row("a_around", ProcessorHook.AROUND, sequence=2),
            Row("a_around", ProcessorHook.AROUND, sequence=3),
        ])),
    )


def test_unknown_key_skipped() -> None:
    section("实现已删的行：跳过并留痕，不让整条链炸")

    log: list[str] = []

    chain = build_chain(
        [
            Row("gone_interceptor", ProcessorHook.BEFORE, sequence=1),
            Row("b_before", ProcessorHook.BEFORE, sequence=2),
        ],
        resolve_processor=resolve_map({
            "b_before": TraceBefore("B", log),
        }),
    )

    check("只剩认识的实现", len(chain.sync_entries) == 1)

    asyncio.run(
        compile_chain(
            chain,
            ToolInvocation(tool_key="smoke.tool"),
            make_base(log),
        )()
    )

    check("链照常跑完", log == ["B:before", "base"])


def test_empty_chain() -> None:
    section("空链：base 直接被调用")

    log: list[str] = []
    chain = build_chain([], resolve_processor=resolve_map({}))

    outcome = asyncio.run(
        compile_chain(
            chain,
            ToolInvocation(tool_key="smoke.tool"),
            make_base(log),
        )()
    )

    check("只有 base 跑", log == ["base"])
    check("outcome 正常", outcome.ok)


def main() -> int:
    """
    同步入口。

    每个用例自己 `asyncio.run` 一次、各拿一个干净的事件循环 ——
    链本身不带任何跨调用的状态，但事件循环隔离能让"某个用例往
    循环上挂了任务没清掉"这类问题不串到下一个用例。
    """

    test_nesting()
    test_nested_around()
    test_before_short_circuit()
    test_around_retry()
    test_arguments_are_live()
    test_async_split()
    test_validation()
    test_unknown_key_skipped()
    test_empty_chain()

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
