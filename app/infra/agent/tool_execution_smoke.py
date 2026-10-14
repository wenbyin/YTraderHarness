"""
工具执行的冒烟测试：链 + 真正调 service 方法。

不需要数据库、不需要模型：
    service 用一个真 @tool 假实现（走真的装饰器与注册表）
    链的行用一个假 loader 直接给（真实那条走服务层读库）
    service 工厂用一个桩（只认这一个契约）

因此它验的是**执行层自己**的逻辑：请求模型怎么包、链怎么驱动、
失败与拒绝怎么落成 ToolOutcome、ASYNC 观察者有没有被起掉。

跑法：
    python -m app.infra.agent.tool_execution_smoke
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel

from app.agent.tools.processor.base import (
    ToolInterceptor,
    ToolObserver,
)
from app.agent.tools.processor.context import (
    ToolCallDenied,
    ToolOutcome,
)
from app.infra.agent.tool_executor import ToolExecutor
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)
from app.services.tool.registry import get_tool_registry
from app.services.tool.registry.decorator import tool
from app.services.tool.registry.processor_registry import (
    get_processor_registry,
)

__all__ = ["main"]


#: 一个假的 tool_id。链挂在 tool_id 上，而假 loader 不看它 ——
#: 给一个真 UUID 是为了让"有 tool_id"这条分支（会去读链）走到。
_DUMMY_ID = UUID("00000000-0000-0000-0000-0000000000aa")


def check(label, condition):
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title):
    print(f"\n{title}")


# ============================================================
# 一个真的 @tool 假服务
# ============================================================


class EchoRequest(BaseModel):
    text: str


#: 服务方法被调到的每一次，参数记在这里 —— 断言"工具到底跑了没有、
#: 拿到的是什么参数"都靠它（比读返回值更直接）。
CALLS: list[str] = []


class EchoService(ABC):
    """
    契约（抽象基类）—— `spec.contract` 靠 MRO 里的抽象类推出来。

    ⚠️ 必须真继承 ABC：只写 @abstractmethod 而不继承 ABC 的话
    `inspect.isabstract()` 是 False，`ToolSpec.contract` 会一路
    回落到 owner 自己 —— 于是"契约 -> 实现"的解析没被验到。
    """

    @abstractmethod
    async def echo(self, req: EchoRequest) -> dict[str, Any]: ...


class EchoServiceImpl(EchoService):

    @tool(key="smoke.echo")
    async def echo(self, req: EchoRequest) -> dict[str, Any]:
        """把传进来的文本回显出来。"""

        CALLS.append(req.text)

        return {"echo": req.text}


class BoomService(ABC):
    """另一条契约 —— 让"每个工具从自己的契约造实例"这件事被验到。"""

    @abstractmethod
    async def boom(self, req: EchoRequest) -> dict[str, Any]: ...


class BoomServiceImpl(BoomService):

    @tool(key="smoke.boom")
    async def boom(self, req: EchoRequest) -> dict[str, Any]:
        """总是失败。"""

        CALLS.append(req.text)

        raise RuntimeError("工具内部炸了")


class StubFactory:
    """
    只认 EchoService 这一个契约的工厂。

    ⚠️ 必须有 `scope()`：执行器是走 scope 造服务的，不是 create()
    --------------------------------------------------------
    `create()` 是基工厂上的方法，而基工厂是 session_factory 模式 ——
    造出来的仓储拿不到 session，第一次查库就
    "AttributeError: 'NoneType' object has no attribute execute"。
    真实工具一执行就查库，所以执行器改走 `scope`（见 tool_executor.py
    的事务边界那一节）。本桩要跟着提供同一个入口，否则它测的就不是
    真实那条路。

    这里刻意**不复用** create()：scope 与 create 在真实实现里是两件事
    （一个开事务、一个不开），桩把它们合成一个会让"执行器到底调的哪个"
    在测试里失去区分度。
    """

    def __init__(self, impl) -> None:
        self.impl = impl
        self.created: list[type] = []
        self.scoped: list[type] = []

    def create(self, contract, **_: Any):
        self.created.append(contract)
        return self.impl()

    @contextlib.asynccontextmanager
    async def scope(self, contract, **_: Any):
        self.scoped.append(contract)
        yield self.impl()


# ============================================================
# 链上的记录与假处理器
# ============================================================


@dataclass
class Row:
    processor_key: str
    hook: ProcessorHook
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    config: dict[str, Any] = field(default_factory=dict)
    sequence: int = 1
    enabled: bool = True


class RewriteBefore(ToolInterceptor):
    """把 text 参数改掉，验证"就地改参数"一路生效到 service。"""

    name = "smoke_rewrite"
    supported_hooks = frozenset({ProcessorHook.BEFORE})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})

    async def before(self, invocation, *, config, runtime=None):
        invocation.arguments["text"] = "改过的"
        return invocation


class DenyBefore(ToolInterceptor):
    name = "smoke_deny"
    supported_hooks = frozenset({ProcessorHook.BEFORE})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})

    async def before(self, invocation, *, config, runtime=None):
        return ToolOutcome(error=ToolCallDenied("审批没过"))


class AsyncObserver(ToolObserver):
    """ASYNC 观察者：起了就不等，但必须真的被起掉。"""

    name = "smoke_async_observer"
    supported_hooks = frozenset({ProcessorHook.AFTER})
    supported_execution_modes = frozenset({ExecutionMode.ASYNC})

    seen: ClassVar[list[str]] = []

    async def after(self, invocation, outcome, *, config, runtime=None):
        AsyncObserver.seen.append(invocation.tool_key)
        return outcome


REAL = get_processor_registry()

#: 本模块定义的假处理器。真实注册表只扫 `app.agent.tools.processor`
#: 那个包，扫不到这里 —— 这正是让本文件不因仓库里处理器增减而失败
#: 的原因（与 tool_processor_smoke.py 同一条做法）。
LOCAL = {
    "smoke_rewrite": RewriteBefore(),
    "smoke_deny": DenyBefore(),
    "smoke_async_observer": AsyncObserver(),
}


class LocalRegistry:
    """
    先查本地假实现，再落到真实注册表。

    只实现 `get(key)` —— executor 对注册表的要求就这一个方法
    （与 `build_chain` 的 `resolve_processor` 是同一个形状）。
    这样既能在同一个用例里用假处理器，也能测真实的那五个。
    """

    def get(self, key: str):
        local = LOCAL.get(key)
        return local if local is not None else REAL.get(key)


def executor_for(impl, loader=None, **kwargs) -> ToolExecutor:
    """造一个执行器。`executor.factory` 留出来给"它走的哪条路"那几条断言用。"""

    factory = StubFactory(impl)

    executor = ToolExecutor(
        registry=get_tool_registry(),
        processor_registry=LocalRegistry(),
        chain_loader=loader,
        factory=factory,
        **kwargs,
    )

    executor.factory = factory  # type: ignore[attr-defined]

    return executor


# ============================================================
# 用例
# ============================================================


def test_plain_execution() -> None:
    section("没有链：直接执行，请求模型包对了")

    CALLS.clear()
    executor = executor_for(EchoServiceImpl)

    outcome = asyncio.run(
        executor.execute("smoke.echo", {"text": "你好"})
    )

    check("成功", outcome.ok)
    check("服务真的被调到、且参数是包好的模型值", CALLS == ["你好"])
    check("返回值透传", outcome.value == {"echo": "你好"})
    check("填了耗时", outcome.duration_ms is not None)
    check("不是缓存命中", outcome.cached is False)

    # ⚠️ 这一条盯的是一个真发生过的 AttributeError：
    #     'NoneType' object has no attribute execute
    # 执行器曾经用 `factory.create()` —— 那是**基工厂**上的方法，而基工厂
    # 是 session_factory 模式，造出来的仓储 `_session=None`，一查库就炸。
    # 且 `new_session()` 那条路**没有任何仓储实现过**（全仓 19 个仓储一律
    # 直接用 self._session），所以唯一的解法是给它一个 scope。
    #
    # 断言"走了 scope、没走 create"而不是只断言执行成功：本桩的两个入口
    # 都能造出可用的服务，只测结果区分不出执行器调的是哪个。
    check(
        "服务是经 scope 造的（不是 create）",
        executor.factory.scoped == [EchoService]
        and executor.factory.created == [],
    )


def test_chain_rewrites_arguments() -> None:
    section("链上的 BEFORE 改参数 -> service 看到改后的值")

    CALLS.clear()

    async def loader(_tool_id):
        return [
            Row("smoke_rewrite", ProcessorHook.BEFORE, sequence=1)
        ]

    executor = executor_for(EchoServiceImpl, loader)

    outcome = asyncio.run(
        executor.execute(
            "smoke.echo", {"text": "原值"}, tool_id=_DUMMY_ID
        )
    )

    check("成功", outcome.ok)
    check("服务拿到改后的值", CALLS == ["改过的"])


def test_before_denies() -> None:
    section("拦截器拒绝：工具一次都没跑")

    CALLS.clear()

    async def loader(_tool_id):
        return [Row("smoke_deny", ProcessorHook.BEFORE, sequence=1)]

    executor = executor_for(EchoServiceImpl, loader)

    outcome = asyncio.run(
        executor.execute(
            "smoke.echo", {"text": "x"}, tool_id=_DUMMY_ID
        )
    )

    check("工具没跑", CALLS == [])
    check("结果是那条拒绝", isinstance(outcome.error, ToolCallDenied))
    check("outcome 不是一个成功的空值", not outcome.ok)


def test_tool_failure() -> None:
    section("工具自己失败：落成 ToolOutcome.error，不往上抛")

    CALLS.clear()
    executor = executor_for(BoomServiceImpl)

    outcome = asyncio.run(
        executor.execute("smoke.boom", {"text": "x"})
    )

    check("工具确实跑了", CALLS == ["x"])
    check("失败被收进 outcome", isinstance(outcome.error, RuntimeError))
    check("耗时照样有（失败也要能观测）",
          outcome.duration_ms is not None)


def test_chain_load_failure_blocks_execution() -> None:
    section("链读不出来：**不执行**工具（绕过链比失败更糟）")

    CALLS.clear()

    async def loader(_tool_id):
        raise RuntimeError("数据库连不上")

    executor = executor_for(EchoServiceImpl, loader)

    outcome = asyncio.run(
        executor.execute(
            "smoke.echo", {"text": "x"}, tool_id=_DUMMY_ID
        )
    )

    check("工具没跑", CALLS == [])
    check("错误被记录", isinstance(outcome.error, RuntimeError))


def test_unknown_tool() -> None:
    section("执行一个不在注册表里的 key")

    executor = executor_for(EchoServiceImpl)

    outcome = asyncio.run(
        executor.execute("nope.missing", {"text": "x"})
    )

    check("是 LookupError", isinstance(outcome.error, LookupError))


def test_real_cache_processor() -> None:
    section("真处理器：cache_interceptor 的第一次未命中、第二次命中")

    CALLS.clear()

    async def loader(_tool_id):
        return [
            Row("cache_interceptor", ProcessorHook.BEFORE, sequence=1),
            Row("cache_interceptor", ProcessorHook.AFTER, sequence=2),
        ]

    executor = executor_for(EchoServiceImpl, loader)

    first = asyncio.run(
        executor.execute(
            "smoke.echo", {"text": "缓存用例"}, tool_id=_DUMMY_ID
        )
    )
    second = asyncio.run(
        executor.execute(
            "smoke.echo", {"text": "缓存用例"}, tool_id=_DUMMY_ID
        )
    )

    check("第一次未命中、真的调了服务", CALLS == ["缓存用例"])
    check("第一次成功", first.ok and not first.cached)
    check("第二次命中缓存", second.cached)
    check("第二次没有再调服务", CALLS == ["缓存用例"])
    check("命中拿到的值与第一次相同",
          second.value == {"echo": "缓存用例"})


def test_async_observer_is_spawned() -> None:
    section("ASYNC 观察者：被起掉、返回值不被等待")

    CALLS.clear()
    AsyncObserver.seen.clear()

    async def loader(_tool_id):
        return [
            Row(
                "smoke_async_observer", ProcessorHook.AFTER,
                execution_mode=ExecutionMode.ASYNC, sequence=1,
            )
        ]

    executor = executor_for(EchoServiceImpl, loader)

    async def run() -> ToolOutcome:
        outcome = await executor.execute(
            "smoke.echo", {"text": "x"}, tool_id=_DUMMY_ID
        )
        # 让 create_task 起来的那个任务跑完（它不等返回值，
        # 所以不能靠 await execute 就认为它跑过了）。
        await asyncio.sleep(0)
        return outcome

    outcome = asyncio.run(run())

    check("执行照常成功", outcome.ok)
    check("ASYNC 观察者确实跑过", AsyncObserver.seen == ["smoke.echo"])


def main() -> int:

    test_plain_execution()
    test_chain_rewrites_arguments()
    test_before_denies()
    test_tool_failure()
    test_chain_load_failure_blocks_execution()
    test_unknown_tool()
    test_real_cache_processor()
    test_async_observer_is_spawned()

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
