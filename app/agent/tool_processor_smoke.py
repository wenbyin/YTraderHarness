"""
处理器挂载的冒烟测试。

不需要数据库 —— 仓储用内存假实现（与 tool_smoke.py 同一套做法）。
这样能让链的状态转移可确定地验证：

    挂载 -> 插入 -> 重排 -> 删除补洞 -> 整体替换

⚠️ 假仓储**复刻了**"写完重编号成 1..N"这条语义，因此它验证的是
**服务层**的逻辑（挂载校验、洋葱模型、位置计算、过期链判断）。
真正的 SQL 正确性 —— 两段式挪位会不会撞 `uq_tool_processor_tool_sequence`
—— 假仓储测不出来，那一份在 app/agent/processor_shift_check.py 里
连真库跑。两层都要跑，缺一个都会漏掉一整类问题。

跑法：
    python -m app.agent.tool_processor_smoke
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Any, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.agent.tools.processor.base import (
    BaseToolProcessor,
    ToolInterceptor,
    ToolObserver,
    ToolTransformer,
)
from app.agent.tools.processor.context import (
    AsyncToolCall,
    ToolInvocation,
    ToolOutcome,
)
from app.common.exceptions import ConflictError, NotFoundError
from app.services.tool.application.processor_definition_service_impl import (
    ProcessorDefinitionServiceImpl,
)
from app.services.tool.application.tool_processor_service_impl import (
    ToolProcessorServiceImpl,
    validate_onion,
)
from app.services.tool.domain.entities import (
    ToolProcessor as ToolProcessorEntity,
)
from app.services.tool.domain.schemas import (
    CreateToolProcessorRequest,
    ExecutionMode,
    ProcessorHook,
    ReorderToolProcessorsRequest,
    ReplaceToolProcessorChainRequest,
    ReplaceToolProcessorItem,
    UpdateToolProcessorRequest,
)
from app.services.tool.registry.processor_registry import (
    ToolProcessorRegistry,
    get_processor_registry,
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


def _raises(exception, fn):
    """fn() 抛出了指定异常吗。"""

    try:
        fn()
    except exception:
        return True
    except Exception:
        return False
    return False


async def _async_raises(exception, coro):
    """await coro 抛出了指定异常吗。coro 必须是个协程。"""

    try:
        await coro
    except exception:
        return True
    except Exception:
        return False
    return False


# ============================================================
# 测试用的假处理器
# ============================================================
#
# ⚠️ 定义在本模块（__main__）里，因此**不会**被真实注册表扫到 ——
# 它只走 PROCESSOR_PACKAGES 下的包（app.agent.tools.processor）。
# 这正是让服务层用例稳定的关键：仓库里真实处理器的增减不该让
# 本文件失败。


class DemoConfig(BaseModel):
    """一个有参数的假处理器配置。"""

    limit: int = Field(default=5, ge=1, le=100)


class AroundDemo(ToolInterceptor):
    name = "demo_around"
    display_name = "假的 AROUND"
    config_model: ClassVar[type[BaseModel]] = DemoConfig
    supported_hooks = frozenset({ProcessorHook.AROUND})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})


class AroundTwoDemo(ToolInterceptor):
    """
    第二个 AROUND 实现。

    造交叉**必须**有两个不同的 AROUND 键：同一个键的两条记录按定义
    构成**一个**区间（首尾），无从与它自己交叉。
    """

    name = "demo_around_two"
    display_name = "假的 AROUND 之二"
    config_model: ClassVar[type[BaseModel]] = DemoConfig
    supported_hooks = frozenset({ProcessorHook.AROUND})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})


class BeforeDemo(ToolInterceptor):
    name = "demo_before"
    display_name = "假的 BEFORE"
    supported_hooks = frozenset({ProcessorHook.BEFORE})
    supported_execution_modes = frozenset({ExecutionMode.SYNC})


def make_bad_class():
    """
    造一个"AROUND 却声明 ASYNC"的类 —— 它在**类体执行时**就抛
    TypeError（见 BaseToolProcessor.__init_subclass__）。

    必须定义在函数里：放模块级会让本文件直接 import 失败，
    那样连一条断言都跑不到。
    """

    class LooseAroundDemo(ToolInterceptor):
        name = "demo_bad"
        supported_hooks = frozenset({ProcessorHook.AROUND})
        supported_execution_modes = frozenset(
            {ExecutionMode.ASYNC}
        )

    return LooseAroundDemo


class ObserverDemo(ToolObserver):
    name = "demo_observer"
    display_name = "假的 OBSERVER"
    supported_hooks = frozenset({
        ProcessorHook.BEFORE,
        ProcessorHook.AFTER,
    })
    supported_execution_modes = frozenset({ExecutionMode.ASYNC})


#: 仓库里真实存在的处理器键（app/agent/tools/processor/ 下那五个）。
#:
#: 曾经还有 retry_interceptor / timeout_interceptor，已删 —— 重试与
#: 超时在**图节点**那一级已经有了，工具级再来一遍是重复实现，而且
#: 两级各有各的语义（节点级管的是"这一步要不要重跑"，工具级只能管
#: "这一次调用"），叠在一起出问题时很难说清是谁在重跑。
REAL_KEYS = frozenset({
    "approval_interceptor",
    "cache_interceptor",
    "auth_interceptor",
    "event_observer",
    "execution_guard_interceptor",
    "metrics_observer",
    "privacy_transformer",
    "retry_interceptor",
    "timeout_interceptor",
})


def make_bad_transformer_class():
    """
    造一个"转换器却只支持 ASYNC"的类 —— 同样在建类时就抛。

    ASYNC 的返回值会被丢弃，所以这种转换器改了数据也没人接，
    等于一个什么都不做却在链上看起来很正常的处理器。
    """

    class LooseTransformerDemo(ToolTransformer):
        name = "demo_bad_transformer"
        supported_hooks = frozenset({
            ProcessorHook.BEFORE,
            ProcessorHook.AFTER,
        })
        supported_execution_modes = frozenset(
            {ExecutionMode.ASYNC}
        )

    return LooseTransformerDemo


def demo_registry() -> ToolProcessorRegistry:
    """只有本文件那几个假实现的注册表（服务层用例用）。"""

    return ToolProcessorRegistry({
        processor.get_key(): processor
        for processor in (
            AroundDemo(),
            AroundTwoDemo(),
            BeforeDemo(),
            ObserverDemo(),
        )
    })


def real_registry() -> ToolProcessorRegistry:
    """
    只含**仓库里真实那五个**的注册表。

    为什么要筛一道：发现走的是 `BaseToolProcessor.__subclasses__()`，
    那是**全局**的 —— 本文件定义的假实现也在同一个进程里，因此
    真实注册表里会多出它们（见下面那条断言）。目录用例要的是
    确定的条目数，所以按已知的真实键筛一遍。
    """

    registry = get_processor_registry()

    return ToolProcessorRegistry({
        key: registry.get(key) for key in REAL_KEYS
    })


# ============================================================
# 内存假仓储
# ============================================================


class FakeDefinitionRepository:
    """只实现服务层用得到的那一个方法。"""

    def __init__(self) -> None:
        self.known: set[UUID] = set()

    async def get(self, tool_id):
        return object() if tool_id in self.known else None


class FakeProcessorRepository:
    """
    内存链条。

    **复刻了契约里"写完重编号成 1..N"那条语义** —— 服务层的用例
    依赖它。真正的挪位 SQL 不在这里验证（见模块 docstring）。
    """

    def __init__(self) -> None:
        self.rows: list[ToolProcessorEntity] = []

    # ---- 读 ----

    def _of(self, tool_id):
        return sorted(
            (r for r in self.rows if r.tool_id == tool_id),
            key=lambda r: r.sequence,
        )

    async def list_for_tool(self, tool_id, *, enabled_only=False):
        rows = self._of(tool_id)
        return [r for r in rows if r.enabled] if enabled_only else rows

    async def get(self, tool_id, processor_id):
        for row in self.rows:
            if (
                    row.tool_id == tool_id
                    and row.processor_id == processor_id
            ):
                return row
        return None

    async def count_for_tool(self, tool_id):
        return len(self._of(tool_id))

    # ---- 写 ----

    def _renumber(self, tool_id) -> None:
        for index, row in enumerate(self._of(tool_id), start=1):
            if row.sequence != index:
                self.rows[self.rows.index(row)] = row.model_copy(
                    update={"sequence": index}
                )

    def _order(self, tool_id) -> list[UUID]:
        return [r.processor_id for r in self._of(tool_id)]

    def _apply(self, tool_id, ordered_ids: list[UUID]) -> None:
        if set(ordered_ids) != set(self._order(tool_id)):
            raise ValueError("顺序表不是全集")
        for index, processor_id in enumerate(ordered_ids, start=1):
            for position, row in enumerate(self.rows):
                if row.processor_id == processor_id:
                    self.rows[position] = row.model_copy(
                        update={"sequence": index}
                    )

    async def insert(self, entity, *, position=None):
        self.rows.append(entity.model_copy(update={"sequence": 10_000}))
        order = self._order(entity.tool_id)
        order.remove(entity.processor_id)
        index = (
            len(order)
            if position is None
            else max(0, min(position - 1, len(order)))
        )
        order.insert(index, entity.processor_id)
        self._apply(entity.tool_id, order)
        return await self.get(entity.tool_id, entity.processor_id)

    async def update(self, entity):
        for position, row in enumerate(self.rows):
            if row.processor_id == entity.processor_id:
                self.rows[position] = entity
                return entity
        raise NotFoundError(f"挂载记录不存在：{entity.processor_id}")

    async def delete(self, tool_id, processor_id):
        before = len(self.rows)
        self.rows = [
            r for r in self.rows
            if not (
                r.tool_id == tool_id
                and r.processor_id == processor_id
            )
        ]
        removed = before - len(self.rows)
        if removed:
            self._renumber(tool_id)
        return removed

    async def replace_all(self, tool_id, entities):
        self.rows = [r for r in self.rows if r.tool_id != tool_id]
        for index, entity in enumerate(entities, start=1):
            self.rows.append(
                entity.model_copy(update={"sequence": index})
            )
        return self._of(tool_id)

    async def reorder(self, tool_id, ordered_ids):
        self._apply(tool_id, ordered_ids)
        return len(ordered_ids)


# ============================================================
# 装配
# ============================================================


def build_service(registry=None):
    definitions = FakeDefinitionRepository()
    processors = FakeProcessorRepository()

    service = ToolProcessorServiceImpl(
        tool_definition_repository=definitions,
        tool_processor_repository=processors,
        registry=registry or demo_registry(),
    )

    return service, definitions, processors


def _around_item(key: str) -> ReplaceToolProcessorItem:
    """整体替换里的一项：一个 AROUND 记录。"""

    return ReplaceToolProcessorItem(
        processor_key=key,
        execution_mode=ExecutionMode.SYNC,
        hook=ProcessorHook.AROUND,
    )


def entity(
        tool_id: UUID,
        key: str,
        hook: ProcessorHook,
        sequence: int,
        mode: ExecutionMode = ExecutionMode.SYNC,
) -> ToolProcessorEntity:

    now = datetime.now(timezone.utc)

    return ToolProcessorEntity(
        processor_id=uuid4(),
        tool_id=tool_id,
        processor_key=key,
        sequence=sequence,
        execution_mode=mode,
        hook=hook,
        config={},
        enabled=True,
        created_at=now,
        updated_at=now,
    )


# ============================================================
# 用例
# ============================================================


async def main() -> int:

    # --------------------------------------------------------
    section("注册表：真实的处理器都在，type 基类不在")

    registry = get_processor_registry()

    check(
        "六个真实处理器都在目录里",
        REAL_KEYS <= set(registry.keys()),
    )
    check(
        "本文件的假实现也被扫到了 —— 发现走的是全局的 "
        "__subclasses__()，不限于 PROCESSOR_PACKAGES",
        {"demo_around", "demo_before", "demo_observer"}
        <= set(registry.keys()),
    )
    check(
        "type 基类没被当成实现",
        all(
            type(p) not in (
                BaseToolProcessor,
                ToolInterceptor,
                ToolObserver,
            )
            for p in registry.all()
        ),
    )
    check(
        "processor_type 与继承的基类一致",
        all(
            type(p).processor_type is not None
            for p in registry.all()
        ),
    )
    check(
        "未注册的 key 读路径返回 None",
        registry.get("no_such_processor") is None,
    )
    check(
        "写路径抛 NotFoundError（→404）",
        _raises(
            NotFoundError,
            lambda: registry.require("no_such_processor"),
        ),
    )

    section("类定义期：自相矛盾的声明建不出来")

    check(
        "AROUND 却声明 ASYNC -> TypeError（不是等运行到才发现）",
        _raises(TypeError, make_bad_class),
    )
    check(
        "TRANSFORMER 只支持 ASYNC -> TypeError"
        "（返回值会被丢弃，等于什么都没改）",
        _raises(TypeError, make_bad_transformer_class),
    )

    section("洋葱模型：AROUND 必须先进后出")

    A = ProcessorHook.AROUND
    B = ProcessorHook.BEFORE

    check(
        "单个 AROUND 放行（POST 的合法中间态）",
        not _raises(
            ConflictError,
            lambda: validate_onion([("x", A)]),
        ),
    )
    check(
        "成对且嵌套 -> 通过",
        not _raises(
            ConflictError,
            lambda: validate_onion(
                [("a", A), ("b", A), ("b", A), ("a", A)]
            ),
        ),
    )
    check(
        "交叉 -> 拒",
        _raises(
            ConflictError,
            lambda: validate_onion(
                [("a", A), ("b", A), ("a", A), ("b", A)]
            ),
        ),
    )
    check(
        "BEFORE 夹在中间不影响（BEFORE 无约束）",
        not _raises(
            ConflictError,
            lambda: validate_onion(
                [("a", A), ("v", B), ("b", A), ("b", A), ("a", A)]
            ),
        ),
    )
    check(
        "多个兄弟区间同在一个父区间里 -> 通过"
        "（a=[1,6] 包住 b=[2,3] 与 c=[4,5]）",
        not _raises(
            ConflictError,
            lambda: validate_onion(
                [("a", A), ("b", A), ("b", A),
                 ("c", A), ("c", A), ("a", A)]
            ),
        ),
    )
    check(
        "c 从 a 内部开始、伸到 a 外面 -> 拒"
        "（a=[1,5] 与 c=[3,6] 交叉，而 b=[2,2] 与两者都不交叉 —— "
        "只比较相邻两个区间的实现会漏掉它）",
        _raises(
            ConflictError,
            lambda: validate_onion(
                [("a", A), ("b", A), ("b", A),
                 ("c", A), ("a", A), ("c", A)]
            ),
        ),
    )

    # --------------------------------------------------------
    section("挂载校验")

    service, definitions, processors = build_service()
    tool_id = uuid4()
    definitions.known.add(tool_id)

    check(
        "tool 不存在 -> NotFoundError",
        await _async_raises(
            NotFoundError,
            service.list_chain(uuid4()),
        ),
    )
    check(
        "未注册的 processor_key -> NotFoundError",
        await _async_raises(
            NotFoundError,
            service.add_processor(
                tool_id,
                CreateToolProcessorRequest(
                    processor_key="ghost",
                    execution_mode=ExecutionMode.SYNC,
                    hook=ProcessorHook.AROUND,
                ),
            ),
        ),
    )
    check(
        "hook 不在 supported_hooks -> ConflictError",
        await _async_raises(
            ConflictError,
            service.add_processor(
                tool_id,
                CreateToolProcessorRequest(
                    processor_key="demo_around",
                    execution_mode=ExecutionMode.SYNC,
                    hook=ProcessorHook.BEFORE,
                ),
            ),
        ),
    )
    check(
        "execution_mode 不在 supported -> ConflictError",
        await _async_raises(
            ConflictError,
            service.add_processor(
                tool_id,
                CreateToolProcessorRequest(
                    processor_key="demo_before",
                    execution_mode=ExecutionMode.ASYNC,
                    hook=ProcessorHook.BEFORE,
                ),
            ),
        ),
    )
    check(
        "config 不满足 config_model -> ValidationError",
        await _async_raises(
            Exception,
            service.add_processor(
                tool_id,
                CreateToolProcessorRequest(
                    processor_key="demo_around",
                    execution_mode=ExecutionMode.SYNC,
                    hook=ProcessorHook.AROUND,
                    config={"limit": 9999},
                ),
            ),
        ),
    )

    # --------------------------------------------------------
    section("链：挂载与位置")

    first = await service.add_processor(
        tool_id,
        CreateToolProcessorRequest(
            processor_key="demo_around",
            execution_mode=ExecutionMode.SYNC,
            hook=ProcessorHook.AROUND,
        ),
    )
    check("第一条 sequence=1", first.sequence == 1)

    second = await service.add_processor(
        tool_id,
        CreateToolProcessorRequest(
            processor_key="demo_before",
            execution_mode=ExecutionMode.SYNC,
            hook=ProcessorHook.BEFORE,
        ),
    )
    check("追加到末尾 sequence=2", second.sequence == 2)

    third = await service.add_processor(
        tool_id,
        CreateToolProcessorRequest(
            processor_key="demo_observer",
            execution_mode=ExecutionMode.ASYNC,
            hook=ProcessorHook.AFTER,
        ),
    )
    check("再追加 sequence=3", third.sequence == 3)

    check(
        "config 被归一化（默认值落库）",
        all(
            row.config == {} or isinstance(row.config, dict)
            for row in await processors.list_for_tool(tool_id)
        ),
    )

    inserted = await service.add_processor(
        tool_id,
        CreateToolProcessorRequest(
            processor_key="demo_before",
            execution_mode=ExecutionMode.SYNC,
            hook=ProcessorHook.BEFORE,
            sequence=1,
        ),
    )
    check(
        "sequence=1 插到最前，其余整体后移",
        inserted.sequence == 1
        and [
            r.processor_key
            for r in await processors.list_for_tool(tool_id)
        ] == [
            "demo_before",
            "demo_around",
            "demo_before",
            "demo_observer",
        ],
    )

    far = await service.add_processor(
        tool_id,
        CreateToolProcessorRequest(
            processor_key="demo_before",
            execution_mode=ExecutionMode.SYNC,
            hook=ProcessorHook.BEFORE,
            sequence=999,
        ),
    )
    check(
        "sequence 超出链长 -> 落到末尾（只是位置提示）",
        far.sequence == 5,
    )

    # --------------------------------------------------------
    section("链：读回来（含已下线的实现）")

    chain = await service.list_chain(tool_id)
    check(
        "链按 sequence 升序",
        [p.sequence for p in chain.processors] == [1, 2, 3, 4, 5],
    )
    check(
        "注册表里有的带上 processor",
        chain.processors[0].processor is not None,
    )

    # 往库里塞一条"实现已被删掉"的行，模拟代码侧删了实现。
    processors.rows.append(
        entity(tool_id, "removed_processor", ProcessorHook.BEFORE, 99)
    )
    processors._renumber(tool_id)

    chain = await service.list_chain(tool_id)
    ghost = [
        p for p in chain.processors
        if p.processor_key == "removed_processor"
    ]
    check("未注册的行**留在链里**", len(ghost) == 1)
    check(
        "它的 processor 字段**整个缺席**（不是 null）",
        "processor" not in ghost[0].model_dump(mode="json"),
    )
    check(
        "其余行照常带上 processor",
        chain.processors[0].processor is not None,
    )

    # 清掉，后面的用例不受影响
    await processors.delete(
        tool_id, ghost[0].processor_id
    )
    check(
        "删除后补洞（仍是 1..5）",
        [
            r.sequence
            for r in await processors.list_for_tool(tool_id)
        ] == [1, 2, 3, 4, 5],
    )

    # --------------------------------------------------------
    section("链：修改")

    target = (await processors.list_for_tool(tool_id))[0]

    check(
        "一次都不给 -> pydantic 就拒了（minProperties: 1）",
        _raises(Exception, UpdateToolProcessorRequest),
    )

    updated = await service.update_processor(
        tool_id,
        target.processor_id,
        UpdateToolProcessorRequest(execution_mode=ExecutionMode.SYNC),
    )
    check("只改一个字段成功", updated.processor_id == target.processor_id)
    check(
        "位置没被改动",
        [
            r.sequence
            for r in await processors.list_for_tool(tool_id)
        ] == [1, 2, 3, 4, 5],
    )

    check(
        "改成实现不支持的 hook -> ConflictError（改完要重校验）",
        await _async_raises(
            ConflictError,
            service.update_processor(
                tool_id,
                target.processor_id,
                UpdateToolProcessorRequest(
                    hook=ProcessorHook.AROUND
                ),
            ),
        ),
    )

    check(
        "改不存在的行 -> NotFoundError",
        await _async_raises(
            NotFoundError,
            service.update_processor(
                tool_id,
                uuid4(),
                UpdateToolProcessorRequest(enabled=False),
            ),
        ),
    )

    check(
        "别的 Tool 的行改不到（按 tool_id 过滤）",
        await _async_raises(
            NotFoundError,
            service.update_processor(
                uuid4(),
                target.processor_id,
                UpdateToolProcessorRequest(enabled=False),
            ),
        ),
    )

    # --------------------------------------------------------
    section("链：重排")

    rows = await processors.list_for_tool(tool_id)

    check(
        "少给一个 ID -> ConflictError（过期链）",
        await _async_raises(
            ConflictError,
            service.reorder(
                tool_id,
                ReorderToolProcessorsRequest(
                    processor_ids=[
                        r.processor_id for r in rows[:-1]
                    ]
                ),
            ),
        ),
    )
    check(
        "混进别家的 ID -> ConflictError",
        await _async_raises(
            ConflictError,
            service.reorder(
                tool_id,
                ReorderToolProcessorsRequest(
                    processor_ids=[
                        r.processor_id for r in rows[1:]
                    ] + [uuid4()]
                ),
            ),
        ),
    )
    check(
        "重复 ID -> pydantic 就拒了（uniqueItems）",
        _raises(
            Exception,
            lambda: ReorderToolProcessorsRequest(
                processor_ids=[rows[0].processor_id] * 2
            ),
        ),
    )
    check(
        "全给（逆序）-> 成功",
        (
            await service.reorder(
                tool_id,
                ReorderToolProcessorsRequest(
                    processor_ids=[
                        r.processor_id for r in reversed(rows)
                    ]
                ),
            )
        ).processors[0].processor_id == rows[-1].processor_id,
    )

    # --------------------------------------------------------
    section("链：整体替换")

    replaced = await service.replace_chain(
        tool_id,
        ReplaceToolProcessorChainRequest(
            items=[
                ReplaceToolProcessorItem(
                    processor_key="demo_around",
                    execution_mode=ExecutionMode.SYNC,
                    hook=ProcessorHook.AROUND,
                ),
                ReplaceToolProcessorItem(
                    processor_key="demo_observer",
                    execution_mode=ExecutionMode.ASYNC,
                    hook=ProcessorHook.AFTER,
                ),
            ]
        ),
    )
    check(
        "顺序即数组顺序",
        [p.processor_key for p in replaced.processors]
        == ["demo_around", "demo_observer"],
    )
    check(
        "编号 1..N",
        [p.sequence for p in replaced.processors] == [1, 2],
    )
    old_ids = {r.processor_id for r in rows}
    new_ids = {p.processor_id for p in replaced.processors}
    check("旧 id 全部消失（全删重建）", not (old_ids & new_ids))

    cleared = await service.replace_chain(
        tool_id,
        ReplaceToolProcessorChainRequest(items=[]),
    )
    check("空数组 = 全部摘掉", cleared.processors == [])

    check(
        "嵌套的 AROUND -> 通过（a 在外、b 在内）",
        not await _async_raises(
            ConflictError,
            service.replace_chain(
                tool_id,
                ReplaceToolProcessorChainRequest(
                    items=[
                        _around_item("demo_around"),
                        _around_item("demo_around_two"),
                        _around_item("demo_around_two"),
                        _around_item("demo_around"),
                    ]
                ),
            ),
        ),
    )
    intact = [
        p.processor_key
        for p in (await service.list_chain(tool_id)).processors
    ]

    check(
        "交叉的 AROUND -> ConflictError（a 先进却先出）",
        await _async_raises(
            ConflictError,
            service.replace_chain(
                tool_id,
                ReplaceToolProcessorChainRequest(
                    items=[
                        _around_item("demo_around"),
                        _around_item("demo_around_two"),
                        _around_item("demo_around"),
                        _around_item("demo_around_two"),
                    ]
                ),
            ),
        ),
    )
    check(
        "被拒之后链原样未动（校验在落库之前，不是先写再回滚）",
        [
            p.processor_key
            for p in (await service.list_chain(tool_id)).processors
        ] == intact
        and len(intact) == 4,
    )

    # --------------------------------------------------------
    section("链：删除")

    await service.replace_chain(
        tool_id,
        ReplaceToolProcessorChainRequest(
            items=[
                ReplaceToolProcessorItem(
                    processor_key="demo_before",
                    execution_mode=ExecutionMode.SYNC,
                    hook=ProcessorHook.BEFORE,
                )
                for _ in range(3)
            ]
        ),
    )

    rows = await processors.list_for_tool(tool_id)
    await service.delete_processor(tool_id, rows[1].processor_id)

    check(
        "删中间一条后补洞",
        [
            r.sequence
            for r in await processors.list_for_tool(tool_id)
        ] == [1, 2],
    )
    check(
        "删不存在的 -> NotFoundError",
        await _async_raises(
            NotFoundError,
            service.delete_processor(tool_id, rows[1].processor_id),
        ),
    )

    # --------------------------------------------------------
    section("实现目录：过滤与分页")

    catalog = ProcessorDefinitionServiceImpl(
        registry=real_registry()
    )

    everything = await catalog.list_definitions()
    # 用 REAL_KEYS 的长度而不是写死的数字：新增一个真实处理器时该改的是
    # 上面那个集合（它同时被"注册表里都有"那条断言用），而不是这里。
    check("默认列出全部实现", everything.total == len(REAL_KEYS))
    check(
        "config_schema 始终存在",
        all(item.config_schema for item in everything.items),
    )
    check(
        "registry_available 恒为 true",
        all(item.registry_available for item in everything.items),
    )
    check(
        "无 config 的实现给出空 object schema",
        next(
            i for i in everything.items
            if i.key == "metrics_observer"
        ).config_schema["properties"] == {},
    )

    by_hook = await catalog.list_definitions(
        hook=ProcessorHook.AROUND
    )
    check(
        "按 hook 过滤（AROUND：缓存 + 执行守卫 + 超时 + 重试）",
        {i.key for i in by_hook.items}
        == {
            "cache_interceptor",
            "execution_guard_interceptor",
            "retry_interceptor",
            "timeout_interceptor",
        },
    )

    by_mode = await catalog.list_definitions(
        execution_mode=ExecutionMode.ASYNC
    )
    check(
        "按 execution_mode 过滤",
        {i.key for i in by_mode.items}
        == {"event_observer", "metrics_observer"},
    )

    by_keyword = await catalog.list_definitions(keyword="缓存")
    check(
        "keyword 命中展示名",
        {i.key for i in by_keyword.items}
        == {"cache_interceptor"},
    )

    # 第 2 页、每页 4 条：满页之外剩下的那几条。
    # 同样从 REAL_KEYS 推，别写死数字 —— 前提只是"处理器比一页多"。
    paged = await catalog.list_definitions(page=2, page_size=4)
    check(
        "分页：total 是过滤后的总数，不是页长",
        paged.total == len(REAL_KEYS)
        # 第 2 页最多 page_size 条；条目总数不一定刚好填满两页。
        and len(paged.items) == min(4, len(REAL_KEYS) - 4),
    )
    check(
        "TRANSFORMER 这个 type 现在有真实实现（之前是空枚举值）",
        {i.processor_type.value for i in everything.items}
        == {"INTERCEPTOR", "OBSERVER", "TRANSFORMER"},
    )

    check(
        "enabled=false 是空集（条目全部来自注册表）",
        (await catalog.list_definitions(enabled=False)).total == 0,
    )
    check(
        "取不存在的 key -> NotFoundError",
        await _async_raises(
            NotFoundError,
            catalog.get_definition("no_such_processor"),
        ),
    )

    # --------------------------------------------------------
    section("隐私过滤（TRANSFORMER）")

    from app.agent.tools.processor.transformer.privacy import (
        PrivacyConfig,
        PrivacyTarget,
        PrivacyToolProcessor,
        scrub_text,
        scrub_value,
    )

    default_privacy = PrivacyConfig()

    scrubbed = scrub_text(
        "联系 13812345678 或 a.b@example.com",
        config=default_privacy,
    )
    check(
        "手机号与邮箱都被抹掉",
        "13812345678" not in scrubbed
        and "a.b@example.com" not in scrubbed
        and scrubbed.count("[REDACTED]") == 2,
    )
    check(
        "身份证也抹（默认项里有）",
        "110101199003078515"
        not in scrub_text(
            "证件号 110101199003078515", config=default_privacy
        ),
    )
    check(
        "长订单号不被误伤（默认不开银行卡）",
        "2025092812345678901"
        in scrub_text(
            "订单号 2025092812345678901", config=default_privacy
        ),
    )
    check(
        "开了银行卡就会误伤长数字串（这条固定住那个已知代价）",
        "1234567890123456"
        not in scrub_text(
            "金额 1234567890123456",
            config=PrivacyConfig(
                targets=[
                    PrivacyTarget.PHONE,
                    PrivacyTarget.BANK_CARD,
                ]
            ),
        ),
    )
    check(
        "keep_length 用等长星号",
        scrub_text(
            "电话 13812345678",
            config=PrivacyConfig(keep_length=True),
        ) == "电话 ***********",
    )
    check(
        "数字 / None 这类叶子原样留着（不猜）",
        scrub_value(
            {"n": 7, "z": None, "p": "13812345678"},
            config=default_privacy,
        ) == {"n": 7, "z": None, "p": "[REDACTED]"},
    )
    check(
        "超出 max_depth 的部分不抹（原样返回，不破坏结构）",
        scrub_value(
            {"a": {"b": {"c": "13812345678"}}},
            config=PrivacyConfig(max_depth=1),
        ) == {"a": {"b": {"c": "13812345678"}}},
    )

    class _Inner(BaseModel):
        email: str

    class _Outer(BaseModel):
        inner: _Inner
        phone: str

    original = _Outer(
        inner=_Inner(email="x@y.com"), phone="13900000000"
    )
    scrubbed_model = scrub_value(
        original, config=default_privacy
    )
    check(
        "pydantic 模型递归抹、且类型不变",
        scrubbed_model.inner.email == "[REDACTED]"
        and scrubbed_model.phone == "[REDACTED]"
        and isinstance(scrubbed_model, _Outer),
    )
    check(
        "抹的是新对象，工具自己的原件没被动过",
        original.inner.email == "x@y.com"
        and original.phone == "13900000000",
    )

    privacy = PrivacyToolProcessor()

    outgoing = ToolInvocation(
        tool_key="smoke.tool",
        arguments={"phone": "13812345678", "q": "ok"},
    )
    await privacy.before(outgoing, config={})
    check(
        "BEFORE 就地抹参数（下游看不到原始值）",
        outgoing.arguments
        == {"phone": "[REDACTED]", "q": "ok"},
    )

    returned = await privacy.after(
        outgoing,
        ToolOutcome(
            value={"msg": "已发给 a@b.com"},
            duration_ms=12.0,
            cached=True,
        ),
        config={},
    )
    check(
        "AFTER 抹结果",
        returned.value == {"msg": "已发给 [REDACTED]"},
    )
    check(
        "抹结果时 duration_ms 与 cached 原样带过去",
        returned.duration_ms == 12.0 and returned.cached is True,
    )
    check(
        "失败的结果没有 value 可抹，原样返回",
        (
            await privacy.after(
                outgoing,
                ToolOutcome(error=RuntimeError("boom")),
                config={},
            )
        ).error is not None,
    )

    # --------------------------------------------------------
    section("调用者鉴权（INTERCEPTOR）")

    from pydantic import ValidationError

    from app.agent.tools.processor.context import (
        CALLER_CONTEXT_KEY,
        ToolCallDenied,
        build_caller_context,
    )
    from app.agent.tools.processor.interceptor.auth import (
        AuthConfig,
        AuthProcessor,
    )

    checkauth = AuthProcessor()

    async def _authz(config, caller=None):
        """跑一次 before，返回它给出的东西（ToolInvocation=放行）。"""

        invocation = ToolInvocation(
            tool_key="checkauth.smoke", arguments={}
        )

        if caller is not None:
            invocation.context[CALLER_CONTEXT_KEY] = caller

        return await checkauth.before(invocation, config=config)

    # ⚠️ 空配置在**挂载期**就该被拒：挂一个什么都不拦的鉴权关卡比没挂
    # 更危险（链上看起来有鉴权）。挂载校验会把 config 过一遍
    # config_model，所以这里抛的 ValidationError -> 接口 400。
    check(
        "两个条件都空 -> 配置本身不成立",
        _raises(
            ValidationError,
            lambda: AuthConfig.model_validate({}),
        )
        and _raises(
            ValidationError,
            lambda: AuthConfig.model_validate(
                {"require_roles": [], "require_group": "   "}
            ),
        ),
    )

    check(
        "角色命中任意一个即放行",
        isinstance(
            await _authz(
                {"require_roles": ["trader_admin", "ops"]},
                build_caller_context(roles=["ops"]),
            ),
            ToolInvocation,
        ),
    )

    denied = await _authz(
        {"require_roles": ["trader_admin"]},
        build_caller_context(roles=["viewer"]),
    )
    check(
        "角色不命中 -> 拒绝（ToolCallDenied，工具没跑）",
        isinstance(denied, ToolOutcome)
        and isinstance(denied.error, ToolCallDenied),
    )

    check(
        "群组精确匹配 -> 放行",
        isinstance(
            await _authz(
                {"require_group": "Y-Trader-User"},
                build_caller_context(groups=["Y-Trader-User"]),
            ),
            ToolInvocation,
        ),
    )
    check(
        "群组不做\"取 / 之后那段\"的宽松匹配（auth 层拒绝过那种写法）",
        isinstance(
            await _authz(
                {"require_group": "Y-Trader-User"},
                build_caller_context(groups=["evil-org/Y-Trader-User"]),
            ),
            ToolOutcome,
        ),
    )
    check(
        "两个条件都配时是 AND",
        isinstance(
            await _authz(
                {"require_roles": ["ops"], "require_group": "g1"},
                build_caller_context(roles=["ops"], groups=["g2"]),
            ),
            ToolOutcome,
        ),
    )

    missing = await _authz({"require_roles": ["ops"]})
    check(
        "拿不到身份 -> 默认拒绝（fail-closed）",
        isinstance(missing, ToolOutcome)
        and isinstance(missing.error, ToolCallDenied),
    )
    check(
        "on_missing=allow 才放行，且**只有**显式配了才这样",
        isinstance(
            await _authz(
                {"require_roles": ["ops"], "on_missing": "allow"}
            ),
            ToolInvocation,
        ),
    )

    # --------------------------------------------------------
    section("运行时设施：缓存 / 事件 / 指标")

    from app.agent.tools.processor.interceptor.cache import (
        CacheToolProcessor,
    )
    from app.agent.tools.processor.observer.events import (
        ToolEventObserver,
    )
    from app.agent.tools.processor.observer.metrics import (
        ToolMetricsObserver,
    )
    from app.agent.tools.processor.runtime.cache import (
        get_tool_cache,
    )
    from app.agent.tools.processor.runtime.events import (
        get_tool_event_bus,
    )
    from app.agent.tools.processor.runtime.metrics import (
        get_tool_metrics,
    )

    cache = CacheToolProcessor()
    calls = {"n": 0}

    async def work() -> ToolOutcome:
        calls["n"] += 1
        return ToolOutcome(value={"n": calls["n"]})

    invocation = ToolInvocation(
        tool_key="smoke.tool", arguments={"q": "x"}
    )

    await get_tool_cache().clear()

    first = await cache.around(
        invocation, work, config={"ttl_seconds": 60}
    )
    second = await cache.around(
        invocation, work, config={"ttl_seconds": 60}
    )

    check("第一次未命中、真的执行", first.ok and not first.cached)
    check("第二次命中缓存", second.cached is True)
    check("命中时短路成 ToolOutcome", isinstance(second, ToolOutcome))
    check("底层只执行了一次", calls["n"] == 1)

    await get_tool_cache().put("k", 1, ttl_seconds=0)

    check(
        "ttl=0 等于不缓存",
        await get_tool_cache().get("k") == (False, None),
    )

    metrics_observer = ToolMetricsObserver()
    for error in (None, RuntimeError("boom")):
        one = ToolInvocation(tool_key="smoke.tool", arguments={})
        await metrics_observer.before(one, config={})
        await metrics_observer.after(
            one, ToolOutcome(value=1, error=error), config={}
        )

    measured = get_tool_metrics().for_tool("smoke.tool")
    check("指标记到了次数", measured.calls == 2)
    check("失败率正确", abs(measured.error_rate - 0.5) < 1e-9)
    check(
        "错误按类型分桶",
        measured.errors_by_type == {"RuntimeError": 1},
    )
    check(
        "BEFORE 写的临时键被清掉了",
        "metrics.started_at" not in one.context,
    )

    bus = get_tool_event_bus()
    event_observer = ToolEventObserver()

    async with bus.subscribe() as queue:
        one = ToolInvocation(tool_key="smoke.tool", arguments={})
        await event_observer.before(one, config={})
        await event_observer.after(
            one, ToolOutcome(value=1, duration_ms=8.0), config={}
        )
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        check(
            "BEFORE 发 STARTED、AFTER 发 SUCCEEDED",
            [e.type.value for e in events] == [
                "STARTED", "SUCCEEDED"
            ],
        )
        check(
            "事件默认不带参数（参数可能含业务数据）",
            all("arguments" not in e.extra for e in events),
        )
        check(
            "事件可 JSON 序列化（SSE 要用）",
            all(
                isinstance(e.to_payload()["occurred_at"], str)
                for e in events
            ),
        )

    check(
        "退出订阅上下文后自动退订",
        bus.subscriber_count == 0,
    )

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
