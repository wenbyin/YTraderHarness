"""
把一条 `tool_processor` 链**编译**成嵌套的调用。

纯函数，不碰数据库、不碰注册表 —— 库里的行与"键 -> 实现"的映射都由
调用方取好传进来（`ToolExecutor` 做那件事）。这样链的语义可以脱离
DB 单独验证，而链的语义恰恰是最容易写错的那部分。

# 一次工具调用长什么样

    BEFORE  处理器先跑；它若短路（缓存命中 / 审批拒绝），工具**不会被调用**
    AROUND  包住它之后的全部内容 + 工具调用本身，可以调 0 次或多次
    AFTER   内层跑完之后跑，拿到 outcome，只读或改写它

编译的方式是**折叠**：从链尾往前，每一层把"内层"当作自己的 next_call。

# ⚠️ AROUND 的两条记录：本模块最需要说清的一件事

`validate_onion`（服务层，挂载期）规定 AROUND 在链上是**两条记录**
（先进 + 后出），两条记录构成的区间必须正确嵌套、不得交叉。它把
AROUND 当成一个"区间"来校验。

但**执行侧只可能有一个调用点**：工具调用只发生一次。两个兄弟 AROUND
（区间不相交）都想"包住这次调用"——这在语义上就无法同时成立。所以
严格按区间来解释是解释不通的；必须选一个确定的读法：

    第一次出现 = 进入：调一次 around()，next_call 是它**之后直到链尾**
                的全部内容（含工具调用）
    第二次出现 = 退出标记：执行期 no-op，它的作用只在挂载期供
                validate_onion 校验嵌套

两个好处：

    嵌套的 AROUND 与 validate_onion 的语义**完全一致**
        （校验保证区间嵌套时，"包裹其后的全部"与"包裹区间内部"
          在工具调用点上是同一个结果）
    兄弟 AROUND 得到确定的行为
        而不是"取决于怎么解释"，也不必为它造一套更复杂的模型

已否决的两条路：

    把退出标记当成第二个 AROUND 再包一层
        同一个实现会被 around() 两次。缓存处理器会重复探针；
        而任何重试实现会**指数级**重试（2 层 = 4 次，3 层 = 8 次）。
    报错拒绝兄弟 AROUND
        它并不违反任何既有约束（validate_onion 明确只禁交叉），
        凭空造一条挂载期没写过的禁令，只会让"配得出来但跑不了"。

# ASYNC 处理器为什么不参与嵌套

`ExecutionMode.ASYNC` 的语义是"起了就不等、返回值丢弃"（见
base.py 的 process 说明）。一个不等返回值的处理器无法"包住"任何东西。

好在它**不可能**是 AROUND：`BaseToolProcessor.__init_subclass__` 在
类体执行时就要求 AROUND 必须支持 SYNC，数据库那侧还有一条
CHECK 约束。所以 ASYNC 只可能是 BEFORE / AFTER 的观察者，被单独
挑出来在链外起掉（见 tool_executor.py）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.agent.tools.processor.base import BaseToolProcessor
from app.agent.tools.processor.context import (
    AsyncToolCall,
    ToolInvocation,
    ToolOutcome,
)
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

__all__ = [
    "ChainEntry",
    "CompiledChain",
    "build_chain",
    "compile_chain",
]

logger = logging.getLogger(__name__)

#: 一个键在链上最多允许几条 AROUND 记录 —— 一对（先进 + 后出）。
_AROUND_RECORDS = 2


@dataclass(frozen=True)
class ChainEntry:
    """链上一个**位置**的处理器（AROUND 因此有两个）。"""

    #: 处理器实现实例 —— 进程级共享单例，**不要**往上挂 runtime。
    processor: BaseToolProcessor

    #: 注册键。错误信息里用它，比 `type(processor).__name__` 好查：
    #: 库里挂的是这个键。
    processor_key: str

    hook: ProcessorHook

    mode: ExecutionMode

    #: 挂载时写下的参数，已由服务层按 config_model 归一化过。
    config: dict[str, Any]

    #: 在链里的位置，1 起。只用于错误信息与排序核对。
    sequence: int


@dataclass(frozen=True)
class CompiledChain:
    """
    拆好的链。

    `sync_entries` 按 sequence 升序，交给 `compile_chain` 折叠；
    ASYNC 的按 hook 分成两组，在链外起掉 —— 分好组是因为
    "BEFORE 的观察者"要在调用前起、"AFTER 的"要在调用后起，
    而它们各自不需要知道对方。
    """

    sync_entries: tuple[ChainEntry, ...] = ()

    async_before: tuple[ChainEntry, ...] = ()

    async_after: tuple[ChainEntry, ...] = ()


def build_chain(
        rows: Sequence[Any],
        *,
        resolve_processor: Callable[[str], BaseToolProcessor | None],
) -> CompiledChain:
    """
    库里的行 -> `CompiledChain`。

    Parameters
    ----------
    rows
        挂载记录，**任意带这几个属性的对象即可**（真实调用方传
        `ToolProcessor` 实体）：`processor_key` / `hook` /
        `execution_mode` / `config` / `sequence` / `enabled`。
        用鸭子类型而不是导入那个实体，是为了让本模块保持"纯"——
        它不该为了一个类型标注把服务层的 domain 拖进来。
    resolve_processor
        键 -> 实现实例。取不到（实现已从代码里删掉）返回 None，
        那一行**跳过并记 WARNING**：行本身是该保留的（历史配置、
        审计），但没有实现就没法执行它。

    Raises
    ------
    ValueError
        链自身不自洽。这三种都在挂载期能防住，但约束是**分别**写在
        类体、CHECK 约束与 validate_onion 里的 —— 一旦有人手改库、
        或将来换了写入路径，这里是最后一道闸。信息里带 sequence，
        因为"哪一行"才是能动手修的东西。
    """

    entries: list[ChainEntry] = []

    for row in rows:

        if not getattr(row, "enabled", True):
            continue

        key = str(getattr(row, "processor_key", "") or "")

        processor = resolve_processor(key)

        if processor is None:
            logger.warning(
                "处理器 %r（第 %s 位）在代码里没有对应实现，已跳过。",
                key,
                getattr(row, "sequence", "?"),
            )
            continue

        # 归一成枚举：库里的行经实体转换后已经是枚举，但写这块的
        # 人手上也可能拿到裸字符串（假仓储、手工构造的行）。
        # `ProcessorHook(hook)` 对枚举成员是恒等，对非法值抛
        # ValueError —— 后者正是我们想要的：能静态发现的错不要
        # 留到运行期变成"走错了分支"。
        entries.append(
            ChainEntry(
                processor=processor,
                processor_key=key,
                hook=ProcessorHook(row.hook),
                mode=ExecutionMode(row.execution_mode),
                config=dict(getattr(row, "config", None) or {}),
                sequence=int(getattr(row, "sequence", 0) or 0),
            )
        )

    _validate(entries)

    sync_entries: list[ChainEntry] = []
    async_before: list[ChainEntry] = []
    async_after: list[ChainEntry] = []

    for entry in entries:

        if entry.mode is not ExecutionMode.ASYNC:
            sync_entries.append(entry)
            continue

        # 上面 _validate 已经保证 ASYNC 不会是 AROUND，这里只分前后。
        if entry.hook is ProcessorHook.BEFORE:
            async_before.append(entry)
        else:
            async_after.append(entry)

    return CompiledChain(
        sync_entries=tuple(sync_entries),
        async_before=tuple(async_before),
        async_after=tuple(async_after),
    )


def _validate(entries: Sequence[ChainEntry]) -> None:
    """
    执行期的链必须自洽。见 `build_chain` 的 Raises。

    ⚠️ 与 `validate_onion` 的分工：那个管**嵌套**（交叉要报错），
    且刻意放行"只有一条 AROUND 记录"—— 那是挂载中的合法中间态
    （POST 一次只建一条）。而**执行期**的链不能是半条，否则
    `compile_chain` 会把一个没有退出标记的 AROUND 当成普通的包裹层，
    行为与挂载者以为的不同，且没有任何迹象。
    """

    around_counts: dict[str, int] = {}

    for entry in entries:

        # ⚠️ 这里**不**校验"同一个键只能有一种 hook"。
        #
        # 那看起来像个合理的自洽性检查，其实会误杀正常配置：
        # privacy_transformer 同时挂 BEFORE + AFTER（抹参数 + 抹结果），
        # 一个实现占多个 hook 位置是设计的一部分。
        #
        # "这个 hook 该实现支持吗"是挂载期的事（服务层的
        # _validate_mount 按 supported_hooks 判），执行期再查一遍
        # 只会把配置错误推迟到运行期才炸。

        if entry.hook is ProcessorHook.AROUND:
            around_counts[entry.processor_key] = (
                around_counts.get(entry.processor_key, 0) + 1
            )

        # ASYNC 不能是 AROUND：不等返回值的处理器包不住任何东西。
        # 类体与 DB 的 CHECK 都拦了这一条，这里兜底 —— 手改库、
        # 或将来换了写入路径时，报错要指向那一行。
        if (
                entry.hook is ProcessorHook.AROUND
                and entry.mode is ExecutionMode.ASYNC
        ):

            raise ValueError(
                f"处理器 {entry.processor_key!r}（第 {entry.sequence} 位）"
                "是 AROUND + ASYNC —— AROUND 要包住整次调用、必须在"
                "调用返回后才知道怎么收尾，因此只能是 SYNC。"
            )

    for key, count in around_counts.items():

        if count == _AROUND_RECORDS:
            continue

        raise ValueError(
            f"处理器 {key!r} 的 AROUND 记录有 {count} 条，"
            f"应该是 {_AROUND_RECORDS} 条（先进 + 后出）。"
            "半条链不能用 —— 常见成因是把其中一条 enabled 关掉了，"
            "或者挂载只做了一半。"
        )


def compile_chain(
        chain: CompiledChain,
        invocation: ToolInvocation,
        base: AsyncToolCall,
        *,
        runtime: Any = None,
) -> AsyncToolCall:
    """
    折叠成一条可调用的链，`base` 在最里层（真正执行工具的那一步）。

    ⚠️ 全程传**同一个** `invocation` 对象，不复制：处理器就地改
    `invocation.arguments`，而 `base` 在**被调用时**才读它 ——
    这既让 BEFORE 改的参数对工具生效，也让 AROUND 重试时每次
    `next_call()` 都拿到当前参数（见 context.py 的契约）。

    因此处理器**返回一个新的 ToolInvocation 是无效的**（`process()`
    只认就地修改）。这是既有契约，不是本模块的选择。
    """

    entries = chain.sync_entries
    entered: set[str] = set()

    def build(pos: int) -> AsyncToolCall:

        if pos >= len(entries):
            return base

        entry = entries[pos]

        if entry.hook is ProcessorHook.AROUND:

            if entry.processor_key in entered:
                # 退出标记。见模块 docstring 的"AROUND 的两条记录"。
                return build(pos + 1)

            entered.add(entry.processor_key)

        # BEFORE / AFTER 的语义差异不在这里 —— 它由
        # `BaseToolProcessor.process()` 按 hook 分发（BEFORE 先跑
        # before() 再决定要不要 next_call；AFTER 先 next_call() 再
        # after()）。本函数只负责**摆好嵌套**，这一层保持中立。
        inner = build(pos + 1)

        async def call(
                entry: ChainEntry = entry,
                next_call: AsyncToolCall = inner,
        ) -> ToolOutcome:

            return await entry.processor.process(
                entry.hook,
                invocation,
                next_call,
                config=entry.config,
                runtime=runtime,
            )

        return call

    return build(0)
