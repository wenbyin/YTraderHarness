"""
Processor 流水线：**既有 Builder 契约与新 Processor 契约之间的桥**。

它不新建执行框架，而是补上既有驱动缺的那一段
------------------------------------------
`BaseContextBuilder.build()` 已经有完整的一套调度：

    先跑 CONTEXT_RESOLVER
    校验 StageResolution.skipped（只能收窄、不能扩张）
    按 PIPELINE_TEMPLATE **固定顺序**跑剩下的阶段
    收尾校验"必须产出 ModelRequest"

因此本流水线**不重复**那些事情（顺序、阶段参与、收尾校验都留在原处）。
它补的是"每个阶段内部"：

    阶段 -> 该阶段上的 Processor（按 order）
        -> gate 判定（when，严格 bool）
        -> 输入绑定（State Path）
        -> 执行 + 观测 + 累积

两套契约定死的地方
----------------
    Builder 侧（不可变，来自 context_builder.yaml）
        enabled_stages / required_stages / PIPELINE_TEMPLATE 顺序
    Processor 侧（自描述，来自 spec.py）
        name / order / stage / when / inputs

它们在 `__init__` 里双向校验，**启动期**就炸：

    实际执行的顺序 ≠ 阶段模板顺序        -> 报错（顺序必须只有一个来源）
    required 阶段上没有 Processor        -> 报错（required 是句空话）
    required 阶段上的 Processor 带 gate  -> 报错（required = 不可被运行时跳过）
    非 required 阶段上的 Processor 无 gate -> 报错（那就没人决定它跑不跑）
    name / order 重复                     -> 报错（顺序与观测都会不确定）

最后四条一起把"gate 与 required 的矛盾"消灭在启动期，而不是留到某次
运行里表现为"配了开关却关不掉"。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.common.context import StageId

from .base import ContextProcessor
from .context import (
    BuildPhase,
    ContextBuildContext,
    ProcessorExecution,
    ProcessorStatus,
)
from .dependencies import ContextDependencies
from .errors import ContextBuildError, ContextProcessorError, ContextStateError
from .messages import is_human_message, is_tool_message
from .spec import ContextProcessorSpec
from .state import MISSING, resolve_inputs, resolve_state_path

__all__ = [
    "ContextProcessorPipeline",
    "PIPELINE_STAGE_ORDER",
]

logger = logging.getLogger(__name__)


#: 阶段的固定顺序（**唯一**来源是既有的 PIPELINE_TEMPLATE）。
#:
#: 直接从既有契约派生，而不是在这里再写一份列表：两份列表一定会漂移，
#: 而漂移的表现是"某个阶段的 Processor 在错误的时机执行" —— 那种错误
#: 在日志里看起来完全正常。
def _stage_order() -> tuple[StageId, ...]:

    from app.agent.context.builder.contracts import PIPELINE_TEMPLATE

    return PIPELINE_TEMPLATE


PIPELINE_STAGE_ORDER: tuple[StageId, ...] = _stage_order()

_STAGE_INDEX: Mapping[StageId, int] = {
    stage: index for index, stage in enumerate(PIPELINE_STAGE_ORDER)
}


@dataclass(frozen=True, slots=True)
class _StageSlot:
    """流水线里一个阶段上的 Processor 列表（按 order）。"""

    stage: StageId
    index: int
    processors: tuple[ContextProcessor, ...]


class ContextProcessorPipeline:
    """
    一组 Processor，按固定阶段顺序执行。

    ⚠️ 实例是**共享的**（一个 Builder 一个，进程里只有一个 Builder 实例）
    ---------------------------------------------------------------
    因此这里只放"从类声明推导出来的不可变东西"（排序后的 Processor 列表、
    校验结果），"这次调用的东西"全在 `ContextBuildContext` 里 —— 与
    `BaseContextBuilder` / `BaseToolProcessor` 的纪律一致。
    """

    def __init__(
            self,
            *,
            processors: Sequence[ContextProcessor],
            required_stages: Iterable[StageId] = (),
    ) -> None:

        self._processors = tuple(processors)
        self._required_stages = frozenset(required_stages)

        self._validate()

        slots: dict[StageId, list[ContextProcessor]] = {}

        for processor in self._ordered():
            slots.setdefault(processor.spec_of().stage, []).append(processor)

        self._slots: Mapping[StageId, _StageSlot] = {
            stage: _StageSlot(
                stage=stage,
                index=_STAGE_INDEX[stage],
                processors=tuple(items),
            )
            for stage, items in slots.items()
        }

    # ========================================================
    # 只读
    # ========================================================

    @property
    def processors(self) -> tuple[ContextProcessor, ...]:
        """全部 Processor，**执行顺序**。"""

        return self._ordered()

    @property
    def stages(self) -> tuple[StageId, ...]:
        """有 Processor 的阶段，按固定顺序。"""

        return tuple(
            stage
            for stage in PIPELINE_STAGE_ORDER
            if stage in self._slots
        )

    def processors_for(self, stage: StageId) -> tuple[ContextProcessor, ...]:
        """某个阶段上的 Processor（测试与目录接口用）。"""

        slot = self._slots.get(stage)

        return slot.processors if slot else ()

    def describe(self) -> list[dict[str, Any]]:
        """
        流水线自描述（进日志 / 将来的 `/context-builders` 目录）。

        字段名贴 `context_builder.yaml` 的 `StagePolicy` / `PipelineNode`，
        因此目录接口从**代码**派生时不必再猜形状。
        """

        return [
            {
                **processor.spec_of().describe(),
                "stage_index": _STAGE_INDEX[processor.spec_of().stage],
                "processor_class": type(processor).__name__,
                "display_name": processor.display_name
                or type(processor).__name__,
                "processor_description": processor.description,
            }
            for processor in self.processors
        ]

    # ========================================================
    # 打开一次构建
    # ========================================================

    def begin(self, state: Any) -> ContextBuildContext:
        """
        为一次 `build()` 建立工作集（在 CONTEXT_RESOLVER 阶段被调用）。

        它做三件事，都是"把请求翻译成可被 State Path 解析的根"：

            1. 组装 `base_state`：图 state ⊕ 运行期派生值（`run.*`）
            2. 判定执行层级（INITIAL / CONTINUATION，规格 §四）
            3. 取出依赖束（缺省 = 进程级默认）

        ⚠️ 派生值**不覆盖**图 state 里已有的键
        ------------------------------------
        `messages` / `current_user_message_id` 这类值：图 state 说了算。
        只有它没给的时候才由这里补（补的是"从 messages 推得出来"的那
        一份）。反向覆盖会让 Agent Builder 显式注入的值被静默顶掉。
        """

        context = self._new_context(state)

        state.processor_context = context

        return context

    # ========================================================
    # 跑一个阶段
    # ========================================================

    async def run_stage(
            self,
            stage: StageId,
            state: Any,
    ) -> None:
        """
        跑某个阶段上的全部 Processor。

        由 Builder 的 `_stage_*` 方法调用（每个阶段一个两行的方法），
        因此执行顺序仍然由**既有驱动**决定 —— 这里是它的执行体，不是
        第二个驱动。
        """

        context = getattr(state, "processor_context", None)

        if context is None:
            # 没经过 begin 就跑到阶段里：接线错误（比如某个实现只覆写了
            # 一半的阶段方法）。报错要说清这一点，而不是 AttributeError。
            raise ContextBuildError(
                f"阶段 {stage.value} 在没有工作集的情况下执行："
                "ContextProcessorPipeline.begin() 没有被调用。"
                "常见原因：Builder 的 _stage_context_resolver 没有委托给"
                "流水线。",
                stage=stage.value,
            )

        slot = self._slots.get(stage)

        if slot is None:
            # 该阶段没有 Processor（比如某个实现只声明了部分阶段）。
            # 这不是错误：Builder 声明的参与关系由它自己负责，
            # 流水线对没有 Processor 的阶段就是"什么都不做"。
            return

        for processor in slot.processors:
            await self._run_processor(processor, context, state)

    # ========================================================
    # internals —— 执行
    # ========================================================

    async def _run_processor(
            self,
            processor: ContextProcessor,
            context: ContextBuildContext,
            state: Any,
    ) -> None:

        spec = processor.spec_of()
        started = time.perf_counter()

        gate = self._gate(spec, context)

        if gate is not True:

            context.record(
                ProcessorExecution(
                    processor=spec.name,
                    order=spec.order,
                    stage=spec.stage.value,
                    when_path=spec.when_path,
                    when_value=gate,
                    status=ProcessorStatus.SKIPPED,
                    duration_ms=_ms(started),
                )
            )

            logger.debug(
                "上下文处理器 %s 被 gate 跳过（%s=%r）。",
                spec.name,
                spec.when_path,
                gate,
            )

            return

        bound = resolve_inputs(context.state_root(), spec.inputs)

        if bound.missing_required:

            raise ContextStateError(
                f"上下文处理器 {spec.name!r} 的必需输入不存在："
                f"{list(bound.missing_required)}。\n"
                + "\n".join(
                    f"    {name} <- {spec.inputs[name].path}"
                    for name in bound.missing_required
                )
                + "\n这些输入由 spec.inputs 声明（dot path）。缺失通常意味着"
                "上游（Agent Builder / 图 state）漏写了它们 —— "
                "不要让它静默变成 None：那会让模型看到一份缺东西的上下文"
                "却以为一切正常。",
                processor=spec.name,
                stage=spec.stage.value,
                details={
                    "missing": list(bound.missing_required),
                    "paths": {
                        name: spec.inputs[name].path
                        for name in bound.missing_required
                    },
                },
            )

        before = _snapshot_shape(context)

        try:
            returned = await processor.process(dict(bound.values), context)

        except ContextBuildError as exc:

            self._handle_failure(exc, processor, context, started, state)
            return

        except Exception as exc:  # noqa: BLE001 - 归一成上下文构建错误

            wrapped = ContextProcessorError(
                f"上下文处理器 {spec.name!r} 执行失败："
                f"{type(exc).__name__}: {exc}",
                processor=spec.name,
                stage=spec.stage.value,
            )

            self._handle_failure(wrapped, processor, context, started, state)
            return

        if isinstance(returned, ContextBuildContext):
            context = returned

        context.record(
            ProcessorExecution(
                processor=spec.name,
                order=spec.order,
                stage=spec.stage.value,
                when_path=spec.when_path,
                when_value=True,
                status=ProcessorStatus.EXECUTED,
                duration_ms=_ms(started),
                missing_inputs=tuple(sorted(bound.missing)),
                detail=_shape_delta(before, context, spec),
            )
        )

    def _handle_failure(
            self,
            error: ContextBuildError,
            processor: ContextProcessor,
            context: ContextBuildContext,
            started: float,
            state: Any,
    ) -> None:
        """
        处理一个 Processor 的失败。

        `spec.optional=True`（Memory / 外部检索那类旁路）
            -> 记 FAILED + 一条 warning，流水线**继续**
        否则
            -> 记 FAILED 后抛出（规格 §二十八：这类错误必须让本次 Run 失败）

        两种情况都先记录再决定：失败也是"这一次上下文构建长什么样"的
        一部分，丢了它 Replay 就缺一块。
        """

        spec = processor.spec_of()

        context.record(
            ProcessorExecution(
                processor=spec.name,
                order=spec.order,
                stage=spec.stage.value,
                when_path=spec.when_path,
                when_value=True,
                status=ProcessorStatus.FAILED,
                duration_ms=_ms(started),
                error=str(error),
            )
        )

        if spec.optional:

            message = f"{spec.name} 失败（已按可选处理）：{error}"

            context.warn(message)

            logger.warning(
                "上下文处理器 %s 失败，按可选处理继续：%s",
                spec.name,
                error,
                exc_info=True,
            )

            return

        logger.error(
            "上下文处理器 %s 失败，本次上下文构建中止：%s",
            spec.name,
            error,
        )

        # 把已经产生的执行记录带进异常：构建失败时也要能回答"跑到哪一步
        # 才坏的"，而那一刻没有快照产出。
        error.details.setdefault(
            "executions",
            [item.as_metadata() for item in context.executions],
        )

        if error.processor is None:
            error.processor = spec.name

        if error.stage is None:
            error.stage = spec.stage.value

        raise error

    def _gate(
            self,
            spec: ContextProcessorSpec,
            context: ContextBuildContext,
    ) -> Any:
        """
        判定一个 Processor 这次要不要执行（规格 §九，**强制**语义）。

            路径不存在            -> ContextStateError（构建失败）
            值是 True             -> 执行
            值是 False            -> 跳过
            值是别的任何东西       -> ContextStateError（**不做隐式 bool**）

        注意 `None` 落在最后一档：`bool(None) is False` 看起来像"跳过"，
        但它的真实含义通常是"上游忘了赋值"，静默跳过正是最难查的那类问题
        （规格 §九情况 4）。
        """

        if spec.when is None:
            return True

        raw = resolve_state_path(context.state_root(), spec.when.path)

        if raw is MISSING:

            raise ContextStateError(
                f"上下文处理器 {spec.name!r} 的 gate 字段不存在："
                f"{spec.when.path}。\n"
                "gate 必须能在 State 里解析到一个 bool —— 缺失意味着"
                "策略层（Agent Builder / Context Policy）没有给出这个开关。\n"
                "**不会**静默跳过：那样一次漏配会表现为"
                "“这个处理器这次恰好没跑”，而没人会发现。",
                processor=spec.name,
                stage=spec.stage.value,
                details={"when": spec.when.path},
            )

        if not isinstance(raw, bool):

            raise ContextStateError(
                f"上下文处理器 {spec.name!r} 的 gate 必须是 bool："
                f"{spec.when.path} = {raw!r}"
                f"（{type(raw).__name__}）。\n"
                "**不做隐式 bool 转换**：'yes' / 1 / None 这类值说明"
                "State 写错了，而 bool('yes') == True 会把一个错误的状态"
                "悄悄变成「照常执行」。",
                processor=spec.name,
                stage=spec.stage.value,
                details={"when": spec.when.path, "value": repr(raw)},
            )

        return raw

    # ========================================================
    # internals —— 组装与校验
    # ========================================================

    def _new_context(self, state: Any) -> ContextBuildContext:
        """把请求翻译成工作集（见 begin 的说明）。"""

        request = state.request

        graph_state = dict(
            getattr(request, "graph_state", None) or {}
        )

        messages = list(
            graph_state.get("messages") or getattr(request, "messages", ()) or ()
        )

        base_state: dict[str, Any] = dict(graph_state)

        # 派生值只在图 state 没给的时候补（见 begin 的说明）。
        base_state.setdefault("messages", messages)

        current_user_message, current_user_message_id = _current_user_message(
            messages
        )

        base_state.setdefault("current_user_message", current_user_message)

        base_state.setdefault(
            "current_user_message_id", current_user_message_id
        )

        phase = _phase(messages)

        conversation_id = _optional_str(
            base_state.get("conversation_id")
            or getattr(request, "thread_id", None)
        )

        model_role = (
            getattr(request, "model_role", "") or ""
        )

        base_state["run"] = {
            **dict(base_state.get("run") or {}),
            "thread_id": getattr(request, "thread_id", None),
            "run_id": _optional_str(getattr(request, "run_id", None)),
            "agent_type": getattr(request, "agent_type", "") or "",
            "builder_type": _builder_type(request),
            "model_role": model_role,
            "phase": phase.value,
            "conversation_id": conversation_id,
            "has_tool_activity": phase is BuildPhase.CONTINUATION,
        }

        dependencies = getattr(request, "dependencies", None)

        if not isinstance(dependencies, ContextDependencies):
            dependencies = None

        return ContextBuildContext(
            base_state=base_state,
            dependencies=dependencies,
            agent_type=str(getattr(request, "agent_type", "") or ""),
            builder_type=_builder_type(request),
            model_role=model_role,
            phase=phase,
            conversation_id=conversation_id,
            run_id=_optional_str(getattr(request, "run_id", None)),
        )

    def _ordered(self) -> tuple[ContextProcessor, ...]:

        return tuple(
            sorted(
                self._processors,
                key=lambda processor: (
                    _STAGE_INDEX[processor.spec_of().stage],
                    processor.spec_of().order,
                    processor.spec_of().name,
                ),
            )
        )

    def _validate(self) -> None:
        """
        启动期校验（见模块 docstring 的五条）。

        全部在这里炸，而不是留到某次运行：
            "顺序只有一个来源"与"gate 与 required 不矛盾"是**声明层面**
            的事实，运行期发现只会得到一个看起来正常、偶尔不对的流水线。
        """

        specs = [processor.spec_of() for processor in self._processors]

        names: dict[str, int] = {}
        orders: dict[int, str] = {}

        for spec in specs:

            if spec.name in names:
                raise ContextBuildError(
                    f"上下文处理器名重复：{spec.name!r}。"
                    "name 是注册键，也是观测里的 processor 名 —— "
                    "重复会让观测分不清是哪一步。"
                )

            names[spec.name] = spec.order

            if spec.order in orders:
                raise ContextBuildError(
                    f"上下文处理器 order 重复：{spec.order} 同时用于 "
                    f"{orders[spec.order]!r} 与 {spec.name!r}。\n"
                    "order 必须全局唯一：它同时是执行顺序与观测里的排序键。"
                )

            orders[spec.order] = spec.name

        # ---- 顺序必须与阶段模板一致 ---------------------------

        by_order = [spec.order for spec in sorted(specs, key=lambda s: s.order)]
        by_stage = [
            spec.order
            for spec in sorted(
                specs,
                key=lambda s: (_STAGE_INDEX[s.stage], s.order, s.name),
            )
        ]

        if by_order != by_stage:
            raise ContextBuildError(
                "上下文处理器的 order 与固定阶段顺序不一致：\n"
                f"    按 order 排：{[f'{s.name}={s.order}' for s in sorted(specs, key=lambda s: s.order)]}\n"
                f"    按阶段排：{[f'{s.name}={s.order}' for s in sorted(specs, key=lambda s: (_STAGE_INDEX[s.stage], s.order, s.name))]}\n"
                "阶段顺序是服务端定义且不可变的（context_builder.yaml），"
                "但 order 也是自描述的一部分 —— 两者必须说同一件事，"
                "否则“谁先跑”就有两个来源。"
            )

        # ---- gate 与 required 必须自洽 ------------------------

        for spec in specs:

            required = spec.stage in self._required_stages

            if required and not spec.unconditional:
                raise ContextBuildError(
                    f"阶段 {spec.stage.value} 被声明为 required，但它上面的"
                    f"处理器 {spec.name!r} 配了 gate（{spec.when_path}）。\n"
                    "required 的含义是「不可被运行时跳过」，而 gate 正是"
                    "运行时的跳过开关 —— 两者只能选一个。"
                )

            if spec.unconditional and not required:
                raise ContextBuildError(
                    f"处理器 {spec.name!r} 声明了无条件执行（when=None），"
                    f"但它所在的阶段 {spec.stage.value} **不是** required。\n"
                    "无条件执行只留给「没有它整个构建就不成立」的阶段"
                    "（入口与出口）—— 其它阶段必须有一个明确的策略开关，"
                    "否则「它这次跑不跑」就没有人能回答。"
                )

        for stage in self._required_stages:

            if stage not in {spec.stage for spec in specs}:
                raise ContextBuildError(
                    f"阶段 {stage.value} 被声明为 required，"
                    "但它上面一个处理器都没有。\n"
                    "required 是句空话，而「必须执行」这件事就这样消失了。"
                )

        for processor in self._processors:

            if not isinstance(processor, ContextProcessor):
                raise ContextBuildError(
                    f"流水线里出现了不是 ContextProcessor 的东西："
                    f"{processor!r}。"
                )

    # ========================================================

    def validate_names(self, names: Iterable[str]) -> None:
        """断言这些名字都在流水线里（给"必须声明某个 Processor"的场景用）。"""

        known = {spec.name for spec in (p.spec_of() for p in self._processors)}

        missing = sorted(set(names) - known)

        if missing:
            raise ContextBuildError(
                f"流水线里没有这些处理器：{missing}。现有的：{sorted(known)}。"
            )


# ============================================================
# internals
# ============================================================


def _ms(started: float) -> int:

    return int((time.perf_counter() - started) * 1000)


def _builder_type(request: Any) -> str:

    value = getattr(request, "builder_type", None)

    return str(getattr(value, "value", value) or "")


def _optional_str(value: Any) -> str | None:

    if value is None:
        return None

    text = str(value)

    return text or None


def _current_user_message(
        messages: Sequence[Any],
) -> tuple[Any, str | None]:
    """
    当前这一轮的用户消息（= 最后一条 Human，及其 id）。

    为什么"当前用户消息"要从 messages 里推，而不是让调用方一定传进来：
    它是**派生事实**（对话里最后一句人话），而"当前这一轮"的定义在图里
    本来就只有一个。调用方想显式指定时，图 state 里放
    `current_user_message_id` 即可覆盖（见 `_new_context` 的 setdefault）。
    """

    for index in range(len(messages) - 1, -1, -1):

        message = messages[index]

        if is_human_message(message):

            identifier = getattr(message, "id", None)

            return message, identifier if isinstance(identifier, str) else None

    return None, None


def _phase(messages: Sequence[Any]) -> BuildPhase:
    """
    判定执行层级（规格 §四）。

        最后一条 Human 之后出现过 ToolMessage -> CONTINUATION
        否则                                  -> INITIAL

    为什么用这个判据而不是"state 里有没有上一次的 context_result"：
    同一 thread 的**下一轮**对话里，上一次运行的 context_result 还留在
    checkpoint 里，用它会把新的一轮误判成 continuation。而"最后一条用户
    消息之后有工具结果"精确对应"ReAct 循环正在跑"这件事（包括从
    interrupt 恢复的场景）。
    """

    last_human = -1

    for index, message in enumerate(messages):

        if is_human_message(message):
            last_human = index

    for message in messages[last_human + 1:]:

        if is_tool_message(message):
            return BuildPhase.CONTINUATION

    return BuildPhase.INITIAL


def _snapshot_shape(context: ContextBuildContext) -> dict[str, int]:
    """执行前的形状（用于算 delta）。"""

    return {
        "sections": len(context.sections),
        "history": len(context.history),
        "tools": len(context.tools),
        "required": len(context.required_messages),
        "overlay": len(context.overlay),
    }


def _shape_delta(
        before: Mapping[str, int],
        context: ContextBuildContext,
        spec: ContextProcessorSpec,
) -> dict[str, Any]:
    """
    这次执行往上下文里加了什么（观测用，不依赖 Processor 自觉上报）。

    它让 Replay 能回答"这一步到底改变了什么"，而不必去比对两次快照。
    """

    after = _snapshot_shape(context)

    detail: dict[str, Any] = {
        "sections_added": after["sections"] - before["sections"],
        "history_messages": after["history"],
        "tools": after["tools"],
        "required_messages": after["required"],
        "overlay_added": after["overlay"] - before["overlay"],
        "inputs": sorted(spec.inputs),
    }

    # Processor 自己的结论（它写在 `context.metadata[自己的名字]` 里）。
    #
    # 与上面那些**算出来**的形状放一起，于是 Replay 既有"改变了什么"、
    # 也有"它自己怎么判断的"（比如 compaction 的 threshold 与
    # compacted 标志）。约定的键名就是 spec.name —— 与观测里的
    # processor 名一致，不必再记第二套命名。
    outcome = context.metadata.get(spec.name)

    if isinstance(outcome, Mapping):
        detail["outcome"] = dict(outcome)

    return detail
