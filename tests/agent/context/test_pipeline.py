"""
流水线行为（规格 §十、§十四、§二十六 + §二十九 的 Processor Pipeline 用例）。

覆盖：order 稳定执行 / gate 四态 / required input / 累积传递 /
可选失败 / 观测记录 / 启动期校验。

这里**不**通过 Builder 跑，而是直接驱动 `ContextProcessorPipeline`
（用自造的小 Processor）—— 那才能精确控制"第几步做什么"，而 Builder
那条路另有 test_agent_invoke_builder.py 覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.agent.context.processor.base import ContextProcessor
from app.agent.context.processor.context import ProcessorStatus

from app.agent.context.processor.errors import (
    ContextBuildError,
    ContextStateError,
)
from app.agent.context.processor.pipeline import ContextProcessorPipeline
from app.agent.context.processor.spec import (
    ContextProcessorSpec,
    StateBinding,
)
from app.common.context import ContextBuilderType, StageId


# ============================================================
# 测试用的最小 Processor
# ============================================================


@dataclass
class _State:
    """假的 `ContextBuildState`（流水线只用到 request / processor_context）。"""

    request: Any
    processor_context: Any = None


@dataclass
class _Request:
    """
    假的 `ContextBuildRequest`。

    直接驱动流水线时不必造真请求 —— 只要它有 `graph_state` 那几个字段。
    """

    graph_state: dict[str, Any] = field(default_factory=dict)
    messages: tuple[Any, ...] = ()
    thread_id: str = "t1"
    run_id: Any = None
    agent_type: str = "test"
    model_role: str = "reasoning"
    dependencies: Any = None
    builder_type: ContextBuilderType = ContextBuilderType.AGENT_INVOKE


class RecordingProcessor(ContextProcessor):
    """把"我执行了"写进 overlay，并可选产出/失败。"""

    spec = ContextProcessorSpec(
        name="recorder",
        order=20,
        stage=StageId.PROFILE_LOADER,
        when="context_policy.record",
        inputs={"value": StateBinding("payload", required=False)},
    )

    def __init__(self, *, label: str = "recorder", order: int = 20,
                 stage: StageId = StageId.PROFILE_LOADER, **_) -> None:

        self._label = label
        self.spec = ContextProcessorSpec(
            name=label,
            order=order,
            stage=stage,
            when="context_policy.record",
            inputs={"value": StateBinding("payload", required=False)},
        )

    async def process(self, inputs, context):
        context.set(f"ran.{self._label}", inputs.get("value"))
        return context


class BoomProcessor(ContextProcessor):
    """总是抛错的 Processor（用来测 Fatal / Optional）。"""

    spec = ContextProcessorSpec(
        name="boom",
        order=20,
        stage=StageId.PROFILE_LOADER,
        when="context_policy.record",
        inputs={},
    )

    def __init__(self, *, label: str = "boom", optional: bool = False) -> None:

        self.spec = ContextProcessorSpec(
            name=label,
            order=20,
            stage=StageId.PROFILE_LOADER,
            when="context_policy.record",
            inputs={},
            optional=optional,
        )

    async def process(self, inputs, context):
        raise RuntimeError("炸了")


class ResolverStub(ContextProcessor):
    """入口：无条件执行，写一个 overlay 键供后面的 Processor 读。"""

    spec = ContextProcessorSpec(
        name="resolver_stub",
        order=10,
        stage=StageId.CONTEXT_RESOLVER,
        when=None,
        inputs={"overrides": StateBinding("context_policy", required=False)},
    )

    async def process(self, inputs, context):

        overrides = inputs.get("overrides") or {}

        context.set("context_policy", {
            "record": overrides.get("record", True),
            "extra": "由 resolver 写入",
        })

        return context


class AssemblerStub(ContextProcessor):
    """出口：无条件执行，产出 draft（内容来自 overlay，验证累积传递）。"""

    spec = ContextProcessorSpec(
        name="assembler_stub",
        order=80,
        stage=StageId.PROMPT_ASSEMBLER,
        when=None,
        inputs={"extra": StateBinding("context_policy.extra")},
    )

    async def process(self, inputs, context):

        from app.agent.model import ModelRequest

        context.draft = ModelRequest(messages=(inputs["extra"],))

        return context


def _pipeline(*processors, required=(StageId.CONTEXT_RESOLVER, StageId.PROMPT_ASSEMBLER)):
    return ContextProcessorPipeline(
        processors=processors,
        required_stages=required,
    )


def _build(pipeline, *, graph_state=None, dependencies=None):
    request = _Request(
        graph_state=dict(graph_state or {}),
        dependencies=dependencies,
    )
    state = _State(request=request)
    context = pipeline.begin(state)
    return state, context


async def _run(pipeline, state):
    """按固定阶段顺序跑完整条流水线（模拟既有驱动）。"""

    from app.agent.context.processor.pipeline import PIPELINE_STAGE_ORDER

    for stage in PIPELINE_STAGE_ORDER:
        await pipeline.run_stage(stage, state)


# ============================================================
# 1. order 稳定执行
# ============================================================


async def test_processors_run_in_order() -> None:
    """按 order 稳定执行（规格 §十四）。"""

    pipeline = _pipeline(
        AssemblerStub(),
        ResolverStub(),
    )

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    await _run(pipeline, state)

    assert [item.get_key() for item in pipeline.processors] == [
        "resolver_stub",
        "assembler_stub",
    ]

    # resolver 先跑：它写出的 overlay 被出口阶段读到（跨阶段累积的证据）
    assert context.overlay["context_policy"]["extra"] == "由 resolver 写入"
    assert context.draft is not None


async def test_order_must_match_stage_template() -> None:
    """
    order 与固定阶段顺序不一致 -> **启动期**报错。

    否则"谁先跑"就有两个来源（阶段模板与 order），而它们的冲突会在
    运行期表现为"某个处理器在错误的时机执行" —— 日志里看起来正常。
    """

    bad = RecordingProcessor(label="late", order=10, stage=StageId.PROFILE_LOADER)
    bad.spec = ContextProcessorSpec(
        name="late", order=90, stage=StageId.PROFILE_LOADER,
        when="context_policy.record",
    )

    early = RecordingProcessor(label="early", order=10)
    early.spec = ContextProcessorSpec(
        name="early", order=20, stage=StageId.SUMMARY_LOADER,
        when="context_policy.record",
    )
    with pytest.raises(ContextBuildError) as excinfo:
        _pipeline(ResolverStub(), bad, early, AssemblerStub())

    assert "order 与固定阶段顺序不一致" in str(excinfo.value)


def test_duplicate_names_and_orders_are_rejected() -> None:

    first = RecordingProcessor(label="a", order=20)
    second = RecordingProcessor(label="a", order=30)

    with pytest.raises(ContextBuildError) as excinfo:
        _pipeline(ResolverStub(), first, second, AssemblerStub())

    assert "重复" in str(excinfo.value)


def test_duplicate_order_is_rejected() -> None:

    first = RecordingProcessor(label="a", order=20)
    second = RecordingProcessor(label="b", order=20)

    with pytest.raises(ContextBuildError) as excinfo:
        _pipeline(ResolverStub(), first, second, AssemblerStub())

    assert "order 重复" in str(excinfo.value)


def test_gate_and_required_must_be_consistent() -> None:
    """
    required 阶段上的处理器**不许**配 gate（规格 §九 + 既有契约的 required）。

    两个方向都校验：required 阶段的处理器必须无条件；无条件处理器必须
    在 required 阶段上。
    """

    gated_resolver = ResolverStub()
    gated_resolver.spec = ContextProcessorSpec(
        name="resolver_stub",
        order=10,
        stage=StageId.CONTEXT_RESOLVER,
        when="context_policy.record",  # <- required 阶段不该有 gate
    )

    with pytest.raises(ContextBuildError) as excinfo:
        _pipeline(gated_resolver, AssemblerStub())

    assert "不可被运行时跳过" in str(excinfo.value)

    unconditional = RecordingProcessor(label="always", order=20)
    unconditional.spec = ContextProcessorSpec(
        name="always", order=20, stage=StageId.PROFILE_LOADER, when=None,
    )

    with pytest.raises(ContextBuildError) as excinfo:
        _pipeline(ResolverStub(), unconditional, AssemblerStub())

    assert "required" in str(excinfo.value)
    assert "无条件执行" in str(excinfo.value)


def test_required_stage_without_processor_is_rejected() -> None:
    """required 阶段上一个处理器都没有 -> 那句 required 是空话。"""

    with pytest.raises(ContextBuildError):
        _pipeline(AssemblerStub())


# ============================================================
# 2. gate 四态
# ============================================================


async def test_gate_true_executes() -> None:
    """when=True -> EXECUTED。"""

    pipeline = _pipeline(ResolverStub(), RecordingProcessor(), AssemblerStub())

    state, context = _build(
        pipeline, graph_state={"context_policy": {"record": True}, "payload": 7}
    )

    await _run(pipeline, state)

    record = next(
        item for item in context.executions if item.processor == "recorder"
    )

    assert record.status is ProcessorStatus.EXECUTED
    assert record.when_path == "context_policy.record"
    assert record.when_value is True
    assert context.overlay["ran"]["recorder"] == 7


async def test_gate_false_skips() -> None:
    """when=False -> SKIPPED（合法行为，不是错误）。"""

    pipeline = _pipeline(ResolverStub(), RecordingProcessor(), AssemblerStub())

    state, context = _build(
        pipeline, graph_state={"context_policy": {"record": False}}
    )

    await _run(pipeline, state)

    record = next(
        item for item in context.executions if item.processor == "recorder"
    )

    assert record.status is ProcessorStatus.SKIPPED
    assert record.when_value is False
    assert "ran" not in context.overlay


async def test_gate_missing_field_fails_the_build() -> None:
    """
    情况 1：gate 路径不存在 -> **FAILED**（不能静默 skip）。
    """

    pipeline = _pipeline(ResolverStub(), RecordingProcessor(), AssemblerStub())

    # resolver 写出的 overlay 里没有 record 这个键
    state, context = _build(pipeline, graph_state={"context_policy": {}})

    # 先让 resolver 之外的东西也别写 record：手工清掉
    with pytest.raises(ContextStateError) as excinfo:
        await pipeline.run_stage(StageId.PROFILE_LOADER, state)

    assert "gate 字段不存在" in str(excinfo.value)
    assert excinfo.value.processor == "recorder"
    assert excinfo.value.details["when"] == "context_policy.record"


async def test_gate_none_fails_the_build() -> None:
    """
    情况 17（规格 §廿九）：when 为 None -> **FAILED**，不得静默 skip。

    `bool(None) == False` 看起来像"跳过"，但它的真实含义通常是"上游忘
    了赋值"。
    """

    pipeline = _pipeline(ResolverStub(), RecordingProcessor(), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    # 把 gate 的值改成 None（模拟上游赋值成了 None）
    context.overlay["context_policy"] = {"record": None}

    with pytest.raises(ContextStateError) as excinfo:
        await pipeline.run_stage(StageId.PROFILE_LOADER, state)

    assert "必须是 bool" in str(excinfo.value)
    assert "None" in str(excinfo.value)


async def test_gate_non_bool_fails_the_build() -> None:
    """
    情况 4：`"yes"` -> FAILED（**不做隐式 bool 转换**）。
    """

    pipeline = _pipeline(ResolverStub(), RecordingProcessor(), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    for bad in ("yes", 1, 0, [], {}):

        context.overlay["context_policy"] = {"record": bad}

        with pytest.raises(ContextStateError) as excinfo:
            await pipeline.run_stage(StageId.PROFILE_LOADER, state)

        assert "必须是 bool" in str(excinfo.value)


# ============================================================
# 3. required input
# ============================================================


async def test_required_input_missing_fails_the_build() -> None:
    """必需输入不存在 -> FAILED（规格 §十二）。"""

    class NeedsUser(ContextProcessor):
        spec = ContextProcessorSpec(
            name="needs_user",
            order=20,
            stage=StageId.PROFILE_LOADER,
            when="context_policy.record",
            inputs={"user_id": StateBinding("user_id")},
        )

        async def process(self, inputs, context):
            return context

    pipeline = _pipeline(ResolverStub(), NeedsUser(), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    with pytest.raises(ContextStateError) as excinfo:
        await pipeline.run_stage(StageId.PROFILE_LOADER, state)

    assert "必需输入不存在" in str(excinfo.value)
    assert excinfo.value.details["missing"] == ["user_id"]
    assert excinfo.value.details["paths"] == {"user_id": "user_id"}


async def test_optional_input_missing_does_not_fail() -> None:

    class OptionalUser(ContextProcessor):
        spec = ContextProcessorSpec(
            name="optional_user",
            order=20,
            stage=StageId.PROFILE_LOADER,
            when="context_policy.record",
            inputs={"user_id": StateBinding("user_id", required=False)},
        )

        async def process(self, inputs, context):
            context.set("saw", inputs["user_id"])
            return context

    pipeline = _pipeline(ResolverStub(), OptionalUser(), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    await pipeline.run_stage(StageId.PROFILE_LOADER, state)

    assert context.overlay["saw"] is None


# ============================================================
# 4. Processor 之间通过 overlay 传递
# ============================================================


async def test_context_is_passed_between_processors() -> None:
    """
    Processor 返回的 context 交给下一个（规格 §廿九 第 23 条）。

    传递的通道是 **overlay**（于是它也走 State Path 解析，而不是
    Python 对象在私有属性上传来传去）。
    """

    pipeline = _pipeline(ResolverStub(), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    await _run(pipeline, state)

    assert context.draft is not None
    assert context.draft.messages == ("由 resolver 写入",)
    assert state.processor_context is context


async def test_processor_may_return_a_replacement_context() -> None:
    """返回一个新的工作集也允许（规格 §六 的签名如此）。"""

    class Replacer(ContextProcessor):
        spec = ContextProcessorSpec(
            name="replacer",
            order=20,
            stage=StageId.PROFILE_LOADER,
            when="context_policy.record",
            inputs={},
        )

        async def process(self, inputs, context):

            context.warn("换了一个工作集")
            return context

    pipeline = _pipeline(ResolverStub(), Replacer(), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    await _run(pipeline, state)

    assert context.warnings == ["换了一个工作集"]


# ============================================================
# 5. Fatal vs Optional（规格 §二十八）
# ============================================================


async def test_optional_processor_failure_continues() -> None:
    """optional=True -> 记 FAILED + warning，流水线继续。"""

    pipeline = _pipeline(ResolverStub(), BoomProcessor(optional=True), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    await _run(pipeline, state)

    record = next(item for item in context.executions if item.processor == "boom")

    assert record.status is ProcessorStatus.FAILED
    assert "炸了" in (record.error or "")
    assert context.warnings and "boom" in context.warnings[0]

    # 后面的阶段照常跑完：draft 产出了
    assert context.draft is not None


async def test_fatal_processor_failure_aborts() -> None:
    """optional=False -> 抛出，且异常里带"跑到哪一步才坏的"。"""

    pipeline = _pipeline(ResolverStub(), BoomProcessor(optional=False), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    with pytest.raises(ContextBuildError) as excinfo:
        await _run(pipeline, state)

    error = excinfo.value

    assert error.processor == "boom"
    assert error.stage == "profile_loader"
    assert "炸了" in str(error)

    executions = error.details["executions"]

    assert any(item["processor"] == "boom" for item in executions)
    assert any(
        item["status"] == ProcessorStatus.FAILED.value
        for item in executions
    )


# ============================================================
# 6. 观测（规格 §二十六）
# ============================================================


async def test_every_processor_records_its_execution() -> None:
    """每个 Processor 都留下 name / order / when / status / duration / detail。"""

    pipeline = _pipeline(ResolverStub(), RecordingProcessor(), AssemblerStub())

    state, context = _build(
        pipeline, graph_state={"context_policy": {"record": True}, "payload": "x"}
    )

    await _run(pipeline, state)

    assert len(context.executions) == 3

    for record in context.executions:

        assert record.processor
        assert record.order > 0
        assert record.stage
        assert record.status is ProcessorStatus.EXECUTED
        assert record.duration_ms >= 0

        metadata = record.as_metadata()

        assert set(metadata) >= {
            "processor", "order", "when", "when_value", "status",
            "duration_ms", "error",
        }

    recorder = next(
        item for item in context.executions if item.processor == "recorder"
    )

    assert recorder.detail["overlay_added"] >= 1
    assert recorder.detail["inputs"] == ["value"]


async def test_processor_detail_reports_what_it_added() -> None:
    """`detail` 由流水线**自己算**（不依赖 Processor 自觉上报）。"""

    class SectionAdder(ContextProcessor):
        spec = ContextProcessorSpec(
            name="sections",
            order=20,
            stage=StageId.PROFILE_LOADER,
            when="context_policy.record",
            inputs={},
        )

        async def process(self, inputs, context):
            context.add_section("profile", "画像", "tier: gold")
            context.add_history(["m1", "m2"])
            return context

    pipeline = _pipeline(ResolverStub(), SectionAdder(), AssemblerStub())

    state, context = _build(pipeline, graph_state={"context_policy": {"record": True}})

    await _run(pipeline, state)

    record = next(
        item for item in context.executions if item.processor == "sections"
    )

    assert record.detail["sections_added"] == 1
    assert record.detail["history_messages"] == 2


# ============================================================
# 7. 快照
# ============================================================


async def test_snapshot_records_processor_statuses() -> None:
    """快照里带每个 Processor 的状态（规格 §廿七）。"""

    pipeline = _pipeline(ResolverStub(), RecordingProcessor(), AssemblerStub())

    state, context = _build(
        pipeline, graph_state={"context_policy": {"record": False}}
    )

    await _run(pipeline, state)

    snapshot = context.snapshot()

    statuses = {
        item["processor"]: item["status"] for item in snapshot.processors
    }

    assert statuses["recorder"] == ProcessorStatus.SKIPPED.value
    assert statuses["resolver_stub"] == ProcessorStatus.EXECUTED.value
    assert snapshot.context_hash


async def test_unconditional_processor_skips_gate_resolution() -> None:
    """`when=None` 的处理器不解析 gate（入口/出口没有策略开关）。"""

    pipeline = _pipeline(ResolverStub(), AssemblerStub())

    state, context = _build(pipeline, graph_state={})

    await _run(pipeline, state)

    resolver = next(
        item for item in context.executions
        if item.processor == "resolver_stub"
    )

    assert resolver.when_path is None
    assert resolver.when_value is True


async def test_run_stage_without_begin_is_a_clear_error() -> None:
    """没经过 begin 就跑阶段 -> 报错说清是接线问题，而不是 AttributeError。"""

    pipeline = _pipeline(ResolverStub(), AssemblerStub())

    with pytest.raises(ContextBuildError) as excinfo:
        await pipeline.run_stage(
            StageId.PROFILE_LOADER, _State(request=_Request())
        )

    assert "工作集" in str(excinfo.value)


async def test_stage_without_processor_is_a_noop() -> None:
    """没有 Processor 的阶段什么都不做（Builder 自己负责阶段参与）。"""

    pipeline = _pipeline(ProcessorStub(), required=())

    state, context = _build(pipeline, graph_state={})

    await pipeline.run_stage(StageId.MEMORY_RETRIEVER, state)

    assert context.executions == []


class ProcessorStub(ContextProcessor):
    """只服务 PROFILE_LOADER 的最小实现（含必需的入口/出口不在这里）。"""

    spec = ContextProcessorSpec(
        name="only_profile",
        order=20,
        stage=StageId.PROFILE_LOADER,
        when="context_policy.record",
        inputs={},
    )

    async def process(self, inputs, context):
        return context
