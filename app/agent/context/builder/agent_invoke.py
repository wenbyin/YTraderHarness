"""
`agent_invoke` 的 Processor 版实现：既有 Builder 契约 + 新 Processor 流水线。

它怎么把两套契约接起来（规格 §五、§二十三、§二十四）
-------------------------------------------------
    BaseContextBuilder.build()      既有驱动：阶段参与、固定顺序、收尾校验
        └─ _stage_<阶段>()          既有扩展点（每个阶段一个方法）
             └─ pipeline.run_stage()  本实现补的那一段：Processor 排序 /
                                      gate / 绑定 / 执行 / 观测 / 累积
                  └─ processor.process(inputs, context)   业务逻辑

**没有第二个驱动**：`build()` 的循环、`StageResolution` 的边界校验、
"必须有产出"的收尾检查全部沿用既有实现。本类只做三件事：

    1. 把 11 个阶段方法**逐个**委托给流水线（薄薄一层，见下）
    2. 每阶段之后把工作集里的 draft 同步进既有的 `ContextBuildState.draft`
    3. 在最后一个阶段之后把快照写进 draft 的 metadata（规格 §十九/§二十七）

⚠️ 为什么逐个覆写阶段方法，而不是覆写 `_run_stage`
---------------------------------------------
`BaseContextBuilder.__init_subclass__` 有一条硬校验：

    声明参与的阶段**必须实现**对应的阶段方法（否则类定义期 TypeError）

它是对的 —— "声明了就要实现"是这份契约的诚实性保证。逐个覆写让这条
校验继续有效，也让"这张图会跑哪 11 个阶段"在类体里一眼看得见；而
覆写 `_run_stage` 会绕过它（`_is_unimplemented` 判的是阶段方法本身）。

每个方法都是一行委托 —— 它们**不是**重复逻辑，而是两套契约之间的桥。

⚠️ 依赖从哪来
-----------
本类的实例是**共享的**（注册表按类型只构造一次），而且
`BaseContextBuilder.__setattr__` 封死了实例属性 —— 因此依赖不能挂在
self 上。它们随请求进来（`ContextBuildRequest.dependencies`），缺省时
用进程级默认（`get_model_registry()` 等）。见 processor/dependencies.py。
"""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from typing import Any, ClassVar

from app.agent.context.processor.context import ContextBuildContext
from app.agent.context.processor.pipeline import (
    PIPELINE_STAGE_ORDER,
    ContextProcessorPipeline,
)
from app.agent.context.processor.registry import create_pipeline
from app.common.context import StageId

from .base import AgentInvokeContextBuilder
from .contracts import ContextBuildState

__all__ = ["AgentInvokeProcessorContextBuilder", "ProcessorContextBuilder"]

#: 本实现声明参与的阶段：**全部** 11 个。
#:
#: 这不是"每个阶段都一定跑"——跑不跑由每个 Processor 的 `when` gate 决定
#: （见 processor/processors/__init__.py 的那张表）。阶段参与是**静态**
#: 事实（这个 Builder 有能力跑这些阶段），gate 是**运行时**决定，两者
#: 刻意分开。
_ALL_STAGES: frozenset[StageId] = frozenset(StageId)

#: 不可被运行时跳过的阶段。
#:
#:     CONTEXT_RESOLVER   入口：策略与预算由它产出（先有鸡才有蛋）
#:     PROMPT_ASSEMBLER   出口：ModelRequest 由它产出（少了它没有产出）
#:
#: 流水线会双向校验这条：required 阶段上的 Processor 必须 `when=None`，
#: 而 `when=None` 的 Processor 必须在 required 阶段上。
_REQUIRED_STAGES: frozenset[StageId] = frozenset({
    StageId.CONTEXT_RESOLVER,
    StageId.PROMPT_ASSEMBLER,
})

#: 最后一个阶段 —— 快照在这里写进 draft 的 metadata（只算一次哈希）。
_FINAL_STAGE: StageId = max(
    _ALL_STAGES, key=lambda stage: PIPELINE_STAGE_ORDER.index(stage)
)


class ProcessorContextBuilder(ABC):
    """
    "用 Processor 流水线实现 Context Builder"的公共基类。

    子类只需定死 `builder_type`（由既有的 type 基类给出）与
    `processors`（不写就用反射发现的全集）。阶段委托、draft 同步、
    快照落 metadata 都在这里，子类不必重复。

    ⚠️ 它不是 `BaseContextBuilder` 的子类
    ------------------------------------
    真正的继承链是：

        BaseContextBuilder
          └── AgentInvokeContextBuilder（既有 type 基类，定死 builder_type）
                └── AgentInvokeProcessorContextBuilder（本包的具体实现）
                      （混入 ProcessorContextBuilder 的实现）

    这样"分类由 type 基类继承而定"这条既有规则继续成立（实现不能自己
    声明 builder_type）。
    """

    #: Processor 集合。None = 反射发现全集（生产路径）。
    #:
    #: 显式传一组是为了测试与定制：不必改包里的文件就能组一条流水线。
    processors_override: ClassVar[Any] = None

    #: 延迟构造的流水线（类级缓存 —— 它是不可变的、进程一份）。
    _pipeline: ClassVar[ContextProcessorPipeline | None] = None

    # ========================================================
    # 流水线
    # ========================================================

    @classmethod
    def pipeline(cls) -> ContextProcessorPipeline:
        """
        取（必要时构造）流水线。

        构造期会做全部自描述校验（顺序一致性 / gate 与 required 的
        矛盾 / 名字与 order 的唯一性），因此"配错了"在**第一次取流水线**
        时就会炸 —— 也就是应用启动期（`create_context_builder` 会在
        lifespan 里被调用），而不是某次真实对话里。
        """

        cached = cls.__dict__.get("_pipeline")

        if cached is not None:
            return cached

        pipeline = create_pipeline(
            required_stages=_REQUIRED_STAGES,
            processors_override=cls.processors_override,
        )

        cls._pipeline = pipeline

        return pipeline

    @classmethod
    def describe_pipeline(cls) -> list[dict[str, Any]]:
        """流水线自描述（目录接口 / 排查用）。"""

        return cls.pipeline().describe()

    # ========================================================
    # 阶段 -> 流水线（桥）
    # ========================================================

    async def _delegate(
            self,
            stage: StageId,
            state: ContextBuildState,
    ) -> None:
        """
        跑一个阶段，并把产出同步回既有契约。

        三步：
            1. 跑该阶段的 Processor（gate / 绑定 / 执行 / 观测在流水线里）
            2. 把工作集里的 draft 同步进 `state.draft`（既有产出槽）
            3. 最后一个阶段之后，把快照与嵌套用量写进 draft 的 metadata
        """

        pipeline = type(self).pipeline()

        if stage is StageId.CONTEXT_RESOLVER and state.processor_context is None:
            # 入口阶段：先建工作集（后续阶段都读它）。
            pipeline.begin(state)

        await pipeline.run_stage(stage, state)

        context: ContextBuildContext | None = state.processor_context

        if context is None or context.draft is None:
            return

        metadata = dict(context.draft.metadata)

        metadata["context_token_usage"] = dict(context.token_usage)

        # 回写图 state 的产出。**合并**两个来源，不是二选一：
        #
        #     summary                                     summary 阶段真的
        #                                                 生成过（summary_dirty）
        #     context.metadata["context_state_updates"]   处理器自己放的
        #                                                 （目前只有 memory）
        #
        # ⚠️ 第二个来源必须挂在 `context.metadata` 上，**不能**让处理器去写
        # `draft.metadata`：memory 处理器在阶段 5 跑，而 draft 要到阶段 8
        # （PROMPT_ASSEMBLER）才存在 —— 本函数在 `draft is None` 时直接
        # return（见上面那句），处理器那时根本没有 draft 可写。
        state_updates = dict(
            context.metadata.get("context_state_updates") or {}
        )

        if context.summary_dirty and context.summary is not None:
            # 摘要**这一次真的生成过**（summary 阶段调过模型）—— 只有这种
            # 情况才需要回写图 state（见 processor/context.py 的
            # summary_dirty）。
            state_updates["summary"] = context.summary.as_state()

        if state_updates:
            metadata["context_state_updates"] = state_updates

        if stage is _FINAL_STAGE:
            snapshot = context.snapshot(
                counter=_token_counter(context)
            )

            metadata["context_snapshot"] = snapshot.as_metadata()
            metadata["context_processors"] = [
                execution.as_metadata() for execution in context.executions
            ]

        state.draft = replace(context.draft, metadata=metadata)

    # ---- 11 个阶段方法（每个一行，见模块 docstring） ----------

    async def _stage_context_resolver(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.CONTEXT_RESOLVER, state)

    async def _stage_profile_loader(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.PROFILE_LOADER, state)

    async def _stage_summary_loader(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.SUMMARY_LOADER, state)

    async def _stage_history_selector(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.HISTORY_SELECTOR, state)

    async def _stage_memory_retriever(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.MEMORY_RETRIEVER, state)

    async def _stage_knowledge_retriever(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.KNOWLEDGE_RETRIEVER, state)

    async def _stage_capability_loader(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.CAPABILITY_LOADER, state)

    async def _stage_prompt_assembler(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.PROMPT_ASSEMBLER, state)

    async def _stage_token_budgeter(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.TOKEN_BUDGETER, state)

    async def _stage_context_compactor(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.CONTEXT_COMPACTOR, state)

    async def _stage_context_validator(
            self, state: ContextBuildState
    ) -> None:
        await self._delegate(StageId.CONTEXT_VALIDATOR, state)


class AgentInvokeProcessorContextBuilder(
    ProcessorContextBuilder,
    AgentInvokeContextBuilder,
):
    """
    主 Agent / ReAct 调用用的 Builder（`agent_invoke`）。

    这是**生产路径**上唯一一个被反射发现的实现（见 context/registry.py）：

        create_context_builder(ContextBuilderType.AGENT_INVOKE)
            -> 本类的一个实例（零参构造）

    它声明参与全部 11 个阶段、其中两个 required；跑不跑由 Processor 的
    gate 与 Context Policy 决定（见 processor/processors/__init__.py）。
    """

    version = "1.0.0"
    display_name = "主 Agent 上下文构建（Processor 流水线）"
    description = (
        "按固定 11 阶段流水线构建主 Agent 的 LLM 上下文："
        "画像 / 摘要 / 历史 / 记忆 / 检索 / 工具 / 组装 / 预算 / 压缩 / 校验。"
    )

    enabled_stages: ClassVar[frozenset[StageId]] = _ALL_STAGES
    required_stages: ClassVar[frozenset[StageId]] = _REQUIRED_STAGES


# ============================================================
# internals
# ============================================================


def _token_counter(context: ContextBuildContext) -> Any:
    """
    取计数器（快照要算 input token）。

    依赖束里没配就用启发式实现 —— 与流水线里其它地方同一条回落规则
    （见 processor/dependencies.py）。
    """

    from app.agent.context.processor.dependencies import dependencies_of

    return dependencies_of(context).resolved_token_counter()
