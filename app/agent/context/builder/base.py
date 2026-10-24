"""
Context Builder 的契约：固定 13 步的调度 + 六个 type 基类。

流水线是**固定**的，顺序不变
------------------------------
    1. Context Resolver     读 builder_type + state，输出 skip 列表 + 参数
    2. Task Resolver        有 current_task 时注入任务执行指令
    3. Skill Loader         按 skill_key 渲染 prompt、解析 capabilities
    4. Profile Loader       ┐
    5. Summary Loader       │
    6. History Selector     │
    7. Memory Retriever     ├ 可能 skip
    8. Knowledge Retriever  │
    9. Capability Loader    │
   10. Prompt Assembler     ┘ 按 state 决定层序，产出 ModelRequest
   11. Token Budgeter
   12. Context Compactor    可能 skip（唯一会自己调 LLM 的阶段）
   13. Context Validator

顺序由 `contracts.PIPELINE_TEMPLATE` 定死，**具体实现改不了** ——
规格里写明 stage order 由服务器定义、客户端必须当作不可变。
实现能改的只有两件事：哪些阶段参与、每个阶段的参数。

这与 tool 模块的处理器链**刻意不同**（那边顺序是数据库里配的）。
差别来自需求本身：处理器链是用户可编排的，而上下文构建的步骤之间存在
真实的先后依赖（没取到历史就谈不上裁剪它），顺序不是一个可配置项。

参与关系：一个上限 + 一次运行的选择
------------------------------------
    enabled_stages    这个 builder 类型**能**跑哪些阶段（静态、类级）
    required_stages   其中哪些**必须**跑（静态、类级）
    resolution.skipped 这一次**不要**跑哪些（动态、每次调用）

    实际执行 ⟺ s ∈ enabled ∧ (s ∈ required ∨ s ∉ skipped)

两处越界是**显式报错**，不是静默求交：
    跳过了没声明的阶段         -> ValueError（resolver 的 bug）
    跳过了声明为 required 的   -> ValueError（required 就是"不可跳过"）

为什么是"报错"而不是"取交集"：静默求交会让两种完全不同的错误长得
一样 —— "resolver 写错了"与"实现能力不够"，在日志里都只是"少跑了一步"。
与 `BaseToolProcessor.__init_subclass__` 的立场一致：能静态或早期发现的
错误，不要留到运行期变成行为差异。

为什么 stage 方法不做成 @abstractmethod
---------------------------------------
与 `BaseToolProcessor` 的三段 hook 同一条理由：每个实现只服务自己
`enabled_stages` 里的那几个阶段。声明成抽象会逼着"只用历史 + prompt"
的 builder 也写九个空壳。因此默认实现是
`raise self._unsupported(stage)`，调错时的信息里带上"本类声明了
哪些阶段"。

`CONTEXT_RESOLVER` 是**唯一**有真实默认值的阶段（默认：不跳过任何
阶段、不带参数）。一个参与关系完全静态的 builder 没有"运行时解析"
这回事，逼它覆写一个空壳正是上面说的那种浪费。

self 上不许放"这次调用的东西"
------------------------------
builder 实例是**共享的**（每个类型只构造一次，见 registry.py），
会被并发的每一次模型调用同时使用。把本次的工作集写到 self 上，等于
让 A 调用读到 B 调用的状态 —— 症状是"偶发地拼错了上下文"，几乎不可能
从日志里看出来。

`self` 上只该有**从类声明推导出来的不可变东西**；凡是"这次调用的
东西"（工作集、请求、解析结果），都走参数 —— 见
`contracts.ContextBuildState`。

阶段**不是图节点**
------------------
规格的 `PipelineNode.independent_lifecycle` 明写："does not automatically
make it a LangGraph node"。十三个阶段在同一次 build() 调用里顺序跑完，
不进 checkpoint、不跨节点、不扇出。把阶段摊成图节点会让一次上下文构建
变成十三个 super-step，而它们之间只有顺序没有并行度。

⚠️ 六个 type 基类**都不是** `inspect.isabstract()`
--------------------------------------------------
本基类故意没有抽象方法，所以六个 type 基类都是"可实例化"的。发现逻辑
按下述**结构规则**把它们排除掉：

    BaseContextBuilder 的直接子类 = type 基类，不是实现
    实现必须挂在 type 基类之下

因此直接实例化某个 type 基类是拿得到一个对象的（它会在 build() 时报
"没声明任何阶段"）。这是刻意的取舍：换来的是"忘了挂到 type 基类下"
这类错误能在 `__init_subclass__` 里被点名。
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, TypeVar

from app.common.context import (
    BuilderExecutionMode,
    ContextBuilderStatus,
    ContextBuilderType,
    StageId,
)

from .contracts import (
    PIPELINE_TEMPLATE,
    ContextBuildRequest,
    ContextBuildState,
    ModelRequest,
    StagePolicy,
    StageResolution,
)

__all__ = [
    "AgentInvokeContextBuilder",
    "BaseContextBuilder",
    "ClassifierContextBuilder",
    "CompactionContextBuilder",
    "MemoryContextBuilder",
    "PlannerContextBuilder",
    "SubAgentContextBuilder",
]


_F = TypeVar("_F", bound=Callable[..., Any])


def _no_default(stage: StageId) -> Callable[[_F], _F]:
    """
    标记：基类对这个阶段**没有**可用的默认实现。

    这个标记只给 `__init_subclass__` 读 —— 它要判断"声明了阶段却
    没写方法"该对哪些阶段生效。让标记**跟着方法走**，而不是在别处
    再维护一份"哪些阶段有默认值"的清单：清单与代码总会漂移，而漂移
    的后果是那条校验**悄悄失效** —— 声明了、方法没写、运行到那一步
    才炸，而报错指向的是运行的某一次请求。

    `CONTEXT_RESOLVER` 不带这个标记（它有真实默认值），所以它不必
    被覆写；将来若给别的阶段补上真实默认值，去掉那个装饰器即可，
    不需要同步改任何一处清单。
    """

    def decorate(func: _F) -> _F:
        func.__context_builder_no_default__ = stage  # type: ignore[attr-defined]
        return func

    return decorate


class BaseContextBuilder(ABC):
    """
    所有 Context Builder 实现的公共契约。

    子类要做三件事（且只有这三件）：

        1. 挂到某个 type 基类之下（分类由继承而来，不能自己声明）
        2. 声明 enabled_stages（以及可选的 required_stages）
        3. 覆写 enabled_stages 里那些带 `_no_default` 标记的阶段方法
    """

    # ========================================================
    # 身份 —— 由 type 基类定死
    # ========================================================

    #: 构建器类型。**由六个 type 基类定死**，实现不能自己声明。
    #:
    #: None = 尚未归类。这样的类进不了发现结果（与 tool 模块
    #: `processor_type is None` 被跳过是同一条规则）。
    builder_type: ClassVar[ContextBuilderType | None] = None

    #: 定义版本，进能力目录（规格的 ContextBuilderDefinition.version），
    #: 用于回放与文档兼容。改了参与关系或参数语义就应该抬它。
    version: ClassVar[str] = "1.0.0"

    #: 给人看的名字，进 ContextBuilderSummary.name。
    #: 留空时目录回落到 builder_type 的值。
    display_name: ClassVar[str] = ""

    #: 一句话说明这个构建器做什么。没有就留空串。
    #: 规格里 description 是 required，所以目录出参时不该是 None。
    description: ClassVar[str] = ""

    #: 定义状态。DISABLED 的定义**仍然注册**（见 registry.py）——
    #: 目录要能按 include_disabled 列出它，跳过它会让目录与实际能取到
    #: 的东西对不上。
    status: ClassVar[ContextBuilderStatus] = (
        ContextBuilderStatus.ACTIVE
    )

    #: 通常跑在哪。由 type 基类给出，实现一般不改。
    execution_mode: ClassVar[BuilderExecutionMode] = (
        BuilderExecutionMode.EMBEDDED_MODEL_INVOCATION
    )

    # ========================================================
    # 参与关系 —— 实现声明这两个
    # ========================================================

    #: 这个实现**声明参与**的阶段。空集合 = 什么都不参与。
    #:
    #: ⚠️ 声明了就必须实现对应的阶段方法 —— 否则在类定义期就
    #: 抛 TypeError，不留到运行期。
    enabled_stages: ClassVar[frozenset[StageId]] = frozenset()

    #: 其中**任何一次运行都不允许跳过**的阶段。
    #:
    #: 用在哪：某些阶段是这个类型的定义性行为（比如 classifier 的
    #: prompt 组装被声明成 required，那么 resolver 就不能因为"这次
    #: 输入很短"把它跳掉）。这是实现对自己行为的承诺。
    #:
    #: 必须 ⊆ enabled_stages。
    required_stages: ClassVar[frozenset[StageId]] = frozenset()

    # ========================================================
    # 阶段 -> 方法名
    # ========================================================

    #: 十三个阶段各自对应的方法名。
    #:
    #: 单列成映射而不是靠命名约定现推，是为了让"某个阶段没有对应
    #: 方法"在**这里**就能看出来 —— 而 `__init_subclass__` 也会
    #: 用它做校验。键集必须与 StageId 相同，由 contracts 的
    #: import 期校验保证模板不漂移，这里由下面的自检保证映射不漏。
    _STAGE_METHODS: ClassVar[Mapping[StageId, str]] = {
        StageId.CONTEXT_RESOLVER: "_stage_context_resolver",
        StageId.TASK_RESOLVER: "_stage_task_resolver",
        StageId.SKILL_LOADER: "_stage_skill_loader",
        StageId.PROFILE_LOADER: "_stage_profile_loader",
        StageId.SUMMARY_LOADER: "_stage_summary_loader",
        StageId.HISTORY_SELECTOR: "_stage_history_selector",
        StageId.MEMORY_RETRIEVER: "_stage_memory_retriever",
        StageId.KNOWLEDGE_RETRIEVER: "_stage_knowledge_retriever",
        StageId.CAPABILITY_LOADER: "_stage_capability_loader",
        StageId.PROMPT_ASSEMBLER: "_stage_prompt_assembler",
        StageId.TOKEN_BUDGETER: "_stage_token_budgeter",
        StageId.CONTEXT_COMPACTOR: "_stage_context_compactor",
        StageId.CONTEXT_VALIDATOR: "_stage_context_validator",
    }

    # ========================================================
    # 无状态 —— 机器保证，不是约定
    # ========================================================

    def __setattr__(self, name: str, value: Any) -> None:
        """
        禁止任何实例属性。

        这不是风格要求，是**并发正确性的保证**：builder 实例是**共享的**
        —— 每个 `builder_type` 只构造一次（见 registry.py），随后并发的
        每一次模型调用都用同一个。只要实例上不可能存在本次调用的状态，
        "共享"就是安全的。

        为什么不靠约定：约定挡不住 `self.foo = ...`，而这类错误的症状是
        "偶发地拼错了上下文" —— 几乎不可能从日志里看出来。把
        `__setattr__` 封死之后，**违约在写下的那一刻就炸**，且指向
        正确的地方。

        那两个"想存东西"的诉求都有去处：

            这次调用的东西   -> ContextBuildState（build() 的参数）
            每个实现固定的东西 -> ClassVar（本来就只有这一种）
        """

        raise AttributeError(
            f"{type(self).__name__} 不允许实例属性"
            f"（试图设置 {name!r}）。\n"
            "Builder 实例是共享的：每个 builder_type 只构造一次，并发的"
            "每次模型调用都用同一个 —— 因此实例上不能有「这次调用」的"
            "状态，否则 A 调用会读到 B 调用的值。\n"
            "本次调用的东西放 ContextBuildState（它是 build() 的参数）；"
            "每个实现固定的东西放 ClassVar。"
        )

    # ========================================================
    # 类定义期校验
    # ========================================================

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        类体执行完的那一刻校验声明是否自洽。

        放在这里而不是服务层：这是一条**静态**约束，违反它的类从
        定义那一刻起就是错的。等服务层（或某次运行）发现，错误信息
        只会指向某次请求，而真正的原因在几天前写下的那个类。

        与 `BaseToolProcessor.__init_subclass__` 是同一种做法、同一个
        理由（那里校验的是 AROUND ⇒ SYNC 之类的规则）。
        """

        super().__init_subclass__(**kwargs)

        # ---- 1. 分类只能来自 type 基类 ------------------------

        if BaseContextBuilder in cls.__bases__:

            # 直接子类**必须**是 type 基类，因此必须定死
            # builder_type。发现逻辑按"直接子类 = type 基类"这条结构
            # 规则排除它们，一个没定类型的直接子类会让那条规则失去
            # 意义（它会被当成实现，或者被静默跳过）。
            if cls.builder_type is None:
                raise TypeError(
                    f"{cls.__name__} 直接继承了 BaseContextBuilder，"
                    "却没有声明 builder_type。直接子类只能是 type "
                    "基类（一个 builder_type 一个），具体实现要挂在"
                    "type 基类之下 —— 分类靠继承定死，不能由实现自己"
                    "声明。"
                )

        elif "builder_type" in cls.__dict__:

            # type 基类的后代：分类已经继承下来了，再写一个不同值
            # 就是在撒谎。
            raise TypeError(
                f"{cls.__name__} 重新声明了 builder_type。"
                f"分类由所继承的 type 基类定死"
                f"（当前是 {cls.builder_type}），实现不能改 —— "
                "与 tool 模块 processor_type、document 模块 role "
                "同一条规则：声明与实际能力不符时，没有第二处可以"
                "撒谎。"
            )

        # ---- 2. 没声明阶段就没什么可校验的 --------------------
        #
        # 六个 type 基类走的就是这条路（今天它们的 enabled_stages
        # 全为空，正是"零具体实现"的诚实状态）。

        if not cls.enabled_stages:
            return

        # ---- 3. 入口与出口必须在 ------------------------------

        if StageId.CONTEXT_RESOLVER not in cls.enabled_stages:
            raise TypeError(
                f"{cls.__name__} 声明了阶段却漏了 "
                f"{StageId.CONTEXT_RESOLVER.value}。它是流水线的**入口**"
                "：skip 列表与各阶段参数都由它产出，少了它，后面每个"
                "阶段拿不到参数（BuildState.resolution 永远是默认值）。"
            )

        if StageId.PROMPT_ASSEMBLER not in cls.enabled_stages:
            raise TypeError(
                f"{cls.__name__} 声明了阶段却漏了 "
                f"{StageId.PROMPT_ASSEMBLER.value}。它是流水线的**出口**"
                "：ModelRequest 由它产出，少了它 build() 必然以"
                "「没有产出任何请求」告终 —— 那是运行期的错，而根因"
                "在这里。"
            )

        # ---- 4. required 不能超出 enabled ---------------------

        stray = cls.required_stages - cls.enabled_stages
        if stray:
            raise TypeError(
                f"{cls.__name__} 把没有声明参与的阶段标成了 required："
                f"{sorted(s.value for s in stray)}。required 是"
                "「已参与的阶段里，哪些不可被运行时跳过」，"
                "对不参与的阶段谈「不可跳过」没有意义 —— "
                "那多半是改名 / 删阶段时漏改了一处。"
            )

        # ---- 5. 声明了就必须实现 ------------------------------

        missing = sorted(
            stage.value
            for stage in cls.enabled_stages
            if _is_unimplemented(cls, stage)
        )

        if missing:
            raise TypeError(
                f"{cls.__name__} 声明参与的阶段没有实现：{missing}。"
                "在 enabled_stages 里写下某个阶段，等于承诺"
                "「这次上下文构建会跑它」—— 而它跑起来只会抛"
                "NotImplementedError。要么实现对应方法，要么把它从"
                "enabled_stages 里去掉（那才是诚实的声明）。"
            )

    # ========================================================
    # 身份与参与关系（能力目录用的访问器）
    # ========================================================

    @classmethod
    def get_key(cls) -> ContextBuilderType:
        """注册键。发现逻辑与能力目录都走这里。"""

        if cls.builder_type is None:
            raise TypeError(
                f"{cls.__name__} 没有 builder_type（尚未归类到任何 "
                "type 基类下），因此没有注册键。"
            )

        return cls.builder_type

    @classmethod
    def stage_policies(cls) -> tuple[StagePolicy, ...]:
        """
        这个实现的参与关系，按**流水线顺序**铺开。

        这是**派生**出来的视图，不是第二份声明 —— 唯一的声明是
        `enabled_stages` / `required_stages` 两个类属性，这里只是
        把它们与固定模板对照着念一遍。将来 `/context-builders`
        接口要的 `stage_policies` 就是这个形状。

        十三个阶段全给出来（包括没参与的），而不是只给参与的那些：
        前端要画的是**完整流水线**，并标出哪些节点这次不亮 —— 只给
        参与的，它就无从知道"少了哪一步"。
        """

        return tuple(
            StagePolicy(
                stage_id=stage,
                enabled=stage in cls.enabled_stages,
                execution_order=order,
                required=stage in cls.required_stages,
            )
            for order, stage in enumerate(PIPELINE_TEMPLATE)
        )

    # ========================================================
    # 执行入口 —— 唯一被调用的方法
    # ========================================================

    async def build(
            self,
            request: ContextBuildRequest,
    ) -> ModelRequest:
        """
        跑完固定流水线，产出交给模型的请求。

        调用方（AgentNode）只认这一个方法，不逐阶段调用 —— 与
        `BaseToolProcessor.process()` 是同一个角色。

        ⚠️ 本方法**不校验 builder_type 与 request.builder_type 是否
        一致**。那是调用方的事（它按类型取实例，本来就不会取错），
        在这里再查一次只会把"调用方拿错了 builder"从一个明显的
        编程错误变成一次运行期异常。
        """

        if not self.enabled_stages:
            raise RuntimeError(
                f"{type(self).__name__} 没有声明任何阶段"
                "（enabled_stages 为空），build() 无从跑起。\n"
                "若这是一个 type 基类 —— 对的，它不该被直接实例化；"
                "具体实现要挂在它之下（见本模块 docstring 里关于"
                "「六个 type 基类都不是 isabstract」的说明）。"
            )

        state = ContextBuildState(request=request)

        # ---- 阶段 1：入口，永远先跑 ---------------------------
        #
        # 它产出 resolution，也就是"后面哪些阶段不跑"，因此不可能
        # 被 resolution 自己跳过 —— 先有鸡才有蛋。它是否在
        # enabled_stages 里由 __init_subclass__ 保证（只要声明了
        # 任何阶段，就必须包含它）。
        await self._run_stage(StageId.CONTEXT_RESOLVER, state)

        resolution = state.resolution

        # ---- 校验 resolver 没有越界 ---------------------------

        undeclared = resolution.skipped - self.enabled_stages

        if undeclared:
            raise ValueError(
                f"{type(self).__name__} 的 "
                f"{StageId.CONTEXT_RESOLVER.value} 想跳过没有声明参与的"
                f"阶段：{sorted(s.value for s in undeclared)}。\n"
                f"本类声明的阶段是："
                f"{sorted(s.value for s in self.enabled_stages)}。\n"
                "skipped 只能**收窄**（在已声明的阶段里挑几个这次不跑），"
                "不能扩张 —— 没声明的阶段本来就不跑，写进来只说明"
                "resolver 手上的阶段名单是错的。"
            )

        forbidden = resolution.skipped & self.required_stages

        if forbidden:
            raise ValueError(
                f"{type(self).__name__} 的 "
                f"{StageId.CONTEXT_RESOLVER.value} 想跳过声明为 "
                f"required 的阶段："
                f"{sorted(s.value for s in forbidden)}。\n"
                "required 的含义就是「任何一次运行都不许跳过」。"
                "要么 resolver 的判断条件写错了，要么这个阶段本来就"
                "不该是 required —— 两处必须改一处，不能悄悄让"
                "运行期赢。"
            )

        # ---- 阶段 2..11：按固定顺序跑 -------------------------

        for stage in PIPELINE_TEMPLATE:

            # 入口已经跑过了，别跑第二遍。
            if stage is StageId.CONTEXT_RESOLVER:
                continue

            if stage not in self.enabled_stages:
                continue

            if stage in resolution.skipped:
                continue

            await self._run_stage(stage, state)

        # ---- 收尾：流水线必须产出东西 -------------------------

        if state.draft is None:
            raise RuntimeError(
                f"{type(self).__name__} 跑完了流水线却没有产出 "
                "ModelRequest。\n"
                f"产出它的是 {StageId.PROMPT_ASSEMBLER.value}"
                f"（它声明参与了吗："
                f"{StageId.PROMPT_ASSEMBLER in self.enabled_stages}）；"
                f"被跳过了吗："
                f"{StageId.PROMPT_ASSEMBLER in state.resolution.skipped}。"
                "\n一次「成功的构建」却什么都没构建出来，不该当成空"
                "结果静默传下去 —— 模型会拿到空消息列表，而报错"
                "离原因很远。"
            )

        return state.draft

    # ========================================================
    # 可观测的产物（规格 §十九）
    # ========================================================

    async def build_with_result(
            self,
            request: ContextBuildRequest,
    ) -> Any:
        """
        跑一次构建，返回**富结果**（含"这次用了哪些上下文"）。

        `build()` 的契约只返回 ModelRequest（那是调用方真正需要的
        东西）。但"这一次 LLM 调用到底看到了什么"必须能被回答 ——
        Debug / Replay / Evaluation / Audit 都建立在它上面。

        默认实现走**一条通道**：请求的 `metadata["context_snapshot"]`
        （Processor 流水线的实现会把快照写进去）。因此即便调用方手上
        只有一个 ModelRequest，信息也不丢：

            ContextBuildResult.from_model_request(
                await builder.build(request)
            )

        没有快照的既有实现会拿到一个只含 `model_request` 的最小结果 ——
        不是错误，只是"这个实现没提供额外观测"。
        """

        from app.agent.context.processor.context import ContextBuildResult

        model_request = await self.build(request)

        return ContextBuildResult.from_model_request(model_request)

    # ========================================================
    # 阶段方法
    # ========================================================

    # ⚠️ 刻意**不带** @_no_default 标记：它是唯一有真实默认值的阶段。
    async def _stage_context_resolver(
            self,
            state: ContextBuildState,
    ) -> None:
        """
        阶段 1：读 builder_type + state，产出 skip 列表与各阶段参数。

        **基类里唯一有真实默认实现的阶段** —— 默认「不跳过任何阶段、
        不带任何参数」。一个参与关系完全静态的 builder 没有"运行时
        解析"这回事，逼它覆写一个空壳没有意义（`_no_default` 的
        docstring 已经说过这条）。

        覆写要点：
            写 `state.resolution = StageResolution(...)`。
            skipped 只能收窄（见 build() 的两处校验）。
            parameters 的键是阶段，值是该阶段的参数 —— 形状由各阶段
            自己定义，本层不认识其中任何一个键。
        """

        state.resolution = StageResolution()

    @_no_default(StageId.TASK_RESOLVER)
    async def _stage_task_resolver(
            self, state: ContextBuildState
    ) -> None:
        """阶段 2：有 current_task 时注入任务执行指令。"""

        raise self._unsupported(StageId.TASK_RESOLVER)

    @_no_default(StageId.SKILL_LOADER)
    async def _stage_skill_loader(
            self, state: ContextBuildState
    ) -> None:
        """阶段 3：按 task.skill_key 切换 prompt 与 tool_keys。"""

        raise self._unsupported(StageId.SKILL_LOADER)

    @_no_default(StageId.PROFILE_LOADER)
    async def _stage_profile_loader(
            self, state: ContextBuildState
    ) -> None:
        """阶段 2：加载用户 / 租户画像。产出写进 state.artifacts。"""

        raise self._unsupported(StageId.PROFILE_LOADER)

    @_no_default(StageId.SUMMARY_LOADER)
    async def _stage_summary_loader(
            self, state: ContextBuildState
    ) -> None:
        """阶段 3：加载既有的对话摘要。"""

        raise self._unsupported(StageId.SUMMARY_LOADER)

    @_no_default(StageId.HISTORY_SELECTOR)
    async def _stage_history_selector(
            self, state: ContextBuildState
    ) -> None:
        """
        阶段 4：按参数取最近若干轮历史。

        参数（走 `state.resolution.parameters`）：取几条、按什么排序。
        """

        raise self._unsupported(StageId.HISTORY_SELECTOR)

    @_no_default(StageId.MEMORY_RETRIEVER)
    async def _stage_memory_retriever(
            self, state: ContextBuildState
    ) -> None:
        """阶段 5：长期记忆检索。"""

        raise self._unsupported(StageId.MEMORY_RETRIEVER)

    @_no_default(StageId.KNOWLEDGE_RETRIEVER)
    async def _stage_knowledge_retriever(
            self, state: ContextBuildState
    ) -> None:
        """阶段 6：知识库检索。"""

        raise self._unsupported(StageId.KNOWLEDGE_RETRIEVER)

    @_no_default(StageId.CAPABILITY_LOADER)
    async def _stage_capability_loader(
            self, state: ContextBuildState
    ) -> None:
        """
        阶段 7：加载本次可用的工具。

        参数：加载哪些工具。产出是 `ModelRequest.tools` 的来源 ——
        用 BaseTool（本项目的 ToolDescriptor 本身就是 StructuredTool，
        为 bind_tools 而生），不要转成 JSON schema dict。
        """

        raise self._unsupported(StageId.CAPABILITY_LOADER)

    @_no_default(StageId.PROMPT_ASSEMBLER)
    async def _stage_prompt_assembler(
            self, state: ContextBuildState
    ) -> None:
        """
        阶段 8：按 state 决定层序，**产出 ModelRequest**（写
        `state.draft`）。

        它是流水线的出口：`build()` 收尾时 draft 仍为 None 就报错。
        """

        raise self._unsupported(StageId.PROMPT_ASSEMBLER)

    @_no_default(StageId.TOKEN_BUDGETER)
    async def _stage_token_budgeter(
            self, state: ContextBuildState
    ) -> None:
        """
        阶段 9：按预算裁剪（参数：预算多少、裁剪优先级）。

        它读 `state.draft` 并就地改写它 —— 排序上刻意排在
        PROMPT_ASSEMBLER **之后**：先拼出来才知道超没超。
        """

        raise self._unsupported(StageId.TOKEN_BUDGETER)

    @_no_default(StageId.CONTEXT_COMPACTOR)
    async def _stage_context_compactor(
            self, state: ContextBuildState
    ) -> None:
        """
        阶段 10：压缩上下文。

        ⚠️ 这是**唯一会自己调 LLM** 的阶段（见
        `contracts.LLM_INVOCATION_STAGES`）。它自己那次调用也得有
        上下文，因此通常**嵌套**调用
        `ContextBuilderType.COMPACTION` 那个 builder（规格的
        `nested_invocations`），而不是在这里裸调模型。
        """

        raise self._unsupported(StageId.CONTEXT_COMPACTOR)

    @_no_default(StageId.CONTEXT_VALIDATOR)
    async def _stage_context_validator(
            self, state: ContextBuildState
    ) -> None:
        """
        阶段 11：校验最终上下文可用。

        失败应当是**抛异常**，而不是悄悄放行 —— 一个校验器放过坏
        上下文，比没有校验器更糟。
        """

        raise self._unsupported(StageId.CONTEXT_VALIDATOR)

    # ========================================================
    # internals
    # ========================================================

    async def _run_stage(
            self,
            stage: StageId,
            state: ContextBuildState,
    ) -> None:
        """按映射取方法并执行。"""

        method = getattr(self, self._STAGE_METHODS[stage])

        # 阶段 Span：只有真正走到 _run_stage 的 stage 才发（build 的循环
        # 已经跳过了未启用 / resolver 声明 skip 的阶段）。
        from app.agent.context.trace import stage_metrics, stage_span

        span = stage_span(
            getattr(state.request, "trace", None),
            stage,
        )

        try:
            await method(state)

        except BaseException as exc:
            if span is not None:
                span.fail(exc)
            raise

        else:
            if span is not None:
                span.complete(payload=stage_metrics(state))

    def _unsupported(self, stage: StageId) -> Exception:
        """
        跑了一个自己没有实现的阶段。

        信息里必须带上 enabled_stages：这条错误的实际成因是
        **声明与实现对不上**。不指明"这个实现声明了什么"的话，
        看日志的人只能去翻代码。
        """

        return NotImplementedError(
            f"{type(self).__name__}"
            f"（{self.builder_type.value if self.builder_type else '未归类'}）"
            f"没有实现 {stage.value}。"
            f"它声明参与的阶段是："
            f"{sorted(s.value for s in self.enabled_stages)}。"
        )


# ============================================================
# 辅助
# ============================================================


def _is_unimplemented(
        cls: type["BaseContextBuilder"],
        stage: StageId,
) -> bool:
    """
    这个类声明了某个阶段，却没写它的实现吗。

    判据有两条，缺一不可：

        基类那个方法带 `_no_default` 标记（基类没有可用默认值）
        且
        子类没有覆写它（拿到的还是基类那个函数对象）

    第一条是必须的：`CONTEXT_RESOLVER` 在基类里**有**真实默认值，
    不覆写是合法的。用标记而不是在别处维护一份"哪些阶段有默认值"
    的清单 —— 清单会漂移，而漂移的后果是这条校验悄悄失效。
    """

    base_method = getattr(BaseContextBuilder, cls._STAGE_METHODS[stage])

    if getattr(
            base_method, "__context_builder_no_default__", None
    ) is None:
        return False

    return getattr(cls, cls._STAGE_METHODS[stage]) is base_method


# ============================================================
# 六个 type 基类
# ============================================================
#
# builder_type 在这里定死，实现不再声明 —— 与 document 模块把 role
# 定死在 role 基类、tool 模块把 processor_type 定死在 type 基类是
# 同一个理由：这是**分类**，不是选项。
#
# execution_mode 也在这里给出：它是**类型**的性质（这个类型的
# builder 通常跑在哪），不是某个实现的选择。
#
# ⚠️ 六者都**不是** `inspect.isabstract()`（基类故意没有抽象方法），
# 所以发现逻辑排除它们靠的是结构规则而不是 isabstract —— 详见
# registry.py。删光某个 type 基类下所有实现不会让它被误当成实现
# 注册，因为它始终是 BaseContextBuilder 的直接子类。


class AgentInvokeContextBuilder(BaseContextBuilder, ABC):
    """
    主 Agent / ReAct 调用前的上下文构建。

    通常包含 history / memory / knowledge / capabilities /
    budgeting，必要时还包括 compaction —— 是六个类型里最全的一个。
    本仓库的 react 图用的就是它（见 graph/react/catalog.py）。
    """

    builder_type: ClassVar[ContextBuilderType] = (
        ContextBuilderType.AGENT_INVOKE
    )

    execution_mode: ClassVar[BuilderExecutionMode] = (
        BuilderExecutionMode.EMBEDDED_MODEL_INVOCATION
    )


class PlannerContextBuilder(BaseContextBuilder, ABC):
    """
    独立规划模型调用的上下文。

    通常只用**任务范围内**的 history 与规划指令，不需要完整执行
    历史 —— 规划要的是目标与约束，不是走过的一长串弯路。
    """

    builder_type: ClassVar[ContextBuilderType] = (
        ContextBuilderType.PLANNER
    )

    execution_mode: ClassVar[BuilderExecutionMode] = (
        BuilderExecutionMode.INDEPENDENT_WORKFLOW
    )


class ClassifierContextBuilder(BaseContextBuilder, ABC):
    """
    路由 / 分类用的窄输入。

    默认**没有**对话历史、memory、knowledge 与工具执行上下文 ——
    它的任务是把一句话分到某一类，多余的东西只会干扰判断并推高
    延迟。因此它是最可能只声明两三个阶段（resolver + assembler）
    的一个类型。
    """

    builder_type: ClassVar[ContextBuilderType] = (
        ContextBuilderType.CLASSIFIER
    )

    execution_mode: ClassVar[BuilderExecutionMode] = (
        BuilderExecutionMode.EMBEDDED_MODEL_INVOCATION
    )


class CompactionContextBuilder(BaseContextBuilder, ABC):
    """
    辅助 LLM 调用的上下文，目的是摘要 / 压缩既有上下文。

    它通常**由别的 builder 嵌套调用**（当预算或策略要求压缩时），
    而不是被图直接调用 —— 见 base.py 里 CONTEXT_COMPACTOR 阶段的
    说明。execution_mode 因此是 EITHER：内嵌是常态，独立跑也行。
    """

    builder_type: ClassVar[ContextBuilderType] = (
        ContextBuilderType.COMPACTION
    )

    execution_mode: ClassVar[BuilderExecutionMode] = (
        BuilderExecutionMode.EITHER
    )


class MemoryContextBuilder(BaseContextBuilder, ABC):
    """
    memory 抽取 / 反思用的上下文。

    可以由**独立后台工作流**执行，因此只应包含该 scope 与策略下
    允许抽取的信息 —— 它是六个类型里唯一一个"输入本身就是敏感面"
    的：别的类型是把资料喂给模型回答用户，这个是把对话喂给模型
    写进长期记忆。
    """

    builder_type: ClassVar[ContextBuilderType] = (
        ContextBuilderType.MEMORY
    )

    execution_mode: ClassVar[BuilderExecutionMode] = (
        BuilderExecutionMode.INDEPENDENT_WORKFLOW
    )


class SubAgentContextBuilder(BaseContextBuilder, ABC):
    """
    隔离到子任务 scope 的上下文。

    通常只看到父 Agent 历史的**简化视图**与部分能力 —— 子 Agent
    要的是"够用且不跑偏"，给它完整父历史既浪费预算也容易让它
    越界去回答不该它回答的问题。
    """

    builder_type: ClassVar[ContextBuilderType] = (
        ContextBuilderType.SUB_AGENT
    )

    execution_mode: ClassVar[BuilderExecutionMode] = (
        BuilderExecutionMode.EMBEDDED_MODEL_INVOCATION
    )
