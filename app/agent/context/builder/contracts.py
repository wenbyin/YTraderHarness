"""
Context Builder **执行期**看到的东西。

这是一份**契约形状**，不是已经接通的数据结构
------------------------------------------------
本模块只定义"一次构建过程中，各阶段之间传递什么"，不接任何
检索 / 预算 / 压缩的实现。与 app/agent/tools/processor/context.py
的立场相同：形状先固定，实现才能各自写，调用方接进来时不必回头改。

为什么用 dataclass 而不是 pydantic
----------------------------------
pydantic 在**类创建时**就要求值注解，会逼出对 langchain_core 的
硬依赖（`ModelRequest.messages` 的类型是 `BaseMessage`）。而本模块
会被发现逻辑 import，而它又会被将来的管理面接口 import —— 那些
场景不该被 langchain 绑住。

这正是 `BaseToolProcessor` 把 `ToolRuntime` 放进 `TYPE_CHECKING`
的同一条理由，连手法都一样：`from __future__ import annotations`
让注解不求值，运行时拿不到 langchain 也不影响 import。

⚠️ 不要为了"校验方便"换成 pydantic：那会让 import 本模块立刻
拉起 openai / httpx 一整串依赖。

ModelRequest 只有一处定义（在 Model 模块）
----------------------------------------
`context_builder.yaml` 的 `FinalOutputContract.type` 只有一个取值
`model_request`，字段是 messages / tool definitions / response
format / model-facing metadata。

那个类型**就是** `app/agent/model/types.py::ModelRequest` —— 本模块
直接复用（见下面那行 import），不再并列第二份。理由：

    一次 LLM 调用只有一个请求类型。Context Builder 产出它、Model
    Runtime 消费它、观测 / 审计 / 测试都读它。两份同形类型一定会漂移，
    而漂移的表现是"某个字段在一个方向上有效、在另一个方向上悄悄丢了"。

于是"上下文构建"到"模型调用"之间**没有转换层**：

    ContextBuilder -> ModelRequest -> ModelRuntime.ainvoke(...)

依赖方向仍然是单向的：本模块 import Model 模块，Model 模块不认识
Context（它只认识自己那套 Model / Provider / Invocation 概念）。

**刻意不加 provider / model 字段**：模型选择属于 Model 模块的 Resolver
（谁建的这个图用哪个 profile，见 graph/react/catalog.py 的 AGENT_TYPE /
MODEL_ROLE），Context Builder 只负责"上下文长什么样"。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.agent.model import ModelRequest
from app.common.context import ContextBuilderType, StageId

if TYPE_CHECKING:
    # 只在类型检查时导入 —— 理由见模块 docstring。
    from langchain_core.messages import BaseMessage

    from app.agent.context.processor.dependencies import ContextDependencies

__all__ = [
    "LLM_INVOCATION_STAGES",
    "PIPELINE_TEMPLATE",
    "ContextBuildRequest",
    "ContextBuildState",
    "ModelRequest",
    "StagePolicy",
    "StageResolution",
]


# ============================================================
# 固定流水线
# ============================================================

#: 十三个阶段的**执行顺序**。
#:
#: ⚠️ 这是**显式字面量**，不是 `tuple(StageId)`
#: -------------------------------------------------
#: 从枚举推导看起来更 DRY，但它把两个不同的东西绑成了一件：
#: "枚举里有哪些成员"和"流水线按什么顺序跑"。绑上之后，有人为了
#: 排版把 StageId 的成员挪一下，流水线顺序就**静默**变了 ——
#: 没有报错，只有行为改变。
#:
#: 写成字面量的代价是"加了新阶段忘了往这里加"，那个由下面那段
#: 启动期校验接住。
#:
#: ⚠️ TASK_RESOLVER / SKILL_LOADER 是 1.1.0 的**显式插入**
#: ----------------------------------------------------
#: 它们必须跑在 PROMPT_ASSEMBLER 之前，因此不是追加到末尾。这是一次
#: 有意的流水线契约变更：所有 builder 的 enabled_stages 与测试一起
#: 更新；不声明这两个阶段的 builder 会按 gate 跳过它们。
PIPELINE_TEMPLATE: tuple[StageId, ...] = (
    StageId.CONTEXT_RESOLVER,
    StageId.TASK_RESOLVER,
    StageId.SKILL_LOADER,
    StageId.PROFILE_LOADER,
    StageId.SUMMARY_LOADER,
    StageId.HISTORY_SELECTOR,
    StageId.MEMORY_RETRIEVER,
    StageId.KNOWLEDGE_RETRIEVER,
    StageId.CAPABILITY_LOADER,
    StageId.PROMPT_ASSEMBLER,
    StageId.TOKEN_BUDGETER,
    StageId.CONTEXT_COMPACTOR,
    StageId.CONTEXT_VALIDATOR,
)

#: 唯一会**自己调 LLM** 的阶段。
#:
#: 规格的 `PipelineNode.llm_invocation` 就是这个意思，也解释了
#: `ContextBuilderType.COMPACTION` 为什么必须独立存在：压缩要给
#: 父调用腾出预算，它自己那次调用也得有上下文，于是它由一个
#: builder 嵌套调用另一个 builder（规格的 `nested_invocations`）。
#:
#: 单列成集合是因为这件事有两个实际后果：
#:     观测    这一阶段的耗时里包含一次 LLM 调用，不能与别的阶段并排比
#: 预算    它自己也要花 token，"省下来的"得减去它花掉的
#:
#: ⚠️ 这里是 SUMMARY_LOADER，不是 CONTEXT_COMPACTOR
#: ---------------------------------------------
#: 生成摘要曾经住在 compactor（"达到 token 阈值就压缩"）。现在压缩只做
#: message 列表的切片，生成归产生它的那个阶段的**语义**：摘要属于
#: SUMMARY_LOADER，它按轮次窗口判断"该不该重新生成"（见
#: processors/summary.py 与 processor/turns.py）。
#:
#: 改这一项会同时改掉观测口径与预算口径 —— 它不是注释，是契约。
LLM_INVOCATION_STAGES: frozenset[StageId] = frozenset(
    {StageId.SUMMARY_LOADER}
)


def _assert_template_complete() -> None:
    """
    模板与枚举必须一一对应。

    漏一个的后果是**无声的**：那个阶段永远不进 `build()` 的循环，
    即便某个实现声明了它、也实现了它 —— 声明通过、方法存在、
    就是从来不被调用。这类"配了但不生效"的错最难查，所以在
    import 期就炸掉，而不是留给某次运行。

    多一个（模板里有、枚举里没有）在 Python 上不可能 —— 字面量
    写错成员名会直接 NameError。这里仍然判等，是因为**顺序**也
    要一致：真按"追加"而不是"重排"来演进的话，两边应当同序。
    """

    if tuple(StageId) != PIPELINE_TEMPLATE:
        raise RuntimeError(
            "PIPELINE_TEMPLATE 与 StageId 不一致：\n"
            f"  枚举：{tuple(s.value for s in StageId)}\n"
            f"  模板：{tuple(s.value for s in PIPELINE_TEMPLATE)}\n"
            "新增阶段必须同时更新 StageId 与 PIPELINE_TEMPLATE；"
            "顺序本身是对外契约（规格里写明 stage order 由服务器定义"
            "且不可变）。1.1.0 的 TASK_RESOLVER / SKILL_LOADER 是为了"
            "让 task 模式在 PROMPT_ASSEMBLER 之前生效而做的显式插入。"
        )


#: import 期就校验，不留给某次运行。
_assert_template_complete()


# ============================================================
# 阶段之间传递的东西
# ============================================================


@dataclass(frozen=True, slots=True)
class StageResolution:
    """
    阶段 1（CONTEXT_RESOLVER）的产出。

    它是**唯一**能让本次运行偏离"实现声明的参与关系"的东西：
    声明说的是"这个 builder 类型能做什么"（静态、类级），这里说的
    是"这一次需要做什么"（动态、每次调用）。两者是不同的问题，
    因此不合并成一个集合。

    skipped 的边界由 base.build() 显式校验，不静默求交：
        跳过了没声明的阶段        -> ValueError
        跳过了声明为 required 的  -> ValueError
    """

    #: 本次运行**不要跑**的阶段。
    #:
    #: 只能收窄，不能扩张：不在实现 enabled_stages 里的阶段本来
    #: 就不跑，写进来是 resolver 的 bug，会被 build() 拦下。
    skipped: frozenset[StageId] = frozenset()

    #: 各阶段的参数。键是阶段，值是那个阶段的参数。
    #:
    #: 用 Mapping 而不是 dict 的裸别名，是为了说清"这里只是读"：
    #: 参数的**形状**由各阶段自己定义（规格里是
    #: StagePolicy.parameters），本模块不认识其中任何一个键。
    parameters: Mapping[StageId, Mapping[str, Any]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ContextBuildRequest:
    """
    一次构建的**输入**。

    thread_id 与 run_id 是给阶段的取数依据：历史、画像、记忆都
    挂在某个会话上，检索要按会话去查。两者都由调用方带进来，而不是
    让阶段自己去 runtime 里摸 —— 阶段不是图节点，拿不到 runtime。

    ⚠️ 后四个字段（graph_state / agent_type / model_role / dependencies）
    -------------------------------------------------------------------
    它们是 Processor 流水线接进来之后加的，**全部带默认值**，因此既有
    调用方（只传前四个）不受影响。

        graph_state    图 state 的**只读快照**（Processor 的 State Path
                       绑定解析的就是它）。它让"上下文从哪来"变成一份
                       可声明、可观测的绑定，而不是 Processor 自己去摸
                       全局。
        agent_type     Agent 类型（模型路由键之一，见 model/resolver.py）。
        model_role     本次模型调用的模型侧角色（reasoning / summarization…）。
        dependencies   本次构建的外部依赖束（ModelRegistry / 上下文源…）。

    为什么 dependencies 是 **request 字段**而不是构造参数：Builder 实例
    是共享的、且 `__setattr__` 被封死（见 builder/base.py 里那段"并发
    正确性的保证"），构造期注入无处存放；而注册表按类反射、零参构造
    实例，也没有注入入口。随请求进、放进工作集，是这套约束下唯一
    干净的做法（见 processor/dependencies.py）。
    """

    builder_type: ContextBuilderType

    #: 会话标识。langgraph 的 checkpoint thread_id 就是它。
    thread_id: str

    #: 进模型之前已有的消息（通常是图 state 里的 messages）。
    messages: tuple[BaseMessage, ...] = ()

    #: 本次运行的 id，用于把检索 / 观测串起来。脱离图运行时为 None。
    run_id: UUID | None = None

    #: 图 state 的只读快照（Processor 的绑定根）。
    graph_state: Mapping[str, Any] = field(default_factory=dict)

    #: Agent 类型（模型路由键）。空串 = 由 Registry 侧按默认处理。
    agent_type: str = ""

    #: 本次模型调用的模型侧角色。空串 = reasoning（主调用）。
    model_role: str = ""

    #: 本次构建的外部依赖（缺省 = 进程级默认）。
    dependencies: "ContextDependencies | None" = None

    #: 本次构建的 Trace 上下文（None = 未开启）。
    #:
    #: 它是 per-call 的，跟着请求进工作集 —— builder 实例是共享的，
    #: 不能把这东西挂到 self 上（见类的说明）。
    trace: Any = None


@dataclass(frozen=True, slots=True)
class StagePolicy:
    """
    一个阶段的**参与关系** —— 规格 `StagePolicy` 的运行时子集。

    只保留运行时真正读的两个位，其余**刻意不在这里**
    --------------------------------------------------
    规格里还有 policy_ref / parameters / execute_when / skip_when /
    input_contract / output_contract，但它们只被尚未存在的
    `/context-builders` 接口消费。现在搬进来等于替那个接口猜形状，
    而猜错的成本是"改契约"而不是"加字段"。

    等到真要出那个接口时，从 base.py 的类属性派生一份
    definition 的视图即可 —— 那时候的字段需求是明确的，不是猜的。
    """

    stage_id: StageId

    #: 这个 builder 类型是否**声明**参与该阶段（静态）。
    enabled: bool

    #: 在固定流水线里的位置，0 起。
    execution_order: int

    #: 声明为 required 的阶段**不可被 runtime 跳过**（见
    #: StageResolution.skipped 的边界校验）。
    required: bool = False


# ============================================================
# ModelRequest
# ============================================================
#
# 请求类型**只有一处定义**：`app.agent.model.types.ModelRequest`。
# 本模块在顶部 import 了它，因此下面两种写法都指向同一个类：
#
#     from app.agent.context.builder.contracts import ModelRequest
#     from app.agent.model import ModelRequest
#
# 这里刻意**不**再写一个同形 dataclass：两份同形类型一定会漂移，而
# 漂移的表现是"某个字段在一个方向上有效、在另一个方向上悄悄丢了"，
# 那种问题在排查时几乎不可能一眼看出来。


@dataclass(slots=True)
class ContextBuildState:
    """
    一次构建的**工作集**。

    每次 build() 新建一个，阶段方法就地改它 —— 这是本模块里唯一
    可变、且**非 frozen** 的类型，因为它就是这个用途。

    ⚠️ 它**不是** langgraph 的 state
    -------------------------------
    阶段不是图节点（规格的 `PipelineNode.independent_lifecycle` 明写
    "does not automatically make it a LangGraph node"）。这个对象
    只在一次 build() 调用里存活，不进 checkpoint、不跨节点。

    ⚠️ 它**不挂在 self 上**
    ----------------------
    builder 实例是**共享的**（每个类型只构造一次，会被并发的每次
    模型调用同时使用）。把本次调用的工作集写到 self 上，等于让 A
    调用读到 B 调用的状态 —— 症状是"偶发地拼错了上下文"，几乎不可能
    从日志里看出来。与 BaseToolProcessor 对 runtime 的警告同一条。

    因此：`self` 上只该有从类声明推导出来的不可变东西；凡是"这次
    调用的东西"，都走参数。
    """

    request: ContextBuildRequest

    #: 阶段 1 的产出。阶段 1 之前是默认值（不跳过任何阶段）。
    resolution: StageResolution = field(
        default_factory=StageResolution
    )

    #: 各阶段产出的中间物，键是阶段。
    #:
    #: 阶段之间**不直接互相调用**，只通过这里交换数据 —— 谁产出的、
    #: 谁消费的，看键就知道。否则十个阶段会退化成一条隐式的调用链，
    #: 而"谁改了这个值"就没人说得清了。
    artifacts: dict[StageId, Any] = field(default_factory=dict)

    #: Processor 流水线的**本次工作集**（`processor/context.py`）。
    #:
    #: None = 这个 Builder 没有在用 Processor 流水线（既有的 stage 实现
    #: 不需要它）。它就是"这次调用的东西"的容器，因此放在这里而不是
    #: 挂在共享的 Builder 实例上 —— 与下面那段"它不挂在 self 上"同一条
    #: 纪律。
    processor_context: Any = None

    #: 待交给模型的请求。
    #:
    #: 由 PROMPT_ASSEMBLER 填充（TOKEN_BUDGETER / CONTEXT_COMPACTOR
    #: 可能改写它）。build() 结束时它仍为 None 说明流水线没产出任何
    #: 东西，那是错误而不是空结果 —— 见 base.build() 的收尾校验。
    draft: ModelRequest | None = None
