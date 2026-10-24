"""
把 ReAct 图接进通用运行时。

**本模块只有"注册图"这一个函数**
--------------------------------
document 那边的 runtime.py 有两个：注册图，以及 `build_dispatch_handler`
（把 DispatchEnvelope 翻成 RunSnapshot + input_state 去跑）。

本图**刻意没有**第二个，而且以后也不会加在这里 —— handler 在
`app/services/session/application/agent_run_dispatch.py`。

为什么它不在这个模块里
--------------------
因为它需要 domain 知识，而那些知识属于 session 模块，不属于图：

    一次运行的主体是谁        会话行（agent_session.session_id）
    运行的状态写到哪          agent_run 表（契约状态，不是观测状态）
    用户输入怎么变成 state    CreateRunRequest.input.text -> HumanMessage

当初悬置的那个问题（"一次 agent 运行的主体是什么"）由 session 模块回答：
**主体是它所属的会话那一行**，session_id 就是 `new_thread_id` 要的那个
UUID 主体。图这边只提供 GraphDefinition 与 context 工厂，不解释这些。

另一个原因更硬：handler 不能用 infra 的 `GraphRunner`（它的终态推导以节点行
为唯一依据，而这条链路不写节点行），所以它本来就不是"照抄 document"的形状。

模型依赖走 ModelRegistry，不是模型实例（规格 §16）
------------------------------------------------
     register_react_graph(model=ChatDeepSeek(...))         ❌ 旧的形状
     register_react_graph(model_registry=model_registry)   ✅ 现在的形状

更不能是 `(provider, model_name, api_key)`。理由：

    Graph 注册阶段只注入 ModelRegistry；
    之后由 LLM Node 在运行时通过 Registry 获取 Runtime。

因此"用哪个模型"这件事在**这张图里根本没有出现过** —— 它由 Model
模块的 Resolver 按 `(agent_type, model_role)` 决定（见
graph/react/catalog.py 的 AGENT_TYPE / MODEL_ROLE）。

工具集由**能力分类**驱动（本图的一条完整链路）
------------------------------------------------
    builder 节点   把用户输入归到能力 code（modules）
    ↓ 查库（ACTIVE、精确匹配 code）
    写 context_policy.tool_keys
    ↓ capability 阶段（第 7 阶段）
    ToolExposureSource 解析成可 bind 的工具对象 -> ModelRequest.tools
    ↓ adapters._bind
    bind_tools
    ↓ 模型要求调用
    tools 节点   同一个 select_specs + 同一个 tool_keys 算允许集，
                 逐个走处理器链执行（app/infra/agent/tool_executor.py）

⚠️ 这条链要在**启动期**把工具注册表扫一遍才算完整（见下面 tool_registry），
否则注册表里空着，表现是"模型一个工具都看不见"，而且不报错。

⚠️ `get_model_registry()` 在**启动期**调
--------------------------------------
它会读 model.yaml 并装配注册表：放在启动期，"目录里少一个模型"这类配置错
会让应用起不来，而不是等某个用户踩到。调用点在 app/main.py 的 lifespan，
与 register_document_pipeline 并排。
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.context.builder.base import BaseContextBuilder
from app.agent.context.registry import create_context_builder
from app.agent.model import ModelRegistry
from app.agent.plan.publish import build_task_board
from app.infra.agent.tool_exposure import RegistryToolExposureSource
from app.infra.graph import GraphDefinition, GraphRegistry, RunContext
from app.services.tool.registry import get_tool_registry

from .catalog import (
    CLASSIFIER_BUILDER_TYPE,
    CONTEXT_BUILDER_TYPE,
    PLANNER_BUILDER_TYPE,
    SUBAGENT_BUILDER_TYPE,
)
from .context import ReactContext
from .definition import build_graph_definition
from .graph import build_react_graph

if TYPE_CHECKING:
    from app.agent.context.updater import ContextUpdater

__all__ = ["NodeRunWriter", "register_react_graph"]

logger = logging.getLogger(__name__)


#: ⚠️ 这里**没有** SUBJECT_TYPE
#: --------------------------
#: document 的 runtime.py 有一个（"document_version"），用来填
#: `RunSnapshot.subject_type`。本图不需要：不写 RunSnapshot，也就没有
#: 主体类别可填。当初为它留的那个 "agent_conversation" 预备值已随本次
#: 接线删掉 —— 主体问题现在有答案了（**会话行**，见 session 模块），
#: 而答案落在 agent_run 这张业务表上，不经过观测层。
#: 将来接节点级观测时，它会连同它的消费者一起回来。

@runtime_checkable
class NodeRunWriter(Protocol):
    """
    本图对 recorder 的**全部**要求：能写节点行。

    比 infra 的 `GraphRunRecorder` 窄得多，这是刻意的 —— 那个协议还包含
    create_run / mark_run_running / finish_run / summarize（运行行的生命周期），
    而运行行是**调用方**的事：谁发起这次运行，谁记录它。图只写"我走到了
    哪个节点"。

    窄类型的好处是可以说清楚"什么算合格"：document 那个完整的
    SqlAlchemyRunRecorder 满足它，session 模块那个只打日志的实现也满足它，
    而一个只有运行行、不写节点的实现则明确不合格。
    """

    async def upsert_node_run(self, update: Any) -> None:
        """写一次节点行的 upsert（node_span 在开始/失败/结束时各调一次）。"""
        ...


def register_react_graph(
        *,
        registry: GraphRegistry,
        checkpointer: BaseCheckpointSaver,
        recorder: NodeRunWriter,
        model_registry: ModelRegistry,
        builder: BaseContextBuilder | None = None,
        context_updater: "ContextUpdater | None" = None,
        store: Any = None,
) -> GraphDefinition:
    """
    编译图、包装成 GraphDefinition、注册进注册表。

    签名与 `register_document_pipeline` 对齐：组合根在 lifespan 里把
    两者并排调用，多出来的是本图特有的依赖（model_registry / builder /
    context_updater）。

    ⚠️ **"类型 -> builder 实现"在这里解析一次**，不在每次模型调用时
    -----------------------------------------------------------------
    本图用哪个 builder_type 是编译期常量（catalog.CONTEXT_BUILDER_TYPE），
    不是运行时数据 —— 所以查表属于启动期的接线，不属于执行路径。
    放在这里还有一个实际好处：没有任何 `agent_invoke` 实现时，
    `require()` 会在**应用启动时**抛错，而不是等第一次真实对话才失败。

    Parameters
    ----------
    checkpointer
        由调用方传入而不是在这里从 provider 取 —— 本函数因此不关心
        用哪种（生产 Postgres、冒烟 InMemory），也更容易在测试里替换。
    recorder
        节点行的写入口，类型是上面那个 **窄协议** `NodeRunWriter` ——
        只要有一个 `upsert_node_run` 就合格，不要求实现运行行的生命周期。
        生产传 session 模块的 `NodeLogRecorder`（只打日志，不落库），
        将来接 SSE 时换成往流里推事件的那个；冒烟脚本传一个内存实现。
    model_registry
        **本图与模型之间的唯一依赖**（规格 §16）。用哪个 provider /
        哪个模型 / 什么降级策略全在它里面，图一概不知情 —— 因此
        "换个模型跑一下"不需要动这张图的任何一行代码。
        生产传 `get_model_registry()`（启动期装配），测试传一个内存
        Registry 即可，不必造真的 provider 客户端。
    builder
        本次用的 Context Builder **实例**。不传就按
        `CONTEXT_BUILDER_TYPE` 解析一次 —— 生产走这条路，
        测试/定制可以显式传一个进来。
    context_updater
        调用之后的回写口子。**现在必然是 None** —— 后置处理还没有
        设计与实现，见 app/agent/context/updater.py。
    """

    # 解析一次。不传就按类型取 —— 拿不到是配置不全，而这是启动期，
    # 早失败正是我们要的。
    context_builder = (
        builder
        if builder is not None
        else create_context_builder(CONTEXT_BUILDER_TYPE)
    )

    # 第二种 builder：Agent Builder 节点的窄输入。
    #
    # ⚠️ 它没有 `classifier_builder` 那样的显式参数
    # -------------------------------------------------
    # 因为它的用途只有一个（分类），没有"测试要注入另一种"的场景 ——
    # 要定制就把 classifier 那个实现换掉，而不是在这里传一个进来。
    # 解析失败同样在启动期炸（"没有任何 classifier 的实现"）。
    classifier_builder = create_context_builder(
        CLASSIFIER_BUILDER_TYPE
    )

    # planner 的窄输入 builder：resolver + summary + history + prompt。
    # 与 classifier 一样在启动期解析，缺实现直接报错。
    planner_builder = create_context_builder(
        PLANNER_BUILDER_TYPE
    )

    # 子代理子图的 llm_node 用同一个 context 取它。在这里解析而不是在
    # 子图首次运行时解析：缺实现不该等到 worker 已经 claim 了任务才炸
    # （那时任务 lease 已经开始走了）。
    subagent_builder = create_context_builder(
        SUBAGENT_BUILDER_TYPE
    )

    state_graph = build_react_graph()

    # 工具注册表：**启动期扫一遍**。
    #
    # @tool 的登记发生在模块被 import 的那一刻（描述符的 __set_name__），
    # 而"这个进程 import 过什么"是运行期状态 —— 不扫的话，能不能暴露
    # 工具就取决于此前有没有谁碰巧 import 过那个 service 模块（比如管理面
    # 跑过一次工具同步）。那是最难查的一类依赖：同一个应用，先点过"同步"
    # 的实例有工具、没点过的没有，代码一模一样。
    #
    # 扫一遍之后，注册表的内容只由**代码库**决定。代价是一次性的包遍历
    # （与同步、冒烟脚本走的是同一个 ToolRegistry.scan）。
    tool_registry = get_tool_registry()
    tool_registry.scan()
    logger.info("工具注册表已装载：%d 个工具。", len(tool_registry))

    tool_source = RegistryToolExposureSource()

    # store 是**跨轮存储**（langgraph 的 BaseStore），工具处理器通过
    # `runtime.store` 读写它。目前唯一的消费者是审批拦截器的批准账本
    # （见 app/agent/tools/processor/interceptor/approval.py）。
    #
    # ⚠️ 不传 store 时 `runtime.store` 是 None，审批拦截器会退回**进程内**
    # 账本并打 warning —— 那份账本不跨重启，重启后 `waiting_input` 的运行
    # 恢复时已经批过的会再问一遍。所以生产必须给一个持久实现
    # （如 langgraph 的 Postgres store）。
    compiled = state_graph.compile(
        checkpointer=checkpointer,
        store=store,
    )

    def context_factory(run_context: RunContext) -> ReactContext:
        """
        每次运行新建一个 context。

        不能复用 —— 它内含本次运行独有的 pipeline_run_id，复用会让
        观测写串到别的运行上（与 DocumentPipelineContext 同一条要求）。

        ⚠️ model_registry 在这里是**引用传递**：Registry 是进程级的，
        不是"每次运行的东西"。放进来只是为了让节点拿得到它（见
        graph/react/context.py 的说明）。
        """
        return ReactContext(
            pipeline_run_id=run_context.pipeline_run_id,
            recorder=recorder,
            model_registry=model_registry,
            builder=context_builder,
            classifier_builder=classifier_builder,
            planner_builder=planner_builder,
            subagent_builder=subagent_builder,
            # 任务板：进程级 Redis 客户端，每次运行包一层 holder。
            # Redis 未装配时是 None —— planner 会把发布降级写进 notes，
            # 而不是让整轮失败（见 plan/publish.py 的失败语义）。
            task_board=build_task_board(),
            context_updater=context_updater,
            tool_source=tool_source,
            # 业务会话。与 caller_identity 同一条取法：直接取属性、
            # 不给 getattr 兜底 —— 它是 RunContext 契约的一部分（默认
            # None），拿不到说明传进来的不是一个 RunContext。
            #
            # ⚠️ 它为 None 的后果比 caller_identity 更**安静**：账本拿不到
            # 会话就不执行工具，于是表现成"模型一个工具都用不了"，而那
            # 与"这次没暴露工具"看起来一模一样。所以这条必须接上。
            session_id=run_context.session_id,
            # 调用者身份由**派发方**交进来（它才有那份快照）：本图不查、
            # 也不解释它的形状，只是原样传给需要的节点（tools 节点把它
            # 交给执行器，供工具级的鉴权处理器判权）。
            #
            # ⚠️ 直接取属性、不给 getattr 兜底：`caller_identity` 是
            # RunContext 契约的一部分（默认 None），拿不到说明传进来的
            # 不是一个 RunContext。悄悄兜成 None 会让"身份没接上"表现成
            # "这个人没角色" —— 而两者的后果都是拒绝，没法从现象区分。
            caller_identity=run_context.caller_identity,
            # 这一轮的职责（= 记忆 scope 的 agent 那一半）。与 session_id /
            # caller_identity 同一条取法：直接取属性、不给 getattr 兜底 ——
            # 它是 RunContext 契约的一部分（默认 None）。拿不到的后果是
            # "需要 agent_id 的那个工具取不到那篇场景"，比"整轮失败"安静得多，
            # 所以这里必须与契约对齐（见 RunContext.duty_key 的说明）。
            duty_key=run_context.duty_key,
            # Trace 聚合键。与 session_id / caller_identity 同一条取法：
            # 直接取属性、不给 getattr 兜底 —— 它是 RunContext 契约的一部分
            # （默认 None）。拿不到的后果是"这次运行静默不采 trace"，
            # 比整轮失败安静得多，所以必须与契约对齐。
            trace_id=run_context.trace_id,
        )

    definition = build_graph_definition(
        compiled=compiled,
        context_factory=context_factory,
    )

    registry.register(definition)

    logger.info(
        "Registered graph %s v%s (fingerprint %s).",
        definition.name,
        definition.version,
        definition.topology_fingerprint[:12],
    )

    return definition
