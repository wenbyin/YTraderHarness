"""
ReAct 图的运行时依赖（LangGraph 的 context_schema）。

传法（容易踩的坑）
------------------
LangGraph 1.x 的 context **不是**放在 config 里传的，而是
ainvoke / astream 的关键字参数：

    await graph.ainvoke(input, config, context=ctx)    ✅
    await graph.ainvoke(input, {"context": ctx})       ❌ 静默变成 None

传错时 `runtime.context` 是 None，报错离原因很远；
`instrumentation._require_context` 会给出能直接照着修的提示。

为什么模型相关的东西是 **ModelRegistry** 而不是模型实例
-----------------------------------------------------
（这一条是本图与 Model 模块之间最重要的边界，见规格 §3、§15、§16）

    register_react_graph(model=ChatDeepSeek(...))     ❌ 旧的形状
    register_react_graph(model_registry=registry)     ✅ 现在的形状

持 Registry 而不是持模型，换来三件事：

    1. 节点不感知 provider / model name / API key（那些是 Model 模块
       内部职责），也就没有"这个节点只会在 deepseek 上跑"这种事。
    2. 模型选择的决定权留在 Model 模块的 Resolver 里 —— 节点只说
       "我是 react 的 reasoning 调用"，而不说"我要 gpt-5"。
    3. 测试可以注入一个假 Registry，不必造真的 provider 客户端。

⚠️ 节点**不要**把 Registry 里的东西缓存到别处
--------------------------------------------
`get_runtime(...)` 每次调用都要走一遍（Registry 内部有缓存，它很便宜）。
在节点或 context 上缓存 runtime 会让"profile 改了但图还在用旧的"
变成一次重启才能解决的问题。

builder / classifier_builder
---------------------------
本次用的 Context Builder **实例**。两个，因为本图有**两种**调用语义：

    builder                 agent 节点：完整上下文（AGENT_INVOKE）
    classifier_builder      builder 节点：窄输入（CLASSIFIER）

⚠️ 是实例，不是按类型查表的注册表
--------------------------------
每个节点用哪个 builder_type 都是**编译期常量**
（catalog.CONTEXT_BUILDER_TYPE / CLASSIFIER_BUILDER_TYPE），不是运行时
数据 —— 没有任何节点在运行时挑 builder。所以"类型 -> 实现"这一步在
**启动时解析一次**就够（见 runtime.register_react_graph），没必要每次
模型调用都查一遍表 —— 那是 document 流水线的形状，那边用哪个能力确实
由数据库里的配置决定。

持实例的另一个好处：解析失败在**启动时**就炸（"没有任何 agent_invoke /
classifier 的实现"），而不是等第一次真实对话才报错。

context_updater
---------------
调用**之后**的回写口子。**可空**，而且现在必然为 None —— 后置处理还
没有实现（见 context/updater.py）。为 None 时节点跳过这一步，而不是去
找一个空实现来调。

本类满足 infra 的 GraphRunContext Protocol
-----------------------------------------
`pipeline_run_id` + `recorder` 两个字段就够 `node_span` 用了
（它是 `runtime_checkable` 的，可以 isinstance 断言）。

⚠️ HITL 不在这里（规格 §10）
---------------------------
这个 context 里没有"要不要人工确认""工具是否高风险"之类的字段，
将来也不该有：HITL 是 **Graph 控制流**（LLM Node 之后的人类输入触发、
Tool Node 之前的人工审批），不是模型调用的依赖。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.agent.context.builder.base import BaseContextBuilder
from app.agent.model import ModelRegistry
from app.infra.graph.recorder import GraphRunRecorder

if TYPE_CHECKING:
    from app.agent.context.processor.sources import ToolExposureSource
    from app.agent.context.updater import ContextUpdater
    from app.agent.plan.publish import TaskBoard

__all__ = ["ReactContext"]


@dataclass(frozen=True, slots=True)
class ReactContext:
    """
    一次运行的依赖容器。

    ⚠️ **每次运行都要新建一个** —— 它内含本次运行独有的
    pipeline_run_id。复用会让观测写串到别的运行上（与
    DocumentPipelineContext 同一条要求）。

    model_registry 是**进程级**的（一个应用一份），放进这里是为了
    让节点拿得到它，而不是因为它是"每次运行的东西" —— 与
    `pipeline_run_id` 的性质完全不同。放这里也让测试能整体替换。
    """

    pipeline_run_id: UUID

    recorder: GraphRunRecorder

    #: 模型注册表 —— 节点通过它拿 ModelRuntime（规格 §3）。
    #:
    #: 类型是 Protocol（`ModelRegistry`），因此本图**不依赖** Model 模块
    #: 的任何具体实现：谁装配的、用的是文件目录还是内存目录，节点都
    #: 不需要知道。
    model_registry: ModelRegistry

    #: agent 节点用的 Context Builder **实例**（不是注册表，理由见类
    #: docstring）。组合根在启动时按 catalog.CONTEXT_BUILDER_TYPE 解析一次。
    builder: BaseContextBuilder

    #: builder 节点（Agent Builder）用的 Context Builder 实例。
    #: 按 catalog.CLASSIFIER_BUILDER_TYPE 解析一次。与上面那个是**两种
    #: 语义**（窄输入 vs 完整上下文），所以是两个实例而不是一个开关。
    classifier_builder: BaseContextBuilder

    #: planner 节点用的 Context Builder 实例。
    #:
    #: 窄输入：摘要 + 历史 + 当前用户输入，不含画像 / 记忆 / 知识 /
    #: 工具。按 catalog.PLANNER_BUILDER_TYPE 在启动期解析一次。
    #: 默认 None 是为了让既有测试 / 只构造旧字段的调用方不被迫补参；
    #: 生产 runtime 必须注入，否则 planner 会降级返回 {}。
    planner_builder: BaseContextBuilder | None = None

    #: 子代理子图（`llm_node`）用的 Context Builder 实例。
    #:
    #: 按 catalog 的 `SUB_AGENT` 类型在启动期解析一次。与 `builder` 是
    #: **两种语义**：主图注入完整对话上下文；子图只看到自己任务 scope
    #: 的 messages / summary / skill。
    #:
    #: 默认 None 是为了让只构造既有字段的测试不被迫补参；生产 runtime
    #: 必须注入，子图的 llm_node 拿不到时会显式报错（不静默退回主图
    #: builder —— 那会让 worker 看到父对话并沿用父提示词）。
    subagent_builder: BaseContextBuilder | None = None

    #: 任务板（Redis）—— planner 把 A2A task 发布到这里。
    #:
    #: None 是**合法状态**：本地没起 Redis、或这版只跑主图顺序执行时，
    #: 发布这一步会降级成「只写 state」，而不是让整轮失败。接上 worker
    #: 之后任务板变成调度权威，那时 None 必须按 fail-closed 处理。
    task_board: "TaskBoard | None" = None

    #: 调用之后的回写口子。None = 不回写（当前唯一的状态）。
    context_updater: "ContextUpdater | None" = None

    #: 工具暴露源 —— capability 阶段靠它把 `tool_keys` 解析成可 bind
    #: 的工具对象（见 context/processor/sources.py 的 Protocol）。
    #:
    #: 与 model_registry 同性质：**进程级、无状态**，放这里是为了让
    #: 节点拿得到，不是为了"每次运行的东西"。测试可以整体替换。
    #:
    #: ⚠️ 本图**恒**注入它（节点在 ContextDependencies 里传下去）。
    #: 那条分支一断，capability 处理器就会退回读 state 的
    #: `exposed_tools` —— 而本图没有任何节点写那个通道，于是症状是
    #: "模型静默地一个工具都看不见"。tools 节点的允许集是同一个源的
    #: 同一个推导，两处必须一起改。
    tool_source: "ToolExposureSource | None" = None

    #: 客户端传入的 Trace 聚合键（None = 未开启 Trace）。
    #:
    #: 节点只读它做两件事：判断要不要产生 trace 事件；把 trace_id 带进
    #: TraceEmitter 的信封。**不要**把它或 emitter 存进共享实例
    #: （ModelRuntime / builder / registry），也不进 graph state。
    trace_id: str | None = None

    #: 这条运行属于哪个**业务会话**（`agent_session.session_id`）。
    #:
    #: ⚠️ 不是 langgraph 的 `thread_id`
    #: ---------------------------------
    #: thread_id 是 langgraph 的内部键（形如 `dev:react:<uuid>`），由
    #: 会话行派生；而 session_id 是会话自己的主键 —— 会话详情页、消息、
    #: 运行列表全按它关联。
    #:
    #: 谁在用：工具执行账本（`tool_execution.session_id`）拿它做
    #: 第一段。**拿不到就不执行工具**（见 tools.py 的 `_LedgerUnavailable`）
    #: —— 一次没有记录的副作用不可追溯，而不可追溯比失败更糟。
    #:
    #: 与 `thread_id` 放这里不同（那个进 config），理由是同一条：它是
    #: 一次运行的属性，不进 state、每轮也不必重写。
    #:
    #: 默认 None：只有 document 那条流水线不经过会话。agent 这条路由
    #: 派发方填（见 agent_run_dispatch 的 RunContext）。
    session_id: UUID | None = None

    #: 这一轮用哪份**职责**（`agent_run.duty_key`）。
    #:
    #: 与 `caller_identity` / `session_id` 同一条取法：由派发方从运行行交
    #: 进来，运行期内不变。tools 节点把它交给 `ToolCaller`（见
    #: app/common/tool_caller.py 的 `agent_id`）—— 工具分支读不到图 state，
    #: 而记忆的 scope 是 (agent_id, user|org)，工具必须按与注入时**同一个**
    #: agent_id 定位，否则模型会"看到索引却取不到那一篇"。
    duty_key: str | None = None

    #: 发起这次运行的调用者的**身份快照**（`{"user_id","groups","roles"}`，
    #: 见 app/agent/tools/processor/context.py 的 build_caller_context）。
    #:
    #: ⚠️ 为什么放在这里而不是 state 通道
    #: 它是**每次运行**的属性，不是每轮的：身份在一次运行里不会变，而
    #: state 每轮都要过一遍 checkpoint。放进 state 等于把同一份数据反复
    #: 写进每一轮的检查点，还顺带进了 Send payload。
    #: `recorder` / `model_registry` 放这里也是同一个理由。
    #:
    #: ⚠️ 它是**快照**：tool 节点把它交给执行器、执行器放进
    #: `invocation.context`，工具级的 checkauth 拦截器据此判权。
    #: 拿不到（None）时受保护的工具会按"没有身份"拒绝 —— fail-closed。
    caller_identity: Mapping[str, Any] | None = None
