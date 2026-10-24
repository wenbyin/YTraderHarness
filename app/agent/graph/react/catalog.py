"""
本图的观测契约：图名、节点目录、拓扑边。

三条硬规则（与 document/pipeline/catalog.py 完全相同）
-----------------------------------------------------
    1. `sequence` **永不重编号** —— 新节点只往后追加
    2. `node_id` 一旦发布**永不复用** —— 节点删了就进
       RETIRED_NODE_SPECS，目录里永久保留
    3. 拓扑有任何变化就抬 `GRAPH_VERSION`

`node_id` 是**稳定契约键**：它会冻结进每一次运行的行里，前端按当时
的拓扑快照渲染历史运行。改一个 id，所有历史行就再也对不上节点了。

为什么 CONTEXT_BUILDER_TYPE 在这里
----------------------------------
两个 builder 常量（`CONTEXT_BUILDER_TYPE` 与 `CLASSIFIER_BUILDER_TYPE`）
都在这里，而不是各自在 state 或 context 里找 —— 这样"哪个节点用哪个
builder"只有一个地方可写，也就没有两处打架的可能。

⚠️ 本图是**两种** builder 类型的宿主（改动前只有一种）
----------------------------------------------------
    agent 节点      AGENT_INVOKE   主循环：完整上下文
    builder 节点    CLASSIFIER     分类：窄输入（只带当前这一轮）

"一张图一种构建语义"那条规则约束的是**一次调用**，不是整张图：每个节点
只读一个编译期常量，没有哪个节点在运行时挑 builder。因此
`ReactContext` 上挂的是**两个**实例（builder / classifier_builder），
各自对应一个常量（见 context.py）。

将来要跑别的类型（比如给 sub_agent 单独开一张图），那仍然是**另一张
图**（另一个 GRAPH_NAME / NODE_SPECS）。

为什么 AGENT_TYPE / MODEL_ROLE 也在这里（规格 §5、§15）
-----------------------------------------------------
节点要在运行时向 Model Registry 要一个 runtime：

    model_registry.get_runtime(
        agent_type=AGENT_TYPE, model_role=<角色>, context=...,
    )

这些值是**这张图的编译期常量**，不是运行时数据：

    AGENT_TYPE              "这张图是哪一种 Agent" —— 由 Agent Builder /
                            JEV 在**选图**时就定死了。写在这里而不是让
                            节点去推导，是为了不让模型选择这件事重新滑回
                            "节点自己判断"。
    MODEL_ROLE              agent 节点这次调用的角色（推理）。
    CLASSIFIER_MODEL_ROLE   builder 节点这次调用的角色（分类）。

⚠️ 两处用的是**同一个 agent_type、不同的 model_role** —— 这正是
model.yaml 里 `react + reasoning` 与 `react + classification` 两条绑定的
由来（后者指向 classification_default 那个分类档）。换分类用的模型是改
model.yaml 的绑定，不是改这里。

Resolver 只拿这两个值查表（`(agent_type, model_role) -> ModelProfile`），
它不调 LLM、不理解用户输入（见 model/resolver.py）。

⚠️ 节点的 `node_id` 与 `agent_type` 是**两件事**
-----------------------------------------------
前者是观测契约键（"agent 这个阶段"），后者是模型路由键（"react 这种
Agent"）。它们今天恰好都指向同一个节点，但改名/拆图的动因完全不同 ——
合并成一个字段会让"重命名一个观测阶段"意外改掉模型选择。
"""

from __future__ import annotations

from uuid import UUID

from app.common.context import ContextBuilderType
from app.agent.model import ModelRole
from app.infra.graph import NodeSpec

__all__ = [
    "AGENT_BUILDER_NODE",
    "AGENT_NODE",
    "AGENT_TYPE",
    "CALLBACK_NODE_IDS",
    "CLASSIFIER_BUILDER_TYPE",
    "CLASSIFIER_MODEL_ROLE",
    "CONTEXT_BUILDER_TYPE",
    "EDGES",
    "FANOUT_NODE_IDS",
    "GRAPH_NAME",
    "GRAPH_VERSION",
    "LINEAR_NODE_IDS",
    "MODEL_ROLE",
    "NODE_SPECS",
    "PLANNER_BUILDER_TYPE",
    "PLANNER_NODE",
    "REACT_TOOL_SCOPE_NAMESPACE",
    "RETIRED_NODE_SPECS",
    "SUBAGENT_BUILDER_TYPE",
    "SUBAGENT_DISPATCH_NODE",
    "SUBAGENT_WORKER_NODE",
    "TASK_LISTENER_NODE",
]


#: 图名。进 GraphRegistry 的键，也是 dispatcher 注册 handler 用的键。
GRAPH_NAME = "agent_react"

#: 图版本。拓扑一变就要抬 —— 历史运行按运行当时的快照渲染。
#:
#: 1.1.0  加入 builder（Agent Builder，前置于主循环）与 planner（占位）
#: 1.2.0  tools 改成 Send 扇出（一个调用一个分支实例）
#: 1.3.0  tools 节点按 ToolNode 规范分两层，并接上真 ToolRuntime
#:        （处理器从此拿得到 tool_call_id / thread_id / store；
#:         审批拦截器不再走 fail-closed 的"通道不可用"分支）
#: 1.4.0  加入 plan_control（enter / exit 两个图控制工具）；
#:        路由从 classification.execution_plan 改为读 plan_status
#: 1.5.0  enter / exit 改为**真工具**（经 tools 节点执行），移除
#:        plan_control；tools 之后按 ToolMessage 里的计划指令条件路由
#: 1.6.0  工具返回 LangGraph `Command`，跳转由 Command.goto 决定；
#:        tools 不再有静态出边 / 条件边（handoff）
#: 1.7.0  planner 真正产出 A2A 任务；新增 task_listener，
#:        拓扑变为 planner -> task_listener -> agent
#: 1.8.0  agent 无 tool_calls 且还有任务时回 task_listener；
#:        全部任务结束才 END
#: 1.9.0  planner 只产出计划（任务板改由 create_team 发布）；planner
#:        之后默认回 task_listener，开发期可开关停在 END
#: 1.10.0 task_listener 在 claim 前检测并行 ready 任务；有并行则回
#:        main agent 做 team_decision，不直接 claim
#: 1.11.0 create_team 改走 Command；新增 subagent_dispatch（预 claim +
#:        动态 Send 扇出）与 subagent_worker（编译进父图的子图节点，
#:        interrupt / custom 事件由父图管理）
#: 1.12.0 planner 变成编译子图（独立 PlannerState / transcript /
#:        checkpoint namespace）；父图节点名不变，只合并计划通道
#:
#: ⚠️ 1.2.0 / 1.3.0 的**拓扑指纹都没变**，版本号是这类改动唯一的痕迹
#: ----------------------------------------------------------------
#: `topology_fingerprint` 只用 `(node_id, sequence)` 与 edges 两样东西
#: （见 infra/graph/ids.py），而这两次改的是**边的语义**（普通边 -> Send
#: 扇出）、节点的形状（linear -> fanout）与节点的**内部实现**，三者都
#: 不在指纹的输入里。
#:
#: 所以"看指纹能发现图被改过"这条在这里**不成立**。真要让它成立得把
#: group 或边的类型纳入指纹 —— 那会让所有历史指纹失效，代价大于收益。
#: 因此：改这类"形状不变、语义变"的东西时，抬版本号是唯一能追溯的动作。
GRAPH_VERSION = "1.12.0"


#: 扇出分支 scope 的 uuid5 命名空间。
#:
#: 与 GRAPH_NAME 同性质：它是**这张图的编译期契约**，不属于任何一次
#: 运行。换一个值等于换一套 branch_key，历史节点行就对不上了。
#:
#: 各图各用一个命名空间（document 那边用持久化主键当 scope，不需要
#: 这个；见 app/infra/graph/ids.py 的 branch_scope_id）。
REACT_TOOL_SCOPE_NAMESPACE = UUID(
    "6f2b1c8e-4a37-5d91-8e0c-2b7a4f61d305"
)


#: 主循环节点。前置于它的 builder 节点是 1.1.0 加的。
AGENT_NODE = "agent"

#: Agent Builder 节点：对用户输入做初次理解与分类，并决定走向。
AGENT_BUILDER_NODE = "builder"

#: 规划节点：生成 TaskDraft，转成 A2A Task 写入 state。
PLANNER_NODE = "planner"

#: planner 节点用的 Context Builder 类型（窄输入：摘要 + 历史 + prompt）。
PLANNER_BUILDER_TYPE = ContextBuilderType.PLANNER

#: 子代理子图的 llm_node 用的 Context Builder 类型。
#:
#: 它属于**另一张图**（app/agent/graph/subagent），但实例由本图的组合根
#: 启动期解析一次后放进共享的 `ReactContext` —— 子图与主图跑在同一次
#: 运行里，共用同一个 context 是刻意的（见 context.py 的说明）。
SUBAGENT_BUILDER_TYPE = ContextBuilderType.SUB_AGENT

#: 任务领取节点：从 state / Redis 任务板取下一个 ready task。
TASK_LISTENER_NODE = "task_listener"

#: 子代理派发节点：从任务板预 claim N 个 ready task，动态 Send 给 worker。
SUBAGENT_DISPATCH_NODE = "subagent_dispatch"

#: 子代理执行节点：一个 Send 分支编译一次子图（子图作为节点）。
#:
#: ⚠️ 它是**子图节点**，不是普通节点函数：interrupt 冒泡、resume、
#: 子图 custom 事件进入父运行流都依赖这一点。
SUBAGENT_WORKER_NODE = "subagent_worker"


#: 本图的 Agent 类型 —— Model Registry 的第一把路由键。
#:
#: ⚠️ 它的词汇表属于 **Agent Builder / JEV**，不属于 Model 模块：Model
#: 模块只把它当作一个字符串键（见 model/resolver.py）。这里写常量只是
#: 为了"写一次、两处引用"（catalog 与 runtime 的文档），而不是在 Model
#: 模块里定义 Agent 的类型系统。
AGENT_TYPE = "react"


#: agent 节点这次模型调用的**模型侧角色**。
#:
#: 主循环是"推理"调用 —— 与将来的摘要调用（summarization）、压缩调用
#: 是同一个 agent_type、不同的 model_role。Resolver 据此选 profile
#: （react + reasoning -> reasoning_default）。
MODEL_ROLE = ModelRole.REASONING


#: 本图用的 Context Builder 类型。
#:
#: 选 AGENT_INVOKE 是因为这张图做的正是"主 Agent / ReAct 调用"：
#: 每次进模型之前都要即时构建上下文。
CONTEXT_BUILDER_TYPE = ContextBuilderType.AGENT_INVOKE


#: Agent Builder 节点用的 Context Builder 类型。
#:
#: 与上面那个是**两种语义**：分类调用要的是窄输入（只带当前这一轮、
#: 不带画像 / 记忆 / 资料 / 工具），与主循环的"尽收眼底"正好相反。
#: 两种都由组合根在启动期各解析一个实例（见 runtime.py 与 context.py）。
CLASSIFIER_BUILDER_TYPE = ContextBuilderType.CLASSIFIER


#: Agent Builder 节点这次模型调用的**模型侧角色**。
#:
#: agent_type 仍是 AGENT_TYPE（"这是 react 这张图上的调用"），换的是
#: 角色 —— 于是 model.yaml 里
#: `react + classification -> classification_default` 那条绑定生效，
#: 分类走千问 flash，主循环仍走推理档。
CLASSIFIER_MODEL_ROLE = ModelRole.CLASSIFICATION


#: 节点目录。`sequence` 从 0 连续，永不重编号。
#:
#: group 是**节点的形状**，不是执行顺序：
#:
#:     linear   每次运行跑一次
#:     fanout   每次运行跑 N 次（Send 扇出的分支，一个调用一个实例）
#:
#: 本图有**一个**扇出节点（tools）与**一个**循环（tools → agent → tools）。
#: 循环关系由 EDGES 表达，不需要额外的 group；扇出则需要 —— 它决定
#: 这个节点要不要按 branch_key 分行（见 CALLBACK_NODE_IDS）。
NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        "agent",
        "模型调用",
        "linear",
        0,
        "构建上下文并调用模型；模型要么要求调工具，要么给出答复。",
    ),
    NodeSpec(
        "tools",
        "工具执行",
        "fanout",
        1,
        "执行**一个**工具调用（每个调用一个分支实例）：按本次暴露的"
        "工具集合挡掉没暴露的名字，走处理器链执行，包成一条 "
        "ToolMessage 放回。扇出是 interrupt 能成立的前提 —— 单节点"
        "循环会在 resume 时重跑已完成的工具，见 routing.py。",
    ),
    # ⚠️ 下面两条的 sequence 是 **2、3**，而它们的执行位置在 agent
    # **之前**。这不矛盾：sequence 是观测目录里的登记序号（追加、永不
    # 重编号，见开头三条硬规则），不是执行顺序 —— 执行顺序由 EDGES 与
    # graph.py 的装配表达。
    NodeSpec(
        "builder",
        "输入分类",
        "linear",
        2,
        "用快模型对用户输入做初次理解与分类（行为 / 模块 / 执行方案 / "
        "能力需求 / 能力配置），并据此决定去主循环还是先规划。",
    ),
    NodeSpec(
        "planner",
        "任务规划",
        "linear",
        3,
        "编译进父图的 planner 子图：独立 transcript 做 ReAct 探索，"
        "task_generator 把结果转成 TaskDraft / A2A tasks，"
        "只把 plan / tasks / current_task / team_decision 交回父图。",
    ),
    NodeSpec(
        "task_listener",
        "任务领取",
        "linear",
        4,
        "完成刚执行完的 current_task（落 COMPLETED / 清 claim），再领取"
        "下一个 ready task；全部终态时清空任务列表。循环由 "
        "listener -> agent -> listener 的图回边表达，节点内不做阻塞 while。",
    ),
    NodeSpec(
        "subagent_dispatch",
        "子代理派发",
        "linear",
        5,
        "从任务板预 claim 最多 subagent_workers 个 ready task（CAS，"
        "owner 按 worker 编号），用 Command.goto=[Send(...)] 动态扇出；"
        "没有 ready task 时回 task_listener 收尾。任务板不可用时 fail-closed。",
    ),
    NodeSpec(
        "subagent_worker",
        "子代理执行",
        "fanout",
        6,
        "一个 Send 分支执行一个已 claim 的任务：subagent 子图作为节点"
        "（task_listener -> llm_node -> tool_node）。interrupt 由父图"
        "冒泡 / resume；子图 output_schema 为空，transcript 不并回父图 "
        "messages，结果只写任务板。",
    ),
)


#: 已从当前图中移除、但历史运行可能仍有其行的节点。
#:
#: 目录里永久保留它们，前端据此解释旧的 node_id。
RETIRED_NODE_SPECS: tuple[NodeSpec, ...] = ()


#: 拓扑边，给拓扑指纹与前端画图用。
#:
#: ⚠️ 这里是**文档**，不是 LangGraph 的装配指令：装配在 graph.py 里
#: 显式写（条件边要先 register 分支，add_edge 表达不了）。两处必须
#: 一致，改动时一起改 —— 指纹算的是这里，跑的是那里。
EDGES: tuple[tuple[str, str], ...] = (
    ("builder", "agent"),       # 非计划模式，或计划仍有未完成任务
    ("builder", "planner"),     # 计划模式且没有任务 / 任务全完成
    ("builder", "tools"),       # 分类判定要规划 -> Send 合成 enter_plan_mode
    ("planner", "task_listener"),   # 规划完，先领取一个 ready task
    # 条件边：默认走上面那条去 task_listener；开发期开关可停在 END。
    # END 在 langgraph 里的节点名就是 "__end__"。
    ("planner", "__end__"),
    ("task_listener", "agent"),     # 领取后交给主循环执行
    ("agent", "task_listener"),     # 无 tool_calls 且还有任务：完成并领取下一个
    ("agent", "tools"),         # 模型要求调工具 —— **Send 扇出**，一个调用一个实例
    # ⚠️ 下面两条是 **Command 动态跳转**，不是装配时挂的静态边：
    # tools 节点没有出边，下一跳由控制器分支的 Command.goto 决定。
    # 写在这里是给拓扑指纹 / 前端画图看"可能去哪"。
    ("tools", "planner"),       # 任意工具 Command(goto="planner")
    ("tools", "agent"),         # 普通工具 / 无 goto / exit / 幂等 enter
    ("tools", "subagent_dispatch"),  # create_team 审批通过 -> 派发
    ("subagent_dispatch", "subagent_worker"),   # 预 claim 后 Send 扇出
    ("subagent_dispatch", "task_listener"),     # 没有 ready task：收尾
    ("subagent_worker", "subagent_dispatch"),   # join 后重新调度下一波
)


#: 按 group 派生的节点集合。
LINEAR_NODE_IDS: frozenset[str] = frozenset(
    spec.node_id for spec in NODE_SPECS if spec.group == "linear"
)

#: 扇出节点集合（每次运行跑 N 次）。
FANOUT_NODE_IDS: frozenset[str] = frozenset(
    spec.node_id for spec in NODE_SPECS if spec.group == "fanout"
)


#: 交给 callback 兜底写入的节点。
#:
#: **排除扇出节点**，与 document 同一条理由：N 个分支在 LangGraph 的
#: callback metadata 里共享同一个 `langgraph_node` 名，callback 分不出
#: 它们属于哪一次调用，硬写只会产出一行 `branch_key=""` 的幽灵聚合行 ——
#: 与 node_span 逐分支写的行并存，计数重复、前端多出一个不存在的节点
#: （见 infra/graph/callbacks.py 的说明）。
#:
#: 表达式保持 `LINEAR_NODE_IDS` 而不是手写差集：这样"哪些节点该被排除"
#: 只有 group 一个来源 —— 给某个节点改成 fanout，它就自动不在网里了。
CALLBACK_NODE_IDS: frozenset[str] = LINEAR_NODE_IDS
