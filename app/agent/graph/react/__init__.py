"""
ReAct 图：一轮对话里模型与工具的循环。

拓扑
----
        START
          │
          ▼
       ┌─ agent ─┐          构建上下文 → 调模型
       │         │
       │   （要调工具？）
       │         │
       │         ▼  Send × N     一个 tool_call 一个分支实例
       │       tools ×N      每个分支执行**一个**工具调用
       │         │
       └─────────┘          扇入：ToolMessage 汇进父状态后回到模型
                 │
                 ▼
                END          模型不再要工具就收尾

所有节点都是**普通 async 函数**（不是类）：langgraph 的 `ToolNode`
之所以是类，是因为它持有构图期的工具列表；这里的工具可用性是每次
调用由 `capability_loader` 阶段决定、随运行而变的，没有
per-instance 状态可持。

⚠️ tools 节点用的是 `ToolNode` 的**规范形状**，不是那个类
-------------------------------------------------------
它内部把一次调用的全部逻辑放在 `_wrap_tool_call(request, execute)`
里 —— 签名与 `awrap_tool_call` 逐字相同，因此将来换成预构建的
`ToolNode` 是一行。今天不直接用它，是因为它一次节点执行会跑完一条
AIMessage 里的**全部** tool_call，与下面那条扇出理由冲突
（见 tools.py 的说明）。

⚠️ tools 是**扇出**节点（一次运行跑 N 次）
-----------------------------------------
不是为了并行，而是 `interrupt()` 的语义要求如此：中断会让节点从**开头
重跑**，单节点里循环执行 N 个工具时，中断在第 3 个会让前两个的真实
副作用再发生一遍。一个调用一个分支之后，已完成的 task 在 checkpoint
里落定，resume 只重跑被中断的那一个（见 routing.route_after_agent）。

本图的构造器返回**未编译**的 StateGraph
----------------------------------------
由组合根决定用哪个 checkpointer 编译（生产 Postgres、冒烟
InMemory），图本身不关心 —— 与 document 流水线同一约定。

模块
----
    state.py        ReactState（messages）
    context.py      每次运行的依赖（ModelRegistry、builder 实例、updater）
    catalog.py      观测契约：GRAPH_NAME / NODE_SPECS / EDGES
                    + 模型路由键 AGENT_TYPE / MODEL_ROLE
    routing.py      agent 之后往哪走（纯函数）
    agent_node.py   AgentNode：ContextBuilder → Model Runtime → ContextUpdater
    tools.py        工具执行（挡掉未暴露的名字 + 驱动处理器链）
    graph.py        拓扑装配
    definition.py   包装成 infra 的 GraphDefinition
    runtime.py      register_react_graph（组合根调的那一个）

模型从哪来
----------
本包**不认识任何 provider / 模型**：`register_react_graph(model_registry=...)`
拿到一个 `app.agent.model.ModelRegistry`，节点在运行时用
`catalog.AGENT_TYPE` + `catalog.MODEL_ROLE` 向它要一个 ModelRuntime。
模型选择因此完全留在 Model 模块内部（见 app/agent/model/ 的包 docstring）。

⚠️ 本包必须有 __init__.py
-------------------------
理由与 app/agent/context/ 相同：任何按包反射的机制都不会走进没有
`__init__.py` 的目录。本包目前没有需要反射的东西，但保持一致的
代价是一个空文件。
"""
