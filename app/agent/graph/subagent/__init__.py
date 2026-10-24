"""子代理（worker）子图。

拓扑
----
    task_listener -> llm_node -> (tool_node × N -> llm_node)*
                  -> task_listener -> ... -> END

它与主图 `react` 的关系：

    主图负责**决定要不要开团队**、把任务发布到任务板；
    worker 负责**执行单个任务**，结果写回任务板，不写父图 state。

三个节点都复用主图的实现：

    task_listener   app/agent/plan/listener.py 的 claim / complete 规则
    llm_node        app/agent/graph/react/agent_node.py 的 run_agent_turn
    tool_node       app/agent/graph/react/tools.py 的 execute_tool_branch

模块
----
    state.py     SubAgentState / SubAgentToolPayload
    catalog.py   GRAPH_NAME / NODE_SPECS / EDGES / 模型路由键
    routing.py   listener / llm 之后的走向
    nodes.py     三个节点
    graph.py     拓扑装配
"""
