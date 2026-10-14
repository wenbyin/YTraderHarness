"""
Agent 侧的图。

与 app/services/document/pipeline/ 的关系
-----------------------------------------
两者都是**业务图**，都注册进 `app/infra/graph` 的同一个
`GraphRegistry`，复用同一套 runner / checkpointer / 观测设施。

    这张（agent 图）     一次对话里模型与工具来回
    document 流水线      一次文档摄入按阶段走

区别在于驱动它们的东西不同：document 流水线是**数据流**（每个节点
把一个产物变成下一个），agent 图是**控制流**（模型自己决定下一步
是调工具还是收尾）。因此 agent 图的 state 里装的是对话本身，而
document 的 state 里只装指向产物的引用 —— 见 react/state.py 的说明。

本包目前只有一张图
------------------
    react/      ReAct 循环：agent → （要调工具就）tools → agent → …
"""
