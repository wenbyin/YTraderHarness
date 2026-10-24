"""子代理子图的 state。

⚠️ 它与 `ReactState` 共享**通道形状**，但不共享 messages
---------------------------------------------------
父图的 `messages` 只属于主 agent。子任务用自己的 `messages`：

    隔离    子任务看不到主对话全文，避免越权回答与预算浪费
    并发    多个 worker 并发写同一 channel 会互相覆盖
    恢复    子图是父图的**节点**，状态存在父 thread 的嵌套 namespace；
            不再是 `{parent}:subagent:{task}:attempt` 的独立 child thread
    体积    transcript 留在子图 namespace，不并回父 `messages`

设计依据见 design_planner/task.md §7.12；合并成节点后的取舍见
`app/agent/graph/subagent/graph.py` 的模块 docstring。

⚠️ 未声明的键会被 LangGraph **静默丢弃**
---------------------------------------
与 ReactState 同一条：往返回值里塞一个没声明的键，不报错也不写入。
ContextBuilder 需要的通道因此必须在这里逐个声明 —— 它们是 Processor
`when` / `inputs` 的绑定根（见 app/agent/context/processor/spec.py）。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

__all__ = ["SubAgentOutput", "SubAgentState", "SubAgentToolPayload"]


class SubAgentOutput(TypedDict):
    """
    子图作为**父图节点**运行时，返回给父图的通道。

    **刻意是空的**。父图只需要知道「这个 Send 分支结束了」—— 结果在
    任务板上，由下一轮 `subagent_dispatch` 重读。任何在这里声明的键都会
    按**父图同名 channel** 合并：

        messages        会把 worker transcript 追加进主对话
        tasks / plan    多个 worker 并发写，最后写者赢（或 InvalidUpdateError）

    换句话说，这个空 schema 就是「子图状态不泄漏到父图」那条隔离的
    正式落点。不要去加 `messages`，哪怕只为了调试。
    """


class SubAgentState(TypedDict):
    """
    一个 worker 的执行状态。

    字段分三组：

        对话        messages（子任务自己的 transcript）
        任务        current_task / tasks / task_id / attempt / skill_key
        上下文      context_policy / summary / memory / retrieved_context /
                    exposed_tools / context_result / task_tool_keys

    后两组与主图同名同形 —— 同一批 Context Processor 直接按这些路径
    绑定，不需要为子图再写一套处理器。
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # ========================================================
    # 任务
    # ========================================================

    #: 父图 subagent_dispatch **预 claim** 好、交给这个 worker 的任务。
    #:
    #: listener 第一次进入时把它转成 current_task（不 claim、不完成）。
    #: 预 claim 用它的理由是：多个 worker 同时抢板会互相踩，父图先 CAS
    #: 好再按 Send 分发，谁跑哪个任务是确定的。
    pending_task: dict[str, Any]

    #: 当前正在执行的 A2A canonical task dict；空 = 没有任务。
    current_task: dict[str, Any]

    #: 一个 worker 只执行一个任务。
    #:
    #: 父图派发时置 True：完成 current_task 后直接退出，不再 claim 下一个，
    #: 避免一个 worker 吞掉整批任务、其他 worker 空转。需要队列排空时置
    #: False（listener 会继续 claim）。
    single_task: bool

    #: 本 worker 看到的任务列表（可能是 Redis 板上的局部快照）。
    tasks: list[dict[str, Any]]

    #: 计划快照。任务板订阅按 plan_id 过滤时用它。
    plan: dict[str, Any]

    #: 当前任务 id / 第几次尝试 / skill key（claim 时写入）。
    task_id: str
    attempt: int
    skill_key: str

    #: worker 身份（claim owner 的后半段）。由派发方传入；
    #: 缺省时用 `pipeline_run_id:worker`。
    worker_id: str

    # ========================================================
    # 上下文（与 ReactState 同名同形）
    # ========================================================

    #: 子任务会话 id。用**主会话 id**：记忆 / 知识 / 工具源的 scope 靠它。
    conversation_id: str

    #: 主体（画像 / 记忆的 actor）。
    user_id: str

    #: 本次上下文构建的策略覆盖项。
    context_policy: dict[str, Any]

    #: 子任务自己的摘要（compaction 产出）。
    summary: dict[str, Any]

    #: L0/L1 召回命中。
    memory: dict[str, Any]

    #: 知识库检索结果。
    retrieved_context: dict[str, Any]

    #: 可暴露给模型的工具对象。
    exposed_tools: list[Any]

    #: 最近一次上下文构建的快照。
    context_result: dict[str, Any]

    #: SkillLoader 算出的工具 key。llm_node 回写，tool_node 的
    #: 允许集读它 —— 两处必须是同一个集合。
    task_tool_keys: list[str]


class SubAgentToolPayload(TypedDict):
    """
    子图 `tool_node` 扇出分支的 Send payload。

    `Send` 的 payload **取代** state（已实测）：分支看不到子图的任何
    字段。因此这里列的是分支真正需要的：

        call           这一次工具调用
        tool_keys      本次暴露的 key 原文（允许集由分支现算）
        call_scope     观测分支身份
        return_to      执行完回哪个节点（llm_node）
        is_controller  本批第一个调用才决定下一跳

    不带 `plan_status` / `plan` / `tasks`：计划控制工具**不暴露给
    worker**（见 design_planner/task.md §7.9）。
    """

    call: dict[str, Any]
    tool_keys: list[str]
    call_scope: str
    return_to: str
    is_controller: bool
