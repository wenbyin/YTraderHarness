"""
会话与运行的取值域。

⚠️ 这里的 `RunStatus` 是**对外契约**的状态，与观测层的
`app/infra/graph/recorder.py::RunStatus` 是**两套**，刻意不合并
------------------------------------------------------------------
    对外契约（本模块）   queued / running / waiting_input /
                        completed / failed / cancelled
    观测层（infra）      pending / running / completed / failed /
                        partial / cancelled

三处差异，每一处都有原因：

    queued  vs  pending
        同义的两种叫法。契约用 queued（队列里等着），观测用 pending
        （还没开始）。纯词汇差异，但一旦写进对外契约就不能随意改 ——
        前端按它做状态机。

    waiting_input
        契约有，观测**没有**。它表示"图停下来等人输入"（HITL 中断）。
        观测层目前没有这个状态 —— HITL 中断点由工具节点 /
        `app/agent/graph/planner/` 子图产生，不在这里建模。
        它出现在契约里是对的（前端要能表达"等你回话"），只是现在还
        产生不出来。

    partial
        观测有，契约**没有**。它是 document 那边的扇出概念（至少一路
        成功、至少一路失败）。Agent 对话没有扇出，这个状态对它没有
        意义，因此不进契约。

映射在哪做：**接派发时做了，但比原来设想的薄**
------------------------------------------------
原计划是"在那一层把观测层状态翻成契约状态"。真接的时候没走那条路：
agent 运行不用 `GraphRunner` / recorder（它的终态推导以**节点行**为唯一
依据，而这条链路不写节点行 —— 见 application/agent_run_dispatch.py），
因此观测层状态根本没出现过，handler 直接判定 `completed` / `failed`。

于是 `partial` 对这条链路是**产生不出来**的（不只是"没意义"）：判定它需要
节点行。将来真接上节点级观测，这个映射要回到这里补。

⚠️ 为什么是 str 枚举而不是 Literal
---------------------------------
与仓库其他地方一致（见 app/services/tool/domain/schemas/enum.py）：
值要落库、要进日志、要在 yaml 与代码之间对齐，字符串最直白；
Literal 只活在类型检查器里。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "MessageKind",
    "MessageRole",
    "MessageStatus",
    "RunStatus",
    "SessionStatus",
]


class SessionStatus(str, Enum):
    """会话状态。对应 session.yaml 的 Session.status。"""

    ACTIVE = "active"

    #: 已归档。语义上是只读。
    #:
    #: 产生它的唯一入口是 `DELETE /sessions/{session_id}`（软删）。
    #: 归档后：
    #:
    #:     默认列表不再返回它（要 status=archived 才查到）
    #:     不能开新运行（create_run 会抛 ConflictError -> 409）
    #:     未结束的运行在归档时被取消
    #:
    #: 没有 unarchive 端点；归档是单向的。
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    """
    运行状态。对应 session.yaml 的 Run.status。

    见模块 docstring：这是对外契约的那一套，与 infra 的 RunStatus 不同。
    """

    #: 已受理、尚未开始执行。建运行的响应里**永远是它** —— 派发要等请求
    #: 事务提交之后才发生（见 run_service_impl.py）。
    QUEUED = "queued"

    RUNNING = "running"

    #: 图停下来等人工输入。当前产生不出来（没有 interrupt 点）。
    WAITING_INPUT = "waiting_input"

    COMPLETED = "completed"

    FAILED = "failed"

    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """
    消息的发出方。对应 messages.yaml 的 Message.role。

    ⚠️ 只有两个取值，因此**工具结果也算 assistant**
    -------------------------------------------------
    图里可能出现的 langchain 消息有四种（human / ai / tool / system），而契约
    只区分"用户说的"与"不是用户说的"。ToolMessage 是工具的执行结果，既不是
    用户写的、也不是模型写的，但契约里没有第三个格子放它 —— 于是归到
    assistant，靠 `kind=tool_activity` 区分（见 MessageKind）。

    system 消息**根本不落库**：图里不产生它（只出现在组装好的 prompt 里，
    不在 ReactState.messages 通道上）。
    """

    USER = "user"

    ASSISTANT = "assistant"


class MessageKind(str, Enum):
    """
    消息的种类。对应 messages.yaml 的 Message.kind。

    ⚠️ 它是给**客户端**的分类，服务端不据此改行为
    ------------------------------------------
    服务端没有"按 kind 分派业务"的逻辑 —— 它决定的是**前端拿这条画成什么**。
    唯一的例外是**落库过滤**：message_recorder 用
    `kind=task_instruction` 允许内部 HumanMessage 落库（普通 HumanMessage
    仍然滤掉，因为用户消息已由建运行事务写过）。这不改变"前端怎么画"。

    取值与生产者
    ------------
        chat            对话正文。模型的正常回复，以及**用户**那条消息
                        （role=user 时同样是 chat）。今天绝大多数行都是它。
        tool_activity   工具调用 / 工具结果：带 `tool_calls` 的 AIMessage
                        与工具执行结果（ToolMessage）都是它。一条既说了话、
                        又调了工具的消息也算它 —— 它的正文里两块都在
                        （文本块 + tool_call 块），前端按 tool_activity
                        渲染工具卡片时文本仍然在旁边，不丢东西。
        notice          系统公告（"上下文已压缩"、运行被取消之类）。
                        目前唯一的生产者是取消时补的那条中断记录
                        （message_recorder.record_cancellation_notice）。
        interaction     人机协同的一次问答（HITL）。
                        `ask_user_question` 的 resume 路径会把问答落成一条
                        `role=user` 的 interaction 消息（content 里同时有
                        人可读的 text 块与结构化的 questions/answers 块）。
                        这样暂停那一轮除了 tool_activity 的 AIMessage 之外，
                        历史里也能直接看到"问了什么、答了什么"。
        task_instruction
                        planner / task_listener 注入的任务执行指令。
                        role=assistant（它不是用户说的），meta 里带 task_id，
                        前端据此把任务卡片从普通聊天里分出来。
        task_result     任务执行模式下最终回答。role=assistant，
                        meta 里带 task_id；与 task_instruction 成对出现。
    """

    CHAT = "chat"

    INTERACTION = "interaction"

    TOOL_ACTIVITY = "tool_activity"

    NOTICE = "notice"

    TASK_INSTRUCTION = "task_instruction"

    TASK_RESULT = "task_result"


class MessageStatus(str, Enum):
    """
    消息的终态。对应 messages.yaml 的 Message.status。

    ⚠️ 不是 RunStatus，也不从它继承
    ------------------------------
    取值几乎一样（都有一个 "cancelled"），但**不是同一组**：

        RunStatus      排队 / 在跑 / 等人 / 结束 —— 是一次**执行**的状态机
        MessageStatus  只表示消息**最终结果**，没有 streaming 这一档

    合并成一个枚举会让"运行是 queued"这种消息上不可能出现的取值混进来，
    而它落的是带 CHECK 的列。因此是两个枚举 + 下面这一处显式映射。

    映射在哪做：`application/message_recorder.py`，由运行的终态推消息的终态。
        运行 completed        -> 消息 completed
        运行 failed           -> 消息 failed
        运行 waiting_input    -> 消息 completed（消息完整，暂停的是 run；
                                "还在等人"由 agent_run.status 表达）
        运行 cancelled        -> 消息 cancelled

    用户消息**永远是 completed**：它是"接收时"写下的完整输入，不存在半截。
    """

    COMPLETED = "completed"

    #: 生成被打断（消息本身不完整）。当前没有生产者：HITL 暂停时
    #: assistant 消息是完整的，不记这个状态。
    INTERRUPTED = "interrupted"

    FAILED = "failed"

    #: 被取消。当前产生不出来（cancel 端点未实现）。
    CANCELLED = "cancelled"
