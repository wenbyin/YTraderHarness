"""
运行事件的**词汇**：一次 run 的进展以什么名字、什么形状对外发布。

这不是日志，是给订阅者看的契约：一次运行从开始到结束会发生什么，
都以固定的事件名出现在一条事件流上。

三个词
------
    topic     事件流的**类别**，本项目目前只有 "agent_run"
    topic_id  该类别下的**一条流**，就是 run_id
    key       topic 与 topic_id 拼出来的键

    topic=agent_run, topic_id=run_123  ->  key = "agent_run:run_123"

也就是**一个 run 一条流**，而不是所有 run 共用一个键。这样订阅范围
与键一一对应，客户端不需要过滤"这条是不是我关心的那个 run" —— 而
过滤恰恰是订阅侧最容易写错、又最难发现写错的一步：少过滤会看到别人
的内容，多过滤会让自己的内容静默地少掉。

本模块**不认识存储**
-------------------
键怎么拼在这里，但"往哪写、怎么写"不在这里 —— 那是
`app/agent/event` 的实现（Redis Stream）。因此这一层可以被任何
存储或传输复用（换 Kafka、换内存队列都不必改事件名）。

与 tools 包里那个 ToolEventBus 的关系
------------------------------------
两者**不是一回事，不要合并**：

    ToolEventBus（agent/tools/processor/runtime/events.py）  本模块
    ------------------------------------------------------  -------------
    进程内（asyncio.Queue，有界、丢最旧）                     跨进程
    工具调用的起止                                          一次运行的完整进展
    STARTED / SUCCEEDED / FAILED / DENIED                   run.started / message.delta / …
    只服务同一个进程里的订阅者                                服务任何连得上的消费者

ToolEventBus **不是** tool.called / tool.completed 的来源：那两条由
工具节点写进图的自定义流（`TOOL_STREAM_KIND`），再经
`RunEventProjector` 投影出来 —— 与 trace 的 fanout.* 同一条路。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, Mapping

__all__ = [
    "STREAM_CLOSING_EVENT_TYPES",
    "TERMINAL_EVENT_TYPES",
    "TOOL_STREAM_KIND",
    "RunEvent",
    "RunEventType",
    "stream_key",
]


#: topic 与 topic_id 之间的分隔符。
#: 只有 stream_key 一处知道它 —— 别处不要再拼一次。
_KEY_SEPARATOR = ":"


def _utc_now() -> datetime:
    """
    带时区的当前时间。

    **必须带时区**：不带时区的 ISO 串被前端 `new Date(...)` 读走时会
    按浏览器本地时区解释，于是同一份数据在不同机器上显示成不同时刻，
    而且错得很隐蔽（差几小时，看着像"延迟"）。
    """

    return datetime.now(timezone.utc)


class RunEventType(str, Enum):
    """
    一次运行里会发生的事。

    ⚠️ **值就是线上的字符串，改一个字就是破坏契约**（订阅者按字面匹配）。
    继承 str 是为了让它能直接比较与进 JSON，不必到处 `.value`。
    """

    # ---- 运行的边界（生产者在 agent_run_dispatch 的 _mark_running / _finish）----
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"

    # ---- 节点的边界（生产者在 React 图的 agent 节点）----
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"

    # ---- 模型消息（生产者在 agent/event/run_event_projector.py）----
    #
    # 三条构成一次模型调用的完整生命周期，订阅端只有三条规则：
    #
    #     message.started   一条消息开始 -> 挂 loading
    #     message.delta     正文增量，**逐 token**，只有可见文本
    #                       （不含思维链、不含 tool-call 参数碎片）-> 追字
    #     message.completed 一条消息结束，data 里带**全文** -> 收 loading
    #
    # completed 是 delta 的**对账点**：delta 是旁路，丢了不会重发；
    # 订阅者拼出来的东西与 completed 的全文不一致时，以全文为准。有这一条，
    # "丢了几个 delta"才不会变成一次错到底的显示。
    #
    # ⚠️ **不是每条消息都有正文** —— 这是 started/completed 必须单独发的原因
    # ------------------------------------------------------------------
    # 一次运行里的模型调用不止"给用户写答复"一种：上下文压缩（summarization）
    # 也在 agent 节点里调模型，而它和主循环**共用同一个 langgraph_node**，
    # 靠节点名分不开。所以正文的放行由 **model_role** 决定（见
    # run_event_projector 的 message_roles）：
    #
    #     角色放行   delta 照发，completed 带全文
    #     角色不放行 **一个 delta 都不发**，completed 的 content 是空串
    #
    # 但**started / completed 照发** —— 它们不是正文，是"这会儿有模型在工作"
    # 的边界信号，前端拿它转 loading 图标。少了它们，内部调用期间界面上
    # 什么都不会发生，看起来像卡住了。
    #
    # 因此订阅端**不能**把"completed 的 content 是空串"读成"模型没说话" ——
    # 它也可能是一条不该给用户看的消息。要区分就读 started / completed 的
    # data 里那两个字段：
    #
    #     role     这次调用的 model_role（"谁在说话"）—— 事实
    #     visible  正文有没有随事件下发（服务端已经判好了）—— 决定
    #
    # 两个都发是刻意的：只发 role，订阅端就得自己抄一份"哪些角色算可见"
    # 的白名单，而那份名单改了、这边没改，症状是前端**静默少显示一段正文**。
    #
    # ⚠️ started / completed **不保证成对**
    # --------------------------------
    # 断线重连按 Last-Event-ID 续读时，起点可能落在一条消息的中途；整条消息
    # 形状（模型没走流式时才触发）更是只有 completed 没有 started。订阅端的
    # 收尾动作因此必须是**幂等**的：多收一条 completed 只是把已经停掉的
    # loading 再停一次。
    MESSAGE_STARTED = "message.started"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_COMPLETED = "message.completed"

    # ---- 思维链（同上，投影层发）----
    #
    # ⚠️ 为什么是**新事件名**，而不是往 message.delta 里塞一个字段
    # -------------------------------------------------------------
    # `message.delta` 的契约明写"只有可见文本（**不含思维链**、不含
    # tool-call 参数碎片）"。把推理并进去就是改这条已经生效的契约 ——
    # 而事件名与载荷是订阅端**按字面**匹配的（见上面的"改一个字就是破坏
    # 契约"）。新事件名是加法：老订阅端不认识就不订阅，一个字都不用改。
    #
    # 两条通道的规则**逐条对应**，只是容器不同：
    #
    #     message.reasoning.delta      {"delta": …}                逐 token
    #     message.reasoning.completed  {"content": …, "role": …}   全文（对账点）
    #
    # 与 message.* 一致：delta 是旁路、丢了不重发，completed 带全文；
    # 订阅端拼出来的与全文不一致时以全文为准。
    #
    # ⚠️ completed 的载荷里**没有 `visible`**（message.completed 有）
    # ------------------------------------------------------------
    # 推理只在可见时发 —— 不可见的调用（压缩 / 分类）**一条推理事件都不发**，
    # 所以那个字段在这里恒为 True，是"一个永远不变的字段"，比缺字段更糟：
    # 订阅端会为它写一条永远走不到的分支。理由与 `node.completed` 不肯把
    # `degraded` 加回来是同一条（见 run_event_projector 的 _on_tasks）。
    #
    # ⚠️ 两条通道的可见性由**同一个**决定管
    # -----------------------------------
    # 也就是 run_event_projector 的 message_nodes × message_roles 两道门。
    # 摘要（summarization）在同一个 agent 节点里调模型，没有角色门的话它的
    # 思维链会直接漏进用户的聊天窗 —— 那是比正文泄漏更糟的一种，因为推理
    # 里往往有内部判据与原始输入。
    #
    # ⚠️ message.reasoning.completed **可能一条都没有**
    # --------------------------------------------
    # 模型没给思维链时（非推理模型、关掉思考的调用）它根本不发 —— 而不是
    # 发一个空壳。"这次没有推理"与"推理是空的"不是一回事，订阅端不该看到
    # 后者。因此**不要**用它来判断"这条消息结束了"：那个信号永远是
    # message.completed（reasoning.completed 先到，且可能被丢）。
    #
    # ⚠️ 两者都**不是**终止事件
    # -----------------------
    # 不进 TERMINAL_EVENT_TYPES / STREAM_CLOSING_EVENT_TYPES —— 推理还在流
    # 的时候把 SSE 连接关掉，用户会以为回答已经结束了。
    MESSAGE_REASONING_DELTA = "message.reasoning.delta"
    MESSAGE_REASONING_COMPLETED = "message.reasoning.completed"

    # ---- 工具调用（来源：工具节点的自定义流，经 RunEventProjector 投影）----
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"

    # ---- 任务板（来源：planner / task_listener 的自定义流）----
    #
    # plan.published  planner 产出完整计划与任务快照（一次性）
    # task.updated    listener claim / 完成某一任务（按 task id 更新）
    # plan.cleared    任务列表清空（完成或 deadlock），前端收起任务面板
    #
    # 三条都是**加法**，不影响 TERMINAL / STREAM_CLOSING；SSE 订阅端
    # 不认识就忽略，认识就按 task id 合并。
    PLAN_PUBLISHED = "plan.published"
    TASK_UPDATED = "task.updated"
    PLAN_CLEARED = "plan.cleared"

    # ---- 子代理（来源：create_team / subagent_dispatch 的自定义流）----
    #
    # subagent.created       一个 worker 实例创建（父图预 claim 成功、
    #                        Send 之前；每 worker 一条）
    # subagent.claimed       某个 worker 预 claim 了一个任务
    # subagent.task.updated  子任务状态变化（SUBMITTED/WORKING -> 终态）
    # subagent.completed     某个 worker 执行完一个任务
    #
    # 与任务板那三条的区别是 **scope**：task.* 是主 agent 的执行进度，
    # subagent.* 带 `worker_id`，前端按 worker 分组渲染并行泳道。不把
    # 它们并进 task.updated 的理由与 message.reasoning.* 一样 —— 事件名
    # 是字面契约，加法比改字段安全。
    #
    # ⚠️ 它们**不在** TERMINAL / STREAM_CLOSING 里：子代理只是运行中间
    # 的并行分支，运行还没结束。
    SUBAGENT_CREATED = "subagent.created"
    SUBAGENT_CLAIMED = "subagent.claimed"
    SUBAGENT_TASK_UPDATED = "subagent.task.updated"
    SUBAGENT_COMPLETED = "subagent.completed"

    # ---- 人工介入（来源：interceptor/approval.py，已在图里）----
    #
    # hitl.requested  运行**暂停**等审批，每个待批中断一条
    #                 （扇出下会同时挂好几个）
    # hitl.resolved   某个中断被回答了，运行接着跑
    HITL_REQUESTED = "hitl.requested"
    HITL_RESOLVED = "hitl.resolved"


#: custom 流里"工具执行"事件的信封标记。
#:
#: 工具节点**不直接写 Redis**：它把工具生命周期写进图的自定义流
#: （`get_stream_writer()`），由 `RunEventProjector` 投影成上面的
#: `tool.called` / `tool.completed` —— 与 trace 的 `fanout.*` 同一条路。
#: `kind` 用来和同一条 custom 通道上的 TraceEvent 区分（后者没有这个键）。
TOOL_STREAM_KIND: Final[str] = "tool"


#: custom 流里“任务板”事件的信封标记。
#:
#: planner / task_listener 不直接写 Redis：把任务生命周期写进图的
#: 自定义流，由 RunEventProjector 投影成 plan.published / task.updated /
#: plan.cleared，再经 publish_run_event 进 SSE 流。
TASK_STREAM_KIND: Final[str] = "task"


#: custom 流里“子代理”事件的信封标记。
#:
#: create_team / subagent_dispatch 不直接写 Redis：把子代理生命周期写进
#: 图的自定义流，由 RunEventProjector 投影成 subagent.created /
#: subagent.claimed / subagent.task.updated / subagent.completed。
#:
#: ⚠️ 子图内部**不要**用这个 kind：这些事件由**父图节点**在派发 / 收波
#: 时统一发。worker 现在虽然是编译进父图的子图（事件会进父流），但它的
#: listener 只写任务板，不重复发这一组，避免同一状态出现两个来源。
SUBAGENT_STREAM_KIND: Final[str] = "subagent"


#: 表示"这次运行结束了"的事件。
#:
#: 订阅端（SSE）靠它决定什么时候把连接**正常关掉** —— 少了这个集合，
#: 一条已经跑完的运行的订阅会永远挂着，因为流上再也不会有新事件。
#:
#: ⚠️ 它放在这一层而不是订阅实现里，是因为"哪些事件算终态"属于**契约**：
#: run.cancelled 目前不存在（见 RunStatus.CANCELLED 的映射说明），
#: 将来若加上，改这里一处即可。
TERMINAL_EVENT_TYPES: frozenset[RunEventType] = frozenset({
    RunEventType.RUN_COMPLETED,
    RunEventType.RUN_FAILED,
})


#: 表示"这条订阅该收尾了"的事件 —— `TERMINAL_EVENT_TYPES` 的**超集**。
#:
#: ⚠️ 与上面那个的区别是真实的，不要合并成一个
#: ------------------------------------------
#:     run.completed / run.failed   运行**终止**了。流上不会再有新事件。
#:     hitl.requested               运行**暂停**等人审批。**没有**终止 ——
#:                                  DB 转 waiting_input，thread 锁仍被这个
#:                                  run 持有，人类批准后同一个 run 会接着
#:                                  往同一条流上写 hitl.resolved 与后续。
#:
#: 那订阅端为什么也要在它上面收尾：审批可能要等几分钟，而客户端需要的
#: 全部信息（interrupt_id + options）已经在事件里了，把连接挂着只是白占
#: 一条长连接。前端收到它之后应**主动** close，POST resume，再重新开流。
#:
#: ⚠️ 判据是"**所有** interrupt 到齐"，不是"收到第一条"
#: ------------------------------------------------
#: 扇出下多个工具可以同时等审批，`_publish_hitl_requested` 是**逐个发**的
#: （agent_run_dispatch.py:756-771）。"批次里最后一条"是**时序**的产物、
#: 不是事件的属性：第二条可能只晚几毫秒、落在下一个批次里 —— 按它关流就
#: 会把兄弟 interrupt 关在门外，而客户端永远凑不齐 resume 需要的 id。
#:
#: 权威信号是 **DB 状态**：派发路径先发完所有 hitl.requested，**之后**才把
#: agent_run.status 写成 waiting_input（:540 发事件、:585 才 _finish）。
#: 因此"状态已是 waiting_input"蕴含"所有 hitl.requested 都已发布"。
#: 订阅端的判定据此是"空闲达阈值 **且** 探活看到 waiting_input"，见
#: app/agent/event/run_stream.py。
STREAM_CLOSING_EVENT_TYPES: frozenset[RunEventType] = (
    TERMINAL_EVENT_TYPES | {RunEventType.HITL_REQUESTED}
)


def stream_key(topic: str, topic_id: str) -> str:
    """
    一次运行的事件流键：`agent_run:run_123`。

    单独成一个函数而不是让调用方各自 f-string 拼：键的形态将来若要变
    （加环境前缀、换分隔符），只有这一处要改。
    """

    return f"{topic}{_KEY_SEPARATOR}{topic_id}"


@dataclass(frozen=True, slots=True)
class RunEvent:
    """
    一条运行事件。

    信封字段（type / topic / topic_id / node）**平铺在顶层**，事件自己的
    内容统一进 `data`。这个形状是为**读取侧**定的：订阅者常常只需要判断
    "这条是不是我关心的那个 run 的 message.delta"，平铺之后不解 JSON 就
    能判断，而不必为了看一眼 topic_id 先把整段正文解出来。
    """

    type: RunEventType

    #: 事件流的类别，即配置里的 key_prefix（目前恒为 "agent_run"）。
    topic: str

    #: 该类别下的一条流，就是 run_id。
    topic_id: str

    #: 事件自有内容，必须 JSON 可序列化。
    data: Mapping[str, Any] = field(default_factory=dict)

    #: 哪个节点发出来的。node.* / message.* 有，run.* 没有。
    node: str | None = None

    #: 产生时刻（UTC）。
    #:
    #: 在**构造**时取，而不是等写进存储时补：排队、重试、背压都会把
    #: 写入推后，用写入时刻会让时间戳挤成一堆，症状是"顺序看着对、时间
    #: 全挤在一起"，看起来像模型突然变快。
    occurred_at: datetime = field(default_factory=_utc_now)

    def to_fields(self) -> dict[str, str]:
        """
        转成"字段 -> 值"的扁平映射（存储侧的一条记录）。

        ⚠️ 每个值只能是 str / bytes / int / float —— Redis Stream 的字段
        值**不能是嵌套结构**。这就是 `data` 必须序列化成 JSON 字符串、
        而不能直接传 dict 的原因。
        """

        return {
            "event": self.type.value,
            "topic": self.topic,
            "topic_id": self.topic_id,
            "node": self.node or "",
            "occurred_at": self.occurred_at.isoformat(),
            # ensure_ascii=False：中文在 redis-cli / 消费端直接可读。
            # 代价是消费端要按 UTF-8 解 —— 而 redis-py 写进去本来就是
            # UTF-8，所以这个代价其实没有额外成本。
            #
            # default=str 是兜底：data 里混进 UUID / datetime 时按 str 转。
            # 不掩盖问题 —— 真混进去了也只是那个字段变成字符串，而不是
            # **整条事件**发不出去（后者会让流静默地少一段，难查得多）。
            "data": json.dumps(
                dict(self.data),
                ensure_ascii=False,
                default=str,
            ),
        }
