"""
运行事件的**键**与**终止判定**。

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

事件的**统一词表**与信封在 `app/common/event/spec.py`（`EventType` /
`Event`）。本模块只保留键、custom 流的 kind、以及两个终止集合。

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

from typing import Final

from .spec import EventType

__all__ = [
    "STREAM_CLOSING_EVENT_TYPES",
    "TERMINAL_EVENT_TYPES",
    "TOOL_STREAM_KIND",
    "TASK_STREAM_KIND",
    "SUBAGENT_STREAM_KIND",
    "stream_key",
]


#: topic 与 topic_id 之间的分隔符。
#: 只有 stream_key 一处知道它 —— 别处不要再拼一次。
_KEY_SEPARATOR = ":"


#: custom 流里"工具执行"事件的信封标记。
#:
#: 工具节点**不直接写 Redis**：它把工具生命周期写进图的自定义流
#: （`get_stream_writer()`），由 `RunEventProjector` 投影成
#: `tool.called` / `tool.completed` —— 与 trace 的 `fanout.*` 同一条路。
#: `kind` 用来和同一条 custom 通道上的 TraceEvent 区分（后者没有这个键）。
TOOL_STREAM_KIND: Final[str] = "tool"


#: custom 流里“任务板”事件的信封标记。
#:
#: planner / task_listener 不直接写 Redis：把任务生命周期写进图的
#: 自定义流，由 RunEventProjector 投影成 plan.published / task.updated /
#: plan.cleared。
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
#: ⚠️ 它放在这一层而不是订阅实现里，是因为"哪些事件算终态"属于**契约**。
#:
#: run.cancelled 与 failed **并列**，不是它的别名：取消是用户意图，
#: 不是错误。chat 放行集合（spec.CHAT_EVENT_TYPES）里两者都在，
#: 读端也必须在这里对齐 —— 否则取消之后 SSE 不会收尾。
TERMINAL_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
    EventType.RUN_CANCELLED,
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
STREAM_CLOSING_EVENT_TYPES: frozenset[EventType] = (
    TERMINAL_EVENT_TYPES | {EventType.HITL_REQUESTED}
)


def stream_key(topic: str, topic_id: str) -> str:
    """
    一次运行的事件流键：`agent_run:run_123`。

    单独成一个函数而不是让调用方各自 f-string 拼：键的形态将来若要变
    （加环境前缀、换分隔符），只有这一处要改。
    """

    return f"{topic}{_KEY_SEPARATOR}{topic_id}"
