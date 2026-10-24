"""
把一次 agent 运行真正交给 ReAct 图。

这是 session 模块与图运行时之间的接缝。dispatcher 收到 `DispatchEnvelope`
后调这里产出的 handler，handler 负责：

    读运行行 -> 组初始 state -> 置 running -> 跑图
             -> 落消息（本轮产出的 assistant 消息）
             -> 映射终态并写回

⚠️ 终态信封会被**直接丢弃**
-------------------------
会话删除会归档并取消未结束的运行；其中“调度器还不认识”的 queued 运行
（post-commit 尚未受理、或进程重启后的孤儿）是在数据库里直接记为
cancelled 的。那之后信封仍可能被投进来 —— handler 在 `_load_run` 之后
先看状态，已终态就记一条日志返回，绝不重新执行一次已经结束的运行。

⚠️ 为什么**不用** infra 的 `GraphRunner`
--------------------------------------
`GraphRunner` 的终态推导（`app/infra/graph/runner.py::_derive_status`）
以**节点行**为唯一依据：

    summary.node_count == 0   ->   RunStatus.FAILED

而这条链路按设计**不写节点行**（只落运行级状态，逐节点进度由将来的 SSE
承担）。硬套 `GraphRunner` 的结果是**每一次成功的运行都被记成 failed** ——
除非让 recorder 伪造一份 summary。终态在这里自己判定（成功 `completed`、
异常 `failed`），比伪造 summary 诚实。

代价说清楚：agent 运行因此**没有** `partial` 这种状态（那是扇出的概念，
一次对话本来也不扇出），也没有拓扑快照与逐节点耗时。这些都是"不写节点行"
的必然结果，不是漏了。

⚠️ 节点级观测：只打日志
----------------------
图的节点走 `node_span`，它只调一个 recorder 方法
`upsert_node_run(...)`，而且包在 `_safe_record` 里（异常被吞掉、只记日志）。
所以这里给一个明确的实现而不是传 `None` —— 传 `None` 也能跑，但每个节点
阶段都会往日志里打一条异常栈，把真正的问题淹掉。

⚠️ 两处已知缺口（本次不做，别当成已完成）
---------------------------------------
1. **进程被杀 -> 运行永远停在 running**。document 那边启动时用
   `recorder.mark_orphans_failed()` 收敛，agent 这边还没有对应的收敛。
   要补就在 lifespan 里加一次（仓储上加一个 mark_stale_running_failed）。
2. ~~**每一轮运行彼此独立**。~~ **已解决**：`thread_id` 现在存在会话行
   上（`agent_session.thread_id`），首轮生成、此后复用 —— 同一会话的多轮
   落在同一条 thread 上，checkpointer 自己就把历史带上了。不再需要
   `ConversationSource`。见 `_initial_state`。

⚠️ 多轮打开之后新暴露的两件事（都没处理，别当成已解决）
----------------------------------------------------
    a. **失败后重试会重复用户消息**。`messages` 是追加通道，而失败那轮
       的 HumanMessage 已经写进 checkpoint 了 —— 重发一次同样的输入会
       再加一条。今天没有去重。
    b. **半截状态会被下一轮继承**。图跑到一半失败时，那条 thread 上的
       state 是残缺的，下一轮就从这个残缺状态继续。以前每轮换 thread，
       这种情况结构上不可能出现。

    两者都是"稳定 thread"的直接代价。要处理得在 resume / 重试语义那层
    决定，不在本模块顺手打补丁。

⚠️ 落消息（第 4 步）带来的三条缺口
--------------------------------
    c. **缺口 a 在 agent_message 表上也有一份**。每次 `create_run` 都插一条
       user 行，因此"失败后重试"会留下两条一模一样的用户消息。要治就得
       与 a 一起治（同一个去重决定），不能只治一边 —— 库里重复而
       checkpoint 不重复（或反过来）比两处都重复更难查。
    d. **"本轮新增"是按条数切出来的，前提是 add_messages 只追加**。
       它还支持 RemoveMessage 墓碑与按 id 替换，那时下标会移位、切片就切
       错了。今天没有节点发墓碑（compaction 只写 summary 通道），所以成立；
       真要用上墓碑时，message_recorder.new_assistant_messages 必须改成
       按消息 id 做差集。
    e. **模型生成到一半报错，那段文字丢了**。agent_node 整段生成完才产出
       AIMessage，中途抛异常时已生成的内容随异常消失。所以"失败"的
       assistant 消息只在**图确实产出过消息**时才有。详见 message_recorder
       的模块 docstring。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from uuid import UUID

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agent.event import (
    RunEventType,
    publish_run_event,
    publish_trace_event,
)
from app.common.event import EventType as TraceEventType
from app.common.event import SpanKind, SpanLink, SpanStatus, TraceError
from app.common.event.trace import LinkRelation
from app.common.event.trace_emitter import stable_span_id, tool_span_id
from app.agent.event.run_event_projector import projector_factory
from app.agent.graph.react.catalog import (
    AGENT_BUILDER_NODE,
    AGENT_NODE,
    AGENT_TYPE,
    CLASSIFIER_MODEL_ROLE,
    MODEL_ROLE,
    NODE_SPECS,
)
from app.agent.graph.react.catalog import (
    GRAPH_NAME as _REACT_GRAPH_NAME,
)
from app.agent.graph.planner.catalog import (
    AGENT_TYPE as _PLANNER_AGENT_TYPE,
    MODEL_ROLE as _PLANNER_MODEL_ROLE,
    NODE_SPECS as _PLANNER_NODE_SPECS,
    PLANNER_LLM_NODE as _PLANNER_LLM_NODE,
)
from app.agent.graph.react.routing import TOOLS_NODE
from app.common.repository.factory import RepositoryFactory
from app.infra.graph import (
    DispatchEnvelope,
    GraphDefinition,
    RunContext,
    get_thread_run_lock,
)
from app.infra.graph.cancellation import get_run_cancellation_registry

from ..domain.entities import AgentRun
from ..domain.schemas import RunStatus
from ..repository.agent_run import AgentRunRepository
from app.agent.tools.hitl_tool import parse_interaction

from .message_recorder import (
    new_assistant_messages,
    record_cancellation_notice,
    record_interaction,
    record_messages,
)

logger = logging.getLogger(__name__)


async def _submit_memory_tasks(
        *,
        run_id: UUID,
        session_id: UUID,
        repositories: RepositoryFactory,
) -> None:
    """
    后台提交 L0 / L1 记忆任务（fire-and-forget）。

    ⚠️ 为什么可以放到 run.completed 之后
    ------------------------------------
    这两条任务读的是**已经提交**的消息与水位，不参与运行的终态判定；
    `_finish` 已经提交了 run 状态与 agent_message。所以它们晚一点跑
    不影响客户端看到的结果，只影响"下一轮之前记忆有没有排上"。

    ⚠️ 进程被 SIGKILL 时可能丢
    --------------------------
    create_task 只在当前进程的事件循环里；进程直接被杀时还没执行的
    任务就丢了。对这两条旁路任务是可接受的：L0 下一轮会重新索引该
    会话（任务侧有幂等键），L1 的游标只在任务成功后才推进，漏一次
    会在下一次触发时补上。要更严格就改成 outbox —— 那是另一套设计。
    """

    # 压测/诊断开关：关闭后不再提交 L0/L1 抽取任务，L2/L3 也不会被
    # 触发。默认开启，mock 模式同样开启；要压测主链路时显式设为 0。
    if os.environ.get(
            "YTRADER_MEMORY_EXTRACTION", "1",
    ).strip().lower() in {"0", "false", "no", "off"}:
        return

    # 延迟 import：见 handler 里同一段说明（避免把 document 模型栈拖进来）。
    from app.services.memory.application.trigger import (
        submit_atom_extract,
        submit_extract,
    )

    try:
        await submit_extract(run_id=run_id, session_id=session_id)

    except Exception:  # noqa: BLE001 —— 旁路任务，失败只留日志
        logger.exception("后台提交 L0 记忆索引任务失败（忽略）：run=%s", run_id)

    try:
        await submit_atom_extract(
            session_id=session_id,
            run_id=run_id,
            repositories=repositories,
        )

    except Exception:  # noqa: BLE001 —— 旁路任务，失败只留日志
        logger.exception("后台提交 L1 记忆抽取任务失败（忽略）：run=%s", run_id)


def _consume_memory_task_exception(task: asyncio.Task) -> None:
    """取走后台任务的异常回执；任务本身已经自己吞过并记日志。"""

    if task.cancelled():
        return

    exc = task.exception()

    if exc is not None:
        logger.error(
            "后台记忆任务提交异常退出：%s",
            exc,
            exc_info=exc,
        )


__all__ = [
    "NodeLogRecorder",
    "REACT_GRAPH_NAME",
    "THREAD_NAMESPACE",
    "TRIGGER_SOURCE",
    "build_dispatch_handler",
]


#: 本模块要派的那张图。
#:
#: 从 react 的 catalog 转出来而不是让调用方各写一遍字符串：图名是那张图的
#: 编译期常量，这里只是个引用。放在这一层是为了让**服务层不必认识**
#: app.agent.graph.react 的路径 —— 它只知道"本模块负责的图"。
REACT_GRAPH_NAME = _REACT_GRAPH_NAME


#: thread_id 的命名空间。
#:
#: 与 document 的 "docver" 不同是**必须的**：两者共用同一个 checkpointer，
#: thread_id 是全库唯一键，撞上就会从别人的 checkpoint 续跑。
THREAD_NAMESPACE = "agentsess"


#: 运行由谁触发。规格里没有这个字段（它是观测层的），写个常量免得两处拼错。
TRIGGER_SOURCE = "api"


#: 已经结束、不该再执行的状态。handler 开工前据此丢弃迟到的信封
#: （典型来源：会话归档把尚未被调度器受理的 queued 运行直接取消了）。
_TERMINAL_STATUSES: frozenset[str] = frozenset({
    RunStatus.COMPLETED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
})


# ============================================================
# 只落 run 级的 recorder
# ============================================================


class NodeLogRecorder:
    """
    节点级观测**不落库**，只打日志。

    这不是"空实现"，是"写到另一个地方"：`node_span` 在节点开始 / 失败 /
    结束时各调一次 `upsert_node_run`，这里把它们记成一条带 run_id、
    node_id、status 与耗时的日志行 —— 排查时够用，且不必为它建一张表。

    将来接 SSE 时，这个类正是把节点事件推到流上的位置（那时它多一个
    "往流里放一条"的动作），所以它不是临时代码。

    ⚠️ 只有 run 级状态落库（`agent_run.status`），所以这里**不做**
    `create_run` / `mark_run_running` / `finish_run` —— 那三个方法由
    `GraphRunner` 调用，而本链路刻意不用它（见模块 docstring）。
    """

    async def upsert_node_run(self, update: Any) -> None:
        """`node_span` 的唯一入口。不要在这里抛 —— 它已被 _safe_record
        包着，抛了也只是多一条异常日志。"""

        # status 是 NodeRunStatus（str 枚举），直接 %s 会打出
        # "NodeRunStatus.RUNNING" 这种带上类名的东西，日志里不好读。
        status = getattr(update, "status", None)

        logger.info(
            "Agent node %s: node_id=%s group=%s run=%s duration_ms=%s",
            getattr(status, "value", status),
            getattr(update, "node_id", "?"),
            getattr(update, "node_group", "?"),
            getattr(update, "pipeline_run_id", "?"),
            getattr(update, "duration_ms", None),
        )


# ============================================================
# dispatcher handler
# ============================================================


def build_dispatch_handler(
        *,
        definition: GraphDefinition,
        repository_factory: RepositoryFactory,
        recorder: NodeLogRecorder,
) -> Callable[[DispatchEnvelope], Awaitable[None]]:
    """
    产出 dispatcher 用的 handler。

    handler 跑在调度器的后台 worker 里，**不在任何请求事务中** ——
    因此每一步状态写入都自开一段短事务（`RepositoryFactory.scope()`）。
    借用请求的 session 是不行的：那个连接早随请求还回池子了。

    Parameters
    ----------
    definition
        react 图的 GraphDefinition（含已编译的图与 context 工厂）。
    repository_factory
        以 session_factory 模式构造的工厂（后台路径的通行做法，
        与 SqlAlchemyRunRecorder 一致）。**不是**请求级 session。
    recorder
        与传给 `register_react_graph(recorder=...)` 的**同一个**实例。
        ReactContext 里的 recorder 其实来自那个闭包，这里再传一次是为了
        让 RunContext 描述的是同一次运行 —— 两个字段不该指向两个东西。
    """

    compiled = definition.compiled

    async def handler(envelope: DispatchEnvelope) -> None:

        run_id: UUID = envelope.pipeline_run_id
        thread_id: str = envelope.thread_id

        # 图是"跑完了"还是"停下来等人"（interrupt）—— 两者的终态**和锁的
        # 去向**都不同（见 finally）。
        #
        # ⚠️ 必须在 try 之前就绑定
        # ----------------------
        # `finally` 要读它。放在 try 里面的话，`_load_run` 抛错那条路径上
        # 它还没被赋值，`finally` 会先炸 NameError —— 而那是**唯一**放锁的
        # 地方，于是锁永远放不掉，这个会话此后每次建运行都 409。少覆盖一条
        # 路径的代价就是永久卡死。
        interrupted = False

        #: 这一段 handler 的起点（trace 的 run 级 duration_ms 用）。
        #: 放在 try 之前：except / finally 路径也要能读到。
        handler_started_ms = int(time.time() * 1000)

        # ⚠️ 这个 try 必须是函数体的**第一条语句**
        # --------------------------------------------
        # 下面的 finally 是**唯一**一处放 thread 锁的地方，它要覆盖：
        #
        #     正常跑完（END）        图抛异常（含 CancelledError）
        #     run 行不存在而抛       输入非法时的提前 return
        #     置 running 失败        写终态失败
        #     取消
        #
        # 少覆盖一条的后果是**永久卡死该会话**：锁不还，此后每次
        # POST /sessions/{id}/runs 都 409，只能重启进程才恢复。
        # 因此不用"按路径分别放锁"——那种写法迟早会漏掉某条 return。
        #
        # ⚠️ 中断是**唯一**一条例外，而且是刻意不放的 —— 见 finally。
        try:

            run = await _load_run(repository_factory, run_id)

            if run is None:
                # 只有接线错了会走到：dispatcher 派了一个本表不认识的 id。
                # 抛出去让 worker 记一条带栈的日志（它不会因此死掉）。
                raise RuntimeError(
                    "Cannot run agent run "
                    f"{run_id}: no such row in agent_run. The dispatcher "
                    "was handed an id that this table does not know."
                )

            # ------------------------------------------------
            # −1. 运行已经终态？
            #
            #    会话删除会把“调度器还不认识”的排队运行直接记为
            #    cancelled（它可能还没被 post-commit 受理，或者进程重启
            #    后没有任务在跑它）。那之后信封仍可能被投进来 —— 这里
            #    直接丢弃，绝不重新执行一次已经结束的运行。
            #
            #    终态判断必须在 `_mark_running` 之前：否则会把 cancelled
            #    覆盖回 running，图随后真的跑起来。
            # ------------------------------------------------

            if run.status in _TERMINAL_STATUSES:

                logger.info(
                    "Discarding dispatch for terminal run %s (status=%s).",
                    run_id,
                    run.status,
                )

                # finally 会走：interrupted 仍是 False，因此照常放锁；
                # release 对不持有的键是 no-op。
                return

            # ------------------------------------------------
            # 0. 排队期间就被取消了？
            #
            #    图**一次都不跑** —— 直接收尾。这条路径存在是因为取消可能
            #    在 handler 开工之前就来了（调度器把它登记下来，而不是等
            #    它跑起来）。不在这儿拦的话，一次已经取消的运行还是会真的
            #    去执行工具。
            #
            #    放在读运行行之后、其它一切之前：连带 `_mark_running` 都
            #    不做 —— 这次运行从来没有"开始过"，与输入非法那条同一立场。
            # ------------------------------------------------

            if get_run_cancellation_registry().is_requested(run_id):

                logger.info(
                    "Agent run %s cancelled while still queued.", run_id
                )

                # ⚠️ 这里**直接**补中断记录，不走"先试着写产出、写不到再补"
                # --------------------------------------------------
                # 图一次都没跑，因此按定义没有任何产出可写 —— 那一步在这里
                # 恒为 0，只是绕一圈。不补的话，会话历史里只剩用户那句话，
                # 而运行却是 cancelled：用户分不清"模型没回"和"我把它停了"。
                await _record_cancellation_notice(
                    repository_factory,
                    session_id=run.session_id,
                    run_id=run_id,
                    business_agent_id=run.duty_key,
                )

                await _finish(
                    repository_factory,
                    run_id,
                    RunStatus.CANCELLED,
                    None,
                    trace_id=run.trace_id,
                    session_id=run.session_id,
                    started_at_ms=handler_started_ms,
                )

                # 提前 return，finally 照跑 —— 锁照样放得掉。
                return

            # ------------------------------------------------
            # 1. 组图的输入。
            #
            #    resume is None  一轮新输入：组初始 state。放在置 running
            #                    **之前** —— 输入不可用时这次运行根本
            #                    没有开始过，它应当是 queued -> failed，
            #                    而不是先亮一下 running 再失败。
            #    resume 非 None  恢复上一次中断：**没有新 state**，
            #                    驱动值是一个 `Command`（见下）。
            # ------------------------------------------------

            resume = _resume_payload(envelope)

            state: Any = None

            if resume is None:

                try:
                    state = _initial_state(run)

                except ValueError as exc:
                    # warning 而不是 exception：这是**有记录的业务结果**（原因
                    # 已经写进 error_message），不是异常现场。堆栈只会指向本
                    # 模块自己的那一行 raise，加不了任何信息。
                    logger.warning(
                        "Agent run %s rejected before start: %s",
                        run_id,
                        exc,
                    )
                    await _finish(
                        repository_factory,
                        run_id,
                        RunStatus.FAILED,
                        exc,
                        trace_id=run.trace_id,
                        session_id=run.session_id,
                        started_at_ms=handler_started_ms,
                    )
                    # 提前 return，但 finally 照跑 —— 锁照样放得掉。
                    return

            else:

                # ⚠️ 恢复**不重新组 state**，这是 LangGraph 的硬要求
                # ------------------------------------------------
                # checkpoint 里那条 thread 上留着一个 pending 任务，它期待的
                # 是"给那个 `interrupt()` 一个答案"，不是"再来一轮输入"。
                # 传新 state 进去是未定义行为（langgraph 会把 HumanMessage
                # 追加到既有历史上，而 pending 任务仍然挂着）。
                #
                # 形状：`{interrupt_id: data}`。实测给对 id 时，`interrupt()`
                # 返回的就是 `data` 本身；给错 id 时 langgraph 把**整个字典**
                # 原样交出去，审批那端认不出形状、按 fail-closed 拒绝 ——
                # 失败方向是"更严"，不是放行。
                logger.info(
                    "Agent run %s resuming %d interrupt(s) (thread %s): %s",
                    run_id,
                    len(resume),
                    thread_id,
                    sorted(resume),
                )

            # ------------------------------------------------
            # 2. 置 running。必须在图开始跑之前提交，否则图跑挂了
            #    就没有"它确实开始过"这个事实。
            # ------------------------------------------------

            await _mark_running(repository_factory, run_id)

            # 诊断：run.started → 第一个 node.started 的空档拆解。
            # 只记慢的（>0.5s），正常请求不刷日志。
            t_run_started = time.monotonic()

            # 事件与库里的状态**同时**成为事实：上面那一行提交之后，
            # "它确实开始过"才成立，这条事件才有依据。
            #
            # 发失败不影响运行 —— publish_run_event 从不抛（见
            # app/common/event/publisher.py 的契约）。
            await publish_run_event(
                RunEventType.RUN_STARTED,
                str(run_id),
                data={"status": RunStatus.RUNNING.value},
            )

            # trace 的 run.started。同一个根 span_id（run_id）贯穿整次
            # 运行；resume 时会再发一次，表示"这一段又开始跑了"。
            await publish_trace_event(
                TraceEventType.RUN_STARTED,
                trace_id=run.trace_id,
                run_id=str(run_id),
                thread_id=str(run.session_id),
                span_id=str(run_id),
                span_kind=SpanKind.RUN,
                span_name="run",
                status=SpanStatus.RUNNING,
                payload={
                    "agent_type": AGENT_TYPE,
                    "graph_name": REACT_GRAPH_NAME,
                },
            )

            if resume is not None:
                # 审批被回答了 —— 与这次运行"确实又开始跑了"同一时刻成为
                # 事实（上面那一行提交之后才发）。
                #
                # 发在这里而不是 API 层：那一层只登记派发，派发可能被拒收
                # （队列满 / 正在关闭），那时说"已恢复"就是假的。
                await _publish_hitl_resolved(
                    run_id,
                    resume,
                    trace_id=run.trace_id,
                    session_id=run.session_id,
                )

            # ------------------------------------------------
            # 3. 跑图。这里不持有任何业务事务。
            # ------------------------------------------------

            run_context = RunContext(
                pipeline_run_id=run_id,
                thread_id=thread_id,
                trigger_source=TRIGGER_SOURCE,
                recorder=recorder,
                # 业务会话：工具执行账本（tool_execution.session_id）用它
                # 做判重键的第一段。这里有 `run`，所以是**唯一**顺手的
                # 地方 —— 再往下就只剩 thread_id 那个 langgraph 内部键了，
                # 而账本要的是会话自己的主键。
                session_id=run.session_id,
                # 建运行那一刻存下来的身份快照（见 agent_run.caller_identity）。
                # 这里是**唯一**能把它交给图的地方 —— 再往后就没有 token 了。
                caller_identity=dict(run.caller_identity or {}),
                # 这一轮用哪份职责。图 state 里有一份同名的（下面图输入那份），
                # 工具分支读不到 state（Send 的 payload 取代它），所以身份的
                # 快照里也带上 —— 记忆的 scope 是 (agent_id, user|org)，工具
                # 要按同一个 agent_id 定位，与注入那两段时用的是同一个值。
                #
                # 空串归一成 None：`duty_key` 为空是允许的（存量运行、或没指定
                # 职责），而"没有职责"与"职责叫空串"是同一件事。
                duty_key=run.duty_key or None,
                # Trace 开关：建运行时存下来的客户端值。resume 时同样
                # 从运行行取 —— 不假设客户端会再传一次。
                trace_id=run.trace_id or None,
            )

            context = definition.context_factory(run_context)

            config: dict[str, Any] = {
                # thread_id 进 config，不是进 context —— agent 节点与
                # checkpointer 都从这里取（没有它会直接抛）。
                "configurable": {
                    "thread_id": thread_id,
                },
                # 让 LangGraph 的根 run_id 等于我们的 run_id，
                # 观测行与 LangSmith trace 因此能对上（与 GraphRunner
                # 同一条）。
                "run_id": str(run_id),
            }

            # 恢复时：先把这次 HITL 问答落成 interaction 消息。
            #
            # 用户回答走 `Command(resume=...)`，不经过 `_initial_state`；
            # 不在这里补，历史里只有 assistant 的 tool_activity，翻不到
            # "问了什么、答了什么"。写失败不阻止 resume（见 helper）。
            if resume is not None:
                await _record_resume_interactions(
                    repository_factory,
                    compiled=compiled,
                    config=config,
                    run=run,
                    resume=resume,
                )

            # ------------------------------------------------
            # 3.5 图跑之前，先数一下这条 thread 上已经有多少条消息。
            #
            # messages 通道是**整条 thread 的历史**（checkpointer 累积的），
            # 不是本轮增量 —— 本轮产出的是这之后的那些。不先数一下，写库时
            # 会把前几轮的消息重复写一遍。见 message_recorder 的说明。
            #
            # 数不出来（checkpointer 读失败）就按 0 处理：那会让本轮的消息
            # 多写几条重复行，但不会让这次运行失败 —— 而一次读不出来的
            # checkpointer，下面的 invoke 马上就会以更明确的方式报出来。
            # ------------------------------------------------

            t_before_count = time.monotonic()
            prior_count = await _message_count(compiled, config)
            t_after_count = time.monotonic()

            if t_after_count - t_before_count > 0.5:
                logger.warning(
                    "Agent run %s: checkpointer aget_state took %.2fs "
                    "before graph start.",
                    run_id,
                    t_after_count - t_before_count,
                )

            error: BaseException | None = None

            # 驱动值：一轮新输入传 state，恢复传 `Command`。
            # ⚠️ 恢复时 `state` 是 None（见第 1 步），**不能**退化成传 None ——
            # langgraph 会把 None 当成"没有输入"，pending 任务仍然挂着，
            # 而运行会被记成 completed。
            graph_input: Any = (
                Command(resume=resume) if resume is not None else state
            )

            # 中断载荷，供 hitl.requested 用（见下）。
            pending_interrupts: list[Any] = []

            try:
                # ⚠️ durability="sync" 是**必须**的，不是保守起见
                # -------------------------------------------
                # 默认值 "async" 意味着 checkpoint 是"提交出去就不管了"，
                # 不等它落盘。后果分两处：
                #
                #   1. 下面第 4 步读终态时，最后一个 super-step 的
                #      AIMessage 可能还没写进 checkpointer —— 那条回复
                #      就丢了（而运行显示 completed）。
                #   2. 更要紧的是**下一轮**：新运行的 prior_count 是读
                #      checkpoint 数出来的，读到旧快照就意味着数少了，
                #      于是上一轮的消息又被写一遍。
                #
                # 代价是每个 super-step 多等一次 checkpoint 落盘。这条
                # 链路一次对话只有几个 super-step，换"不丢消息、不重复
                # 消息"是划算的。
                #
                # ⚠️ 为什么从 `ainvoke` 换成 v3 流
                # ---------------------------------
                # `StreamTransformer` **只**在 v3 通道里存在：`StreamMux`
                # 仅在 pregel/main.py:3533(sync)/3589(async) 构造，而
                # `ainvoke` 走的是 `self.astream(version="v1")`，完全不经过
                # mux（它的签名甚至只收 `version: Literal["v1","v2"]`）。
                # 要拿到协议事件流，只能换驱动 —— 没有别的入口。
                #
                # 换的只是**投影层**，不是执行引擎：`_apregel_stream_v3`
                # 内部就是 `self.astream(..., version="v2", **kwargs)`
                # （main.py:3601-3611），`**kwargs` 原样透传。所以上面的
                # durability、下面的 context / store / interrupt 语义全部照旧。
                #
                # ⚠️ 这一行**不启动图**。`await astream_events(...)` 只拿到流
                # 对象；图由后面 `await run_stream.output()` 驱动。忘了驱动的
                # 症状是 run 行安静地停在 running —— 与"进程被杀"那个已知缺口
                # 长得一样，很难区分，所以驱动那一行不能省。
                #
                # ⚠️ 名字**不能**叫 `run`：这个函数里 `run` 是 AgentRun 实体
                # （下面 `run.session_id` / `run.caller_identity` 都在用），
                # 用同名会把实体遮掉 —— 那是运行期 AttributeError，静态检查
                # 看不出来。
                run_stream = await compiled.astream_events(
                    graph_input,
                    config,
                    version="v3",
                    context=context,
                    durability="sync",
                    # ⚠️ 运行事件的**唯一**生产者就是这一个投影层。
                    # 节点里已经没有 publish_run_event 了（见 agent_node 的
                    # "本节点不再生产运行事件"）。
                    #
                    # `message_nodes` 只放 agent 节点：协议里的 messages
                    # 事件**不只来自它** —— builder 的结构化输出也会产生
                    # （实测 19 条，正文是路由 JSON）。不限制的话那段 JSON
                    # 会当作助手正文流进聊天窗。
                    #
                    # `message_roles` 是**第二道门**，拦的是同一个节点里的
                    # 另一次调用：上下文压缩（`summarization`）就在 agent
                    # 节点里面跑，节点名一样，只有角色分得开。少了它，压缩
                    # 结果会以两三千字的形式当助手正文推给用户（实测过）。
                    #
                    # `node_facts` 补协议给不出的领域字段：`tasks` 事件只有
                    # 节点名，没有 agent_type / model_role。它们是图的编译期
                    # 常量，这里注入即可。
                    transformers=[
                        projector_factory(
                            run_id=str(run_id),
                            message_nodes={AGENT_NODE},
                            message_roles={MODEL_ROLE},
                            # 节点目录里的身份字段（node_id / label /
                            # group / sequence）。以前只注入了 agent 一个，
                            # trace 需要给每个 node.* 都带上。
                            node_facts=_node_facts(),
                            # Trace：客户端传了才开。custom 模式始终请求
                            # （工具事件走它）；不开 trace 只是少请求
                            # tools 通道、不写 trace topic。
                            trace_id=run.trace_id or None,
                            session_id=str(run.session_id),
                        ),
                    ],
                )

                t_after_stream = time.monotonic()

                # ⚠️ `async with` 必须在 try **里面**：`__aexit__` 会走
                # `abort()`，那里面有 await。放到 try 外面的话，那一步若是
                # 抛出（取消路径上可能），会跳过下面的
                # `_record_produced_messages` / `_finish`，run 行就永远停在
                # running —— 正是这个 except 要防的那件事。
                async with run_stream:
                    t_before_output = time.monotonic()
                    pre_graph = t_before_output - t_run_started

                    if pre_graph > 0.5:
                        logger.warning(
                            "Agent run %s: %.2fs between run.started and "
                            "graph output start "
                            "(aget_state %.2fs, astream_events setup %.2fs, "
                            "__aenter__ %.2fs).",
                            run_id,
                            pre_graph,
                            t_after_count - t_before_count,
                            t_after_stream - t_after_count,
                            t_before_output - t_after_stream,
                        )

                    # 驱动器。返回值不再被消费 —— 终态改从
                    # `run_stream.interrupted()` 取，理由见 `_terminal_status`。
                    #
                    # ⚠️ 这一行**必须留着**：图是靠它驱动的。`astream_events(...)`
                    # 本身不启动任何东西（见上面那段说明），少了它 run 行会
                    # 安静地停在 running。
                    await run_stream.output()

                    interrupted = await run_stream.interrupted()

                    # 谁在等人。每个 `Interrupt` 带 `value`（审批载荷 —— 就是
                    # 拦截器交给 `interrupt()` 的那个字典）与 `id`（回答它时
                    # 要带上的那个）。`hitl.requested` 靠这两样，见下。
                    #
                    # 只在真的中断时取：没中断时它是空列表，取了也没有意义。
                    if interrupted:
                        pending_interrupts = await run_stream.interrupts()

            except BaseException as exc:  # noqa: BLE001
                # 与 GraphRunner 同一条理由：刻意捕获 BaseException 而不是
                # Exception —— asyncio.CancelledError 也要走终结路径，
                # 否则 run 行会永远停在 running。
                error = exc

                # ⚠️ 用户主动取消**不是失败**，日志这一侧也必须分岔
                # --------------------------------------------------
                # 终态那边已经分过了（下面把 error 置空、状态记 cancelled），
                # 但这里若一律按"运行失败"打一条 ERROR + 完整堆栈，日志里
                # 看到的仍然是一堆 `failed: CancelledError` —— 与
                # agent_run.error_* 要躲开的是同一个信号污染（见下面
                # cancelled 分支的说明），只是换了个地方漏。
                #
                # 判据用登记表而不是异常类型：取消**可能**以别的形态冒出来
                # （图在收尾时抛的别的东西），而"这次运行被登记为取消了"才
                # 是真相。关闭应用时 worker 被取消走的是另一条路（那里没有
                # 登记），因此不会误判成用户取消。
                if get_run_cancellation_registry().is_requested(run_id):
                    logger.info(
                        "Agent run %s cancelled by request "
                        "(interrupted at %s).",
                        run_id,
                        type(exc).__name__,
                    )

                else:
                    logger.exception(
                        "Agent run %s failed: %s",
                        run_id,
                        type(exc).__name__,
                    )

            # ------------------------------------------------
            # 3.9 图停下来了 —— 把"在等谁点头"发出去。
            #
            # ⚠️ 发在这里而不是拦截器里
            # ----------------------
            # `interrupt_id` 是 langgraph 在 `interrupt()` 抛出时才生成的，
            # 拦截器那一边（它正是抛出者）拿不到它。而客户端回答时必须带上
            # 这个 id（见 ResumeRequest.interrupt_id）—— 所以只能在这儿取。
            #
            # 只在 interrupted 时发。发失败不影响运行（publish_run_event
            # 从不抛，见 app/common/event/publisher.py 的契约）。
            # ------------------------------------------------

            if interrupted:
                await _publish_hitl_requested(
                    run_id,
                    pending_interrupts,
                    trace_id=run.trace_id,
                    session_id=run.session_id,
                )

            # ------------------------------------------------
            # 4. 先写本轮产出的消息，再写终态
            #
            # 顺序不能反。_finish 会发终止事件，而客户端收到
            # run.completed 后紧接着就会来查 messages —— 反过来的话存在一个
            # 真实的窗口：它看到"运行结束了"，却读不到回复。
            #
            # 写入排在 finally 放锁**之前**（本段还在 try 里），因此 seq 的
            # 取号仍在同一把 thread 锁内 —— 与建运行那一步写用户消息时
            # 是同一个前提。
            # ------------------------------------------------

            # ⚠️ 取消 = **强行中断**，不回滚
            # --------------------------------
            # 任务在某个 await 点上被砍断，checkpoint 上留下"跑到哪儿算哪儿"
            # 的半截状态 —— 就这样留着。不回滚的理由：
            #
            #   1. 快照必须**在跑之前**取（取消来了再取，拿到的已经是写完了
            #      的状态），于是每次运行都多一次 checkpoint 读 —— 只为了
            #      一条少数路径
            #   2. 回滚其实是"分叉"（在旧 checkpoint 之下写一份新 head），
            #      旧的那些都还在。看着干净了，实际是一份只增不减的历史
            #   3. 半截状态本来就是**既有缺口**（见模块 docstring 的 (b)：
            #      "失败后下一轮从残缺状态继续"），取消让它更容易触发，
            #      但没有新引入什么
            #
            # 因此取消与正常收尾**走同一条路**，只在终态上分岔 ——
            # 本轮已经产出的消息照常落库，带上 `cancelled` 的标记。
            # `message_status` 里本来就有这一格（MessageStatus.CANCELLED），
            # 所以 agent_message 与 checkpoint 说的是同一件事，而不是
            # "库里少几条、模型却看得见"。
            cancelled = get_run_cancellation_registry().is_requested(run_id)

            if cancelled:
                terminal_status = RunStatus.CANCELLED
                # ⚠️ 用户主动取消不是"失败"，error_* 上不该留一个
                # CancelledError —— 那会让运维在失败列表里看到一堆用户
                # 正常取消的运行，而失败率这个信号就废了。
                error = None
            else:
                terminal_status = _terminal_status(error, interrupted)

            written = await _record_produced_messages(
                repository_factory,
                compiled,
                config,
                prior_count,
                session_id=run.session_id,
                run_id=run_id,
                status=terminal_status,
                # 这一轮用什么职责是建运行那一刻定下的（run.duty_key），
                # 派发这边只是把它带给消息 —— 不重新解析一次。
                business_agent_id=run.duty_key,
            )

            # ⚠️ 取消了、而本轮**一条 message 都没写成** -> 补一条中断记录。
            #
            # 走到这里的典型是"跑了一半被取消、但还没有一条完整的
            # AIMessage"（agent_node 整段生成完才产出，见 message_recorder
            # 的"已知缺口"）。症状同样是：会话历史里只剩用户那句话。
            #
            # ⚠️ 只有取消才补
            # --------------
            # 正常结束与失败**不补**：失败有自己的 error_* 落库、有
            # run.failed 事件，而"模型什么都没说"对失败而言是**真实的**
            # 信息，给它补一条公告反而把真相盖住了。取消不同 —— 那是
            # **外部打断**，不留下痕迹用户就分不清"模型没回"和"我停了它"。
            if cancelled and written == 0:
                await _record_cancellation_notice(
                    repository_factory,
                    session_id=run.session_id,
                    run_id=run_id,
                    business_agent_id=run.duty_key,
                )

            await _finish(
                repository_factory,
                run_id,
                terminal_status,
                error,
                trace_id=run.trace_id,
                session_id=run.session_id,
                started_at_ms=handler_started_ms,
            )

            # ---- L0 记忆索引：把这一轮交给后台任务 ----
            #
            # ⚠️ 位置在 `_finish` **之后**，三个理由缺一不可
            # -------------------------------------------
            #   1. 消息已提交 —— `_record_produced_messages` 在上面跑完了，
            #      任务即使被立刻投递也读得到这一轮的全部消息
            #   2. run 已终态 —— `_finish` 刚把 status 落库并提交，所以
            #      任务不需要"等 run 结束"这一层重试
            #   3. **失败不回滚任何东西** —— 此时这次运行已经收尾完毕，
            #      一个旁路任务提交失败只该留下一条日志
            #
            # ⚠️ 它**不抛异常**（见 memory/application/trigger.py）：
            # 记忆索引是旁路，它失败不该让一次成功的运行变成失败的
            # handler。吞掉的是异常，不是问题 —— 那里留了一条 warning。
            #
            # ⚠️ 为什么不是图里的 deferred node
            # ------------------------------
            # deferred node 在**图返回之前**执行，而消息写入在**图返回
            # 之后**（本文件 :630 附近）—— 于是它提交的任务可能先跑、
            # 读到一条消息都没有。放在这里顺序是确定的。这也正是
            # message_recorder 的 docstring 把消息落库点选在 handler 里
            # 的同一条理由（"每条消息的落库时机应该是它自己的结局确定的
            # 那一刻"），而运行终局的那一瞬就在这一行。
            #
            # ⚠️ **waiting_input 不提交** —— 它不是运行的结尾
            # --------------------------------------------
            # `_terminal_status` 的 docstring 早就把这条写清楚了："图停下来了，
            # 但这次运行还没完"。收尾会再发生一次（resume 跑完），所以在这里
            # 建索引的**直接**后果是：索引的是**半程**（用户那句话 + 动手前的
            # 旁白），而真正回答了问题的最后一条还没生成。
            #
            # 实测现场：一次运行 5 条消息，索引里只有 seq=1/2，而 seq=5
            # （真正回答了"平台有哪些工具"的那条）缺席。
            #
            # ⚠️ 当初还有第二个后果，现在已经不成立了
            # ------------------------------------
            # 那时幂等键是 `memory.extract:{run_id}`，而**终态的任务会永久
            # 占住它的键** —— 于是"半程"那次提交跑完之后，resume 之后那次
            # 提交只会拿回那条已完成的任务，后半程永远进不了索引。
            # 2026-10-10 起终态会释放幂等键（见 task_impl 的「幂等键的
            # 生命周期」），所以那个后果没有了：resume 之后那次会排一条**新的**，
            # 把后半程补上。
            #
            # 规则本身**不变**，因为第一条理由还在：半程索引进来的是没有答案的
            # 半截对话，那比"稍后再索引"糟。
            #
            # 取消**要**提交：取消意味着这次运行真的结束了，而 "取消" 与
            # "停下等人" 是两件事（见 finally 里放锁的那段对照）。
            #
            # 已知代价：停在 interrupt 上**再也没被 resume** 的运行（被放弃
            # 的那条路），它的消息不会被索引。这是可接受的 —— 那一轮的结论
            # 本来就没产生，把它当成一条"记忆"会让检索到一个没有答案的
            # 半截对话。真要补，按 run_id 重跑一次 `memory.extract` 即可。
            #
            # ⚠️ import 放在函数里是**刻意的**
            # ------------------------------
            # memory.application 的 __init__ 会拉起它的服务实现，而那个
            # 实现又 import document 的流水线注册表与 embedder。写成模块级
            # import 就等于：任何 import 本模块的地方（包括 session 自己的
            # 冒烟脚本）都会连带拉起 document 的模型栈 —— 一条谁也没打算
            # 建立的依赖。与 bootstrap/task.py 的 _register_handlers 里
            # 那几行放在函数内是同一个理由。
            if terminal_status is not RunStatus.WAITING_INPUT:

                # ---- L0/L1 记忆任务：移出 handler 关键路径 ----
                #
                # 它们发生在 run.completed 之后，读的是**已经提交**的消息
                # 与水位，不参与终态判定。改成 create_task 之后，thread 锁
                # 可以立刻释放、worker 可以接下一轮，不再被这 1.6s 挡住。
                #
                # ⚠️ 代价：进程被直接杀掉时，排队中的提交会丢（见
                # `_submit_memory_tasks` 的 docstring）。这两条都是自愈的
                # 旁路任务，接受这个窗口。
                task = asyncio.create_task(
                    _submit_memory_tasks(
                        run_id=run_id,
                        session_id=run.session_id,
                        repositories=repository_factory,
                    ),
                    name=f"memory-submit:{run_id}",
                )
                task.add_done_callback(_consume_memory_task_exception)

        finally:
            # 唯一一处放锁。理由与覆盖范围见上面 try 的说明。
            #
            # ⚠️ 唯一一条**不放**的路径：图停在 interrupt 上等人
            # -------------------------------------------------
            # thread_id 是**按会话稳定**的（见 _resolve_thread_id）。锁一放，
            # 下一个请求就能在同一条 thread 上 `ainvoke` —— 而 checkpointer
            # 那里留着一个 pending 任务，langgraph 此刻期望的是
            # `Command(resume=...)`。两者撞上会损坏那份 checkpoint。
            #
            # 所以"在途"的定义延伸到**等人期间**：锁由这次运行一直持有，直到
            # resume 跑完、或取消、或超时收敛把运行判掉
            # （见 agent_run_service_impl._schedule_resume —— 恢复路径因此
            # **不重新取锁**，它继承的就是这一把）。
            #
            # 代价说清楚：等人工输入期间这个会话建不了新运行（409）。那是对
            # 的 —— 那条 thread 上确实还挂着一件没做完的事。
            #
            # ⚠️ 同步调用是刻意的：这一刻任务可能正被取消，finally 里
            # 任何一个 await 都可能立刻再抛一次，而锁必须放掉。
            # release 对"不是自己持有的键"是 no-op，因此重复放也安全。
            # 取消请求到这一刻才算用完 —— 上面每个分支都读过它。
            cancel_requested = (
                get_run_cancellation_registry().is_requested(run_id)
            )

            # 只有"停下等人工输入、且没人取消"才留锁（理由见上）。
            # 取消意味着**这次运行真的结束了**（状态已记成 cancelled），
            # 所以锁必须放掉 —— 与中断那条正好相反。
            if not interrupted or cancel_requested:
                get_thread_run_lock().release(thread_id, run_id=run_id)

            # 登记表按 run_id 攒，不清理就是一条内存泄漏（症状是"跑得越久
            # 占用越大"）。收尾是唯一确定会跑到的地方，所以放在这里。
            get_run_cancellation_registry().discard(run_id)

    return handler


# ============================================================
# 人工介入（interrupt / resume）
# ============================================================


def _resume_payload(
        envelope: DispatchEnvelope,
) -> dict[str, dict[str, Any]] | None:
    """
    envelope 里"这是一次恢复"的标记：`{interrupt_id: data, ...}`。

    写入方见 `AgentRunServiceImpl._schedule_resume`；形状正好就是
    `Command(resume=...)` 要的那个映射，中间不再翻译一次。

    ⚠️ 一次可以带**多个** —— 扇出下多个工具同时等审批是常态（实测
    `next=('tools','tools')`）。契约层要求一次给全（见 `ResumeRequest`），
    这里只管把它原样交给引擎。

    ⚠️ 形状不对时**当作没有**标记（返回 None），也就是按一轮新输入跑。
    ------------------------------------------------------------
    看着像"悄悄降级"，但那正是**安全的那一边**：`_initial_state` 会用运行行
    里存着的原始输入重新组一次 state、图从头跑；而一个形状不对的 resume
    本来也没有任何可用的语义。反过来（硬按 resume 跑）才会拿半截数据去驱动
    一个 pending 任务。

    何况这个键只有本模块自己的 `_schedule_resume` 会写 —— 形状不对说明有人
    手改了 envelope，或者两个模块的契约漂移了。
    """

    resume = (envelope.payload or {}).get("resume")

    if not isinstance(resume, dict) or not resume:
        return None

    answers: dict[str, dict[str, Any]] = {}

    for interrupt_id, data in resume.items():

        if (
                not isinstance(interrupt_id, str)
                or not interrupt_id
                or not isinstance(data, dict)
        ):
            logger.warning(
                "envelope 里的 resume 标记形状不对（%r），"
                "本次按一轮新输入处理。",
                resume,
            )
            return None

        answers[interrupt_id] = data

    return answers


async def pending_interrupt_ids(
        definition: GraphDefinition,
        *,
        thread_id: str,
) -> list[str]:
    """
    这条 thread 上当前**所有**待处理的中断 id。

    ⚠️ 真相只有一处：checkpoint
    --------------------------
    事件流（`hitl.requested`）也带着这些 id，但它是**旁路** —— Redis 流有
    TTL，过期就没了；而"这次 resume 该带哪些 id"是个**校验**问题，拿一个
    会过期的来源去校验，会在流过期之后把所有正常的 resume 判成 409。

    所以读 checkpoint：`aget_state(...).tasks[*].interrupts[*].id`（实测
    扇出下每个 pending 分支各一个 task、各一个 interrupt）。

    ⚠️ 它读的是**引擎的内部结构**，这一点是明知故犯
    -------------------------------------------
    `StateSnapshot.tasks` 不是 langgraph 的对外协议。用它是为了让
    "少了/多了"能在**请求期**同步报出来 —— 否则客户端拿到 200，而那次
    resume 只是把图重新停了一遍（实测少给一个 id 就是这个结果）。

    替代方案是"不校验、让客户端自己对齐"，代价是那种静默的空推进。这里
    选了可诊断性，并且把读法收在这一个函数里 —— 引擎升级改了结构，
    只需要改这里。
    """

    snapshot = await definition.compiled.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )

    return [
        str(item.id)
        for task in (getattr(snapshot, "tasks", None) or ())
        for item in (getattr(task, "interrupts", None) or ())
    ]


def _approval_outcome(data: Any) -> str:
    """
    resume 值 -> Trace 的三种审批结论。

    只做**词表映射**，不替 ApprovalDecision 判语义（那个仍由拦截器做）：
    approve/allow/always -> approved；reject/deny -> rejected；
    其余（含无法识别）-> rejected（fail-closed，与拦截器一致）。
    """

    if isinstance(data, bool):
        return "approved" if data else "rejected"

    if isinstance(data, dict):
        data = data.get("decision", data.get("choice"))

    if isinstance(data, str):
        normalized = data.strip().lower()
        if normalized in ("approve", "allow", "always"):
            return "approved"
        if normalized in ("reject", "deny"):
            return "rejected"

    return "rejected"


async def _publish_hitl_requested(
        run_id: UUID,
        interrupts: Sequence[Any],
        *,
        trace_id: str | None = None,
        session_id: UUID | None = None,
) -> None:
    """
    "这次运行在等人工审批" —— 每个中断一条。

    `data` 就是拦截器交给 `interrupt()` 的那个字典（`ask` / `tool_key` /
    `arguments` / `options` …）**原样透传**，再补一个 `interrupt_id`：客户端
    回答时必须把它带回来（见 ResumeRequest.interrupt_id）。

    ⚠️ 这里**不解释**那个字典的内容
    ----------------------------
    审批有几个选项、每个选项什么含义，是**工具侧处理器**的契约
    （processor/interceptor/approval.py 的 `ApprovalDecision`）。派发层把它
    当不透明载荷搬运 —— 这样换一种需要人工介入的处理器（比如"补充信息"时
    的提问）时，这里一行都不用改。
    """

    for item in interrupts:

        value = getattr(item, "value", None)

        data: dict[str, Any] = (
            dict(value) if isinstance(value, dict) else {"value": value}
        )

        interrupt_id = str(getattr(item, "id", "") or "")

        data["interrupt_id"] = interrupt_id

        await publish_run_event(
            RunEventType.HITL_REQUESTED,
            str(run_id),
            data=data,
            node=TOOLS_NODE,
        )

        if not trace_id:
            continue

        tool_call_id = data.get("tool_call_id")
        tool_name = str(data.get("tool_key") or "")
        failed_task_id = data.get("task_id")

        approval_span_id = stable_span_id(
            "approval", trace_id, interrupt_id
        )

        approval_parent = (
            tool_span_id(str(trace_id), str(tool_call_id))
            if tool_call_id is not None
            else (
                str(failed_task_id)
                if failed_task_id is not None
                else str(run_id)
            )
        )

        await publish_trace_event(
            TraceEventType.HITL_APPROVAL_REQUESTED,
            trace_id=trace_id,
            run_id=str(run_id),
            thread_id=None if session_id is None else str(session_id),
            span_id=approval_span_id,
            parent_span_id=approval_parent,
            span_kind=SpanKind.APPROVAL,
            span_name="approval",
            status=SpanStatus.RUNNING,
            payload={
                "approval_id": interrupt_id,
                "action_name": tool_name,
            },
            event_id=stable_span_id(
                "event", trace_id, interrupt_id, "approval_requested"
            ),
        )

        if tool_call_id is not None:
            await publish_trace_event(
                TraceEventType.TOOL_INTERRUPTED,
                trace_id=trace_id,
                run_id=str(run_id),
                thread_id=None if session_id is None else str(session_id),
                span_id=tool_span_id(str(trace_id), str(tool_call_id)),
                parent_span_id=(
                    str(failed_task_id)
                    if failed_task_id is not None
                    else str(run_id)
                ),
                span_kind=SpanKind.TOOL,
                span_name=tool_name or "tool",
                status=SpanStatus.INTERRUPTED,
                payload={
                    "tool_name": tool_name,
                    "tool_call_id": str(tool_call_id),
                    "interrupt_id": interrupt_id,
                },
                event_id=stable_span_id(
                    "event", trace_id, interrupt_id, "tool_interrupted"
                ),
            )


async def _publish_hitl_resolved(
        run_id: UUID,
        resume: dict[str, dict[str, Any]],
        *,
        trace_id: str | None = None,
        session_id: UUID | None = None,
) -> None:
    """
    "那些中断被回答了" —— 每个一条，与 `hitl.requested` 对称。

    一次恢复可以覆盖多个中断（扇出下多个工具同时等审批是常态），所以这里
    是循环而不是发一条。一条事件带一个 `interrupt_id`，客户端据此把它跟
    自己发出去的那份答案对上。

    ⚠️ 它**不代表审批通过**
    --------------------
    只代表有人答了。答的是什么在 `data` 里，而"这个回答意味着放行还是拒绝"
    由处理器判 —— 它可能把 `always` 降级成仅这次，也可能因为选项不在
    `options` 里而拒绝（见 approval.py）。这里不替它下结论，只搬运。
    """

    for interrupt_id, data in resume.items():

        await publish_run_event(
            RunEventType.HITL_RESOLVED,
            str(run_id),
            data={"interrupt_id": interrupt_id, "data": data},
            node=TOOLS_NODE,
        )

        if not trace_id:
            continue

        approval_span_id = stable_span_id(
            "approval", trace_id, str(interrupt_id)
        )

        await publish_trace_event(
            TraceEventType.HITL_DECISION,
            trace_id=trace_id,
            run_id=str(run_id),
            thread_id=None if session_id is None else str(session_id),
            span_id=approval_span_id,
            span_kind=SpanKind.APPROVAL,
            span_name="approval",
            payload={
                "approval_id": str(interrupt_id),
                "decision": _approval_outcome(data),
                "actor_type": "user",
            },
            event_id=stable_span_id(
                "event", trace_id, str(interrupt_id), "hitl_decision"
            ),
        )

        await publish_trace_event(
            TraceEventType.HITL_RESUMED,
            trace_id=trace_id,
            run_id=str(run_id),
            thread_id=None if session_id is None else str(session_id),
            span_id=approval_span_id,
            span_kind=SpanKind.APPROVAL,
            span_name="approval",
            status=SpanStatus.COMPLETED,
            links=[
                SpanLink(
                    relation=LinkRelation.RESUMED_FROM,
                    span_id=approval_span_id,
                )
            ],
            payload={
                "approval_id": str(interrupt_id),
                "resumed_span_id": approval_span_id,
            },
            event_id=stable_span_id(
                "event", trace_id, str(interrupt_id), "hitl_resumed"
            ),
        )




def _terminal_status(
        error: BaseException | None,
        interrupted: bool,
) -> RunStatus:
    """
    这次运行该怎么收尾。

        error 非空      -> failed
        interrupted     -> waiting_input（图停下来等人）
        其余            -> completed（走到了 END）

    ⚠️ 检测来源换了，语义没换
    ------------------------
    以前判的是 `result["__interrupt__"]`，现在判 `run.interrupted()`。两者
    同源 —— 都是 values 事件 `params["interrupts"]` 里那个字段 —— 但走 v3
    之后 `result` 这条路**不通了**：

        `run.output()` 返回的是 `AsyncGraphRunStream._latest`，而它取的是
        values 事件的 `params["data"]`。v2 的 `_output` 在产出那条 chunk 时
        已经把 interrupt **pop 出去了**：

            ints = payload.pop(INTERRUPT, ())          # main.py:4224
            yield {"...", "data": payload, "interrupts": ints}

        于是 data 里**永远不会**有 `__interrupt__`。继续读那个键，interrupt
        会被静默记成 COMPLETED，还会发出错误的 `run.completed`。

        顺带说明今天为什么一直没暴露：v1 是靠 `invoke` 在返回前手动把
        `{**latest, INTERRUPT: interrupts}` 塞回去才有的（main.py:3946-3955），
        v3 不经过那段。也就是说这个键本来是 v1 的补丁，不是 values 自带的。

    参数名从 `result` 改成 `interrupted` 是刻意的：让"这里吃的是布尔"在
    调用点就看得见，而不是继续传一个没人再需要的 dict。

    ⚠️ `waiting_input` **不是**本次运行的结尾
    --------------------------------------
    它是"图停下来了，但这次运行还没完"。三处后果：

        锁       **不放**（见 handler 的 finally）—— 那条 thread 上还挂着
                 一个 pending 任务，谁都不能在上面再 invoke
        事件     **不发终止事件**（见 `_publish_terminal`），所以 SSE 流
                 不会关 —— 后面还有 hitl.resolved 与恢复后的事件
        消息     本轮已产出的 assistant 消息按 `interrupted` 落库
                 （见 message_recorder 的 status 映射）

    三者合起来才构成"运行停在中途"这个事实。任何一处按"结束"处理，都会
    让客户端以为可以开始下一轮 —— 而那时 checkpoint 是坏的。
    """

    if error is not None:
        return RunStatus.FAILED

    if interrupted:
        return RunStatus.WAITING_INPUT

    return RunStatus.COMPLETED


# ============================================================
# Helpers
# ============================================================


def _initial_state(run: AgentRun) -> dict[str, Any]:
    """
    用户输入 -> ReAct 图的初始 state。

    只填四个通道，其余**不填**：`ReactState` 是 TypedDict，多写的键会被
    LangGraph 静默丢掉，所以"没填"和"填了没用"在这里是等价的 —— 与其写
    一堆空值装作上下文很丰富，不如只给确定有内容的那几个。

        messages        这一轮用户说的话。
        conversation_id 会话身份 = session_id。
        user_id         发起这次运行的用户（agent_run.user_id）。空串是
                        **约定的"没有用户"**，画像处理器据此静默跳过
                        （见 ReactState 的说明）。
        duty_key        这一轮用哪份职责提示词（agent_run.duty_key）。空串
                        同样是约定的"没指定"，builder 节点据此回落默认职责
                        —— 存量运行（那一列为 NULL）走的就是这条。

    ⚠️ user_id 直接读 run，不额外查会话
    ---------------------------------
    建运行时它就被写到 `agent_run.user_id` 上了（见
    agent_run_service_impl.create_run），而这里手头只有 run 实体 ——
    派发路径**没有请求上下文**，为了拿它再去读一次 agent_session 是多余的，
    也会平白给后台路径接上一个它不该有的依赖。

    ⚠️ conversation_id 仍然显式给 session_id，但理由变了
    ------------------------------------------------
    以前 thread_id 按**运行**生成，conversation_id 显式给 session_id 是
    为了不让会话身份跟着"每次运行都变的那个东西"走。

    现在 thread_id 按**会话**稳定（存在 `agent_session.thread_id` 上），
    两者已经等价 —— 但这里仍然显式写 session_id：它表达的是"这一段对话
    属于谁"，与"state 存在哪条 thread 下"是两件事，不该因为今天恰好相等
    就把它们的来路合并（将来一个会话开多条 thread 时，这个区别就回来了）。

    多轮历史因此可以工作了：`add_messages` 会把这一轮的 HumanMessage 追加
    到该 thread 已有的 state 上。代价见模块 docstring 的 a / b 两条。

    ⚠️ **不要**在这里写 `plan_status` / `plan`
    -----------------------------------------
    新运行的初始 state 会**合并**到同一条 thread 的 checkpoint 上。写
    `plan_status="off"` 等于每一轮都把正在进行的计划模式重置一次；而写
    `plan={}` 会把 planner 已产出的任务清空。模式状态由 tools 节点
    （执行 enter / exit 工具后结算）与 planner（计划产出）写，派发方
    只负责不要碰它们。

    Raises:
        ValueError: `input.text` 缺失或只有空白。规格与 DTO 都要求必填，
                   走到这里说明有人绕过了接口直接写库。
    """

    text = str(run.input.get("text") or "").strip()

    if not text:
        raise ValueError(
            "input.text 是空的，没有可处理的内容。"
            "（规格要求它是必填的非空字符串。）"
        )

    return {
        "messages": [HumanMessage(content=text)],
        "conversation_id": str(run.session_id),
        # None -> ""：图的契约里 user_id 是字符串，空串才是"没有用户"
        # 那个约定值。加这列之前建的行（user_id 为 NULL）走的就是这条。
        "user_id": run.user_id or "",
        # 这一轮用哪份**职责提示词** —— 建运行那一刻定下的，这里只是搬运。
        # 空串（存量运行 / 解析失败回落之外的场合）在 builder 节点那边等价于
        # "没指定"，会回落到默认职责，与这里之前的恒默认行为一致。
        "duty_key": run.duty_key or "",
    }


async def _load_run(
        repository_factory: RepositoryFactory,
        run_id: UUID,
) -> AgentRun | None:
    """只读，因此 begin=False（不开显式事务）—— 与 recorder 的
    summarize 同一条规矩。"""

    async with repository_factory.scope(begin=False) as repos:
        return await repos.create(AgentRunRepository).get(run_id)


async def _message_count(
        compiled: Any,
        config: dict[str, Any],
) -> int:
    """
    invoke 之前，这条 thread 上已有的消息条数。

    读失败按 0 处理而不是往上抛：这是"为了写库不重复"而做的一次准备，让它
    有能力**搞砸一次本来能跑通的运行**是不划算的。退化的后果只是多写几条
    重复消息（而且下面的 invoke 十有八九会以更明确的方式报出同一个问题）。
    """

    try:
        snapshot = await compiled.aget_state(config)

    except Exception:  # noqa: BLE001 - 见 docstring
        logger.exception(
            "读取图状态失败（thread %s），本轮消息条数按 0 处理 —— "
            "可能因此重复写入前面几轮的消息。",
            config.get("configurable", {}).get("thread_id"),
        )
        return 0

    return len(snapshot.values.get("messages") or [])


async def _record_resume_interactions(
        repository_factory: RepositoryFactory,
        *,
        compiled: Any,
        config: dict[str, Any],
        run: AgentRun,
        resume: Mapping[str, Any],
) -> None:
    """
    这次 resume 里若有 ask_user_question，把问答落成 interaction 消息。

    ⚠️ 为什么不抛
    -----------
    写历史是旁路；写失败不该阻止图 resume（用户已经把答案给了，图还能
    继续）。但必须留日志 —— 否则"历史里为什么没有这次问答"又成了一个
    只能靠翻 checkpoint 猜的问题。
    """

    try:
        snapshot = await compiled.aget_state(config)

    except Exception:  # noqa: BLE001 - 见 docstring
        logger.exception(
            "Agent run %s: 读取中断快照失败，本次 HITL 问答不落库。",
            run.run_id,
        )
        return

    by_id = {
        str(getattr(item, "id", "")): item
        for item in (getattr(snapshot, "interrupts", ()) or ())
    }

    for interrupt_id, data in resume.items():

        item = by_id.get(str(interrupt_id))

        if item is None:
            continue

        payload = getattr(item, "value", None)

        if (
                not isinstance(payload, Mapping)
                or payload.get("type") != "ask_user_question"
        ):
            # 审批等其它中断：不是问答，不写 interaction。
            continue

        try:
            questions, answers, annotations = parse_interaction(
                payload, data
            )

            if not questions:
                continue

            written = await record_interaction(
                repository_factory,
                session_id=run.session_id,
                run_id=run.run_id,
                business_agent_id=run.duty_key,
                questions=questions,
                answers=answers,
                annotations=annotations,
                interrupt_id=str(interrupt_id),
            )

            if written:
                logger.info(
                    "Agent run %s: HITL 问答已写为 interaction 消息"
                    "（%d 个问题）。",
                    run.run_id,
                    len(questions),
                )

        except Exception:  # noqa: BLE001 - 见 docstring
            logger.exception(
                "Agent run %s: HITL 问答落库失败（interrupt %s）。",
                run.run_id,
                interrupt_id,
            )


async def _record_produced_messages(
        repository_factory: RepositoryFactory,
        compiled: Any,
        config: dict[str, Any],
        prior_count: int,
        *,
        session_id: UUID,
        run_id: UUID,
        status: RunStatus,
        business_agent_id: str | None,
) -> int:
    """
    把本轮产出的消息写进 agent_message 表，返回**实际写入的条数**。

    返回值不是装饰：取消路径要靠它判断"这一轮是不是一条都没写成" ——
    那种情况下才补一条中断记录（见 `record_cancellation_notice`）。

    ⚠️ 本函数**绝不抛**
    -----------------
    它在 `_finish` 之前调用，而 `_finish` 是写运行终态的地方 —— 让它抛出去
    的后果不是"消息没写"，是**运行永远停在 running**（锁在 finally 里照样
    会放，于是下一次请求面对的是一个卡住的运行）。一次消息写入的失败不该
    有那个后果。

    因此这里吞掉 Exception 并记日志（而不是 BaseException）：任务被取消时
    CancelledError 必须继续往上走，那是 handler 的终结路径要处理的事，
    不该在这里被拦下。

    失败可见性：日志里有完整堆栈。消息与运行状态不一致这件事本身也查得出来
    —— 运行 completed 而 agent_message 表里没有对应的 assistant 行。

    ⚠️ 失败时返回 0
    --------------
    不能假装"写成功了"（那会让调用方跳过补记录），也不能返回一个负数之类
    的哨兵（调用方还得再学一个约定）。返回 0 让"写失败"与"没有产出"汇到
    同一条路：取消时补一条中断记录。**宁可多一条公告，也别静默什么都没有。**
    """

    try:
        snapshot = await compiled.aget_state(config)

        produced = new_assistant_messages(
            prior_count,
            snapshot.values.get("messages") or [],
        )

        written = await record_messages(
            repository_factory,
            session_id=session_id,
            run_id=run_id,
            status=status,
            messages=produced,
            business_agent_id=business_agent_id,
        )

        if written:
            logger.info(
                "Agent run %s persisted %d message(s).",
                run_id,
                written,
            )

        return written

    except Exception:  # noqa: BLE001 - 见 docstring
        logger.exception(
            "写入运行 %s 的消息失败（运行状态照常写回）。",
            run_id,
        )

        return 0


async def _record_cancellation_notice(
        repository_factory: RepositoryFactory,
        *,
        session_id: UUID,
        run_id: UUID,
        business_agent_id: str | None,
) -> None:
    """
    补一条取消中断记录。**绝不抛**。

    与 `_record_produced_messages` 同一条理由：它在 `_finish` 之前，而
    `_finish` 是写运行终态的地方 —— 让它抛出去的后果不是"少一条记录"，
    是**运行永远停在 running**。

    ⚠️ 与 `_record_produced_messages` 的差别：那里"没有产出"是常态、值得
    静默；这里**不该发生**（调用方已经判过"一条都没写成"），所以成功也
    记一条 info —— 它出现就说明这一轮确实什么都没留下。
    """

    try:
        await record_cancellation_notice(
            repository_factory,
            session_id=session_id,
            run_id=run_id,
            business_agent_id=business_agent_id,
        )

        logger.info(
            "Agent run %s cancelled without producing any message; "
            "记录了一条中断公告。",
            run_id,
        )

    except Exception:  # noqa: BLE001 - 见 docstring
        logger.exception(
            "写入运行 %s 的取消记录失败（运行状态照常写回）。",
            run_id,
        )


def _node_facts() -> dict[str, dict[str, Any]]:
    """
    节点名 -> trace / run 事件需要的领域字段。

    `tasks` 协议事件只给节点名；node_id / label / group / sequence 是图的
    编译期常量（NODE_SPECS），agent_type / model_role 则只有两个模型节点
    才有。全部在这里注入，投影层不需要 import 图。
    """

    facts: dict[str, dict[str, Any]] = {
        spec.node_id: {
            "node_id": spec.node_id,
            "label": spec.label,
            "group": spec.group,
            "sequence": spec.sequence,
        }
        for spec in NODE_SPECS
    }

    # planner 子图的节点也要有 group / label：projector 用 group 判断
    # 哪些节点是 fanout（planner_tool 的 trace 不能挂到 run 上）。
    for spec in _PLANNER_NODE_SPECS:
        facts.setdefault(spec.node_id, {
            "node_id": spec.node_id,
            "label": spec.label,
            "group": spec.group,
            "sequence": spec.sequence,
        })

    facts.setdefault(AGENT_NODE, {}).update({
        "agent_type": AGENT_TYPE,
        "model_role": MODEL_ROLE,
    })
    facts.setdefault(AGENT_BUILDER_NODE, {}).update({
        "agent_type": AGENT_TYPE,
        "model_role": CLASSIFIER_MODEL_ROLE,
    })
    facts.setdefault(_PLANNER_LLM_NODE, {}).update({
        "agent_type": _PLANNER_AGENT_TYPE,
        "model_role": _PLANNER_MODEL_ROLE,
    })

    return facts


async def _mark_running(
        repository_factory: RepositoryFactory,
        run_id: UUID,
) -> None:

    async with repository_factory.scope() as repos:
        await repos.create(AgentRunRepository).mark_running(run_id)


async def _finish(
        repository_factory: RepositoryFactory,
        run_id: UUID,
        status: RunStatus,
        error: BaseException | None,
        *,
        trace_id: str | None = None,
        session_id: UUID | None = None,
        started_at_ms: int | None = None,
) -> None:
    """
    写终态。错误信息**必须落库**：规格的 Run 只回 status，日志又会滚掉，
    否则一次失败的运行在库里只剩 `failed` 一个词。

    ⚠️ 终态**事件**也在这里发，而不是让两个调用点各自记得发
    ------------------------------------------------
    写终态与发终态是同一件事的两面。分开写，就必然会有一条路径漏发，
    而漏发的表现是"订阅端永远等不到结束"（SSE 靠终止事件关流）——
    那是最难查的一类问题：没有报错，只是永远转圈。
    """

    error_type = None if error is None else type(error).__name__

    error_message = (
        None
        if error is None
        else f"{type(error).__name__}: {error}"
    )

    async with repository_factory.scope() as repos:
        await repos.create(AgentRunRepository).finish(
            run_id,
            status.value,
            error_type=error_type,
            error_message=error_message,
        )

    await _publish_terminal(
        run_id,
        status,
        error_type=error_type,
        error_message=error_message,
        trace_id=trace_id,
        session_id=session_id,
        started_at_ms=started_at_ms,
    )


async def _publish_terminal(
        run_id: UUID,
        status: RunStatus,
        *,
        error_type: str | None,
        error_message: str | None,
        trace_id: str | None = None,
        session_id: UUID | None = None,
        started_at_ms: int | None = None,
) -> None:
    """
    终态事件。

    ⚠️ 契约里**没有** run.cancelled / run.waiting_input，因此这里有映射，
    代价要说清楚：

        completed       -> run.completed
        failed          -> run.failed
        cancelled       -> run.failed（data 里 status=cancelled 表明真相）
        waiting_input   -> **不发** —— 图停下来等人，这次运行并没有结束

    最后一条值得单独说：waiting_input 时流上不会有终止事件，订阅端因此
    会继续挂着。那**是对的**（运行确实没完，后续还有 hitl.* 与恢复后的
    事件）。它今天也产生不出来：react 图里没有 interrupt 点，见
    `_terminal_status` 与 domain/schemas/enum.py 的标注。

    提前退出的路径（图还没跑就被拒）同样走这里，因此会看到一条没有
    run.started 的 run.failed —— 那是事实：它确实没开始过。
    """

    payload: dict[str, Any] = {"status": status.value}

    if error_type is not None:
        payload["error_type"] = error_type
        payload["error_message"] = error_message

    if status is RunStatus.COMPLETED:
        event_type = RunEventType.RUN_COMPLETED

    elif status in (RunStatus.FAILED, RunStatus.CANCELLED):
        event_type = RunEventType.RUN_FAILED

    else:
        # waiting_input（以及理论上不该出现的 queued / running）：
        # 这次运行没有结束，不发终止事件。
        return

    await publish_run_event(event_type, str(run_id), data=payload)

    # ---- Trace 的 run 终止事件 ----
    #
    # 与 run 事件的分叉点：契约里 cancelled 折叠成 run.failed，而 trace
    # 词表有独立的 run.cancelled；waiting_input 两边都不发（这次运行没完）。
    duration_ms = (
        None
        if started_at_ms is None
        else max(0, int(time.time() * 1000) - started_at_ms)
    )

    trace_error = (
        TraceError(
            type=error_type or "Error",
            message=error_message or "",
        )
        if status is RunStatus.FAILED
        else None
    )

    if status is RunStatus.COMPLETED:
        trace_type = TraceEventType.RUN_COMPLETED
        trace_status = SpanStatus.COMPLETED
        trace_payload: dict[str, Any] = {}

    elif status is RunStatus.CANCELLED:
        trace_type = TraceEventType.RUN_CANCELLED
        trace_status = SpanStatus.CANCELLED
        trace_payload = {"reason": error_message or "cancelled"}

    else:
        trace_type = TraceEventType.RUN_FAILED
        trace_status = SpanStatus.FAILED
        trace_payload = {}

    await publish_trace_event(
        trace_type,
        trace_id=trace_id,
        run_id=str(run_id),
        thread_id=None if session_id is None else str(session_id),
        span_id=str(run_id),
        span_kind=SpanKind.RUN,
        span_name="run",
        status=trace_status,
        duration_ms=duration_ms,
        payload=trace_payload,
        error=trace_error,
    )
