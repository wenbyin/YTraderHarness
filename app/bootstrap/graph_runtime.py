"""
图的执行侧：进程内图调度器 + 两个 handler 的注册 + 生命周期。

与 app/bootstrap/task.py 是同一个形状（一个带端口的运行时、启动时集中
注册 handler、自带启停），差别只在注册的是什么：那边是任务 Handler
（逻辑名 -> 服务方法），这边是图 handler（图名 -> 跑图的协程）。

    task.py    统一异步任务基础设施
    graph_runtime.py  图（文档管线 / Agent 对话）的执行

它**不拥有**图本身 —— 图由 app/bootstrap/pipelines.py 装配好，这里只
负责"谁来跑它们"。所以入参是那个 Pipelines 引用束。

⚠️ 停摆顺序里的位置由组合根决定
----------------------------
本模块只提供"关自己"的那一步。它与事件流、数据源之间的先后约束跨模块，
因此在 main.py 的 finally 里显式排：图调度器要排在事件发布器**之前**停
（还在收尾的图任务要把 run.completed 这类终止事件发出去，先停发布器会
让订阅端永远等不到终止事件）。
"""

from __future__ import annotations

import logging
from typing import Any

from app.common.repository.factory import RepositoryFactory
from app.infra.graph.dispatcher_impl import (
    InProcessGraphDispatcher,
    build_graph_dispatcher,
    get_graph_dispatcher,
    reset_graph_dispatcher,
)
from app.services.document.dependencies import make_pipeline_state_loader
from app.services.document.pipeline.runtime import build_dispatch_handler
from app.services.session.application.agent_run_dispatch import (
    build_dispatch_handler as build_agent_run_dispatch_handler,
)

from .pipelines import Pipelines

__all__ = [
    "aclose_graph_runtime",
    "build_graph_runtime",
    "start_graph_runtime",
]

logger = logging.getLogger(__name__)

#: 本模块建出来的那一个调度器。
#:
#: 与 app/infra/graph 的进程级访问器（build_graph_dispatcher /
#: get_graph_dispatcher）**不是**同一件事：那个是给事件流探针在运行期
#: 反查用的，这个是本模块为了能关掉自己而留的引用。两者指向同一个对象，
#: 但生命周期归这里管 —— 探针那边拿不到也不该由它去关。
_dispatcher: InProcessGraphDispatcher | None = None


def build_graph_runtime(
        *,
        pipelines: Pipelines,
        session_factory: Any,
) -> InProcessGraphDispatcher:
    """
    建图调度器并注册两个 handler。**不启动**（见 start_graph_runtime）。

    两个 handler 的形状不同，是两条链路的观测方式不同导致的：

        document    走 GraphRunner + SqlAlchemyRunRecorder（写节点行）
        react       不用 GraphRunner（终态推导以节点行为唯一依据，
                    而这条链路不写节点行），因此薄得多：只翻 agent_run
                    的状态
    """

    global _dispatcher

    dispatcher = InProcessGraphDispatcher()

    dispatcher.register(
        pipelines.document_definition.name,
        build_dispatch_handler(
            definition=pipelines.document_definition,
            recorder=pipelines.recorder,
            resume_state_loader=make_pipeline_state_loader(
                session_factory
            ),
            settings=pipelines.settings,
        ),
    )

    # Agent 对话的 handler。
    #
    # repository_factory 以 session_factory 模式构造 —— handler 跑在后台
    # worker 里，没有请求级 session 可借（与 SqlAlchemyRunRecorder 同一条）。
    dispatcher.register(
        pipelines.react_definition.name,
        build_agent_run_dispatch_handler(
            definition=pipelines.react_definition,
            repository_factory=RepositoryFactory(
                session_factory=session_factory
            ),
            recorder=pipelines.node_recorder,
        ),
    )

    _dispatcher = dispatcher

    return dispatcher


async def start_graph_runtime() -> None:
    """
    启动调度器的消费循环，并登记进程级引用。

    登记那一步是给事件流探针用的（见
    app/services/session/application/run_liveness.py）：它要问
    "本进程手里还有这个 run 吗"。

    ⚠️ 走进程级访问器而不是把 dispatcher 传下去
    ----------------------------------------
    事件流的读取端在 lifespan 里排在**更早**的位置，那时 dispatcher 还
    不存在。想直接传就得把两段的顺序倒过来 —— 那只是把耦合换个方向，
    而且读取端本来也不该依赖调度器。所以在这里登记一次，探针被调用时
    （流空闲时）再取。
    """

    if _dispatcher is None:
        raise RuntimeError(
            "图调度器还没装配。请先调用 build_graph_runtime() —— "
            "它由 lifespan 在装配流水线之后调用。"
        )

    await _dispatcher.start()

    build_graph_dispatcher(_dispatcher)

    logger.info("[Graph] 图调度器已启动并登记进程级引用")


async def aclose_graph_runtime() -> None:
    """
    停掉调度器并清掉进程级引用。

    **没装配过时是无操作**：它会在 lifespan 的 finally 里被调用，而那里
    可能连装配都没走到。

    清掉引用是必须的 —— 别让它在下一次启动前还指着一个已经停掉的调度器
    （那种情况下 is_active 恒为 False，所有流都会被判成"执行器没了"）。
    """

    global _dispatcher

    dispatcher = _dispatcher if _dispatcher is not None else get_graph_dispatcher()

    if dispatcher is not None:
        try:
            await dispatcher.aclose()
        except Exception:
            logger.exception("Failed to stop graph dispatcher.")

    reset_graph_dispatcher()

    _dispatcher = None
