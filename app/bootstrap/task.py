"""
统一异步任务基础设施的装配、Handler 注册与生命周期。

lifespan 里只剩三行（建、进池子、启动）：

    task_infrastructure = build_task_infrastructure(session_factory)
    build_factories(session_factory, pool={..., **task_infrastructure.pool_entries})
    await start_task_infrastructure()

⚠️ 为什么建与启动分成两步（而不是 build 里顺手 start）
----------------------------------------------------
Worker 执行任务时才去取**应用级服务工厂**（见
app/infra/task/worker.py::Worker.services，那样才解开了"工厂要先于
dispatcher、dispatcher 又要先于工厂"的死结）。而应用级工厂由
build_factories 建立，在本模块之后。若装配时就 start()，一条**积压任务**
（上次进程崩前留下的，扫描循环第一轮就会捞它）会在工厂还不存在时被
执行，拿到的是一个明确的 RuntimeError。

所以：build 放在 build_factories **之前**（池子要收它的两个单例），
start 放在 build_factories **之后**。这个顺序是硬约束，不是风格。

本模块是业务 Handler 的**唯一注册点**
------------------------------------
见 _register_handlers()。放这里而不是 app/infra/task 下的原因见
app/bootstrap/__init__.py：infra 不允许 import 业务模块。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.common.repository.base import SessionFactory
from app.common.repository.factory import RepositoryFactory
from app.common.task.protocols import TaskDispatcher
from app.infra.task import (
    DEFAULT_LEASE_SECONDS,
    DbPollingTaskDispatcher,
    OutboxRelay,
    RetryPolicy,
    TaskHandlerRegistryImpl,
    Worker,
)

__all__ = [
    "TaskInfrastructure",
    "aclose_task_infrastructure",
    "build_task_infrastructure",
    "get_task_dispatcher",
    "get_task_handler_registry",
    "get_task_infrastructure",
    "get_task_outbox_relay",
    "reset_task_infrastructure",
    "start_task_infrastructure",
]

logger = logging.getLogger(__name__)

#: 装配后的任务运行时。lifespan 写一次，之后只读。
#:
#: 用模块级单例而不是 app.state：Worker / Dispatcher 在**后台路径**上被
#: 调用（没有 Request 可依附），而 app/services/dependencies.py 也已经
#: 为同一类原因走 get_service_factory() 的进程级访问器。这里保持一致。
_infrastructure: "TaskInfrastructure | None" = None


@dataclass(frozen=True, slots=True)
class TaskInfrastructure:
    """
    一整套路就绪的任务运行时。

    frozen：装配完之后这些引用不该被换掉（换掉等于悄悄跑两个 dispatcher，
    而旧的还在轮询、还在写库）。
    """

    registry: TaskHandlerRegistryImpl
    worker: Worker
    dispatcher: DbPollingTaskDispatcher
    relay: OutboxRelay
    repositories: RepositoryFactory

    @property
    def pool_entries(self) -> dict[str, Any]:
        """
        要放进 ServiceFactory 池子的键值。

        两个键都是有讲究的：

        handler_registry
            TaskService 提交前按它校验 handler 与参数类型，
            让"拼错了"在 submit 那一刻报错，而不是等 Worker 读库时才发现。

        task_dispatcher
            TaskService 提交后登记一个**提交后**投递钩子，省掉一个轮询
            周期的延迟。⚠️ 键名不是 "dispatcher" —— 那个名字已经被
            InProcessGraphDispatcher 占着，而工厂按**参数名**优先匹配
            （见 TaskServiceImpl 的构造参数说明）。
        """

        return {
            "handler_registry": self.registry,
            "task_dispatcher": self.dispatcher,
        }

    async def start(self) -> None:
        """
        拉起两条常驻循环：投递器的消费者/扫描，与 outbox relay。

        两者都是"先跑一轮再睡"，所以上一次进程崩时留下的到期任务与未
        投递记录在这一刻就开始被捞回来 —— 这就是"服务重启恢复"。
        """

        await self.dispatcher.start()
        await self.relay.start()

    async def aclose(self) -> None:
        """
        停掉两条循环。

        顺序：relay 先、dispatcher 后 —— relay 只负责投递，先停它就不会
        在 dispatcher 已经停掉之后还往它的队列里塞东西。

        没有写回中的任务会因此丢状态：Worker 的每次写回都在自己的短事务里，
        而 aclose() 会等消费者协程走完（见 DbPollingTaskDispatcher.aclose）。
        """

        await self.relay.aclose()
        await self.dispatcher.aclose()


def _register_handlers(
        registry: TaskHandlerRegistryImpl,
) -> None:
    """
    ★ 所有业务 Handler 的**集中注册点**（spec §3）。

    显式登记，不做动态 import —— 理由见 app/infra/task/registry.py
    （Task.handler 是持久化数据，不该成为一条代码执行的入口）。

    注册的形状：

        registry.register(
            handler="hitl.timeout",           # 逻辑名，稳定契约
            service_contract=HitlService,     # 契约，不是 Impl
            method="timeout",
            params_type=HitlTimeoutParams,
        )

    注册期会校验四件事并在启动时报错：handler 非空、params_type 是
    Pydantic Model 子类、method 在**契约**上存在、方法签名装得下
    (params, ctx)。写错不会拖到第一条任务跑起来才炸。

    ⚠️ handler 名是**数据**：库里已有的行按字面存着它。改名 = 让那些行
    解析失败（判 FAILED）。要改就新旧名字都注册一段时间。

    注册期会校验四件事并在启动时报错：handler 非空、params_type 是
    Pydantic Model 子类、method 在**契约**上存在、方法签名装得下
    (params, ctx)。写错不会拖到第一条任务跑起来才炸。

    ⚠️ 这里在 import 期就把 memory 的应用层拉进来
    ----------------------------------------
    注册要的是**契约**（MemoryService）与**参数模型**，两者都在
    app/services/memory 下。这不是"bootstrap 依赖业务"的例外 ——
    本模块的定位就是业务 Handler 的唯一注册点（见模块 docstring），
    所以它必然 import 业务契约。
    """

    from app.services.memory.application import (
        MemoryAtomService,
        MemoryPersonaService,
        MemorySceneService,
        MemoryService,
    )
    from app.services.memory.domain.tasks import (
        MEMORY_EXTRACT_HANDLER,
        MEMORY_ATOM_EXTRACT_HANDLER,
        MEMORY_SCENE_EXTRACT_HANDLER,
        MEMORY_PERSONA_EXTRACT_HANDLER,
        MemoryExtractParams,
        MemoryAtomExtractParams,
        MemorySceneExtractParams,
        MemoryPersonaExtractParams,
    )

    # L0 原文记忆的索引：一次运行收尾时提交（见
    # app/services/memory/application/trigger.py 与 agent_run_dispatch
    # 的收尾段）。它是本仓库接入的**第一条**真 Handler。
    registry.register(
        handler=MEMORY_EXTRACT_HANDLER,
        service_contract=MemoryService,
        method="message_extract",
        params_type=MemoryExtractParams,
    )

    # L1 原子记忆的抽取：**同样**在收尾时提交，但触发条件不是"每轮"
    # 而是"攒够 N 轮 / 空闲 60s"（见 trigger.submit_atom_extract）。
    #
    # ⚠️ 它是本仓库**第一条会调云端模型**的后台任务 —— 因此 lease_seconds
    # 必须大于"一次抽取最坏耗时"（默认 15 分钟，够；但换更长的模型或
    # 更大的批量时要重新看，见 build_task_infrastructure 的参数说明）。
    #
    # ⚠️ 抽取与去重是**同一个 handler**（去重接在 extract 末尾），所以这里
    # 只注册一条。理由见 memory_atom_service_impl 的模块 docstring。
    registry.register(
        handler=MEMORY_ATOM_EXTRACT_HANDLER,
        service_contract=MemoryAtomService,
        method="memory_atom_extract",
        params_type=MemoryAtomExtractParams,
    )

    # L2 场景整合：**不是**在运行收尾时提交的 —— 它排在 L1 那条任务的
    # 末尾（"确实抽出了东西"之后），而且要再延迟一会儿（见 trigger.py 的
    # `submit_scene_extract` 与那两个延迟常量）。
    #
    # 同样是云端模型调用，所以 lease_seconds 那条约束与 L1 一样适用：
    # 它要读该 scope 的**全部场景全文**再写回，比 L1 更慢一点。
    registry.register(
        handler=MEMORY_SCENE_EXTRACT_HANDLER,
        service_contract=MemorySceneService,
        method="memory_scene_extract",
        params_type=MemorySceneExtractParams,
    )

    # L3 画像生成：**同样不是**在运行收尾时提交的 —— 它挂在 L2 handler 的
    # 末尾，L2 一完成就排（`submit_persona_extract` **不判定**）。多数排出来
    # 的任务会在执行侧空转跳过（没有变化场景、也没举手、也没满轮次），
    # 那次跳过很便宜：一条 SELECT 走人，不调模型。
    #
    # 同样是云端模型调用（要读该 scope 的**全部变化场景 + 整篇画像**），
    # 所以 lease_seconds 那条约束与 L1/L2 一样适用。输入的规模与 L2 同量级，
    # 不需要单独放大。
    registry.register(
        handler=MEMORY_PERSONA_EXTRACT_HANDLER,
        service_contract=MemoryPersonaService,
        method="memory_persona_extract",
        params_type=MemoryPersonaExtractParams,
    )

    logger.debug(
        "[Task] 已注册 %d 个 Handler：%s",
        len(registry),
        ", ".join(registry.known()) or "（一个都没有）",
    )


def build_task_infrastructure(
        session_factory: SessionFactory,
        *,
        policy: RetryPolicy | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        concurrency: int = 4,
        sweep_interval: float = 15.0,
        relay_interval: float = 10.0,
) -> TaskInfrastructure:
    """
    装配任务运行时。**不启动**（见模块 docstring 的顺序说明）。

    参数都是可调开关，默认值按"正确优先于激进"取：

        lease_seconds    默认 15 分钟。必须大于最慢 Handler 的预期运行
                         时长，否则它会被当成"执行者已死"而重复执行。
                         按最慢的那个 Handler 调，不是按平均。
        concurrency      进程内并发执行的任务数。
        sweep_interval   兜底扫描间隔：延迟任务到点、以及"该跑却没跑"的
                         任务，最坏等这么久。
        relay_interval   未投递记录的补投间隔。

    Raises:
        RuntimeError: 已经装配过。重复装配会造出**第二套** dispatcher ——
            旧的还在轮询、还在写库，两个实例抢同一批任务（claim 是 CAS，
            所以不会重复执行，但一半的投递会白白落空）。宁可在这里炸，
            也不要那种"看起来正常"的重复。
    """

    global _infrastructure

    if _infrastructure is not None:
        raise RuntimeError(
            "任务基础设施已经装配过。要重新装配请先调用 "
            "reset_task_infrastructure()（它会丢掉引用，但**不会**停掉"
            "已经在跑的循环 —— 先 aclose_task_infrastructure()）。"
        )

    repositories = RepositoryFactory(session_factory=session_factory)

    registry = TaskHandlerRegistryImpl()
    _register_handlers(registry)

    worker = Worker(
        repositories=repositories,
        registry=registry,
        policy=policy,
        lease_seconds=lease_seconds,
    )

    dispatcher = DbPollingTaskDispatcher(
        worker=worker,
        repositories=repositories,
        concurrency=concurrency,
        sweep_interval=sweep_interval,
    )

    relay = OutboxRelay(
        dispatcher=dispatcher,
        repositories=repositories,
        interval=relay_interval,
    )

    _infrastructure = TaskInfrastructure(
        registry=registry,
        worker=worker,
        dispatcher=dispatcher,
        relay=relay,
        repositories=repositories,
    )

    logger.info(
        "[Task] 已装配：并发 %d，扫描 %.1fs，补投 %.1fs，租约 %.0fs",
        concurrency,
        sweep_interval,
        relay_interval,
        lease_seconds,
    )

    return _infrastructure


def get_task_infrastructure() -> TaskInfrastructure:
    """
    取整套任务运行时。

    没有时**明确报错**而不是现场装一套 —— 现场装出来的会缺 Handler 注册
    与池子条目，失败点会跑到某个任务被执行时。真正的原因只有一个：
    应用没跑过 lifespan。
    """

    if _infrastructure is None:
        raise RuntimeError(
            "任务基础设施还没装配。它由 lifespan 调用 "
            "build_task_infrastructure() 建立 —— "
            "请确认应用正常启动，或在测试里显式装配一次。"
        )

    return _infrastructure


def get_task_handler_registry() -> TaskHandlerRegistryImpl:
    """取 Handler 注册表（只读自省用：known() / resolve()）。"""

    return get_task_infrastructure().registry


def get_task_dispatcher() -> TaskDispatcher:
    """
    取任务投递器。

    业务代码**不应该**用它 —— 那是绕过 TaskService 直接投递，等于把
    业务钉在投递方式上（spec §4 明令禁止依赖 MQ / Worker）。
    它暴露出来是给运维端点、探针与测试用的。
    """

    return get_task_infrastructure().dispatcher


def get_task_outbox_relay() -> OutboxRelay:
    """取投递保证的 relay（观测未投递积压用）。"""

    return get_task_infrastructure().relay


async def start_task_infrastructure() -> None:
    """
    拉起任务执行侧。必须在 build_factories() **之后**调用 —— 见模块
    docstring。在它之前调用，积压任务会撞上一个还不存在的服务工厂。
    """

    await get_task_infrastructure().start()


async def aclose_task_infrastructure() -> None:
    """
    停掉任务运行时。

    **没装配过时是无操作**（不是错误）：它会在 lifespan 的 finally 里被
    调用，而那里可能连装配都没走到（比如第 1 步数据源就失败了）。
    那正是最需要安静收场的时候。
    """

    if _infrastructure is None:
        return

    await _infrastructure.aclose()


def reset_task_infrastructure() -> None:
    """
    丢掉装配引用。

    ⚠️ 只丢引用，**不停循环**。要停请先 aclose_task_infrastructure()。
    之所以不在这里顺手停：本函数是同步的（给测试的 fixture 与 reload 用），
    而停循环必须 await。留一个同步的 `reset` 却暗地里需要 await，只会让
    调用方以为它安全。
    """

    global _infrastructure

    _infrastructure = None
