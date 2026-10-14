"""
Worker —— 通用执行流程。它**只知道 Task**（spec §12）。

    load Task -> 校验状态 -> claim -> resolve handler -> 构造 service
      -> 校验 params -> invoke -> 原子写回结果 -> ack / retry

这里没有、也**不允许**有 HITL / Run / Memory / Tool 的任何分支。判断
"这是不是 HITL 的任务"的唯一方式是通过 registry 解析出来的
HandlerDefinition —— Worker 连 handler 名字都不需要理解。

⚠️ 事务边界：执行期间**不持有**事务
---------------------------------
Worker 分三段，每段各自开一个短事务：

    1. claim            一个短事务，提交
    2. 执行 handler      由 ServiceFactory.scope 自己开一个事务
    3. 写回结果          一个短事务，提交

第 2 段是业务 Handler 的运行期：它可能跑几秒、几十秒，甚至去调一个
慢上游。如果把三段包在一个事务里，那个事务会一直开着 —— 它持有的行锁
会挡住 Sweeper 的租约回收、挡住别的实例、也会让连接池被长事务占满。
runner 那条链路（agent_run）踩过同一个坑，做法一致：执行期间不持有事务。

⚠️ 租约到期而 Handler 还在跑 = 重复执行
------------------------------------
本实现**不续约**。Handler 跑得比 lease_seconds 久的话，Sweeper 会认为
执行者已死、把任务收回，于是它可能被第二个 Worker 同时再跑一遍。

这不是疏忽，是取舍：续约要引入一个心跳协程、一套续约失败的判定，以及
"续约失败但 Handler 其实快跑完了"的边界处理 —— 而 spec §16 明确第一版
不要这些。代价被两件事兜住了：

    1. lease_seconds 默认给得足够大（15 分钟），
       并且可以在组合根按最慢的 Handler 调大；
    2. **至少一次执行**本来就是本基础设施的契约（spec §11）——
       业务 Handler 涉及外部副作用时必须自己幂等，而"重复执行"是
       它本来就要承受的，不是租约带来的新问题。
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from uuid import UUID

from app.common.factories import get_service_factory
from app.common.repository.factory import RepositoryFactory
from app.common.service.factory import ServiceFactory
from app.common.task.exceptions import (
    TaskParamsError,
    UnknownHandlerError,
)
from app.common.task.models import TaskContext
from app.common.task.protocols import TaskHandlerRegistry
from app.common.task.types import TaskStatus
from app.services.task.repository import TaskStore

from .retry import RetryPolicy
from .submitter import ScopedTaskSubmitter

__all__ = ["TaskRunOutcome", "Worker"]

logger = logging.getLogger(__name__)

#: 落库的 error 文本上限。
#:
#: 上游异常偶尔会带上整个响应体（几千行 HTML）。列是 Text，存得下，
#: 但查询接口会把它吐给前端、日志也会带上它。截断到够定位问题的长度，
#: 并明确标出被截断 —— 悄悄截断会让人以为那就是全文。
_ERROR_MAX = 2000

#: 默认租期（秒）。见模块 docstring 的说明：必须大于最慢 Handler 的
#: 预期运行时长，否则它会被当成"执行者已死"而被重复执行。
DEFAULT_LEASE_SECONDS = 900.0

#: 配置类错误：日志不打堆栈（消息本身就是完整诊断）。
_CONFIG_ERRORS: tuple[type[BaseException], ...] = (
    UnknownHandlerError,
    TaskParamsError,
)


def _default_identity() -> str:
    """Worker 标识，写进 tasks.leased_by。

    主机名 + PID：多实例部署时靠它区分"这条卡住的任务当时是谁在跑"。
    同一进程里起多个 Worker 时不区分 —— 那种情况下它们本来就是同一个
    执行者（同一个进程崩了就一起崩），区分没有意义。
    """

    try:
        host = socket.gethostname()
    except OSError:  # pragma: no cover - 极少数环境取不到主机名
        host = "unknown-host"

    return f"{host}:{os.getpid()}"


def _describe(
        exc: BaseException,
) -> str:
    """异常 -> 落库的一句话。

    带上类型名：光有消息的话，"TimeoutError" 与 "ValueError" 都可能是
    "请求超时了"这类文本，排查时分不出来到底是哪一层出的问题。
    """

    text = f"{type(exc).__name__}: {exc}"

    if len(text) > _ERROR_MAX:
        text = text[:_ERROR_MAX] + "…（已截断）"

    return text


@dataclass(frozen=True, slots=True)
class TaskRunOutcome:
    """
    一次 run() 的结果。**不是**给业务看的东西 —— 它是给 Dispatcher
    与冒烟用的（"这次到底跑了没有、跑成什么样"）。

    ⚠️ 特意不叫 TaskResult：Task 上没有"执行结果"这一列，本对象也不
    落库。Handler 的产出应该通过它自己的业务表落地（写 agent_message、
    写 pipeline_run…），而不是塞回任务表 —— 那会让任务表变成一张
    什么都装的表。

    ran 与 status 分开
    ----------------
    "没抢到"（重复投递）与"跑失败了"是两件完全不同的事，用一个 status
    表达会把前者写成 FAILED —— 于是下游看到"任务失败了"，而真相是
    "另一个 Worker 正在跑它，而且跑得挺好"。这类假失败会直接污染告警。

        没抢到      ran=False, status=None  （状态不归这次调用管）
        跑了，失败   ran=True,  status=FAILED
        跑了，成功   ran=True,  status=SUCCESS
    """

    task_id: UUID

    #: 这次调用**是否真的执行了** Handler。
    ran: bool

    #: 执行之后任务的状态。没执行（未抢到）时为 None。
    status: TaskStatus | None

    #: 人类可读的一句话：跑成了 / 为什么没跑 / 为什么失败。
    detail: str


class Worker:
    """
    单条任务的执行者。**每次只处理一条**，没有循环、没有队列 ——
    循环与并发在 Dispatcher 里（见 dispatcher.py）。

    这样切开是为了让"执行一条任务"可以被独立调用与断言：冒烟里直接
    `await worker.run(task_id)` 就能验完整条 Handler 解析链，不必启动
    任何后台协程。
    """

    def __init__(
            self,
            *,
            repositories: RepositoryFactory,
            registry: TaskHandlerRegistry,
            services: ServiceFactory | None = None,
            policy: RetryPolicy | None = None,
            lease_seconds: float = DEFAULT_LEASE_SECONDS,
            identity: str | None = None,
    ) -> None:

        self._repositories = repositories
        self._registry = registry
        self._services = services
        self._policy = policy if policy is not None else RetryPolicy()
        self._lease_seconds = lease_seconds
        self._identity = (
            identity if identity is not None else _default_identity()
        )

    @property
    def identity(self) -> str:
        """写进 tasks.leased_by 的标识。"""

        return self._identity

    @property
    def services(self) -> ServiceFactory:
        """
        服务工厂。没显式注入时取**应用级**那一个。

        ⚠️ 这个"延迟解析"不是省事，是为了解开组合顺序上的死结
        --------------------------------------------------
        业务服务必须由 ServiceFactory 构造，而 ServiceFactory 必须等
        dispatcher / 注册表等单例都就绪**之后**才能 build（它会把池子里的
        实例抓走）。可本地的投递器又需要 Worker，Worker 又需要工厂 ——
        谁先谁后就绕回去了。

        让 Worker 在**执行时**才去取工厂就断了这个环：组合根可以先建
        Worker 与投递器，把它们放进池子，再 build_factories。任务只可能
        productive 地存在于此之后（提交本身要经过 TaskService，而它来自
        那个工厂），所以执行时它一定已经就绪。

        取应用级单例而不是现场造一个，与 app/services/dependencies.py 的
        service_dependency 是同一条理由：后台路径没有 Request 可以依附，
        而 get_service_factory() 本来就为这种场景提供（见
        app/common/factories.py）。它取不到时会明确报错，而不是造一个
        缺依赖的工厂。
        """

        if self._services is not None:
            return self._services

        return get_service_factory()

    # ========================================================
    # 执行一条
    # ========================================================

    async def run(
            self,
            task_id: UUID,
    ) -> TaskRunOutcome:

        # -----------------------------------------------------
        # 1. claim
        # -----------------------------------------------------
        #
        # 抢不到就直接返回，**这不是错误**：至少一次投递意味着同一条
        # task_id 会被投多次（post-commit 钩子 + outbox relay + 兜底扫描），
        # 而每条任务只该跑一次 —— 抢不到正是"别人正在跑/已经跑过"。
        async with self._repositories.scope() as repos:

            claimed = await repos.create(TaskStore).claim(
                task_id,
                leased_by=self._identity,
                lease_seconds=self._lease_seconds,
            )

        if claimed is None:

            return TaskRunOutcome(
                task_id=task_id,
                ran=False,
                # 刻意不填状态：没抢到意味着这条任务不归这次调用管，
                # 它的真实状态在库里 —— 猜一个填上去只会制造假信息。
                status=None,
                detail="没抢到（重复投递，或状态已不是可执行）——已忽略",
            )

        # -----------------------------------------------------
        # 2. 执行
        # -----------------------------------------------------
        try:

            definition = self._registry.resolve(claimed.handler)

            params = definition.parse_params(claimed)

            # Handler 拿到的 ctx 里有"正在跑的那条任务"与一个 submit 口子，
            # 于是它可以派生下一步（spec §8）：
            #
            #     hitl.timeout -> run.resume
            #
            # TaskContext 在这里现造：它是**这一次执行**的上下文，
            # 不是可复用的对象，也不该被 Handler 存起来。
            ctx = TaskContext(
                task=claimed,
                task_service=ScopedTaskSubmitter(
                    services=self.services,
                ),
            )

            await self._invoke(definition, params, ctx)

        except asyncio.CancelledError:

            # ⚠️ 取消**不是失败**，绝不能写 failed。
            #
            # 状态其实已经由 TaskService.cancel 置成 cancelled 了；这里
            # 什么都不写，只把取消继续往上抛 —— 吞掉 CancelledError 会让
            # 调用方（Dispatcher 的停止流程）以为它正常结束了。
            #
            # 也不用担心下面的失败分支会误写：mark_failure 带
            # `status == running` 守卫，而此刻状态已是 cancelled，
            # 它本来就写不进去。
            logger.info(
                "[Task] %s 执行被取消（handler=%s）",
                task_id,
                claimed.handler,
            )

            raise

        except BaseException as exc:  # noqa: BLE001 - 见下

            # 捕获 BaseException（不是 Exception）：SystemExit、
            # KeyboardInterrupt 这类也要留下痕迹再放行。否则一条让进程
            # 退出的任务在库里什么都没写，看起来像"任务凭空消失"。
            await self._mark_failed(claimed, exc)

            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise

            return TaskRunOutcome(
                task_id=task_id,
                ran=True,
                # 这次确实是失败了 —— 但注意可能是"失败且已排入重试"，
                # 那种情况下库里的状态是 retryable 而不是 failed。
                # 想拿准确状态应当回查（冒烟就是这么做的）。
                status=TaskStatus.FAILED,
                detail=_describe(exc),
            )

        # -----------------------------------------------------
        # 3. 写回成功
        # -----------------------------------------------------
        async with self._repositories.scope() as repos:

            updated = await repos.create(TaskStore).mark_success(task_id)

        if updated is None:

            # 执行成功了，但状态写不进去 —— 只可能是这几秒里被取消了
            # （mark_success 的守卫），或者被 Sweeper 当成租约过期收走了。
            # 如实说出来，别假装成功。
            logger.warning(
                "[Task] %s 执行成功但状态写回无效"
                "（多半是期间被取消 / 租约被收回）",
                task_id,
            )

            return TaskRunOutcome(
                task_id=task_id,
                ran=True,
                status=TaskStatus.CANCELLED,
                detail="执行完成，但状态写回时已不是 running（被取消或租约被收回）",
            )

        return TaskRunOutcome(
            task_id=task_id,
            ran=True,
            status=TaskStatus.SUCCESS,
            detail="执行成功",
        )

    # ========================================================
    # internals
    # ========================================================

    async def _invoke(
            self,
            definition,
            params,
            ctx,
    ) -> None:
        """
        构造业务 Service 并调用它的方法。

        这里与 app/services/dependencies.py::service_dependency 是同一套
        规则，只是面向"后台路径"而不是"请求路径"：

            有仓储依赖 -> ServiceFactory.scope（顺带开事务）
            没有仓储    -> ServiceFactory.create（不必白开一个事务）

        不直接 `service = definition.service_contract()`：服务实例必须由
        工厂装配（仓储、池子里的单例都靠它注入），手工构造会绕开那些
        依赖，症状是 Handler 一跑就报缺参数。
        """

        contract = definition.service_contract

        services = self.services

        if services.needs_session(contract):

            async with services.scope(contract) as service:

                await definition.invoke(service, params, ctx)

            return

        await definition.invoke(
            services.create(contract),
            params,
            ctx,
        )

    async def _mark_failed(
            self,
            claimed,
            exc: BaseException,
    ) -> None:
        """
        失败写回：决定重试还是判死，然后落库。

        判断只在这里做一次。TaskStore.mark_failure 收的是**结论**
        （retry_delay_seconds 是不是 None），不是 attempt / max_attempts ——
        那样"能不能重试"就只有一处判断，不会两边不一致。
        """

        retryable = self._policy.is_retryable(
            exc,
            attempt=claimed.attempt,
            max_attempts=claimed.max_attempts,
        )

        delay = (
            self._policy.delay_for(claimed.attempt)
            if retryable
            else None
        )

        if retryable:
            logger.warning(
                "[Task] %s 执行失败，将在 %.1fs 后重试"
                "（第 %d/%d 次）：%s",
                claimed.task_id,
                delay,
                claimed.attempt,
                claimed.max_attempts,
                _describe(exc),
            )
        else:
            # 配置类错误（handler 没注册、参数对不上）不打堆栈：它们的
            # 消息本身就是完整的诊断，堆栈只会把日志刷屏而没有任何新信息。
            # 其余情况打堆栈 —— 那才是唯一能定位"到底哪一行炸了"的东西。
            logger.error(
                "[Task] %s 执行失败且不再重试"
                "（第 %d/%d 次）：%s",
                claimed.task_id,
                claimed.attempt,
                claimed.max_attempts,
                _describe(exc),
                exc_info=not isinstance(
                    exc, _CONFIG_ERRORS
                ),
            )

        async with self._repositories.scope() as repos:

            await repos.create(TaskStore).mark_failure(
                claimed.task_id,
                error=_describe(exc),
                retry_delay_seconds=delay,
            )
