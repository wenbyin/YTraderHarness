"""
Task 核心依赖的那些**端口**：它只认这些协议，不认具体实现。

为什么在 app.common 而不是 infra/task
------------------------------------
依赖方向是：

    common/task  <-  services/task  <-  infra/task

Worker（infra）要按 TaskStore 契约读写、按 TaskHandlerRegistry 契约解析
Handler；而 TaskContext（common）要拿到一个能 submit 的东西。这些类型
若定义在下游，上游就都得反向 import。放这里，三个方向都只是**读**它。

因此本模块只有 Protocol 与不可变的定义结构，没有任何实现、没有 import
任何业务或 infra 模块 —— app.common 必须保持纯叶子。

⚠️ 为什么这里只有 TaskSubmitter，没有完整的 TaskService
---------------------------------------------------
spec §4 把 TaskService 写成一个 Protocol。落到本仓库它落成了
app/services/task/application/task_service.py 里的 **ABC**，原因是
FastAPI 那边用 `Service[TaskService]` 取服务，而它走
app/common/introspect.py::resolve_impl —— 靠 `__subclasses__()` 找实现。
Protocol 没有子类，resolve_impl 对它必然失败。所以完整契约只能是 ABC，
而 ABC 住在 services 下，common 里就不能引用它。

于是这里留下**窄的那一半**：Handler 派生下一步只需要 submit，看不到
cancel / retry / get。这既解开了循环依赖，也正好是接口隔离 ——
见 TaskContext 的说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel

from .exceptions import TaskParamsError

if TYPE_CHECKING:  # pragma: no cover - 仅为注解
    # ⚠️ 只在类型检查期导入。models.py 在**运行期** import 本模块
    # （TaskContext 要用 TaskSubmitter），所以本模块反过来只能在
    # TYPE_CHECKING 下 import 它 —— 否则就是一个模块级循环导入：
    # models 加载到一半就去加载 protocols，而 protocols 要的那个
    # Task 此时还不存在。
    #
    # 之所以能这么做：本模块开了 `from __future__ import annotations`，
    # 所有注解都是惰性字符串，运行期不求值。Protocol 的 runtime_checkable
    # 也只检查方法在不在，不看注解。
    from .models import Task, TaskContext

__all__ = [
    "HandlerDefinition",
    "TaskDispatcher",
    "TaskHandlerRegistry",
    "TaskSubmitter",
]


@runtime_checkable
class TaskSubmitter(Protocol):
    """
    "能把一条 Task 交出去"的最小接口。

    TaskContext 只依赖它。业务模块若只需要排一条后台任务，也可以只声明
    这个类型，从而完全不依赖 services/task 下的任何东西。
    """

    async def submit(
            self,
            task: Task,
    ) -> UUID:
        """
        提交一条 Task，返回它的 task_id。

        对同一个 idempotency_key 重复提交返回**同一个** task_id。
        """

        ...


@runtime_checkable
class TaskDispatcher(Protocol):
    """
    把一条**已经落库**的 Task 投递出去。

    存在这个端口的唯一目的：让业务与 Worker 都不感知底层是 DB 轮询还是
    MQ（spec §6）。第一版实现是 DB Polling（infra/task/dispatcher.py），
    将来换 MQ 只换实现，调用方一行不动。

    契约的核心是"**至少一次**投递，且投递物只有 task_id"：
    实现不得把 Task 参数、Graph State 塞进传输层 —— 那些以 DB 为准。
    """

    async def dispatch(
            self,
            task_id: UUID,
    ) -> None:
        """
        投递一条 Task。

        只保证"尽力让某个 Worker 尽快去处理它"，因此**允许重复投递**，
        也允许在进程重启时丢掉尚未处理的投递（这时靠 task_outbox 重投）。
        调用方不得依赖它一定成功。
        """

        ...

    async def start(self) -> None:
        """启动投递侧的常驻循环（消费线程 / 轮询协程）。"""

        ...

    async def aclose(self) -> None:
        """停掉常驻循环，等在途工作收尾。"""

        ...


@dataclass(frozen=True, slots=True)
class HandlerDefinition:
    """
    一个已注册的 Handler：把**逻辑名字**翻译成"调谁、传什么"。

    为什么 Task 里存的是 handler 字符串而不是 import path
    --------------------------------------------------
    "hitl.timeout" 是稳定契约：改类名、拆模块、换实现都不影响库里已有的
    行。反过来，存 "app.services.hitl.application.XService.timeout" 则把
    一次重构变成一次数据迁移，而且**允许任意字符串动态 import 本身就是
    一个洞** —— 谁能写 task.handler 谁就能导入任意模块（spec §3）。

    所以解析只能走一个显式登记过的注册表。

    与 spec 的差异：这里存的是 service_contract（服务**契约**）而不是
    service_factory。本仓库没有 per-service 的工厂类，服务实例一律由
    app/common/service/factory.py::ServiceFactory.scope(contract) 产出
    —— 顺带把事务边界也一并管了。Worker 因此持有 ServiceFactory，
    在调用前开一个短事务。见 infra/task/worker.py。
    """

    handler: str

    #: 参数类型。Worker 用它把库里的 params_json 校验成具体模型 ——
    #: 这是"参数写错在运行时立刻炸"而不是"Handler 收到一堆 None"的地方。
    params_type: type[BaseModel]

    #: 要调用的服务契约（不是实现）。ServiceFactory 靠它 resolve_impl。
    service_contract: type

    #: 服务上要调用的方法名。
    method: str

    def parse_params(
            self,
            task: Task,
    ) -> BaseModel:
        """
        把 Task 里那个"还没被解释过"的参数校验成 params_type。

        走 params_dict 而不是直接拿 task.params：从库里装载时后者是
        RawTaskParams，pydantic 不会把 A 类实例当成 B 类校验。

        ⚠️ 除了交给 pydantic，这里还要自己查一遍"有没有字段被丢掉"
        ------------------------------------------------------
        参数类若是普通 BaseModel，pydantic 默认 extra="ignore"：多出来的
        字段**既不报错、也不保留**，只是消失。于是"调用方传错了参数类"
        这种错误会表现为"Handler 收到一份缺参数的输入"，一路跑到业务
        逻辑深处才炸 —— 或者更糟，不炸，只是做错了事。

        model_validate 之后 model_dump() 的键集合就是**真正被接收**的
        字段；与原始 dict 一比，多出来的那些就是被静默丢掉的。

        Raises:
            pydantic.ValidationError: 参数类型/取值不符。
            TaskParamsError: 有字段被静默丢弃（多半是传错了参数类）。
                两者都是**致命**错误（提交方与注册方对不上），
                Worker 不重试。
        """

        raw = task.params_dict

        params = self.params_type.model_validate(raw)

        dropped = set(raw) - set(params.model_dump())

        if dropped:
            raise TaskParamsError(
                f"handler {self.handler} 的参数类型 "
                f"{self.params_type.__name__} 没有接收这些字段："
                f"{sorted(dropped)}。"
                "参数类通常没开 extra='forbid'，于是它们被静默丢弃、"
                "Handler 会拿到一份缺参数的输入。"
                "请确认提交的是该 handler 注册的那个参数类。"
            )

        return params

    async def invoke(
            self,
            service: Any,
            params: BaseModel,
            ctx: "TaskContext",
    ) -> None:
        """
        调用 service.<method>(params, ctx)。

        ⚠️ ctx 是**必需**的第二个参数，不是可选的
        --------------------------------------
        spec §8 要求 Handler 能派生下一步：

            await ctx.task_service.submit(next_task)

        把 ctx 做成"看签名决定传不传"会让 Handler 的契约变成隐式的 ——
        作者写 `async def timeout(self, params)` 与写
        `async def timeout(self, params, ctx)` 都能跑，但前者一旦想派生
        子任务就撞 AttributeError/TypeError，而那个错误会落进"Handler
        执行失败"，被当成可重试的业务错误重试几次。

        统一要求两个参数之后，注册期就能校验签名（见
        infra/task/registry.py），写错在**启动时**报错。不需要 ctx 的
        Handler 照常声明它、不用即可。

        方法不存在时抛 AttributeError 就够了吗？不够 —— 那会混进
        "Handler 执行失败"里被当成可重试的业务错误。但这里也**不**导入
        UnknownHandlerError：注册表在 register() 时就校验过方法存在
        （见 infra/task/registry.py），所以走到这里还缺方法只能是有人在
        运行期往实例上动过手脚，属于真异常。
        """

        await getattr(service, self.method)(params, ctx)


@runtime_checkable
class TaskHandlerRegistry(Protocol):
    """
    handler 逻辑名 -> HandlerDefinition 的解析表。

    ⚠️ 必须是**启动时集中注册**的显式表：不允许按名字动态 import。
    见 infra/task/registry.py 的实现与理由。
    """

    def resolve(
            self,
            handler: str,
    ) -> HandlerDefinition:
        """
        Raises:
            UnknownHandlerError: 这个名字没注册过。错误里会带上已知的
                全部名字，因为"handler 拼错了"是这条路径上最常见的原因。
        """

        ...
