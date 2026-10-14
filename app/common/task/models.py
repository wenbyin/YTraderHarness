"""
Task 本身，以及 Handler 执行时拿到的 TaskContext。

这是整套 Task Infrastructure 的**中心数据类型**：Dispatcher 靠它投递、
Worker 靠它执行、查询接口靠它回答"这条任务怎么样了"。因此它必须
可序列化、可持久化、可反序列化，并且**小**。

为什么不携带大对象
----------------
Task 里只放业务 ID 与执行所需的最小参数，大状态（Agent State、
Message History、Tool Result）在执行阶段按 ID 从 DB / Store 重新加载。

理由是 Task 会被反复读出：Dispatcher 每轮轮询要读、Worker claim 要读、
查询接口要读、重试要读。往里面塞一份 graph_state，等于把最热的那些
查询全部变重，而且**一旦图状态变了，历史 Task 行里的那份就是过期的**
—— 两处各有一份状态，迟早对不上。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
)

from .protocols import TaskSubmitter
from .types import TaskStatus

__all__ = [
    "RawTaskParams",
    "Task",
    "TaskContext",
]


def _utcnow() -> datetime:
    """
    统一用带时区的 UTC 时间。

    库里是 TIMESTAMPTZ，裸 datetime 会被当成本地时间写进去 —— 而
    execute_at 是"到点了没有"的判据，一次本地/UTC 混淆就足以让延迟
    任务提前或推迟若干小时执行。
    """

    return datetime.now(timezone.utc)


class RawTaskParams(BaseModel):
    """
    "还没被解释过"的任务参数：原始 dict 的**保真**载体。

    为什么需要它
    -----------
    Task.params 的类型在**读出**时是不知道的：库里只有一列 params_json，
    该把它校验成哪个 Pydantic 类，取决于 task.handler 指向哪个
    HandlerDefinition —— 而那要等 Registry 解析完才知道。

    所以读出路径分两步：

        TaskStore    把 params_json 装进 RawTaskParams（本类）
        Worker       definition.params_type.model_validate(task.params_dict)

    extra="allow" 是这里的关键：它让 pydantic 把未声明的键原样收进
    __pydantic_extra__，model_dump() 再原样还原。少了它，参数会在
    "装载"这一步被静默丢掉，而 Worker 拿到的是一个空壳 —— 那种失败
    在冒烟里表现为"Handler 收到了空参数"，离原因很远。

    业务代码**不应该**用它构造 Task：提交时必须给出具体的 params 类，
    这样参数名/类型写错在校验期就报错，而不是等 Worker 跑起来才发现。
    """

    model_config = ConfigDict(extra="allow")


class Task(BaseModel):
    """
    一次"未来需要执行的动作"。

    最小可用构造（其余字段都有默认值）：

        Task(
            handler="hitl.timeout",
            params=HitlTimeoutParams(hitl_id=hitl_id),
            execute_at=expire_at,
        )

    params 必须是 Pydantic Model
    ---------------------------
    不接受裸 dict（见 _reject_bare_dict）。dict 作为业务参数类型意味着
    "参数校验推迟到运行时"，而失败要等到 Worker 跑起来、事务已经开过、
    甚至外部副作用已经发生之后才暴露。用 Model 则提交的那一刻就报错。

    Lineage
    -------
        parent_task_id   直接父任务
        root_task_id     整条链的起点（自己派生的第一条）
        causation_id     由**什么**触发 —— 通常是父任务 ID，也可以是
                         一个业务事件 ID（那样就能回答"这次 HITL 最终
                         触发了哪些任务"）

    单父，不做 DAG：spec 明确第一版只要 parent -> child，不要 fan-in。
    想表达"B、C 都因为 A 才跑"用 causation_id，不需要多父。
    """

    model_config = ConfigDict(from_attributes=True)

    task_id: UUID = Field(
        default_factory=uuid4,
        description="任务的全局唯一 ID。稳定，且全链路复用同一个值。",
    )

    handler: str = Field(
        description=(
            "稳定的逻辑 Handler Name，如 hitl.timeout。"
            "不是 import path，也不是类名 —— 见 infra/task/registry.py。"
        ),
    )

    # ⚠️ SerializeAsAny 不是装饰性的，**去掉它会静默丢参数**
    # ------------------------------------------------------
    # 注解是 BaseModel 这个基类，而实际值是 HitlTimeoutParams 之类的子类。
    # pydantic 默认按**注解**（基类）的 schema 序列化，而基类一个字段都
    # 没有 —— 于是 Task.model_dump() 得到的是 params={}，参数全没了。
    #
    # 危险之处在于它不报错：入库时 params_json 写成一个空对象，
    # Worker 再从库里读出来校验，报的是"缺 hitl_id"这种离原因很远的话。
    # SerializeAsAny 让它按**运行时实际类型**序列化，model_dump 才是保真的。
    #
    # （Task.params_dict 走的是 model_dump 实例本身，本来就不受影响；
    # 但任何用 Task.model_dump() 落库的写法都必须靠这一行才对。）
    params: SerializeAsAny[BaseModel] = Field(
        description=(
            "业务参数，必须是 Pydantic Model。"
            "只放业务 ID 与执行所需的最小参数。"
        ),
    )

    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="当前状态。流转规则见 types.py。",
    )

    execute_at: datetime | None = Field(
        default=None,
        description=(
            "什么时候可以开始执行。None 表示立即。"
            "延迟执行与定时执行都用它，不为任何业务单独造 scheduler。"
        ),
    )

    attempt: int = Field(
        default=0,
        description="已经执行过几次（含正在执行的那次）。提交时为 0。",
    )

    #: 定时器代际。提交时为 1，**每重排一次 +1**。
    #:
    #: 它的作用是让"已经生成的那个定时"能识别出自己是否已被取代：
    #:
    #:     定时器绑定的 (task_id, generation, execute_at)
    #:     醒来时读回库里的 generation
    #:     不一致 -> 说明这条定时在睡眠期间被重排过，本次唤醒作废，
    #:              改按**新的**执行时间重新绑定（见 infra/task/dispatcher.py）
    #:
    #: 为什么不能只比 execute_at
    #: ----------------------
    #: 把时间重排到**同一个时刻**是完全合法的一次重排（业务想表达
    #: "按现在的时刻重算"）。那时 execute_at 前后相等，只有计数器能
    #: 区分"旧定时"与"新定时"。
    #:
    #: 它**不**参与 claim 的判断 —— 到期与否由 execute_at 说了算，那是
    #: 硬保证；generation 只影响定时器"该不该重排"这个决定。见
    #: TaskStore.reschedule 的说明。
    generation: int = Field(
        default=1,
        description=(
            "定时器代际。提交时为 1，每重排一次 +1。"
            "用于让已生成的定时识别自己是否被取代。"
        ),
    )

    max_attempts: int = Field(
        default=3,
        description=(
            "总执行次数上限。attempt 达到它之后失败就是终态 FAILED，"
            "不再重试。"
        ),
    )

    parent_task_id: UUID | None = Field(
        default=None,
        description="直接父任务。为 None 表示这是链条的起点。",
    )

    root_task_id: UUID | None = Field(
        default=None,
        description=(
            "整条链的根任务 ID。根任务自己的这个字段等于 task_id。"
            "按它查就能拿到一条后台执行链上的全部任务。"
        ),
    )

    causation_id: UUID | None = Field(
        default=None,
        description=(
            "触发源。可以是父任务 ID，也可以是一个业务事件 ID —— "
            "后者用于回答「这次 HITL 最终引发了哪些任务」。"
        ),
    )

    idempotency_key: str | None = Field(
        default=None,
        description=(
            "幂等键。同一个 key 重复提交只会得到同一条 Task，"
            "返回的是既有 task_id 而不是新建一条。"
            "有外部副作用的 Task 应该带上它 —— 至少一次投递意味着"
            "重复执行是常态，不是异常。\n\n"
            "⚠️ 默认是**一次性**的：任务进终态之后那个键仍然占着它。"
            "要让它**可以反复用**（同一个键代表\"此刻哪条任务代表这件事\"，"
            "任务一结束就释放），把 `reusable_key` 置 True —— 见那个字段。"
        ),
    )

    reusable_key: bool = Field(
        default=False,
        description=(
            "这个幂等键在任务结束后**可以被复用**吗。\n\n"
            "False（默认）= 一次性键：任务终结之后键仍归它，重复提交拿回"
            "那条已终结的（老行为，L0 与既有调用方都靠它）。\n\n"
            "True = 键在进终态时被释放（写回 NULL），于是同一个键可以"
            "**反复用**："
            "一个会话长期占着 `memory.l1.extract:{session_id}`，上一次跑完"
            "就释放，下一次提交建新的。L1 的\"每会话一条、反复重排\"靠它。\n\n"
            "⚠️ 为什么是开关而不是统一行为：这是**共享基础设施**，统一改会"
            "顺带改掉 L0 的行为。默认 False 让老调用方一行不动。"
        ),
    )

    created_at: datetime = Field(
        default_factory=_utcnow,
        description="创建时间。",
    )

    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="最后更新时间。",
    )

    started_at: datetime | None = Field(
        default=None,
        description="最近一次开始执行的时间（重试会覆盖它）。",
    )

    #: 租约相关。它们只对 running 的任务有意义：claim 时写入，进终态或
    #: 被收回时清空。
    #:
    #: 放在实体上（而不是只存在 DB 里）是因为"这条卡住了，谁在跑"必须
    #: 能从查询接口看出来 —— 一个状态是 running 但租约早已过期的任务，
    #: 与一个正常执行中的任务，运维需要能区分。
    lease_expires_at: datetime | None = Field(
        default=None,
        description=(
            "当前这次执行的租约到期时刻。过期而状态仍是 running，"
            "说明执行它的 Worker 崩了，任务正等着被收回重跑。"
        ),
    )

    leased_by: str | None = Field(
        default=None,
        description="持有租约的 Worker 标识。",
    )

    finished_at: datetime | None = Field(
        default=None,
        description="进入终态的时间。",
    )

    error: str | None = Field(
        default=None,
        description=(
            "最近一次失败的原因（异常类型 + 消息）。"
            "重试成功后**仍然保留** —— 排查「它为什么重试过」要有据可查。"
        ),
    )

    # ========================================================
    # 校验
    # ========================================================

    @field_validator("params", mode="before")
    @classmethod
    def _reject_bare_dict(
            cls,
            value: Any,
    ) -> Any:
        """
        拒绝裸 dict。

        mode="before" 是必须的：若交给 pydantic 默认流程，一个 dict 会被
        当成 BaseModel 去校验，而 BaseModel 没有字段 —— 结果是一个**空
        模型通过校验**，参数在提交这一步就没了。那种失败要到 Worker
        执行时才表现为"参数缺失"，离真正的原因很远。

        读库时要装参数请用 RawTaskParams（它显式声明了 extra="allow"，
        是有意为之的保真容器，不是"随手塞个 dict"）。
        """

        if isinstance(value, dict):
            raise ValueError(
                "Task.params 必须是 Pydantic Model，不接受裸 dict。"
                "提交时请给出具体的 params 类（如 HitlTimeoutParams）；"
                "从库里装载请用 RawTaskParams。"
            )

        return value

    # ========================================================
    # 派生
    # ========================================================

    def child(
            self,
            *,
            handler: str,
            params: BaseModel,
            execute_at: datetime | None = None,
            max_attempts: int | None = None,
            idempotency_key: str | None = None,
            causation_id: UUID | None = None,
    ) -> "Task":
        """
        派生一条子 Task，自动接上 lineage。

        放在 Task 上而不是让每个 Handler 自己填三个字段：lineage 是
        **每条链上每一步都要做对**的事，而漏填不会报错 —— 只会让
        "这条任务为什么产生"这个问题在半年后查不出来。

        causation_id 默认取父任务 ID。若这次派生的触发源是别的东西
        （比如一个业务事件），显式传进来覆盖它。
        """

        return Task(
            handler=handler,
            params=params,
            execute_at=execute_at,
            max_attempts=(
                self.max_attempts
                if max_attempts is None
                else max_attempts
            ),
            idempotency_key=idempotency_key,
            parent_task_id=self.task_id,
            # 根本继承父的；父没有（说明父自己就是根）就取父的 ID。
            root_task_id=self.root_task_id or self.task_id,
            causation_id=(
                self.task_id
                if causation_id is None
                else causation_id
            ),
        )

    # ========================================================
    # 参数访问
    # ========================================================

    @property
    def params_dict(self) -> dict[str, Any]:
        """
        参数的原始 dict 视图。

        Worker 用它去做 model_validate —— 注意**不是**把 self.params
        直接丢给 model_validate：从库里装载时它是 RawTaskParams，
        pydantic 不会把 A 类的实例校验成 B 类。先转 dict 再校验，
        类型信息就不会丢。
        """

        return self.params.model_dump(mode="json")

    def seal_root(self) -> "Task":
        """
        把 root_task_id 补齐成自己。

        由 TaskService.submit() 调用：只有到提交这一步才知道"没人给我
        接过 lineage"，也就是我是一条链的起点。放在这里而不是
        default_factory，是因为调用方显式覆盖 root_task_id 时必须认得出来。
        """

        if self.root_task_id is None:
            self.root_task_id = self.task_id

        return self


class TaskContext:
    """
    Handler 执行时的上下文。

    刻意不是一个 Pydantic 模型：它持有**活的** TaskSubmitter，不是为了
    序列化而生的东西。把它做成模型会引诱人把整条链的 context 存起来，
    而链的可靠性恰恰依赖"每个 Handler 只知道自己那一步"。

    只给 submit，不给 cancel/get/retry
    --------------------------------
    类型是 TaskSubmitter 而不是完整的 TaskService，这是**故意的**：

        1. Handler 派生下一步只需要 submit。
        2. cancel / retry 是**链路之外**的操作（用户点的、运维点的），
           让 Handler 能随手取消或重试自己的任务，只会制造出
           "任务把自己取消掉"这种循环。

    接口小，就越不容易用错。
    """

    __slots__ = ("task", "task_service")

    def __init__(
            self,
            *,
            task: Task,
            task_service: TaskSubmitter,
    ) -> None:

        self.task = task
        self.task_service = task_service

    async def submit(
            self,
            *,
            handler: str,
            params: BaseModel,
            execute_at: datetime | None = None,
            max_attempts: int | None = None,
            idempotency_key: str | None = None,
            causation_id: UUID | None = None,
    ) -> UUID:
        """
        派生并提交下一条 Task，返回它的 task_id。

        这是 Handler 里"任务接任务"的唯一入口（spec §8）：

            hitl.timeout -> run.resume

        它只是 self.task.child(...) + submit 的组合，存在的意义是
        **让 Handler 没有机会漏填 lineage**。
        """

        child = self.task.child(
            handler=handler,
            params=params,
            execute_at=execute_at,
            max_attempts=max_attempts,
            idempotency_key=idempotency_key,
            causation_id=causation_id,
        )

        return await self.task_service.submit(child)

    def __repr__(self) -> str:

        return (
            f"TaskContext(task_id={self.task.task_id!r}, "
            f"handler={self.task.handler!r})"
        )
