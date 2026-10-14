from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


class Task(Base):
    """
    一条待执行/已执行的 Task。**本表是 Task 的事实来源**（spec §6）。

    MQ 只是投递手段，Worker 只是执行手段；状态、参数、lineage、重试计数
    全部以本表为准。因此任何一次状态推进都必须是**一条原子 UPDATE**，
    而不是"读到内存、改完再整体写回" —— 后者在多 Worker 并发下会把
    别人刚写进去的结果覆盖掉。

    与 app/common/task/models.py::Task 的关系
    ----------------------------------------
    那个是**抽象层**的 Task（业务构造、Handler 接收的东西）；本表是它
    的持久化形态。两者刻意不合并：

        params         <->  params_json     （列里只有序列化结果，
                                             类型要等 Registry 解析出
                                             handler 才知道）
        纯 Python 类型  <->  DB 列 + 索引 + 状态机约束

    转换在 repository/task_impl.py 的 _to_task / _to_model 里。

    params_json 只放最小参数（spec §2）
    ---------------------------------
    业务 ID + 执行所需的最小参数。不放 Agent State、Message History、
    Tool Result —— 那些在执行阶段按 ID 从 DB / Store 重新加载。

    理由不是"省空间"，而是**避免同一份状态有两处副本**：Task 会被反复
    读出（轮询、claim、查询接口、重试），一旦里面冻了一份图状态，它就会
    与真正的图状态不一致，而历史 Task 行里的那份**永远是错的**、且没有
    任何东西会去纠正它。

    ⚠️ lease_expires_at 与 leased_by 是 spec 建议字段之外的**加法**
    ----------------------------------------------------------
    spec §5 建议的字段清单里没有它们，但 spec §18 的验收条件要求
    "服务重启恢复"与"Worker 多实例并发执行"，二者都要求一条：
    **被崩溃的 Worker 抢走、却永远不会结束的任务必须能回到可执行状态**。

    没有租约时，Worker 在 claim 之后、写回结果之前被杀，那条任务就永远
    停在 RUNNING —— 既不会被重试（RUNNING 不是可 claim 状态），也不会
    有任何告警（"看起来"有人正在处理）。这正是既有 agent_run 那条链路
    已知的缺口（见 application/agent_run_dispatch.py 的 docstring）。

        lease_expires_at   claim 时写 now() + 租期。Sweeper 把租约过期而
                           状态仍是 RUNNING 的行收回成 RETRYABLE。
        leased_by          持有租约的 Worker 标识（进程名 + 实例数）。
                           排查"这条卡住了，谁在跑"时的第一手信息。

    不加 `worker_id` 之类的所有权表、也不加 heartbeat 列：租约到期即失效，
    不需要额外的续约协议。

    ⚠️ generation 同样是加法：为"防抖的延迟任务"准备
    ---------------------------------------------
    延迟任务的执行时间可能被**反复重排**（典型是审批超时：每次有新动静
    就把截止时间往后推）。重排必须让**上一个已经生成的定时**失效，
    否则它到点就会把任务跑掉 —— 那正是防抖要避免的事。

    generation 就是这个失效令牌：重排时 +1，定时器醒来读回它、与自己
    绑定的值比对，不一致就说明自己已被取代，改按新时间重排。

    为什么不只比 execute_at：把时间重排到**同一个时刻**是合法的一次
    重排，那时只有计数器能区分新旧定时。

    没有迁移工具
    -----------
    建表靠 app/infra/datasource/initializer.py 的 Base.metadata.create_all()，
    它**幂等但不会改已有的表**。所以本表若在已有库上要加列，必须手动
    ALTER（见 models.sql 末尾的说明），否则 ORM 声明了新列、SELECT 会带上
    它，症状是**每一次任务查询都 500**。
    """

    __tablename__ = "tasks"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="任务全局唯一 ID。稳定，全链路复用。",
    )

    handler: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment=(
            "逻辑 Handler Name（如 hitl.timeout）。"
            "不是 import path —— 解析走 infra/task/registry.py 的显式注册表。"
        ),
    )

    params_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment=(
            "Pydantic params 序列化后的结果。只有业务 ID 与最小参数，"
            "不放图状态/消息历史。"
        ),
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment=(
            "pending / running / retryable / success / failed / cancelled。"
            "取值定义在 app/common/task/types.py::TaskStatus —— 这里存 .value，"
            "不是 Python 枚举名。"
        ),
    )

    execute_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "可执行时刻。NULL = 立即。延迟执行与定时执行都用它，"
            "不为任何业务单独造 scheduler。"
        ),
    )

    attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="已执行次数（含正在执行的那次）。",
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        comment="执行次数上限。attempt 达到它之后失败即终态，不再重试。",
    )

    # ⚠️ 后加的列，见类 docstring 里那段 ALTER 说明。
    generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment=(
            "定时器代际。提交时 1，每重排一次 +1。"
            "让已生成的定时能识别自己是否已被取代（见 reschedule）。"
        ),
    )

    #: 自引用外键，只表达 Task -> Task 的单父结构。
    #:
    #: 刻意**不加** CASCADE（与 capability 的自引用外键同一个取舍）：
    #: 删父任务时级联删掉整条子链，会让"某次 HITL 触发的任务"无声消失，
    #: 而那种消失没有任何地方会记录。
    #:
    #: spec §9 明确第一版不做多依赖 fan-in，所以这里是单列外键而不是
    #: 一张关联表。将来真要 DAG，加关联表即可，这一列可以保留为
    #: "主父"。
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tasks.task_id"),
        nullable=True,
        comment="直接父任务。NULL 表示这是链条的起点。",
    )

    #: 根任务 ID。**不是外键**，两个理由：
    #:   1. 它是 lineage 的缓存（查整条链走它），不是结构约束；
    #:   2. 让它成为外键会给"根任务"加上"被所有后代钉住、不可删除"
    #:      的隐含语义，而那是另一件事，不该由一个查询用的列顺带决定。
    root_task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="整条链的根任务 ID。根任务自己的这个值等于 task_id。",
    )

    #: 触发源。**也不是外键**：它可以是父任务的 ID，也可以是一个**业务
    #: 事件 ID**（spec §9）—— 后者不在本表里，外键会立刻拦下它。
    causation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "触发源。可以是父任务 ID，也可以是业务事件 ID —— "
            "用于回答「这次 HITL 最终引发了哪些任务」。"
        ),
    )

    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "幂等键。唯一性由下面那个**部分**唯一索引保证"
            "（NULL 不参与唯一，否则所有不带 key 的任务会互相冲突）。"
        ),
    )

    # ⚠️ 后加的列，见类 docstring 里那段 ALTER 说明。
    reusable_key: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
        comment=(
            "这个键在任务**结束后可以被复用**吗（终态时释放它）。\n\n"
            "False（默认）= 老语义：键是**一次性**的，同一条任务结束之后\n"
            "那个键仍然占着它（重复提交拿回那条已终结的）。\n"
            "True = 键的含义变成\"此刻哪个任务代表这件事\"，任务一进终态就\n"
            "把它写回 NULL，于是同一个键**可以反复用**（L1 的\"每会话一条、\n"
            "反复重排\"靠它）。\n\n"
            "⚠️ 为什么做成开关而不是统一行为：那是**共享基础设施**，统一改会\n"
            "顺带改掉 L0 的行为（`memory.l0.extract:{run_id}` 在任务跑完之后\n"
            "再收尾一次就会多排一条）。默认 False 让老调用方一行不动。\n\n"
            "理由与全部代价见 repository/task_impl.py 顶部的\n"
            "「幂等键的生命周期」。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后更新时间。",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="最近一次开始执行的时间（重试会覆盖它）。",
    )

    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "当前这次执行的租约到期时刻。Sweeper 靠它收回崩溃 Worker "
            "留下的 RUNNING 行。终态时置 NULL。"
        ),
    )

    leased_by: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="持有租约的 Worker 标识。用于排查「卡住的这条谁在跑」。",
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="进入终态的时间。",
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "最近一次失败的原因（异常类型 + 消息）。重试成功后**仍然保留**，"
            "排查「它为什么重试过」要有据可查。"
        ),
    )

    # ========================================================
    # Constraints / Indexes
    # ========================================================

    __table_args__ = (
        # 轮询与 claim 的主索引，对应那条最热的查询：
        #
        #   WHERE status IN ('pending','retryable')
        #     AND (execute_at IS NULL OR execute_at <= now())
        #   ORDER BY execute_at NULLS FIRST, created_at
        #
        # 列顺序 (status, execute_at) 而不是反过来：status 是等值条件，
        # 放前面才能把范围扫描限制在"待跑"的那一小部分行上 ——
        # 反过来写会先按 execute_at 扫全表。
        Index("idx_tasks_due", "status", "execute_at"),

        # 幂等的部分唯一索引。
        #
        # ⚠️ postgresql_where 不能省：普通唯一索引下 NULL 虽然可以重复
        # （SQL 里 NULL != NULL），但语义会变得很脆 —— 一旦有人给
        # idempotency_key 写了空字符串，所有这样的行就会互相冲突。
        # 显式加上 "IS NOT NULL" 才把"不传 key 就不参与幂等"这件事
        # 写成了约束，而不是依赖 NULL 的比较规则。
        Index(
            "uq_tasks_idempotency",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),

        Index("idx_tasks_parent", "parent_task_id"),

        # 查整条执行链：WHERE root_task_id = :id ORDER BY created_at
        Index("idx_tasks_root", "root_task_id"),

        # 租约回收用：WHERE status = 'running' AND lease_expires_at <= now()
        Index("idx_tasks_lease", "status", "lease_expires_at"),
    )


#: 跨模块引用时用的别名。
#:
#: repository/task_impl.py 同时要 import 本类与抽象层的 Task，两个都叫
#: Task 会撞名。仓库里已有的写法是 import 时改名（见
#: capability_impl.py 的 `Capability as CapabilityModel`），这里额外导出一个
#: 语义明确的别名，让调用方可以直接 `from ...task import TaskModel`，
#: 少一次改名。两个名字指向同一个类，不会产生第二张表。
TaskModel = Task
