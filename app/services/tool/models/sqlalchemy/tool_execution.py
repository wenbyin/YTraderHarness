from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ...domain.schemas.enum import ToolExecutionStatus


class ToolExecution(Base):
    """
    一次工具调用的执行记录 —— 幂等判重与审计的载体。

    表结构与约束逐条对应 tool_execution.sql，那份 sql 是契约。

    它在回答什么问题
    ----------------
    「这个副作用到底发生过没有？」之前进程崩了、工具跑一半被杀、
    审批放行后重跑，这三件事都没有任何持久痕迹可查。

    判重的键是 (session_id, run_id, tool_key, param_hash)
    -------------------------------------------------
    session_id 取的是 **agent_session.session_id**（会话行自己的主键），
    不是 langgraph 的 thread_id。

    ⚠️ 为什么用 session_id 而不是 thread_id
    -------------------------------------
    本仓里两者是**一对一**的（`agent_session.thread_id` 按会话稳定、
    存在会话行上，见 agent_run_service_impl._resolve_thread_id），所以
    分区的粒度一样。区别在**身份归属**：会话详情页、消息、运行列表
    全都按 session_id 关联，而 thread_id 是 langgraph 的内部键
    （形如 `dev:react:<uuid>`）—— 排查的人手里拿到的是前者。
    存 thread_id 意味着每次查账本都要先反查一次会话，而那个映射
    就在调用方手边（`run.session_id`）。

    run_id 在这一段里同样关键：**resume 不换 run_id**
    （agent_run_service_impl._schedule_resume 沿用原 run）—— 所以
    HITL 中断后重跑那次能查到中断前留下的那一行；而同一条会话的
    **下一轮是新 run_id**，同参数可以再调，这是想要的。

    ⚠️ 为什么**不**建 run_id 的外键
    ------------------------------
    本表要在 agent_run 的生命周期之外活着。加了外键只有两种收场：

        级联删除    删一次历史 run 就把「这单副作用发生过」的证据删了
        NO ACTION   一条审计记录把 run 的清理**卡住**

    两者对一张执行账本都是坏的。tool_processor 那种挂载记录可以用
    CASCADE（Tool 没了，链就没有载体了），这里不行 —— 记录的价值
    恰恰在于它的主体可能已经没了。

    ⚠️ result 存的是**脱敏之后**的那一份
    -----------------------------------
    写库发生在处理器链**跑完之后**（见 graph/react/tools.py 的
    _wrap_tool_call —— 它拿到的 ToolOutcome 是整条链含 AFTER 的
    产物），所以 privacy_transformer 的脱敏已经生效。存进来的就是
    模型当时看到的那份。

    **不要**改成从 _base_call 里顺手取 —— 那会绕过 AFTER，把
    未脱敏的原始结果永久写进库。

    ⚠️ 没有 DB 触发器维护 updated_at
    -------------------------------
    与 tool_processor 同一条理由（见那里的说明）：项目统一用
    onupdate=func.now()，而 create_all 建不出触发器。两套机制
    并存会让「到底谁在改 updated_at」没法回答。
    """

    __tablename__ = "tool_execution"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="执行记录 ID。",
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment=(
            "所属会话（agent_session.session_id）。与 run_id 一起决定"
            "「哪条会话上的哪一次运行」。"
            "⚠️ 是会话行自己的主键，不是 langgraph 的 thread_id —— "
            "见类 docstring。"
        ),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment=(
            "本次运行（agent_run 主键，即 pipeline_run_id）。"
            "⚠️ 刻意不建外键 —— 见类 docstring。"
        ),
    )

    tool_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="被调用的工具，形如 document.retrieve_documents。",
    )

    param_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "调用参数的 sha256 十六进制摘要（canonical_json + "
            "sha256_text，与 cache 拦截器同一套）。"
            "用摘要而不是原文：参数可能很长，而这四列是唯一键。"
        ),
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ToolExecutionStatus.PENDING.value,
        comment=(
            "执行状态。取值与语义见 domain/schemas/enum.py 的 "
            "ToolExecutionStatus —— ⚠️ 那里的每个取值都标了它"
            "「能不能重跑」，那才是策略的依据。"
        ),
    )

    result: Mapped[dict | list | str | int | float | bool | None] = (
        mapped_column(
            JSONB(none_as_null=True),
            nullable=True,
            comment=(
                "工具返回值，经规范化后的 JSON 形态（可查询）。"
                "⚠️ 是**脱敏之后**的那一份，见类 docstring。"
                "⚠️ none_as_null=True 是必须的：否则 Python None 会被"
                "写成 JSON 'null' 字面量（非 SQL NULL），"
                "让 upsert 的 COALESCE 守卫失效。"
                "工具正常返回 None 是**合法的成功结果**，"
                "与「这列没写过」是两回事 —— 后者是 update 时"
                "是否覆盖的判据，因此不能混。"
            ),
        )
    )

    error: Mapped[dict | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
        comment=(
            "失败 / 拒绝的那段说明，形状 "
            "{'type': 异常类名, 'message': 给模型看的文案}。"
            "完整堆栈只进日志，不入库 —— 与 checkauth 的"
            "「文案刻意含糊、细节留日志」同一口径"
            "（interceptor/auth.py 的 _deny）。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="这条记录第一次建出来的时间（claim 那一刻）。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment=(
            "最后一次状态转移的时间 —— 也是判断"
            "「这行是不是还活着」的依据。"
        ),
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        # 状态取值必须与枚举一致。这道拦的是绕过应用的写入 ——
        # 服务层每个校验在这里都有一份对应的表达（与 tool_processor
        # 同一条立场）。
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', "
            "'UNKNOWN', 'CANCELLED', 'REJECTED', 'INTERRUPT')",
            name="ck_tool_execution_status",
        ),
        CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'",
            name="ck_tool_execution_error_object",
        ),
        # 判重与抢锁都落在这条上：并发同键靠 INSERT ... ON CONFLICT
        # 串行化，**不是**先查再插（那样两个并发分支会都执行）。
        #
        # ⚠️ 用 UniqueConstraint 而不是 Index(unique=True)：只有真正的
        # **约束**才能被 `ON CONFLICT ON CONSTRAINT <名字>` 引用，
        # 唯一索引只能走列推断那条形式。写死名字比列推断稳 ——
        # 与 document_pipeline_node_run 的 uq_document_pipeline_node_run_identity
        # 同一写法。
        UniqueConstraint(
            "session_id",
            "run_id",
            "tool_key",
            "param_hash",
            name="uq_tool_execution_identity",
        ),
        # 运行详情页：「这一次运行调了哪些工具」。
        Index(
            "idx_tool_execution_run_created",
            "run_id",
            "created_at",
        ),
        # 按工具查历史执行（运维排查「这个工具最近失败了几次」）。
        Index(
            "idx_tool_execution_tool_key",
            "tool_key",
            "created_at",
        ),
        # 部分索引：只覆盖「没有走到终态的行」。
        # 那个进程被杀留下的遗留行（本期还没有 reaper，见枚举的说明）
        # 以及将来的重试扫描都只会查这一段，比全表索引小得多。
        Index(
            "idx_tool_execution_unsettled",
            "created_at",
            postgresql_where=text(
                "status IN ('PENDING', 'RUNNING')"
            ),
        ),
    )
