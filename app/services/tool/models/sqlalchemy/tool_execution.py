from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

from ...domain.schemas.enum import ToolExecutionStatus


class ToolExecution(Base):
    """
    一次工具执行尝试 —— 守卫计数与审计的 append-only 账本。

    ⚠️ 沿用 `tool_execution` 表名，在原表上扩展
    ------------------------------------------
    旧表是「一个判重键一行、重试时重置」，唯一约束
    `(session_id, run_id, tool_key, param_hash)` 恰好禁止了
    「一次执行一条记录」。新版直接在这张表上：
        - 删掉旧唯一约束
        - 增加 scope / execution / counted / tool_call_id 等列
        - 用 `(run_id, tool_call_id)` 部分唯一索引做 replay 去重
    迁移脚本见 `app/services/tool/tool_execution_migration.sql`，
    需**手工执行**（`create_all` 只建缺失的表，不改已存在的表）。

    它在回答什么问题
    ----------------
    「这个维度上已经发生过几次、这次还让不让做？」以及
    「上一次到底成功、失败，还是不知道跑没跑？」

    一次执行一行，保存每一次尝试
    --------------------------
    与旧账本不同，重试不会覆盖上一行 —— 否则计数永远停在 1，
    守卫的「超过 N 次」就没有依据。

    判重仍然存在，但换了键
    --------------------
    `(run_id, tool_call_id)` 的部分唯一索引保证：同一次模型调用在
    LangGraph 重放 / HITL resume 时复用同一行，不会重复执行、也不会
    重复占用额度。没有 tool_call_id 的图外直调不做 replay 去重。

    计数键
    ------
    `match_key` = 五段（tool_key + scope_type + scope_key +
    execution_type + param_hash）的定长摘要。加锁、计数、索引都走它，
    避免多列组合比较把 NULL / 大小写之类的差异引进判定。

    ⚠️ `counted` 与 `status` 是两个维度
    --------------------------------
    status 描述发生了什么；counted 描述是否占额度。被拒绝 / 中断 /
    取消 / 守卫拦下的行 counted=False —— 它们没有真正执行。
    预留时先写 True，让并发的其它分支看见「有人正在跑」；落终态时
    再按结果修正。

    ⚠️ run_id 刻意不建外键
    ---------------------
    本表要在 agent_run 的生命周期之外活着：级联删除会删掉「副作用
    发生过」的证据，NO ACTION 会把 run 的清理卡住。

    ⚠️ result 存的是脱敏之后的那一份
    -------------------------------
    写库发生在处理器链跑完之后，privacy_transformer 已生效。不要
    改成从 _base_call 里顺手取。
    """

    __tablename__ = "tool_execution"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="执行记录 ID。",
    )

    tool_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="被调用的工具，形如 document.retrieve_documents。",
    )

    tool_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="tool_definition 主键；还没同步进库的工具为空。",
    )

    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "所属会话（agent_session.session_id）。"
            "⚠️ 是会话行自己的主键，不是 langgraph 的 thread_id。"
            "图外直调时为空。"
        ),
    )

    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "本次运行（agent_run 主键，即 pipeline_run_id）。"
            "⚠️ 刻意不建外键 —— 见类 docstring。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "调用者（caller_identity.user_id）。"
            "拿不到时为空 —— 不回落到会话反查。"
        ),
    )

    scope_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="SESSION / RUN / USER / GLOBAL。",
    )

    scope_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="session_id / run_id / user_id / '*'。",
    )

    execution_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="METHOD / INVOKE。",
    )

    param_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="",
        comment=(
            "INVOKE 时的参数 sha256；METHOD 时为空串。"
            "与 cache 拦截器同一套 canonical_json + sha256。"
        ),
    )

    match_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "五段维度组合的 sha256。计数、advisory lock、索引都用它。"
        ),
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ToolExecutionStatus.PENDING.value,
        comment=(
            "执行状态。取值与语义见 domain/schemas/enum.py 的 "
            "ToolExecutionStatus。"
        ),
    )

    counted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment=(
            "这次尝试是否占用额度。被拒绝/中断/取消/守卫拦下为 "
            "False。预占时先 True，落终态时修正。"
        ),
    )

    tool_call_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "模型给的调用 id。与 run_id 一起做 replay/resume 去重。"
        ),
    )

    result: Mapped[dict | list | str | int | float | bool | None] = (
        mapped_column(
            JSONB(none_as_null=True),
            nullable=True,
            comment=(
                "工具返回值，规范化后的 JSON 形态。"
                "⚠️ 是**脱敏之后**的那一份。"
                "none_as_null=True 让 Python None 写成 SQL NULL，"
                "而不是 JSON 'null' 字面量。"
            ),
        )
    )

    error: Mapped[dict | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
        comment=(
            "失败 / 拒绝的说明，形状 "
            "{'type': 异常类名, 'message': 给模型看的文案}。"
            "完整堆栈只进日志，不入库。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="这条记录建出来的时间（预占那一刻）。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后一次状态转移的时间。",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', "
            "'UNKNOWN', 'BLOCKED', 'CANCELLED', 'REJECTED', "
            "'INTERRUPT')",
            name="ck_tool_execution_status",
        ),
        CheckConstraint(
            "scope_type IN ('SESSION', 'RUN', 'USER', 'GLOBAL')",
            name="ck_tool_execution_scope",
        ),
        CheckConstraint(
            "execution_type IN ('METHOD', 'INVOKE')",
            name="ck_tool_execution_execution",
        ),
        CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'",
            name="ck_tool_execution_error_object",
        ),
        # replay / resume 去重：同一次模型调用在图中重跑时复用同一行。
        # 部分索引：没有 tool_call_id 的图外直调不参与去重。
        Index(
            "uq_tool_execution_call_id",
            "run_id",
            "tool_call_id",
            unique=True,
            postgresql_where=text(
                "tool_call_id IS NOT NULL AND run_id IS NOT NULL"
            ),
        ),
        # 守卫计数：按维度 + 时间窗口。只覆盖 counted 的行 ——
        # 被拒绝/中断的行不占额度，让索引更小。
        Index(
            "idx_tool_execution_match",
            "match_key",
            "created_at",
            postgresql_where=text("counted"),
        ),
        # 运行详情页：这一次运行调了哪些工具。
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
        # 部分索引：只覆盖没有走到终态的行（崩溃遗留 / reaper 扫描）。
        Index(
            "idx_tool_execution_unsettled",
            "created_at",
            postgresql_where=text(
                "status IN ('PENDING', 'RUNNING')"
            ),
        ),
    )
