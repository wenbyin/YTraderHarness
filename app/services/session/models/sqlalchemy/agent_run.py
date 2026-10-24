from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


class AgentRun(Base):
    """
    一次 Agent 运行。

    追加型历史：每次运行一行。状态随执行被改写（queued -> running ->
    completed...），但行本身不复用 —— 下一次运行是新的 run_id。

    删会话时本表随之删除（ON DELETE CASCADE）：运行没有独立于会话的
    存在理由，而留下无主运行会让"某个会话发生过什么"查出一堆孤儿。

    ⚠️ 本表**只落运行级**状态，不是观测表
    ------------------------------------
    `app/infra/graph/recorder.py` 那套观测（拓扑快照、逐节点行、
    create_run / mark_run_running / upsert_node_run）**不写这里**：
    它的实现写在 document 模块（SqlAlchemyRunRecorder），针对
    document_pipeline_run 那张表，且断言 subject_type 必须是
    document_version。

    Agent 这条链路目前只翻状态：queued -> running -> completed / failed
    （见 application/agent_run_dispatch.py）。**节点级进度不落库** ——
    它由将来的 SSE 事件流承担，那时再加表/加行，而不是现在先建一张
    没人读的节点表。

    ⚠️ error_* 、user_id 与 caller_identity 是**加出来的**，已有库要手动补
    -------------------------------------------------------------------
    仓库没有迁移工具，initialize_database 用的是
    `Base.metadata.create_all`，它只建缺失的表、**不改已存在的表**
    （见 app/infra/datasource/initializer.py）。已有的库要执行：

        ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
            error_type VARCHAR(128);
        ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
            error_message TEXT;
        ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
            user_id VARCHAR(255);
        ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
            caller_identity JSONB NOT NULL DEFAULT '{}'::jsonb;
        ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
            trace_id VARCHAR(255);

    症状各不相同：没有 error_* 两列，失败路径的写回会报 column does
    not exist；没有 user_id，则**每一次运行查询都 500** —— ORM 一旦
    声明了列，SELECT 就会带上它，这一条必须**先跑 ALTER 再上新代码**。

    caller_identity 同理（也是 SELECT 会带上的一列），但它的默认值让
    后果轻一些：补跑 ALTER 之后存量运行拿到 `{}`，工具级鉴权对它们
    一律按"没有身份"处理（fail-closed），而不是报错。
    """

    __tablename__ = "agent_run"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="运行 ID。也是 LangGraph thread_id 的一部分。",
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "agent_session.session_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属会话。删会话即删它的全部运行。",
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment=(
            "queued / running / waiting_input / completed / failed / "
            "cancelled（对外契约那一套，见 domain/schemas/enum.py）。"
        ),
    )

    input: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="用户这次的输入载荷。",
    )

    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="调用方附加信息。列名不用 metadata，见 AgentSession 的说明。",
    )

    error_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "失败时的异常类名（如 NotImplementedError）。"
            "未结束或成功时是 NULL。"
        ),
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "失败时的异常文本。规格的 Run 不回它（只有 status），"
            "排查时只能从这里看 —— 所以它必须落库，而不是只进日志。"
        ),
    )

    # ⚠️ 后加的列，见类 docstring 里那三句 ALTER。
    caller_identity: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "发起运行时调用者的身份快照（groups / roles）。"
            "内部字段，不进对外契约 ——"
            "见 domain/entities/agent_run.py 的字段说明。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "发起这次运行的用户 ID（Casdoor 的 sub）—— 注意是**发起人**，"
            "不是所属会话的 user_id：存量会话（user_id 为 NULL、对谁都可见）"
            "上仍然能记录真实的执行者。\n\n"
            "它同时是给 ReAct 图用的：派发时写进初始 state 的 user_id"
            "（见 application/agent_run_dispatch.py 的 _initial_state）。"
        ),
    )

    # ⚠️ 后加的列，已有库要手动补（DDL 见 models.sql 末尾）。
    duty_key: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "这次运行用哪份**职责提示词**（duty_key）。\n\n"
            "建运行那一步解析一次（见 application/agent_run_service_impl.py "
            "的 create_run 与 app/services/prompt），此后这一轮里不再变："
            "派发时进图 state（builder 节点据此取提示词），写消息时填进"
            " agent_message.business_agent_id。\n\n"
            "之所以存下来而不是各处理解一次：同一轮运行里「用的哪份职责」"
            "必须只有一个答案，否则运行行、提示词、消息三处可能对不上。\n\n"
            "存量运行为 NULL —— 那时还没有这个概念，主循环会回落到默认职责。"
        ),
    )

    # ⚠️ 后加的列，已有库要手动补（DDL 见 models.sql 末尾）。
    trace_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "客户端传入的 Trace 聚合键。NULL = 这次运行未开启 Trace。"
            "trace 事件的 Redis 键是 trace:<trace_id>:events。"
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

    __table_args__ = (
        Index(
            "idx_agent_run_session",
            "session_id",
            "created_at",
        ),
    )
