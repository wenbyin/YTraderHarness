from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


class AgentSession(Base):
    """
    一个 Agent 会话。

    长期存在的主体：一次对话可以开很多次运行，运行结束会话还在。
    运行侧通过外键指向本表，本表**不**反向聚合运行（不需要 relationship：
    读一个会话不该顺便把它的全部历史拖出来）。

    ⚠️ 属性名 `meta` 而不是 `metadata`
    ---------------------------------
    列名与属性名都是 meta。SQLAlchemy 的声明式基类把 `metadata` 这个
    属性名留给了 MetaData 对象，同名会直接报 InvalidRequestError。
    对外契约里仍然是 metadata（见 session.yaml），改名只发生在契约边界。

    名字带 agent_ 前缀的理由见模块 __init__.py：session 这个词在本仓库里
    已经被数据库会话占满了。
    """

    __tablename__ = "agent_session"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="会话 ID。也是 agent_run 的主体外键。",
    )

    agent_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="这个会话属于哪种 Agent。取值由 Agent 目录决定（尚未落地）。",
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="active / archived。见 domain/schemas/enum.py。",
    )

    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="调用方附加信息。列名不用 metadata，理由见类 docstring。",
    )

    # ⚠️ 后加的列。create_all 只建缺失的表、不改已存在的表 ——
    # 已有的库必须手动补，DDL 见 models.sql 末尾。
    thread_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "本会话在 LangGraph checkpointer 里的线程键，全库唯一。"
            "首轮运行时生成并落库，此后不再变 —— 多轮历史靠它累积。"
            "可空是为了不要求给已有会话回填：第一次发运行时才生成。"
        ),
    )

    # ⚠️ 也是后加的列，已有库同样要手动补（DDL 见 models.sql 末尾）。
    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "会话归属的用户 ID —— Casdoor 的 sub（见 app/common/auth），"
            "与 file.uploaded_by 同一口径。\n\n"
            "⚠️ 它是 session 接口的过滤键，但两种口径不同：\n\n"
            "    按 id 读      `user_id = 当前用户 OR user_id IS NULL`\n"
            "    列表 / 删除   `user_id = 当前用户`（严格）\n\n"
            "NULL 是刻意保留的存量语义（加这列之前的会话对谁都可见），"
            "但无主会话不出现在任何人的列表里、也不能被删除。"
            "新建会话必须写入。"
        ),
    )

    # ⚠️ 也是后加的列，已有库同样要手动补（DDL 见 models.sql 末尾）。
    org_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "会话所属的组织 —— JWT 的 org claim（claim 名可配，见 "
            "app/common/auth 的 org_claim）。\n\n"
            "用途只有一个：**团队 / 组织归属的记录**。它不参与鉴权"
            "（那是 groups 的事），也不参与可见性过滤（过滤键仍然是"
            " user_id）。\n\n"
            "NULL 表示这次登录的 token 里没有组织信息 —— 这是允许的"
            "（Casdoor 各版本与各应用的 Token fields 配置不一样），"
            "不是漏写。"
        ),
    )

    # ⚠️ 也是后加的列，已有库同样要手动补（DDL 见 models.sql 末尾）。
    task_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "预留位：这个会话是为哪一件**工作任务**服务的。\n\n"
            "⚠️ **当前恒为空** —— 还没有任何地方写它（没有输入来源）。"
            "留着是为了将来接「工作任务」那条线时不必再动一次 DDL。\n\n"
            "⚠️ 别与 `tasks.task_id`（后台任务主键，UUID）混起来："
            "**同名不同物**。这一列是业务侧的工作任务标识，而它的形态要等"
            "那条线定了才知道 —— 所以先按字符串留，不是 UUID。"
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
        comment=(
            "最后更新时间。除了本行字段被改，**发一条消息也会把它推到当下**"
            " —— 消息写在 agent_message，本行一个字都没变，所以那条路径是"
            "显式碰它（见 repository/agent_session.py 的 touch）。"
        ),
    )

    __table_args__ = (
        Index("idx_agent_session_agent", "agent_id"),
        Index("idx_agent_session_status", "status"),
        # 唯一而不是普通索引：thread_id 是 checkpointer 里的**全局键**，
        # 两个会话共用一条会互相续跑对方的对话。见 agent_run_dispatch.py
        # 里 THREAD_NAMESPACE 那段说明。
        Index("uq_agent_session_thread", "thread_id", unique=True),
    )
