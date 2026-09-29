from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.models import Base

from ...domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)


class ToolProcessor(Base):
    """
    挂在某个 Tool 上的一条处理器记录。

    一个 Tool 一条链，链由本表的行按 sequence 排序构成。
    表结构与约束逐条对应 tool_processor.sql，那份 sql 是契约。

    这里存的是**挂载**，不是实现
    ----------------------------
    处理器实现（重试、超时、审批）由代码拥有，通过
    registry/processor_registry.py 反射发现。本表只存"哪个 Tool 挂了
    哪个实现、参数是什么"。因此：

        改实现      只发版，不动这张表
        调参数      只改这张表，不发版

    processor_key 在代码里已不存在时**保留这一行**
    ---------------------------------------------
    读链时 `processor` 字段整个省掉（见 application/_assembly.py），
    而不是删行 —— 删掉等于替人做决定，而且实现回滚时那行已经没了。
    这与 tool_definition 的 source_state=MISSING 是同一种立场：
    事实（有人配过）与现状（代码里还有没有）分开表达。

    ⚠️ 一条 AROUND 处理器在本表里是**两行**
    ----------------------------------------
    先进一行、后出一行，各带自己那份 config。两行之间必须与其它
    AROUND 的区间正确嵌套（洋葱模型），校验在服务层 —— 表结构表达
    不了"括号要配对"这种跨行约束。

    为什么不用 DB 触发器维护 updated_at
    -----------------------------------
    tool_processor.sql 里带了一个 trigger 函数，这里**刻意没有镜像**
    它：项目统一的写法是 `onupdate=func.now()`（见 tool_documentation.py
    与 tool.yaml 的其余四张表），而 `create_all` 也建不出触发器。
    两套机制并存会让"到底谁在改 updated_at"没法回答。
    """

    __tablename__ = "tool_processor"

    processor_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="处理器挂载记录 ID。",
    )

    tool_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "tool_definition.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment=(
            "所属 Tool 定义 ID。级联删除：Tool 没了，链就没有载体了。"
        ),
    )

    processor_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment=(
            "代码里注册的处理器实现键，例如 cache_interceptor。"
            "⚠️ 实现改名后这里的旧值**不会被自动改**，读链时表现为"
            " processor 字段缺席。"
        ),
    )

    sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment=(
            "在链里的位置，**从 1 开始连续**。"
            "服务层每次写完都会重编号 1..N，因此前端传来的值只是"
            "位置提示，不保证被原样存下。"
        ),
    )

    execution_mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ExecutionMode.SYNC.value,
        comment=(
            "SYNC=参与结果；ASYNC=旁路不等。"
            "结构属性，不放进 config —— 它改变的是执行语义，"
            "不是处理器的业务参数。"
        ),
    )

    hook: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ProcessorHook.AROUND.value,
        comment=(
            "BEFORE / AFTER / AROUND —— 挂在调用的哪一段。"
            "同样是结构属性，不放进 config。"
        ),
    )

    config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "处理器自己的业务参数，形状由实现的 config_model 决定"
            "（见 /processors 的 config_schema）。"
            "**不要**把 sequence / hook / execution_mode 塞进来。"
        ),
    )

    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment=(
            "关掉的行留在链里占位（前端看得到它排第几），"
            "执行时跳过。"
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

    # ========================================================
    # Relationships
    # ========================================================

    tool: Mapped["ToolDefinition"] = relationship(
        back_populates="processors",
    )

    # ========================================================
    # Constraints
    # ========================================================
    #
    # ⚠️ 这是本项目**第一处** CheckConstraint（其余四张表只有索引
    # 与唯一约束）。约束逐条照抄 tool_processor.sql —— DB 是绕过
    # 应用写入时的最后一道防线，服务层每个校验都在这里有一份
    # 对应的表达。

    __table_args__ = (
        CheckConstraint(
            "sequence >= 1",
            name="ck_tool_processor_sequence",
        ),
        CheckConstraint(
            "execution_mode IN ('SYNC', 'ASYNC')",
            name="ck_tool_processor_execution_mode",
        ),
        CheckConstraint(
            "hook IN ('BEFORE', 'AFTER', 'AROUND')",
            name="ck_tool_processor_hook",
        ),
        CheckConstraint(
            "jsonb_typeof(config) = 'object'",
            name="ck_tool_processor_config_object",
        ),
        # AROUND 必然同步：一个不被 await 的包装器无法决定何时、
        # 是否调用工具。服务层先拦一道（抛 ConflictError → 409），
        # 这道拦的是绕过应用的写入。
        CheckConstraint(
            "hook <> 'AROUND' OR execution_mode = 'SYNC'",
            name="ck_tool_processor_around_sync",
        ),
        # 一个 Tool 的链里每个位置只能有一行。
        Index(
            "uq_tool_processor_tool_sequence",
            "tool_id",
            "sequence",
            unique=True,
        ),
        Index(
            "idx_tool_processor_tool_enabled_sequence",
            "tool_id",
            "enabled",
            "sequence",
        ),
        Index(
            "idx_tool_processor_processor_key",
            "processor_key",
        ),
        # 部分索引：只覆盖"启用中的异步处理器"。
        # 异步处理器的运维查询（"这个 Tool 有哪些旁路"）不会去
        # 扫同步的那些，因此这个索引比全表索引小得多。
        Index(
            "idx_tool_processor_async",
            "tool_id",
            "sequence",
            postgresql_where=text(
                "execution_mode = 'ASYNC' AND enabled = TRUE"
            ),
        ),
    )
