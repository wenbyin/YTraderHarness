from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base

__all__ = ["ToolDefinitionState"]


class ToolDefinitionState(Base):
    """
    一个 Tool 的**运行时开关**：有行 = 显式禁用，无行 = 启用。

    为什么单独一张表
    ---------------
    `tool_definition` 是代码侧事实，同步会整行覆盖它；这个开关是
    人写的运行时状态，两者生命周期不同。放进同一行意味着同步要
    特意绕开那一列 —— 而"忘了绕开"的症状是"禁用过的工具同步一次
    又自己开了"，且不会有任何报错。

    为什么**没有** enabled 列
    ------------------------
    判据只有"在不在"：插入即禁用，删除即恢复。这样"只过滤显式
    声明禁用"这条语义是表结构本身保证的，不需要读写两边各自记得
    判断一个布尔值。

    为什么**没有**外键
    -----------------
    `remove_missing=true` 会物理删除 `tool_definition` 的行。若这里
    ON DELETE CASCADE，同步一次就会把人工设的禁用悄悄清掉 —— 而
    fail-closed 才是开关该有的方向。代价是可能留下指向 MISSING /
    已删除工具的孤儿行；键是稳定的 `tool_key`，同名工具以后回来
    仍然处于禁用状态，这正是想要的。

    ⚠️ `tool_key` 而不是 `tool_id`：`tool_key` 是同步与对外契约
       认的稳定业务标识（见 tool_definition.tool_key 的注释）。
    """

    __tablename__ = "tool_definition_state"

    tool_key: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        comment=(
            "被显式禁用的 Tool 的稳定业务标识，"
            "与 tool_definition.tool_key 对应。"
        ),
    )

    disabled_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="最后执行禁用的人。",
    )

    disabled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="禁用时间。",
    )
