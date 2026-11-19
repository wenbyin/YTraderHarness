from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


class ToolPresentation(Base):
    """
    一个工具的**展示描述**：有行 = 显示，无行 = 不显示（默认）。

    为什么单独一张表
    ---------------
    `tool_definition_state` 的语义是「有行 = 禁用」，默认启用；
    这里恰好相反 —— 默认隐藏，有行才显示。两种默认方向塞进同一行，
    必然要加一个布尔列，于是「看这一列还是看这一行在不在」变成
    读写两边的共同负担（见 tool_definition_state.py 的 docstring）。

    另外两条写路径也不同：下线是 DELETE 一行；显示是 upsert 一整份
    描述符。合并之后，「恢复上线」会顺手删掉展示配置。

    为什么没有外键
    -------------
    与 tool_definition_state 同一条理由：`remove_missing=true` 会物理
    删除 definition 行。这里若 ON DELETE CASCADE，同步一次就把人工配的
    展示悄悄清掉。键用稳定的 `tool_key`，工具以后回来时配置还在。

    ⚠️ 默认隐藏是**产品默认值**，不是缺失值
       切默认之前必须先把现有可见工具的 presenter 灌好，
       否则上线瞬间所有工具事件从聊天窗消失。
    """

    __tablename__ = "tool_presentation"

    tool_key: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        comment="稳定业务标识，与 tool_definition.tool_key 对应。",
    )

    renderer: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="inline / tips / card / table —— 前端组件注册表的键。",
    )

    display_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="给人看的标题；为空时前端回落到工具 name。",
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="点路径模板，如「已查询 {input.symbol} 共 {output.count} 条」。",
    )

    show_input: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="false 时事件载荷里不出现 input 键。",
    )

    show_output: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="false 时事件载荷里不出现 output 键。",
    )

    props: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "renderer 专属内容。共享键：fields（JSON Pointer 白名单）、"
            "max_bytes（截断提示）。"
        ),
    )

    updated_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="最后修改人。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后修改时间。",
    )

    # ========================================================
    # Constraints
    # ========================================================

    __table_args__ = (
        CheckConstraint(
            "renderer IN ('inline', 'tips', 'card', 'table')",
            name="ck_tool_presentation_renderer",
        ),
        CheckConstraint(
            "jsonb_typeof(props) = 'object'",
            name="ck_tool_presentation_props_object",
        ),
    )
