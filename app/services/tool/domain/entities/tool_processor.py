"""
Tool 处理器挂载的领域实体。

    ToolProcessor      一个 Tool 的链上的一行

处理器**实现**不是实体 —— 它归代码（见
app/agent/tools/processor/base.py），由注册表反射发现，从不落库。
本实体只描述"哪个 Tool 挂了哪个实现、什么参数、排在第几"。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import ExecutionMode, ProcessorHook

__all__ = ["ToolProcessor"]


class ToolProcessor(BaseModel):
    """
    一条挂载记录。

    枚举字段用 ProcessorHook / ExecutionMode 而不是裸 str ——
    ORM 那边存的是字符串（见 models/sqlalchemy/tool_processor.py），
    `from_attributes=True` 会让 pydantic 按值转换，于是服务层拿到的
    是**枚举**，比较与排序都不必自己造一套字符串常量。
    转换失败（库里存了非法值，理论上被 CHECK 挡住）会在这里炸，
    而不是在某个 if 判断里静默走错分支。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    processor_id: UUID = Field(
        description="挂载记录 ID。",
    )

    tool_id: UUID = Field(
        description="所属 Tool 定义 ID。",
    )

    processor_key: str = Field(
        description=(
            "代码里注册的实现键，例如 cache_interceptor。"
            "它可能指向一个**已经不存在**的实现 —— 那种行照样保留，"
            "只是读链时不带 processor 字段。"
        ),
    )

    sequence: int = Field(
        ge=1,
        description=(
            "在链里的位置，从 1 开始连续。服务层每次写完都会重编号 "
            "1..N，因此它不保证等于写请求里传来的值。"
        ),
    )

    execution_mode: ExecutionMode = Field(
        description="SYNC=参与结果；ASYNC=旁路不等。",
    )

    hook: ProcessorHook = Field(
        description="BEFORE / AFTER / AROUND。",
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
        description="该实现自己的业务参数。",
    )

    enabled: bool = Field(
        default=True,
        description="关掉的行留在链里占位，执行时跳过。",
    )

    created_at: datetime = Field(
        description="创建时间。",
    )

    updated_at: datetime = Field(
        description="最后更新时间。",
    )
