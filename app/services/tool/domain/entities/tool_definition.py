"""
Domain entity for ToolDefinition.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.enum import SourceState, SourceType

__all__ = ["ToolDefinition"]


class ToolDefinition(BaseModel):
    """
    一个 Tool 的**代码侧事实快照**。

    这一行的所有内容都由代码产生 —— 见 tool.yaml 的 ToolManifest：
    那里的字段无一例外标着 readOnly，没有任何写入口。

    为什么这么设计
    --------------
    工具是**代码的一部分**：方法签名、docstring、参数模型都在代码里。
    如果允许人工改这里的字段，代码一升级就会与库里的记录对不上，
    而且没人知道该信哪一份。

    所以：definition 归代码，说明文字归人。
    人工想改描述，改的是 ToolDocumentation 的 override 字段
    （见 tool_documentation.py），同步永远不会碰它。

    这条分工是"自注册 + 同步"能安全整行覆盖的前提。
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    # ========================================================
    # Identity
    # ========================================================

    tool_id: UUID = Field(
        description="Tool 定义的全局唯一 ID。",
    )

    tool_key: str = Field(
        description=(
            "Tool 的稳定业务标识，例如 document.list。"
            "由代码声明，跨版本不变 —— 同步按它比对，"
            "不用 tool_id 是因为后者在重新发现时会变。"
        ),
    )

    name: str = Field(
        description="展示名。默认取方法名，可由声明覆盖。",
    )

    # ========================================================
    # Source
    # ========================================================

    source_type: SourceType = Field(
        description="来源种类。本期只产生 LOCAL_METHOD。",
    )

    source_ref: str | None = Field(
        default=None,
        description=(
            "来源对象，例如 service/module/mcp-server/openapi spec。"
            "LOCAL_METHOD 下是所属业务模块名（如 document）。"
        ),
    )

    entrypoint: str | None = Field(
        default=None,
        description=(
            "实际方法入口，形如 python.module:Class.method。"
            "**只用于展示与审计** —— 执行时不 import 这个字符串，"
            "而是持有代码里那个描述符对象（见 registry/decorator.py）。"
        ),
    )

    # ========================================================
    # Schema
    # ========================================================

    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="代码产生的输入 schema（JSON Schema）。",
    )

    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="代码产生的输出 schema。返回注解不是模型时为 None。",
    )

    source_docstring: dict[str, Any] | None = Field(
        default=None,
        description=(
            "代码 docstring 的原始/解析结果。"
            "保留成结构而不是一整段字符串，是为了让"
            "参数级说明能与 document 里的 parameter_path 对上。"
        ),
    )

    # ========================================================
    # Change Detection
    # ========================================================

    manifest_hash: str = Field(
        description=(
            "上面这些字段的稳定摘要。同步靠它判断"
            "created / updated / unchanged。"
        ),
    )

    source_state: SourceState = Field(
        default=SourceState.ACTIVE,
        description="ACTIVE=代码里还在；MISSING=代码里已找不到。",
    )

    # ========================================================
    # Audit
    # ========================================================

    discovered_at: datetime = Field(
        description="首次发现时间。",
    )

    last_synced_at: datetime = Field(
        description="最近一次同步时间。",
    )

    created_at: datetime = Field(
        description="创建时间。",
    )

    updated_at: datetime = Field(
        description="最后更新时间。",
    )
