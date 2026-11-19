"""
工具展示描述符的对外契约。

对应 tool.yaml：
    ToolPresentation / ToolPresentationState
    PUT    /tools/{tool_id}/presentation
    DELETE /tools/{tool_id}/presentation

⚠️ 这里只说**显示什么**
-----------------------
没有时机、合并、消失、错误升级之类的字段 —— 那些属于客户端。
后端唯一决定的是：这条工具调用出不出现在事件流里、以什么形态
出现、带哪些字段。

⚠️ 没有 presenter 行 = 不显示
-----------------------------
默认值是隐藏，所以「开启显示」必须是显式的。清除走 DELETE，
不存在一个 renderer=hidden 的取值（见 ToolRenderer 的说明）。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from app.common.schemas import APIModel

from .enum import ToolRenderer

__all__ = [
    "ToolPresentation",
    "ToolPresentationState",
]


class ToolPresentation(APIModel):
    """
    一个工具的展示描述符。

    字段与 tool.yaml 的 ToolPresentation 逐字段对应。
    """

    renderer: ToolRenderer = Field(
        description="渲染形态，前端组件注册表的键。",
    )

    display_name: str | None = Field(
        default=None,
        description=(
            "给人看的标题。为空时回落到工具的 name；"
            "永远不要把 tool_key 直接露给用户。"
        ),
    )

    summary: str | None = Field(
        default=None,
        description=(
            "短句模板，投影时用输入/输出填空。占位符是点路径，"
            "形如 {input.symbol} / {output.count}；"
            "解析不到的路径渲染成空串，不让模板拖垮一次调用。"
        ),
    )

    show_input: bool = Field(
        default=False,
        description="false 时事件载荷里不出现 input 键（不是置 null）。",
    )

    show_output: bool = Field(
        default=False,
        description="false 时事件载荷里不出现 output 键（不是置 null）。",
    )

    props: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "renderer 专属内容。所有 renderer 共享四个键："
            "output_path（从结果里取一段当 output）、"
            "max_rows（数组最多带几条）、"
            "fields（JSON Pointer 白名单）与 max_bytes（截断提示）。"
            "其余键由 renderer 自己解释，客户端不认识时忽略。"
        ),
    )

    @field_validator("props")
    @classmethod
    def _validate_props(
            cls,
            value: dict[str, Any],
    ) -> dict[str, Any]:
        """
        只校验两个共享键的形状，其余原样放行。

        ⚠️ 校验太严会把「renderer 自己的参数」也锁死 ——
        那种参数只能由前端组件定义，后端无权规定它的形状。
        """

        fields = value.get("fields")

        if fields is not None:

            if (
                    not isinstance(fields, list)
                    or not all(isinstance(item, str) for item in fields)
            ):
                raise ValueError(
                    "props.fields 必须是 JSON Pointer 字符串数组。"
                )

        output_path = value.get("output_path")

        if output_path is not None:

            if (
                    not isinstance(output_path, str)
                    or not output_path.startswith("/")
            ):
                raise ValueError(
                    "props.output_path 必须是 JSON Pointer 字符串"
                    "（以 / 开头，例如 /bars）。"
                )

        max_rows = value.get("max_rows")

        if max_rows is not None:

            if (
                    not isinstance(max_rows, int)
                    or isinstance(max_rows, bool)
                    or max_rows < 0
            ):
                raise ValueError(
                    "props.max_rows 必须是非负整数。"
                )

        max_bytes = value.get("max_bytes")

        if max_bytes is not None:

            if (
                    not isinstance(max_bytes, int)
                    or isinstance(max_bytes, bool)
                    or max_bytes < 0
            ):
                raise ValueError(
                    "props.max_bytes 必须是非负整数。"
                )

        return value


class ToolPresentationState(APIModel):
    """
    写入 / 清除之后的状态。

    presented=False 时 presentation 为 None —— 前端只看一个布尔
    就能决定开关，不必区分「没配」与「配了空对象」。
    """

    tool_key: str = Field(description="稳定业务标识。")

    presented: bool = Field(
        description="是否有 presenter 行；false = 该工具的调用不进流。",
    )

    presentation: ToolPresentation | None = Field(
        default=None,
        description="当前描述符；presented=false 时为空。",
    )
