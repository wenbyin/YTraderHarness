"""
阶段 7：CAPABILITY_LOADER —— 暴露本次可用的工具（规格 §二十四）。

⚠️ 它只决定"给模型看哪些工具"
--------------------------
不决定权限，也不执行（规格 §二十二）。权限与执行属于 Tool 模块 /
Tool Node / 审批拦截器 —— Context Builder 越界那一层的后果是
"看起来在管权限，其实是把手上的列表照抄了一份"。

数据从哪来
---------
    配了 ToolExposureSource -> 按 `context_policy.tool_keys` 解析出工具对象
    没配                    -> 用 state 里绑定的 `exposed_tools`

⚠️ 这是**必需**处理器（不是 optional）
----------------------------------
"策略说了要暴露这些工具，但没有任何东西能解析它们"是一个**承诺兑现
不了**的接线错误：模型会以为有工具、而 Tool Node 一个都找不到。
因此这一条 FAILED（与 tools.py 里"不返回假 ToolMessage"同一条立场）。

空工具集是完全合法的（"这次没有工具" —— 比如纯问答的调用）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..messages import tool_name
from ..spec import ContextProcessorSpec, StateBinding

__all__ = ["CapabilityProcessor"]


class CapabilityProcessor(ContextProcessor):
    """把本次可用的工具放进上下文。"""

    spec = ContextProcessorSpec(
        name="capability",
        order=70,
        stage=StageId.CAPABILITY_LOADER,
        when="context_policy.include_capabilities",
        inputs={
            "exposed_tools": StateBinding("exposed_tools", required=False),
            "tool_keys": StateBinding("context_policy.tool_keys", required=False),
            # 人工写的工具说明覆盖（预载在 Agent Builder 那一轮，
            # 见 agent_builder_node._tool_overrides_for_keys）。形状与
            # ToolQueryService.tool_documentation_overrides 的返回一致。
            # 不在这里查库：本处理器每次模型调用都跑。
            "tool_overrides": StateBinding(
                "context_policy.tool_overrides", required=False
            ),
            "conversation_id": StateBinding("conversation_id", required=False),
        },
        description="暴露本次调用可用的工具（权限与执行不在这里）。",
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        deps = dependencies_of(context)

        tool_keys = _keys(inputs.get("tool_keys"))

        if deps.tool_source is not None and tool_keys:

            tools = list(
                await deps.tool_source.resolve(
                    tool_keys=tool_keys,
                    conversation_id=inputs.get("conversation_id"),
                    overrides=_overrides(inputs.get("tool_overrides")),
                )
            )

        elif tool_keys and deps.tool_source is None:

            # 策略点名要这些工具，却没有任何解析入口 —— 承诺兑现不了。
            # 静默给一个空列表会让模型以为"这些工具不存在"，而真正的问题
            # 是接线漏了 source（或者 tool_keys 写错了地方）。
            raise LookupError(
                f"context_policy.tool_keys 声明了 {tool_keys}，"
                "但没有配置 ToolExposureSource，无法把它们解析成工具对象。\n"
                "两条路：在 ContextDependencies 里注入一个 tool_source，"
                "或直接把工具对象放进图 state 的 exposed_tools。"
            )

        else:

            raw = inputs.get("exposed_tools")

            tools = (
                list(raw)
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes))
                else []
            )

        # 去重（同一个工具在两条途径里都出现是常见的手误）。
        unique: list[Any] = []
        seen: set[str] = set()

        for tool in tools:

            name = tool_name(tool)

            if name in seen:
                continue

            seen.add(name)
            unique.append(tool)

        context.add_tools(unique)

        context.set(
            "capabilities",
            {
                "tool_count": len(unique),
                "tool_names": sorted(seen),
                "requested_keys": tool_keys,
                "source": (
                    "tool_source"
                    if deps.tool_source is not None and tool_keys
                    else "state"
                ),
            },
        )

        return context


# ============================================================
# internals
# ============================================================


def _overrides(raw: Any) -> dict[str, Any] | None:
    """
    从 state 通道里取出的覆盖，收成"源的 resolve 能吃的形状"。

    `context_policy` 会被 checkpoint、也可能由调用方手写，所以这里
    只做**形状守卫**，不做内容校验（深层是 `for_wire` 的事）：不是一个
    mapping 就当作"没有覆盖"。

    认不出的值不抛：这一格坏了最多是工具说明回落到代码侧，而为此
    中断整轮对话（模型一个工具都拿不到）显然更糟 —— 与
    `_tool_keys_for_capabilities` 的降级立场一致。
    """

    if not isinstance(raw, Mapping):
        return None

    return {
        str(key): value
        for key, value in raw.items()
        if isinstance(value, Mapping)
    } or None


def _keys(raw: Any) -> list[str]:

    if isinstance(raw, str):
        return [raw] if raw.strip() else []

    if isinstance(raw, Sequence):

        return [
            str(item) for item in raw
            if isinstance(item, str) and item.strip()
        ]

    return []
