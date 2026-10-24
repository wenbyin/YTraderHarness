"""Planner 的 Context Builder：resolver + summary + history + 工具 + 预算。

两段策略
--------
planner 子图有两个模型调用节点，共用这个 builder，用 context_policy
区分：

    ReAct 探索段
        include_capabilities=True + tool_keys -> 模型可调只读/运行时工具

    task_generator 段
        include_capabilities=False + tool_keys=[] -> 严格 JSON，无工具

为什么保留同一套阶段
--------------------
阶段顺序、gate 校验、绑定解析、快照格式都是“上下文怎么长出来”的公共
契约。两段只是 policy 不同，不需要两套 builder。

    CONTEXT_RESOLVER   策略 / 预算
    SUMMARY_LOADER     已有摘要（需要时才生成）
    HISTORY_SELECTOR   最近消息 + 当前用户输入
    CAPABILITY_LOADER  按 tool_keys 解析工具（planner 探索段）
    TOKEN_BUDGETER     裁剪超预算的历史 / 工具结果
    PROMPT_ASSEMBLER   产出 ModelRequest

入口与出口是 required；其余由 context_policy 的开关决定。
"""

from __future__ import annotations

from typing import ClassVar

from app.common.context import StageId

from .agent_invoke import ProcessorContextBuilder
from .base import PlannerContextBuilder

__all__ = ["PlannerProcessorContextBuilder"]


_STAGES: frozenset[StageId] = frozenset({
    StageId.CONTEXT_RESOLVER,
    StageId.SUMMARY_LOADER,
    StageId.HISTORY_SELECTOR,
    StageId.CAPABILITY_LOADER,
    StageId.TOKEN_BUDGETER,
    StageId.PROMPT_ASSEMBLER,
})

_REQUIRED: frozenset[StageId] = frozenset({
    StageId.CONTEXT_RESOLVER,
    StageId.PROMPT_ASSEMBLER,
})


class PlannerProcessorContextBuilder(
        ProcessorContextBuilder,
        PlannerContextBuilder,
):
    """Planner 模型调用的上下文（窄输入）。"""

    version = "1.1.0"
    display_name = "Planner 上下文构建"
    description = (
        "组装 planner 的摘要、历史、当前用户输入；探索段按 "
        "context_policy.tool_keys 解析工具，task_generator 段关闭工具。"
        "不注入画像 / 记忆 / 知识。"
    )

    enabled_stages: ClassVar[frozenset[StageId]] = _STAGES
    required_stages: ClassVar[frozenset[StageId]] = _REQUIRED
