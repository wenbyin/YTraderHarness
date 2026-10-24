"""阶段 3：SKILL_LOADER —— task 的 skill 与 capabilities。

它接在 TaskResolver 后面，只在 `task_mode` 为真时执行：

    1. 从 current_task.metadata.skill_key 加载 SkillSnapshot
    2. 用统一结构渲染 system prompt（instructions + 能力清单）
    3. effective = skill_capabilities ∪ task_capabilities
    4. capability code -> tool_keys，写进 context_policy
    5. 没有 skill / 没有 resolver 时降级，但**不回落到主 agent 工具**

为什么 system_prompt 与 tool_keys 都在这里切换
--------------------------------------------
TaskResolver 只告诉模型“你在执行任务”，但模型的角色和工具仍可能
是主 agent 的。这里把两样一起换掉，任务执行才真正成立：

    system_prompt  skill.instructions（或通用任务提示词）
    tool_keys      effective capabilities 解析出来的工具

⚠️ 能力不是授权
------------
这里只决定“给模型看哪些工具”。真正的权限仍在 ToolExecutor 的
checkauth / approval 链上。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.agent.context.skills import (
    TASK_EXECUTION_FALLBACK_PROMPT,
    SkillSnapshot,
    capabilities_of,
    skill_key_of,
)
from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..spec import ContextProcessorSpec, StateBinding

__all__ = ["SkillLoaderProcessor"]

logger = logging.getLogger(__name__)


class SkillLoaderProcessor(ContextProcessor):
    """按 task.skill_key 切换 system prompt 与 tool_keys。"""

    spec = ContextProcessorSpec(
        name="skill_loader",
        order=16,
        stage=StageId.SKILL_LOADER,
        when="task_mode",
        inputs={
            "current_task": StateBinding(
                "current_task", required=False
            ),
        },
        optional=True,
        description=(
            "加载当前 task 的 skill：渲染 system prompt，并把 "
            "skill + task capabilities 解析成 tool_keys。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        task = inputs.get("current_task")

        if not isinstance(task, Mapping) or not task:
            return context

        deps = dependencies_of(context)

        skill_key = skill_key_of(task)
        task_capabilities = capabilities_of(task)

        snapshot: SkillSnapshot | None = None

        if skill_key:

            try:
                snapshot = await deps.resolved_skill_source().load(
                    skill_key
                )
            except Exception as exc:  # noqa: BLE001 - 可选阶段，降级
                logger.warning(
                    "加载 skill %r 失败（%s: %s），使用通用任务提示词。",
                    skill_key,
                    type(exc).__name__,
                    exc,
                )
                context.warn(
                    f"skill {skill_key!r} 加载失败：{type(exc).__name__}"
                )

        if snapshot is None:
            prompt = TASK_EXECUTION_FALLBACK_PROMPT

            if skill_key:
                context.warn(
                    f"skill {skill_key!r} 未找到，使用通用任务提示词"
                )
            else:
                context.warn("task 没有 skill_key，使用通用任务提示词")

            effective = list(dict.fromkeys(task_capabilities))

        else:
            prompt = snapshot.render_prompt()

            effective = list(dict.fromkeys([
                *(item.code for item in snapshot.capabilities),
                *task_capabilities,
            ]))

        # system prompt：TaskResolver 已经加了任务 section，这里换角色。
        context.set("context_policy.system_prompt", prompt)
        context.set("context_policy.include_capabilities", True)

        if context.policy is not None:
            context.policy = context.policy.with_(
                system_prompt=prompt,
                include_capabilities=True,
            )

        # capabilities -> tool_keys。失败时给空工具集，而不是回落到
        # 主 agent 的 exposed_tools —— 那会把主对话的工具带进任务。
        tool_keys: list[str] = []

        try:
            tool_keys = list(
                await deps.resolved_capability_resolver().tool_keys(
                    effective
                )
            )
        except Exception as exc:  # noqa: BLE001 - 可选阶段，降级
            logger.warning(
                "解析 task capabilities 失败（%s: %s），本次任务无工具。",
                type(exc).__name__,
                exc,
            )
            context.warn(
                f"capability 解析失败：{type(exc).__name__}"
            )

        context.set("context_policy.tool_keys", tool_keys)

        if context.policy is not None:
            context.policy = context.policy.with_(
                tool_keys=tuple(tool_keys),
            )

        if not tool_keys:
            # capability 处理器在 tool_keys 为空时会回落到 state 的
            # exposed_tools；任务模式必须显式清空，否则会拿到主 agent
            # 的工具集。
            context.set("exposed_tools", [])

        context.set(
            "skill",
            {
                "skill_key": skill_key,
                "loaded": snapshot is not None,
                "effective_capabilities": effective,
                "tool_keys": tool_keys,
                "snapshot": (
                    snapshot.as_state() if snapshot is not None else None
                ),
            },
        )

        # 路由 / tools 节点读的是**图 state**，不是本次 ContextBuild 的
        # overlay。把 task 的工具 key 交给 agent 节点回写，否则模型看到
        # 了 skill 工具、tools 节点却按主 agent 的 tool_keys 拒绝它。
        updates = dict(
            context.metadata.get("context_state_updates") or {}
        )
        updates["task_tool_keys"] = list(tool_keys)
        context.metadata["context_state_updates"] = updates

        return context
