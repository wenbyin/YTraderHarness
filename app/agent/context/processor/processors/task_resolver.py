"""阶段 2：TASK_RESOLVER —— 注入任务执行上下文或任务决策提示。

它解决两个问题
------------
    有 current_task
        主 agent 正在执行某个任务：告诉它负责人、任务内容、任务列表
        进度，并把任务指令作为**当前这一轮**的输入。

    没有 current_task，但任务列表还有待办
        task_listener 把「要不要开子代理」的决策交给 main agent：
        提示它当前有无并行、可以自己顺序处理或调用 create_team。
        这一轮**不执行任务内容**。

⚠️ 两种情况的输入不同
-------------------
    有 current_task     required HumanMessage 是完整任务指令
    没有 current_task   required HumanMessage 是 team_decision

SkillLoader 仍然只在 `task_mode`（有 current_task）时执行；决策回合
不需要加载某个任务的 skill。

⚠️ gate 是 `task_context`，不是 `task_mode`
----------------------------------------
`task_context` 由 CONTEXT_RESOLVER 派生：有 current_task 或有待办任务。
它比 `task_mode` 宽一档，正好覆盖「还没 claim、等 main agent 决策」的回合。

⚠️ 它只读、只拼提示词
-------------------
不查库、不加载 skill、不 claim 任务。claim 仍在 task_listener。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.context.skills import (
    TASK_EXECUTION_FALLBACK_PROMPT,
    capabilities_of,
    skill_key_of,
)
from app.agent.plan.checks import pending_tasks, ready_tasks
from app.agent.plan.task_prompt import (
    render_task_header,
    task_instruction_message,
    task_results_message,
    team_decision_message,
)
from app.agent.tools.task_tool import CREATE_TEAM_TOOL_KEY
from app.common.context import StageId

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..spec import ContextProcessorSpec, StateBinding

__all__ = ["TaskResolverProcessor"]


class TaskResolverProcessor(ContextProcessor):
    """有 current_task 或待办任务时注入任务相关提示。"""

    spec = ContextProcessorSpec(
        name="task_resolver",
        order=15,
        stage=StageId.TASK_RESOLVER,
        # task_context = 有 current_task 或有待办任务，由 CONTEXT_RESOLVER 派生。
        when="task_context",
        inputs={
            "current_task": StateBinding(
                "current_task", required=False
            ),
            "tasks": StateBinding("tasks", required=False),
            "plan": StateBinding("plan", required=False),
            "tool_keys": StateBinding(
                "context_policy.tool_keys", required=False
            ),
            "synthesis_pending": StateBinding(
                "synthesis_pending", required=False
            ),
        },
        optional=True,
        description=(
            "有 current_task 时注入任务说明与进度；没有 current_task 但"
            "有待办任务时注入「自己处理 / 开子代理」的决策提示；"
            "全部任务完成时注入任务结果，供 main agent 汇总。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        current_task = inputs.get("current_task")
        tasks = _tasks(inputs.get("tasks"))
        plan = inputs.get("plan")

        if not isinstance(plan, Mapping):
            plan = {}

        if isinstance(current_task, Mapping) and current_task:
            return self._current_task(
                context,
                task=current_task,
                tasks=tasks,
            )

        if inputs.get("synthesis_pending"):
            return self._synthesis(
                context,
                plan=plan,
                tasks=tasks,
            )

        return self._decision(
            context,
            plan=plan,
            tasks=tasks,
            tool_keys=_tool_keys(inputs.get("tool_keys")),
        )

    # --------------------------------------------------------

    def _current_task(
            self,
            context: ContextBuildContext,
            *,
            task: Mapping[str, Any],
            tasks: list[Mapping[str, Any]],
    ) -> ContextBuildContext:

        task_id = str(task.get("id") or "").strip()
        skill_key = skill_key_of(task)
        capabilities = capabilities_of(task)
        owner = _owner(task)
        position, total = _position(task, tasks)

        # system 里只放短的“当前在做什么”；完整任务数据放在下面的
        # required HumanMessage 里。
        header = render_task_header(
            task,
            owner=owner,
            position=position,
            total=total,
        )

        # 先把 system 切到任务模式；SkillLoader 随后会用 skill 的
        # instructions 覆盖它。这样即使 skill 加载失败，主 agent 也不会
        # 继续用“回复用户”的职责提示词。
        context.set(
            "context_policy.system_prompt",
            TASK_EXECUTION_FALLBACK_PROMPT,
        )

        if context.policy is not None:
            context.policy = context.policy.with_(
                system_prompt=TASK_EXECUTION_FALLBACK_PROMPT,
            )

        context.add_section(
            "task",
            "当前任务",
            header,
            source_ids=((task_id,) if task_id else ()),
        )

        # required 消息在 assembled_messages() 里排在 history 之后，
        # 让任务指令成为“当前这一轮”的输入。
        context.add_required_message(
            task_instruction_message(
                task,
                owner=owner,
                position=position,
                total=total,
            )
        )

        context.set(
            "task",
            {
                "task_id": task_id,
                "skill_key": skill_key,
                "capabilities": capabilities,
                "owner": owner,
                "position": position,
                "total": total,
                "kind": "task_instruction",
                "message_type": "task_instruction",
            },
        )

        return context

    # --------------------------------------------------------

    def _synthesis(
            self,
            context: ContextBuildContext,
            *,
            plan: Mapping[str, Any],
            tasks: list[Mapping[str, Any]],
    ) -> ContextBuildContext:
        """所有任务已完成：把结果注入 main agent 的汇总回合。"""

        # system 里只放短提示；完整结果在 required HumanMessage 里，
        # 避免同一份长文本在 system + 当前输入里重复两遍。
        context.add_section(
            "task",
            "任务结果",
            "所有任务已完成。完整结果见当前输入。",
            source_ids=tuple(
                str(task.get("id") or "") for task in tasks
                if task.get("id")
            ),
        )

        context.add_required_message(
            task_results_message(tasks, plan)
        )

        context.set(
            "task",
            {
                "kind": "synthesis",
                "task_ids": [
                    str(task.get("id") or "") for task in tasks
                    if task.get("id")
                ],
                "message_type": "task_results",
            },
        )

        return context

    # --------------------------------------------------------

    def _decision(
            self,
            context: ContextBuildContext,
            *,
            plan: Mapping[str, Any],
            tasks: list[Mapping[str, Any]],
            tool_keys: list[str],
    ) -> ContextBuildContext:

        pending = pending_tasks(tasks)

        if not pending:
            return context

        ready = ready_tasks(tasks)
        can_create_team = CREATE_TEAM_TOOL_KEY in tool_keys

        context.add_section(
            "task",
            "待办任务",
            (
                "当前没有正在执行的任务，但计划里还有待办任务。"
                "请判断是否有并行收益：自己按顺序处理，或开启子代理。"
            ),
            source_ids=tuple(
                str(task.get("id") or "") for task in ready
                if task.get("id")
            ),
        )

        context.add_required_message(
            team_decision_message(
                plan=plan,
                tasks=tasks,
                pending_count=len(pending),
                ready_count=len(ready),
                can_create_team=can_create_team,
            )
        )

        context.set(
            "task",
            {
                "kind": "team_decision",
                "pending_count": len(pending),
                "ready_count": len(ready),
                "can_create_team": can_create_team,
                "message_type": "team_decision",
            },
        )

        return context


# ============================================================
# internals
# ============================================================


def _tasks(raw: Any) -> list[Mapping[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []

    return [
        task for task in raw
        if isinstance(task, Mapping)
    ]


def _tool_keys(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []

    return [str(key) for key in raw]


def _owner(task: Mapping[str, Any]) -> str | None:
    metadata = task.get("metadata")

    if not isinstance(metadata, Mapping):
        return None

    claim = metadata.get("claim")

    if not isinstance(claim, Mapping):
        return None

    owner = str(claim.get("owner") or "").strip()

    return owner or None


def _position(
        task: Mapping[str, Any],
        tasks: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """
    当前任务在任务列表里的位置（从 1 开始）与总数。

    找不到（旧 state / 任务列表还没回写）时返回 (0, total)，渲染层会
    省略进度行，而不是给出一个错误的位置。
    """

    task_id = str(task.get("id") or "")

    for index, item in enumerate(tasks, start=1):

        if str(item.get("id") or "") == task_id:
            return index, len(tasks)

    return 0, len(tasks)
