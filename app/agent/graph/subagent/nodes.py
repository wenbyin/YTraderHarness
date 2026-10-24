"""子代理子图的三个节点。

    task_listener   完成上一个任务（写回任务板）→ 领下一个 ready task
    llm_node        复用主图 run_agent_turn，换 SUB_AGENT builder 与
                    subagent 模型档位
    tool_node       复用主图 execute_tool_branch，换自己的观测身份

⚠️ 这里是**另一张图的节点**，不是主图节点的复制
---------------------------------------------
执行 / 审批 / 记账依然只走 `ToolExecutor` + processor chain；上下文
构建依然只走同一套 13 阶段流水线。本模块只做三件与图有关的事：

    1. 读写 SubAgentState 的通道（主图节点读写 ReactState）
    2. 任务领取 / 完成的**外部副作用**（任务板）
    3. 观测身份（llm_node / tool_node）

为什么 listener 在 worker 里
--------------------------
worker 连续吃任务时，`listener -> llm_node -> tool_node -> llm_node ->
listener` 是一条**有限**回边：每轮只 claim 一次，没有 ready task 就
END。节点里不写 `while True` —— 那会让当前 super-step 永远不结束，
父图的 join 也永远等不到它。

⚠️ 任务之间清空 messages
-----------------------
一个 worker 连续执行两个任务时，第二个任务不能看到第一个任务的工具
结果与中间推理（否则上下文污染、预算被吃）。listener 在完成上一任务
并 claim 到下一个任务时，用 `RemoveMessage` 清掉旧 transcript，只留下
新任务指令。`summary` / `memory` 通道不受影响。

⚠️ worker 不发 plan 事件
-----------------------
主图 listener 会 `publish_plan_cleared`（无任务时收掉前端计划面板）。
worker 不碰 `task_events`：计划面板属于父图，worker 只写任务板。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agent.context.skills import skill_key_of
from app.agent.graph.react.agent_node import run_agent_turn
from app.agent.graph.react.context import ReactContext
from app.agent.graph.react.tools import command_update, execute_tool_branch
from app.agent.plan.listener import (
    TERMINAL,
    WORKING,
    TaskListener,
    build_task_subscriber,
    complete_task,
    status_state,
    task_position,
)
from app.agent.plan.task_prompt import (
    last_ai_text,
    task_instruction_message,
)
from app.infra.graph import node_span

from .catalog import (
    AGENT_TYPE,
    CONTEXT_BUILDER_TYPE,
    LLM_NODE,
    MODEL_ROLE,
    NODE_SPECS,
    SUBAGENT_TOOL_SCOPE_NAMESPACE,
)
from .state import SubAgentState, SubAgentToolPayload

__all__ = ["llm_node", "task_listener_node", "tool_node"]

logger = logging.getLogger(__name__)

_SPEC_LISTENER = next(
    spec for spec in NODE_SPECS if spec.node_id == "task_listener"
)
_SPEC_LLM = next(spec for spec in NODE_SPECS if spec.node_id == LLM_NODE)
_SPEC_TOOL = next(spec for spec in NODE_SPECS if spec.node_id == "tool_node")

#: task history / metadata 里保留的摘要上限。完整结果进 artifact。
_SUMMARY_LIMIT = 4000

#: 允许从上下文构建回写进 **SubAgentState** 的键。
#:
#: 与主图 `agent_node._STATE_UPDATE_KEYS` 同形但**不同目标**：主图写
#: ReactState，这里写 SubAgentState。不要把这些键写回父图。
_STATE_UPDATE_KEYS: tuple[str, ...] = (
    "summary",
    "memory",
    # SkillLoader 算出的任务工具 key；子图 routing 与 tool_node 都读它。
    "task_tool_keys",
)


# ============================================================
# task_listener
# ============================================================


async def task_listener_node(
        state: SubAgentState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """
    完成 current_task，再 claim 下一个 ready task。

    数据源由 `build_task_subscriber` 决定：worker 恒以「任务板优先」
    的姿态调用（`subagent_mode="on"`）；Redis 未装配时退回 state ——
    这让子图能脱离 Redis 做冒烟，而生产上任务板是权威。

    Returns
    -------
    dict
        `current_task` 为空时，`route_after_listener` 会走 END。
    """

    async with node_span(
            runtime,
            node_id=_SPEC_LISTENER.node_id,
            label=_SPEC_LISTENER.label,
            group=_SPEC_LISTENER.group,
            sequence=_SPEC_LISTENER.sequence,
            input_summary={
                "task_count": len(_tasks(state)),
                "current_task": (
                    (state.get("current_task") or {}).get("id")
                ),
                "worker_id": state.get("worker_id"),
            },
    ) as span:

        context = runtime.context
        current = _current_task(state)
        tasks = _tasks(state)
        messages = tuple(state.get("messages") or ())

        # 父图预 claim 的任务：**收养**它，不是 claim、也不是完成。
        # 这是 dispatch 扇出的入口 —— 若在这里走完成分支，任务会在
        # 执行前就被标记 COMPLETED。
        pending = _pending_task(state)

        if pending:
            position, total = task_position(pending, tasks)
            owner = _owner(state, context)
            new_messages = _clear_messages(messages)
            new_messages.append(
                task_instruction_message(
                    pending,
                    owner=owner,
                    position=position,
                    total=total,
                )
            )

            claim = _claim_of(pending)

            span.set_output_summary({
                "completed": False,
                "claimed": pending.get("id"),
                "source": "dispatch",
                "attempt": _attempt(claim),
            })

            return {
                "pending_task": {},
                "current_task": dict(pending),
                "task_id": str(pending.get("id") or ""),
                "attempt": _attempt(claim),
                "skill_key": skill_key_of(pending),
                "messages": new_messages,
            }

        full_result = last_ai_text(messages, limit=None)
        summary = full_result[:_SUMMARY_LIMIT]

        completed_task: dict[str, Any] | None = None

        if isinstance(current, Mapping) and current:

            if status_state(current) in WORKING:

                finished_id = str(current.get("id") or "")

                # 只完成**这一条**：完整任务列表由 task 快照维护，这里
                # 不需要把整份列表传进去再找回来。
                completed_task = complete_task(
                    [dict(current)],
                    task_id_value=finished_id,
                    summary=summary,
                    result_text=full_result,
                    context_id=_context_id(state, context),
                )[0]

                tasks = [
                    completed_task
                    if str(task.get("id") or "") == finished_id
                    else task
                    for task in tasks
                ]

            elif status_state(current) in TERMINAL:
                # 已经是终态（重复进入 listener）：不再写一次。
                pass

        # 先写权威任务板，再 claim 下一个。顺序不能反：claim 成功之后
        # 崩溃而没有完成写回，会让同一个任务被另一个 worker 重跑。
        if completed_task is not None:
            await _write_board_completion(context, completed_task)

        # 一任务一 worker：完成即退出，不再 claim 下一批。下一批由父图
        # subagent_dispatch 重新按并发上限扇出。
        if completed_task is not None and bool(state.get("single_task")):
            span.set_output_summary({
                "completed": True,
                "claimed": None,
                "remaining": len(tasks),
                "exit": "single_task",
            })

            return {
                "current_task": {},
                "task_tool_keys": [],
                "task_id": "",
                "attempt": 0,
                "skill_key": "",
                "tasks": tasks,
            }

        owner = _owner(state, context)
        subscriber = build_task_subscriber(
            {
                "tasks": tasks,
                "current_task": {},
                # worker 恒按「任务板优先」取源；没有 Redis 时
                # build_task_subscriber 会退回 StateTaskSubscriber。
                "subagent_mode": "on",
            },
            context_id=_context_id(state, context),
            plan_id=_plan_id(state),
        )

        claimed = await TaskListener(subscriber).claim_next(owner=owner)

        if claimed is None:
            span.set_output_summary({
                "completed": completed_task is not None,
                "claimed": None,
                "remaining": len(tasks),
            })

            return {
                "current_task": {},
                "task_tool_keys": [],
                "task_id": "",
                "attempt": 0,
                "skill_key": "",
                "tasks": tasks,
            }

        current_tasks = (
            claimed.tasks if claimed.tasks is not None else tasks
        )
        position, total = task_position(claimed.task, current_tasks)
        claim = _claim_of(claimed.task)

        # 完成上一任务并领到新任务：清掉旧 transcript（见模块 docstring）。
        new_messages: list[Any] = []

        if completed_task is not None:
            new_messages.extend(_clear_messages(messages))

        new_messages.append(
            task_instruction_message(
                claimed.task,
                owner=owner,
                position=position,
                total=total,
            )
        )

        span.set_output_summary({
            "completed": completed_task is not None,
            "source": claimed.source,
            "claimed": claimed.task.get("id"),
            "remaining": len(current_tasks),
            "attempt": claim.get("attempt"),
        })

        return {
            "current_task": claimed.task,
            "tasks": current_tasks,
            "task_tool_keys": [],
            "task_id": str(claimed.task.get("id") or ""),
            "attempt": _attempt(claim),
            "skill_key": skill_key_of(claimed.task),
            "messages": new_messages,
        }


# ============================================================
# llm_node
# ============================================================


async def llm_node(
        state: SubAgentState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """
    用子任务自己的上下文调用一次模型。

    ⚠️ 没有 current_task 就**报错**，不静默降级
    -----------------------------------------
    缺它时 ContextBuilder 的 `task_mode=False`，TASK_RESOLVER /
    SKILL_LOADER 会静默跳过，worker 会带着默认策略跑 —— 症状是「模型
    答得很泛」，而没有任何报错。这比当场失败难查得多。

    ⚠️ 没有 subagent_builder 也报错
    ------------------------------
    不回落到主图 builder：那会让 worker 看到父对话、沿用父职责提示词，
    等于「子代理」只是换了个节点名。
    """

    task = _current_task(state)

    if not isinstance(task, Mapping) or not task:
        raise RuntimeError(
            "llm_node 没有 current_task：worker 必须在 task_listener "
            "领到任务之后才进入模型调用。检查子图入口 / 路由。"
        )

    context = runtime.context
    builder = getattr(context, "subagent_builder", None)

    if builder is None:
        raise RuntimeError(
            "ReactContext.subagent_builder 未注入：子图的 llm_node 需要 "
            "ContextBuilderType.SUB_AGENT 的实例（由 register_react_graph "
            "在启动期解析）。不要回落到主图 builder。"
        )

    return await run_agent_turn(
        state,
        runtime,
        spec=_SPEC_LLM,
        builder=builder,
        builder_type=CONTEXT_BUILDER_TYPE,
        agent_type=AGENT_TYPE,
        model_role=MODEL_ROLE,
        state_update_keys=_STATE_UPDATE_KEYS,
        tag_task_result=True,
    )


# ============================================================
# tool_node
# ============================================================


async def tool_node(
        state: SubAgentToolPayload,
        runtime: Runtime[ReactContext],
        config: RunnableConfig,
) -> dict[str, Any] | Command:
    """
    执行一个工具调用，结果回到 `llm_node`。

    执行本身在主图的 `execute_tool_branch` —— 校验允许集、驱动处理器
    链（审批 / 脱敏 / 缓存 / 鉴权）、记观测都在那里。本函数只把结果
    包成 state 更新，并在控制器分支上决定下一跳。

    ⚠️ 第三个参数 `config` 是 langgraph 注入的（`ToolRuntime` 要它拿
    `configurable.thread_id`，审批账本按 thread/run 归档）。
    """

    result = await execute_tool_branch(
        state,
        runtime,
        config,
        spec=_SPEC_TOOL,
        scope_namespace=SUBAGENT_TOOL_SCOPE_NAMESPACE,
    )

    update: dict[str, Any] = {"messages": [result.message]}

    if result.command is not None:
        update.update(command_update(result.command))

    # 非控制器分支只写消息，不决定跳转（一个 super-step 只能有一个
    # Command），与主图 tools 节点同一条规则。
    if not state.get("is_controller", True):
        return update

    if result.command is not None and result.command.goto:

        # 原样保留 graph（工具可能返回 Command.PARENT）；只补上本批
        # 汇总后的 update。
        return Command(
            graph=result.command.graph,
            goto=result.command.goto,
            update=update,
        )

    return Command(
        goto=str(state.get("return_to") or LLM_NODE),
        update=update,
    )


# ============================================================
# internals
# ============================================================


def _current_task(state: Mapping[str, Any]) -> Mapping[str, Any]:
    task = state.get("current_task")

    return task if isinstance(task, Mapping) else {}


def _pending_task(state: Mapping[str, Any]) -> dict[str, Any]:
    """父图预 claim、还没被这个 worker 收养的任务。"""

    task = state.get("pending_task")

    return dict(task) if isinstance(task, Mapping) and task else {}


def _clear_messages(messages: tuple[Any, ...]) -> list[Any]:
    """
    整个 transcript 的墓碑。

    只对**有 id** 的消息发墓碑：没有 id 的消息本来就不在 `add_messages`
    的归约里（绕过 reducer 写入的消息不会进 state）。
    """

    return [
        RemoveMessage(id=message.id)
        for message in messages
        if getattr(message, "id", None)
    ]


def _tasks(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("tasks")

    if not isinstance(raw, (list, tuple)):
        return []

    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _claim_of(task: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = task.get("metadata")

    if not isinstance(metadata, Mapping):
        return {}

    claim = metadata.get("claim")

    return claim if isinstance(claim, Mapping) else {}


def _attempt(claim: Mapping[str, Any]) -> int:
    try:
        return max(1, int(claim.get("attempt") or 1))
    except (TypeError, ValueError):
        return 1


def _owner(
        state: Mapping[str, Any],
        context: ReactContext,
) -> str:
    """
    claim owner：`{worker_id}`，缺省用运行 id 兜底。

    owner 必须**稳定且唯一**：它写进 claim，用来判断「这个任务是不是
    我领的」。worker 之间用 `worker_id` 区分（派发方传入）。
    """

    worker_id = str(state.get("worker_id") or "").strip()

    if worker_id:
        return worker_id

    return f"{context.pipeline_run_id}:worker"


def _context_id(
        state: Mapping[str, Any],
        context: ReactContext,
) -> str:
    """任务板索引键用的会话 id（与主图 listener 同一条取法）。"""

    value = str(state.get("conversation_id") or "").strip()

    if value:
        return value

    if context.session_id is not None:
        return str(context.session_id)

    return str(context.pipeline_run_id)


def _plan_id(state: Mapping[str, Any]) -> str | None:
    plan = state.get("plan")

    if not isinstance(plan, Mapping):
        return None

    value = str(plan.get("plan_id") or "").strip()

    return value or None


async def _write_board_completion(
        context: ReactContext,
        task: Mapping[str, Any],
) -> None:
    """
    把完成态写回 Redis 任务板。

    ⚠️ 失败只记日志、不抛
    --------------------
    worker 的成果已经在上面的 `complete_task` 里成形；这里抛会让一次
    已经完成的工作变成 failed，而任务板上的 lease 本来就会到期重试。
    但**不能静默**：写失败意味着父图看不到这次完成，必须留下痕迹。

    ⚠️ 没有任务板是合法状态
    --------------------
    `context.task_board` 为 None = 本地只跑 state source（冒烟 / 没起
    Redis）。此时完成态留在 SubAgentState 里，不假装写成功。
    """

    board = getattr(context, "task_board", None)

    if board is None:
        return

    complete = getattr(board, "complete", None)

    if complete is None:
        logger.warning(
            "任务板实现没有 complete()，worker 的完成态未写回任务板。"
        )
        return

    try:
        written = await complete(
            task_id=str(task.get("id") or ""),
            task=task,
        )

    except Exception:  # noqa: BLE001 - 见 docstring 的失败语义
        logger.exception(
            "任务完成写回任务板失败（task_id=%s）。",
            task.get("id"),
        )
        return

    if not written:
        logger.warning(
            "任务板里没有这一行，完成态未写回（task_id=%s）。",
            task.get("id"),
        )
