"""Planner 子图的三个节点。

    planner_llm      ReAct 回合：按需查能力 / 技能 / 文档，或向用户澄清
    planner_tool     执行一次工具调用，结果回到 planner_llm
    task_generator   不接工具，严格 JSON 输出 TaskDraft -> A2A tasks

⚠️ planner_llm 不写父图
----------------------
它写的是 `PlannerState.messages`；子图 `output_schema=PlannerOutput`
只把 plan / tasks / current_task / team_decision 交回父图。工具调用与
中间推理留在子图 namespace，不串进主对话。

⚠️ task_generator 必须看到 planner 的 transcript
-----------------------------------------------
否则 ReAct 阶段的探索结果会被丢掉。这里把 `PlannerState.messages`
作为历史，再追加一条临时 HumanMessage 要求输出 JSON；临时消息不写回
state，避免重跑时堆叠。
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.context.trace import ContextTrace
from app.agent.event.task_events import publish_plan_published
from app.agent.graph.react.agent_node import run_agent_turn
from app.agent.graph.react.capability_defaults import (
    default_capability_codes,
    parent_codes_of,
    resolve_capability_exposure,
)
from app.agent.graph.react.context import ReactContext
from app.agent.graph.react.tools import command_update, execute_tool_branch
from app.agent.graph.react.trace import node_trace
from app.agent.model import ModelTrace
from app.agent.plan import (
    PlanDraft,
    find_dependency_cycle,
    normalize_payload,
    task_to_dict,
    to_a2a_tasks,
    validate_plan,
)
from app.agent.prompt import get_prompt_provider
from app.infra.agent.capability_resolver import ServiceCapabilityResolver
from app.infra.graph import node_span

from .catalog import (
    AGENT_TYPE,
    CONTEXT_BUILDER_TYPE,
    MODEL_ROLE,
    NODE_SPECS,
    PLANNER_EXCLUDED_TOOL_KEYS,
    PLANNER_LLM_NODE,
    PLANNER_NODE_IDS,
    PLANNER_TOOL_SCOPE_NAMESPACE,
    TASK_GENERATOR_NODE,
    TOOL_NODE,
)
from .state import PlannerState, PlannerToolPayload

__all__ = [
    "planner_llm_node",
    "planner_tool_node",
    "task_generator_node",
]

logger = logging.getLogger(__name__)


_SPEC_LLM = next(s for s in NODE_SPECS if s.node_id == PLANNER_LLM_NODE)
_SPEC_TOOL = next(s for s in NODE_SPECS if s.node_id == TOOL_NODE)
_SPEC_TASK = next(s for s in NODE_SPECS if s.node_id == TASK_GENERATOR_NODE)

_PLANNER_REACT_PROMPT_KEY = "planner_react"
_PLANNER_TASK_PROMPT_KEY = "planner"

#: 目录最多带多少条进提示词。能力树的完整 parent 关系另取全量。
_CATALOG_LIMIT = 100


# ============================================================
# planner_llm
# ============================================================


async def planner_llm_node(
        state: PlannerState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """一次 ReAct 规划回合。"""

    context = runtime.context
    builder = getattr(context, "planner_builder", None)

    if builder is None:
        raise RuntimeError(
            "planner_llm 需要 ReactContext.planner_builder；"
            "不要回落到主图 builder。"
        )

    notes = list(state.get("planner_notes") or ())
    policy = _planner_policy(state)
    resolved = bool(state.get("planner_tools_resolved"))

    if not resolved:

        tool_keys, tool_notes = await _planner_tool_keys()
        policy["tool_keys"] = tool_keys
        notes.extend(tool_notes)
        resolved = True

    else:

        existing = state.get("context_policy") or {}
        raw = existing.get("tool_keys") if isinstance(existing, dict) else ()

        policy["tool_keys"] = [
            str(item) for item in raw
        ] if isinstance(raw, (list, tuple)) else []

    turn_state = dict(state)
    turn_state["context_policy"] = policy

    try:
        update = await run_agent_turn(
            turn_state,
            runtime,
            spec=_SPEC_LLM,
            builder=builder,
            builder_type=CONTEXT_BUILDER_TYPE,
            agent_type=AGENT_TYPE,
            model_role=MODEL_ROLE,
            # planner 的总结不是任务成果；也不把 summary/memory 写回子图。
            state_update_keys=(),
            tag_task_result=False,
        )

    except GraphBubbleUp:
        # interrupt / 父图跳转必须冒泡，不能被降级吞掉。
        raise

    except Exception as exc:  # noqa: BLE001 - 探索失败降级给 task_generator
        logger.warning(
            "planner ReAct 回合失败（%s: %s），交给 task_generator 降级。",
            type(exc).__name__,
            exc,
        )
        notes.append(f"planner 探索失败：{type(exc).__name__}: {exc}")
        update = {
            "messages": [
                AIMessage(
                    content=(
                        "规划探索阶段失败，请基于当前对话直接生成计划。"
                    )
                )
            ]
        }

    update["context_policy"] = policy
    update["planner_tools_resolved"] = resolved
    update["planner_notes"] = _dedupe(notes)

    return update


def _planner_policy(state: PlannerState) -> dict[str, Any]:
    """planner ReAct 回合的 ContextPolicy。"""

    prompt = get_prompt_provider().node_prompt(
        _PLANNER_REACT_PROMPT_KEY,
        plan_snapshot=_plan_snapshot(state),
    )
    plan = state.get("plan") or {}

    return {
        "system_prompt": prompt,
        "include_summary": bool(state.get("summary")),
        "include_history": True,
        "include_user_profile": False,
        "include_memory": False,
        "include_retrieved_context": False,
        "include_capabilities": True,
        # 首次解析后由调用方覆盖。
        "tool_keys": [],
        "enforce_token_budget": True,
        "allow_compaction": False,
        "validate": False,
        "model_metadata": {
            "planner": True,
            "phase": "react",
            "plan_id": str(plan.get("plan_id") or ""),
            "revision": _revision_of(plan),
        },
    }


async def _planner_tool_keys() -> tuple[list[str], list[str]]:
    """
    planner 默认能力 -> tool_keys。

    与主 agent 同一条策略链：
        DEFAULT_CAPABILITIES["planner"]  -> 默认能力子树
        MODEL_CAPABILITIES["planner"]    -> 全局能力
        ServiceCapabilityResolver        -> code -> tool_key
    最后过滤掉会返回父图 Command 的工具。
    """

    notes: list[str] = []

    try:
        from app.common.factories import get_service_factory
        from app.services.capability.application import CapabilityService

        factory = get_service_factory()

        async with factory.scope(CapabilityService, begin=False) as service:
            items = await service.list_capabilities(is_active=True)

    except Exception as exc:  # noqa: BLE001 - 目录是增强，不是运行前提
        logger.warning(
            "planner 读取能力目录失败（%s: %s），本次无工具。",
            type(exc).__name__,
            exc,
        )
        return [], [f"planner 能力目录不可用：{type(exc).__name__}"]

    codes = [item.code for item in items]
    parent_of = parent_codes_of(items)

    if "AGENT_RUNTIME" not in codes:
        notes.append("planner 默认能力根 AGENT_RUNTIME 不在能力目录里。")

    mode_defaults = default_capability_codes(
        "planner",
        codes=codes,
        parent_of=parent_of,
    )
    exposure = resolve_capability_exposure(
        "planner",
        mode_defaults=mode_defaults,
        requested=(),
        codes=codes,
        parent_of=parent_of,
    )

    try:
        keys = await ServiceCapabilityResolver().tool_keys(
            list(exposure.effective)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "planner 解析能力 -> 工具失败（%s: %s），本次无工具。",
            type(exc).__name__,
            exc,
        )
        return [], [*notes, f"planner 能力解析失败：{type(exc).__name__}"]

    filtered = [
        key for key in keys
        if key not in PLANNER_EXCLUDED_TOOL_KEYS
    ]

    return filtered, notes


# ============================================================
# planner_tool
# ============================================================


async def planner_tool_node(
        state: PlannerToolPayload,
        runtime: Runtime[ReactContext],
        config: RunnableConfig,
) -> dict[str, Any] | Command:
    """执行一个工具调用；结果回到 planner_llm。"""

    result = await execute_tool_branch(
        state,
        runtime,
        config,
        spec=_SPEC_TOOL,
        scope_namespace=PLANNER_TOOL_SCOPE_NAMESPACE,
    )

    return_to = str(state.get("return_to") or PLANNER_LLM_NODE)

    # 非控制器分支只写消息；一个 super-step 只能有一个 Command。
    if not state.get("is_controller", True):
        return {"messages": [result.message]}

    if result.command is None:
        return Command(
            goto=return_to,
            update={"messages": [result.message]},
        )

    if _command_is_local(result.command):
        update: dict[str, Any] = {"messages": [result.message]}
        update.update(command_update(result.command))
        return Command(
            graph=result.command.graph,
            goto=result.command.goto,
            update=update,
        )

    # 工具返回了父图节点的 Command（或 Command.PARENT）：子图里 goto
    # 不到。把它变成一条明确的错误结果，让模型换一种做法继续规划。
    logger.warning(
        "planner 工具返回了子图外的 Command，已拦截：goto=%r graph=%r",
        getattr(result.command, "goto", None),
        getattr(result.command, "graph", None),
    )
    return Command(
        goto=return_to,
        update={"messages": [_rejected_command_message(result.message)]},
    )


def _command_is_local(command: Any) -> bool:
    """Command 是否只落在 planner 子图内。"""

    if getattr(command, "graph", None) is not None:
        return False

    goto = getattr(command, "goto", None)

    if not goto:
        return False

    targets = goto if isinstance(goto, (list, tuple)) else [goto]

    return all(
        str(target) in PLANNER_NODE_IDS
        for target in targets
    )


def _rejected_command_message(original: ToolMessage) -> ToolMessage:
    """把被拒绝的 Command 转成模型可读的错误工具结果。"""

    return ToolMessage(
        content=(
            "该工具返回了父图控制指令，在 planner 子图内不可执行。"
            "请改用不跨图跳转的工具，或直接总结探索结论。"
        ),
        tool_call_id=str(getattr(original, "tool_call_id", "") or "unknown"),
        name=getattr(original, "name", None),
    )


# ============================================================
# task_generator
# ============================================================


async def task_generator_node(
        state: PlannerState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """把 planner 的探索结果收成严格 JSON 的 TaskDraft。"""

    context = runtime.context
    builder = getattr(context, "planner_builder", None)

    if builder is None:
        raise RuntimeError(
            "task_generator 需要 ReactContext.planner_builder；"
            "不要回落到主图 builder。"
        )

    async with node_span(
            runtime,
            node_id=_SPEC_TASK.node_id,
            label=_SPEC_TASK.label,
            group=_SPEC_TASK.group,
            sequence=_SPEC_TASK.sequence,
            input_summary={
                "message_count": len(state.get("messages") or ()),
            },
    ) as span:

        try:
            update = await _generate(state, runtime, builder)

        except GraphBubbleUp:
            raise

        except Exception as exc:  # noqa: BLE001 - 保持旧计划，不炸整轮
            logger.warning(
                "task_generator 失败（%s: %s），保留旧计划。",
                type(exc).__name__,
                exc,
            )
            span.set_output_summary({
                "degraded": True,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return {}

        span.set_output_summary({
            "plan_id": update["plan"]["plan_id"],
            "revision": update["plan"]["revision"],
            "task_count": len(update["tasks"]),
            "degraded": False,
        })
        return update


async def _generate(
        state: PlannerState,
        runtime: Runtime[ReactContext],
        builder: Any,
) -> dict[str, Any]:
    context = runtime.context

    skills, skill_notes = await _skill_catalog()
    capabilities, capability_notes = await _capability_catalog()

    plan_id = _plan_id(state)
    revision = _next_revision(state)

    prompt = get_prompt_provider().node_prompt(
        _PLANNER_TASK_PROMPT_KEY,
        skill_catalog=_render_catalog(skills, key="skill_key"),
        capability_catalog=_render_catalog(capabilities, key="code"),
        plan_snapshot=_plan_snapshot(state),
    )

    # 把规划探索的 transcript 交给 task_generator；临时指令不写回 state。
    messages = tuple(state.get("messages") or ()) + (
        HumanMessage(
            content=(
                "请根据以上探索结果，只输出最终的 PlanDraft JSON；"
                "不要调用工具，不要输出解释性正文。"
            )
        ),
    )

    policy = _task_generator_policy(prompt, state)

    graph_state: dict[str, Any] = {
        "messages": messages,
        "context_policy": policy,
    }

    if state.get("summary"):
        graph_state["summary"] = state["summary"]

    try:
        emitter, parent_span_id = node_trace(runtime)
    except Exception:  # noqa: BLE001 - 图外运行没有 writer，不是错误
        emitter, parent_span_id = None, ""

    request = ContextBuildRequest(
        builder_type=CONTEXT_BUILDER_TYPE,
        thread_id=str(
            state.get("conversation_id")
            or context.session_id
            or context.pipeline_run_id
        ),
        messages=messages,
        run_id=context.pipeline_run_id,
        graph_state=graph_state,
        agent_type=AGENT_TYPE,
        model_role=MODEL_ROLE,
        dependencies=ContextDependencies(
            model_registry=context.model_registry,
            tool_source=getattr(context, "tool_source", None),
        ),
        trace=(
            ContextTrace(
                emitter=emitter,
                parent_span_id=parent_span_id,
            )
            if emitter is not None
            else None
        ),
    )

    model_request = await builder.build(request)

    # 严格 JSON 输出：与思考模式互斥，调用期关掉。
    model_request = replace(model_request, disable_thinking=True)

    model_runtime = context.model_registry.get_runtime(
        agent_type=AGENT_TYPE,
        model_role=MODEL_ROLE,
    )

    response = await model_runtime.ainvoke(
        model_request,
        trace=(
            ModelTrace(
                emitter=emitter,
                parent_span_id=parent_span_id,
            )
            if emitter is not None
            else None
        ),
    )

    payload = _parse_payload(response.message)
    draft = PlanDraft.model_validate(normalize_payload(payload))

    validated = validate_plan(
        draft,
        known_skill_keys=(
            None
            if skill_notes
            else {item["skill_key"] for item in skills}
        ),
        known_capabilities=(
            None
            if capability_notes
            else {item["code"] for item in capabilities}
        ),
    )

    context_id = _context_id(state, context)

    proto_tasks = to_a2a_tasks(
        validated.tasks,
        context_id=context_id,
        plan_id=plan_id,
        revision=revision,
        source_run_id=str(context.pipeline_run_id),
    )

    task_dicts = [task_to_dict(task) for task in proto_tasks]

    cycle = find_dependency_cycle(task_dicts)

    if cycle is not None:
        raise ValueError(
            "planner 输出的任务依赖存在环："
            + " -> ".join(cycle)
        )

    plan_snapshot = {
        "plan_id": plan_id,
        "revision": revision,
        "goal": draft.goal,
        "tasks": [
            {
                "id": item["id"],
                "local_id": item["metadata"].get("plan_task_id"),
                "status": "pending",
            }
            for item in task_dicts
        ],
        "open_questions": list(draft.open_questions),
        "notes": _dedupe([
            *draft.notes,
            *validated.warnings,
            *skill_notes,
            *capability_notes,
            *(state.get("planner_notes") or ()),
        ]),
    }

    publish_plan_published(
        plan_id=plan_id,
        revision=revision,
        goal=draft.goal,
        tasks=task_dicts,
        current_task_id=None,
        publish_id=f"{plan_id}:{revision}",
    )

    return {
        "plan": plan_snapshot,
        "tasks": task_dicts,
        "current_task": {},
        "team_decision": "",
    }


def _task_generator_policy(
        prompt: str,
        state: PlannerState,
) -> dict[str, Any]:
    plan = state.get("plan") or {}

    return {
        "system_prompt": prompt,
        "include_summary": bool(state.get("summary")),
        "include_history": True,
        "include_user_profile": False,
        "include_memory": False,
        "include_retrieved_context": False,
        "include_capabilities": False,
        "tool_keys": [],
        "enforce_token_budget": True,
        "allow_compaction": False,
        "validate": False,
        "model_metadata": {
            "planner": True,
            "phase": "task_generator",
            "plan_id": str(plan.get("plan_id") or ""),
            "revision": _revision_of(plan),
        },
    }


# ============================================================
# 目录（只读、可降级）
# ============================================================


async def _skill_catalog() -> tuple[list[dict[str, str]], list[str]]:
    """ACTIVE 技能目录；失败返回空目录 + 一条 note。"""

    try:
        from app.common.factories import get_service_factory
        from app.services.skill.application import SkillService
        from app.services.skill.domain.schemas import SkillStatus

        factory = get_service_factory()

        async with factory.scope(SkillService, begin=False) as service:
            page = await service.list_skills(
                status=SkillStatus.ACTIVE,
                page_size=_CATALOG_LIMIT,
            )

        return (
            [
                {
                    "skill_key": item.skill_key,
                    "name": item.name,
                    "description": item.description,
                }
                for item in page.items
            ],
            [],
        )

    except Exception as exc:  # noqa: BLE001 - 目录是增强，不是运行前提
        logger.warning(
            "Planner 读取技能目录失败（%s: %s），按空目录继续。",
            type(exc).__name__,
            exc,
        )
        return [], [f"技能目录不可用：{type(exc).__name__}"]


async def _capability_catalog() -> tuple[list[dict[str, str]], list[str]]:
    """ACTIVE 能力目录；失败返回空目录 + 一条 note。"""

    try:
        from app.common.factories import get_service_factory
        from app.services.capability.application import CapabilityService

        factory = get_service_factory()

        async with factory.scope(CapabilityService, begin=False) as service:
            items = await service.list_capabilities(is_active=True)

        return (
            [
                {
                    "code": item.code,
                    "name": item.name,
                    "description": item.description or "",
                }
                for item in items[:_CATALOG_LIMIT]
            ],
            [],
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Planner 读取能力目录失败（%s: %s），按空目录继续。",
            type(exc).__name__,
            exc,
        )
        return [], [f"能力目录不可用：{type(exc).__name__}"]


def _render_catalog(
        items: list[dict[str, str]],
        *,
        key: str,
) -> str:
    if not items:
        return "（无）"

    lines: list[str] = []

    for item in items:
        name = item.get("name") or item.get(key) or ""
        description = item.get("description") or ""
        value = item.get(key) or ""

        line = f"- {value}：{name}"
        if description:
            line += f" —— {description}"

        lines.append(line)

    return "\n".join(lines)


# ============================================================
# 解析与 state 工具
# ============================================================


def _parse_payload(message: Any) -> dict[str, Any]:
    """从模型响应里取 JSON object（文本 / 结构化实例都认）。"""

    if isinstance(message, BaseModel) and not isinstance(
        message, BaseMessage
    ):
        payload: Any = message.model_dump()
    else:
        payload = getattr(message, "content", message)

    if isinstance(payload, list):
        # 有的 provider 把 content 拆成 [{"type": "text", "text": ...}]
        chunks: list[str] = []

        for block in payload:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                text = block.get("text")

                if isinstance(text, str):
                    chunks.append(text)

        payload = "".join(chunks)

    if isinstance(payload, str):
        text = payload.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        payload = json.loads(text)

    if not isinstance(payload, dict):
        raise ValueError(
            f"planner 输出不是 JSON object：{type(payload).__name__}"
        )

    return payload


def _plan_id(state: PlannerState) -> str:
    plan = state.get("plan") or {}
    existing = str(plan.get("plan_id") or "").strip()

    return existing or str(uuid4())


def _next_revision(state: PlannerState) -> int:
    plan = state.get("plan") or {}

    return max(_revision_of(plan), 0) + 1


def _revision_of(plan: Any) -> int:
    if not isinstance(plan, dict):
        return 0

    try:
        return int(plan.get("revision") or 0)
    except (TypeError, ValueError):
        return 0


def _context_id(
        state: PlannerState,
        context: ReactContext,
) -> str:
    value = str(state.get("conversation_id") or "").strip()

    if value:
        return value

    if context.session_id is not None:
        return str(context.session_id)

    return str(context.pipeline_run_id)


def _plan_snapshot(state: PlannerState) -> str:
    plan = state.get("plan") or {}

    if not plan:
        return "（无）"

    try:
        return json.dumps(plan, ensure_ascii=False)
    except (TypeError, ValueError):
        return "（无法序列化；忽略旧计划）"


def _dedupe(items: Any) -> list[str]:
    """保持顺序去重；只收非空字符串。"""

    result: list[str] = []
    seen: set[str] = set()

    for item in items or ():
        text = str(item).strip()

        if not text or text in seen:
            continue

        seen.add(text)
        result.append(text)

    return result
