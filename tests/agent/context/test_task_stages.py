"""TASK_RESOLVER / SKILL_LOADER 的任务上下文切换。"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.agent.context.builder.agent_invoke import (
    AgentInvokeProcessorContextBuilder,
)
from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.context.skills import SkillCapability, SkillSnapshot
from app.agent.graph.react.routing import _tool_keys
from app.agent.model import ModelRole
from app.common.context import ContextBuilderType

from context_fakes import FakeModelRegistry

__all__: list[str] = []


class _FakeSkillSource:
    async def load(self, skill_key: str):
        if skill_key != "research":
            return None

        return SkillSnapshot(
            skill_key="research",
            name="研究",
            description="研究技能",
            instructions="这是研究技能的 SOP。",
            capabilities=(
                SkillCapability(code="web.search", name="搜索"),
            ),
        )


class _FakeCapabilityResolver:
    async def tool_keys(self, codes):
        return [f"tool.{code}" for code in codes]


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeToolSource:
    async def resolve(self, *, tool_keys, conversation_id, overrides=None):
        return [_FakeTool(key) for key in tool_keys]


def _task() -> dict:
    return {
        "id": "plan-1:t1",
        "status": {"state": "TASK_STATE_WORKING"},
        "history": [
            {
                "role": "ROLE_USER",
                "parts": [
                    {
                        "data": {
                            "description": "研究 A 并输出结论",
                            "params": {"symbol": "600519"},
                        }
                    }
                ],
            }
        ],
        "metadata": {
            "skill_key": "research",
            "capabilities": ["document.retrieve"],
            "blocked_by": [],
            "block_policy": "all_succeeded",
            "claim": {"owner": "run-1:main", "token": "x"},
        },
    }


def _pending_task(
        task_id: str,
        description: str,
        *,
        blocked_by: list[str] | None = None,
) -> dict:
    return {
        "id": task_id,
        "status": {"state": "TASK_STATE_SUBMITTED"},
        "history": [
            {
                "role": "ROLE_USER",
                "parts": [{"data": {"description": description}}],
            }
        ],
        "metadata": {
            "skill_key": "research",
            "capabilities": [],
            "blocked_by": blocked_by or [],
            "block_policy": "all_succeeded",
            "claim": {},
        },
    }


def _completed_task(task_id: str, summary: str) -> dict:
    return {
        "id": task_id,
        "status": {"state": "TASK_STATE_COMPLETED"},
        "metadata": {
            "skill_key": "research",
            "result_summary": summary,
            "claim": {},
        },
        "artifacts": [
            {
                "artifactId": f"{task_id}:result:1",
                "name": "task_result",
                "parts": [{"text": summary}],
            }
        ],
    }


def _plan(*task_ids: str) -> dict:
    return {
        "plan_id": "plan-1",
        "revision": 1,
        "goal": "测试计划",
        "tasks": [
            {"id": task_id, "status": "pending"}
            for task_id in task_ids
        ],
    }


def _state(*, task: bool = True) -> dict:
    state = {
        "messages": (HumanMessage("原始用户请求", id="u1"),),
        "conversation_id": "c1",
        "user_id": "u1",
        "context_policy": {
            "system_prompt": "主 agent 职责",
            "include_history": True,
            "include_user_profile": False,
            "include_summary": False,
            "include_memory": False,
            "include_retrieved_context": False,
            "include_capabilities": True,
            "enforce_token_budget": False,
            "allow_compaction": False,
            "validate": False,
        },
    }

    if task:
        state["current_task"] = _task()
        state["tasks"] = [state["current_task"]]
        state["plan"] = _plan("plan-1:t1")

    return state


def _request(state: dict) -> ContextBuildRequest:
    return ContextBuildRequest(
        builder_type=ContextBuilderType.AGENT_INVOKE,
        thread_id="thread-1",
        messages=tuple(state["messages"]),
        run_id=uuid4(),
        graph_state=state,
        agent_type="react",
        model_role=ModelRole.REASONING,
        dependencies=ContextDependencies(
            model_registry=FakeModelRegistry(),
            skill_source=_FakeSkillSource(),
            capability_resolver=_FakeCapabilityResolver(),
            tool_source=_FakeToolSource(),
        ),
    )


@pytest.mark.asyncio
async def test_task_mode_switches_prompt_and_tools() -> None:
    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state())
    )

    request = result.model_request
    system = request.messages[0]
    assert system.type == "system"
    assert "研究技能的 SOP" in str(system.content)
    assert "当前任务" in str(system.content)
    assert "plan-1:t1" in str(system.content)
    assert "负责人：main（你）" in str(system.content)
    assert "任务列表：总计 1，当前第 1 个" in str(system.content)
    assert "研究 A 并输出结论" in str(system.content)

    # required 任务指令排在历史之后，成为“当前这一轮”；完整 payload
    # 只在这里出现一次（system 里只有短头）。
    last = request.messages[-1]
    assert isinstance(last, HumanMessage)
    assert last.id == "task:plan-1:t1:instruction"
    assert "任务执行模式" in str(last.content)
    assert "负责人：main（你）" in str(last.content)
    assert "任务列表：总计 1，当前第 1 个" in str(last.content)
    assert "研究 A 并输出结论" in str(last.content)
    assert "600519" in str(last.content)
    assert last.additional_kwargs["kind"] == "task_instruction"

    assert {tool.name for tool in request.tools} == {
        "tool.web.search",
        "tool.document.retrieve",
    }

    updates = request.metadata["context_state_updates"]
    assert updates["task_tool_keys"] == [
        "tool.web.search",
        "tool.document.retrieve",
    ]

    stages = {
        item["stage"]: item["status"]
        for item in result.processor_executions
    }
    assert stages["task_resolver"] == "executed"
    assert stages["skill_loader"] == "executed"


@pytest.mark.asyncio
async def test_no_task_skips_task_stages_and_keeps_main_prompt() -> None:
    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(_state(task=False))
    )

    request = result.model_request
    system = request.messages[0]
    assert "主 agent 职责" in str(system.content)
    assert "当前任务" not in str(system.content)

    stages = {
        item["stage"]: item["status"]
        for item in result.processor_executions
    }
    assert stages["task_resolver"] == "skipped"
    assert stages["skill_loader"] == "skipped"


@pytest.mark.asyncio
async def test_pending_tasks_inject_team_decision_prompt() -> None:
    state = _state(task=False)
    state["tasks"] = [
        _pending_task("plan-1:t1", "研究 A"),
        _pending_task("plan-1:t2", "研究 B"),
    ]
    state["plan"] = _plan("plan-1:t1", "plan-1:t2")
    state["context_policy"]["tool_keys"] = ["agent.create_team"]

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(state)
    )

    request = result.model_request
    system = str(request.messages[0].content)

    assert "待办任务" in system
    assert "当前没有正在执行的任务" in system

    last = request.messages[-1]
    assert isinstance(last, HumanMessage)
    assert last.additional_kwargs["kind"] == "team_decision"
    assert "总计 2，待处理 2，可并行 2" in str(last.content)
    assert "create_team(worker_num=2)" in str(last.content)
    assert "自己处理" in str(last.content)

    stages = {
        item["stage"]: item["status"]
        for item in result.processor_executions
    }
    assert stages["task_resolver"] == "executed"
    assert stages["skill_loader"] == "skipped"


@pytest.mark.asyncio
async def test_completed_tasks_inject_results_for_synthesis() -> None:
    state = _state(task=False)
    state["tasks"] = [
        _completed_task("plan-1:t1", "短信分支结果正文"),
        _completed_task("plan-1:t2", "邮件分支结果正文"),
    ]
    state["plan"] = _plan("plan-1:t1", "plan-1:t2")
    state["plan"]["open_questions"] = ["会议日期待确认"]
    state["synthesis_pending"] = True
    state["hist_synthesis_keys"] = ["plan-1:1"]
    state["context_policy"]["tool_keys"] = []
    state["context_policy"]["include_capabilities"] = False

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(state)
    )

    request = result.model_request
    system = str(request.messages[0].content)

    assert "任务结果" in system
    assert "完整结果见当前输入" in system

    last = request.messages[-1]
    assert isinstance(last, HumanMessage)
    assert last.additional_kwargs["kind"] == "task_results"
    assert "不要重新发布任务" in str(last.content)
    assert "短信分支结果正文" in str(last.content)
    assert "邮件分支结果正文" in str(last.content)
    assert "会议日期待确认" in str(last.content)

    # 汇总回合不加载任务 skill，也不再绑定工具。
    stages = {
        item["stage"]: item["status"]
        for item in result.processor_executions
    }
    assert stages["task_resolver"] == "executed"
    assert stages["skill_loader"] == "skipped"
    assert request.tools == ()


@pytest.mark.asyncio
async def test_pending_tasks_without_parallel_advise_sequential() -> None:
    state = _state(task=False)
    state["tasks"] = [
        _pending_task("plan-1:t1", "研究 A"),
        _pending_task("plan-1:t2", "研究 B", blocked_by=["plan-1:t1"]),
    ]
    state["plan"] = _plan("plan-1:t1", "plan-1:t2")
    state["context_policy"]["tool_keys"] = ["agent.create_team"]

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(state)
    )

    last = result.model_request.messages[-1]

    assert "可并行 1" in str(last.content)
    assert "当前没有可并行任务" in str(last.content)
    assert "create_team(" not in str(last.content)


@pytest.mark.asyncio
async def test_pending_tasks_without_create_team_do_not_promise_it() -> None:
    state = _state(task=False)
    state["tasks"] = [
        _pending_task("plan-1:t1", "研究 A"),
        _pending_task("plan-1:t2", "研究 B"),
    ]
    state["plan"] = _plan("plan-1:t1", "plan-1:t2")
    state["context_policy"]["tool_keys"] = ["agent.enter_plan_mode"]

    result = await AgentInvokeProcessorContextBuilder().build_with_result(
        _request(state)
    )

    last = result.model_request.messages[-1]

    assert "没有开启子代理的工具" in str(last.content)
    assert "create_team(" not in str(last.content)


def test_routing_prefers_task_tool_keys_only_in_task_mode() -> None:
    state = {
        "current_task": {"id": "plan-1:t1"},
        "task_tool_keys": ["skill.tool"],
        "context_policy": {"tool_keys": ["main.tool"]},
    }

    assert _tool_keys(state) == ["skill.tool"]

    state["current_task"] = {}
    assert _tool_keys(state) == ["main.tool"]
