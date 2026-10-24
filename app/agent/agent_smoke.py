"""
ReAct 图 + Agent Builder 节点的冒烟测试。

不需要数据库，也**不调真模型** —— 用 `graph.py` 特意留出的
`NODE_FUNCTIONS` 那条缝把节点换成桩（见 graph.py 里那段"单列成字典而不是
在装配时直接引用函数，是为了让冒烟脚本能 monkey-patch 掉某个节点"），
模型侧则用一个假的 Registry 顶掉。

跑法：
    python -m app.agent.agent_smoke
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.context.registry import create_context_builder
from app.agent.graph.react import agent_builder_node as builder_module
from app.agent.graph.react.context import ReactContext
from app.agent.graph.react.graph import NODE_FUNCTIONS, build_react_graph
from app.agent.graph.react.state import ReactState
from app.agent.model import ModelRequest, ModelResponse
from app.agent.prompt import (
    DefaultPromptProvider,
    get_prompt_provider,
    reset_prompt_provider,
    set_prompt_provider,
)
from app.agent.state import (
    MODULE_OTHER,
    BehaviorCategory,
    ExecutionPlan,
)
from app.common.context import ContextBuilderType, StageId

USER_TEXT = "帮我查一下这个季度的财报，并做个对比"


# ============================================================
# 桩
# ============================================================


class _FakeRecorder:
    """把观测收在内存里 —— 只为让 node_span 有地方写。"""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, Any]] = []

    async def upsert_node_run(self, update: Any) -> None:
        self.nodes.append((update.node_id, update.status))

    async def create_run(self, snapshot: Any) -> None:
        pass

    async def mark_run_running(self, run_id: Any, *, started_at: Any):
        pass

    async def finish_run(self, run_id: Any, outcome: Any) -> None:
        pass

    async def summarize(self, run_id: Any) -> Any:
        return None

    async def mark_orphans_failed(self, *, older_than: Any) -> int:
        return 0


class _FakeRuntime:
    """假的 ModelRuntime：ainvoke 直接回一段结构化 payload。"""

    #: 最近一次请求 —— 用来断言主循环是否打开了思考。
    last_request: Any = None

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def ainvoke(self, request: Any, **kwargs: Any) -> Any:
        type(self).last_request = request

        content = (
            self._payload
            if isinstance(self._payload, str)
            else json.dumps(self._payload, ensure_ascii=False)
        )

        return ModelResponse(
            message=AIMessage(content=content),
            provider="fake",
            model="fake",
        )


class _FakeRegistry:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def get_runtime(self, **kwargs: Any) -> _FakeRuntime:
        return _FakeRuntime(self._payload)


class _FakeBuilder:
    """
    冒充 Context Builder。

    默认**不碰流水线**，直接给一个 ModelRequest —— 分类节点的关注点是
    "拿到产出之后怎么处理"，上下文怎么组装由 builder 自己的用例覆盖。
    boom=True 时模拟调用失败，用来验降级。
    """

    def __init__(self, *, boom: bool = False) -> None:
        self.boom = boom
        self.requests: list[ContextBuildRequest] = []

    async def build(self, request: ContextBuildRequest) -> ModelRequest:
        self.requests.append(request)
        if self.boom:
            raise RuntimeError("假装分类模型挂了")
        return ModelRequest(messages=(HumanMessage(USER_TEXT),))


class _FakeRuntimeContext:
    """
    冒充 LangGraph 的 Runtime。

    `node_span` 只从它取 `.context`，`execution_ids` 只取
    `.execution_info`（取不到就降级成"没有 thread_id"）—— 所以一个带
    `context` 属性的普通对象就够，不必造真的 Runtime。
    """

    def __init__(self) -> None:
        self.context = ReactContext(
            pipeline_run_id=uuid4(),
            recorder=_FakeRecorder(),
            model_registry=_FakeRegistry(None),
            builder=_FakeBuilder(),
            classifier_builder=_FakeBuilder(),
        )


#: 最近一次分类调用的请求（见 `_run_graph` 末尾）。
#:
#: 只给"提示词里到底拼了什么"那几条断言用 —— 那是 `_system_prompt` 的
#: 产出，而它不落 state（分类调用不进对话历史），没有别的取法。
_LAST_CLASSIFIER_REQUEST: list[Any] = []


class _FakeProvider:
    """
    假的 PromptProvider：只换**职责**那一半。

    节点提示词转发给真的默认实现 —— 这一节要验的是"职责提示词有没有落到
    context_policy 上"，不该顺手把分类提示词也换掉（那样就验不出
    "两条路各拿各的提示词"了）。
    """

    def __init__(self, duty: str) -> None:
        self._duty = duty
        self._real = DefaultPromptProvider()
        #: 每一次被问到"用哪个职责"时收到的 key —— 用来钉住
        #: "state 里的 duty_key 真的传下来了"这条线。
        self.requested: list[str | None] = []

    def node_prompt(self, model_role: str, /, **variables: Any) -> str:
        return self._real.node_prompt(model_role, **variables)

    async def duty_prompt(self, duty_key: str | None = None) -> str:
        self.requested.append(duty_key)
        return self._duty

    async def duty_key(self) -> str:
        return "general"


def _payload(plan: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "behavior": BehaviorCategory.QA.value,
        "modules": [],
        "execution_plan": plan,
        "capability_needs": [],
        "capability_config": {},
    }
    base.update(overrides)
    return base


# ============================================================
# 断言工具（与 tool_smoke.py 同一套）
# ============================================================


def check(label: str, condition: Any) -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title: str) -> None:
    print(f"\n{title}")


# ============================================================
# 跑图
# ============================================================


async def _run_graph(
        *,
        payload: Any = None,
        boom: bool = False,
        candidates: list[str] | None = None,
        duty_key: str | None = None,
        agent_tool_calls: list[dict[str, Any]] | None = None,
        initial_extra: dict[str, Any] | None = None,
) -> tuple[list[str], ReactState]:
    """
    跑一次真图，节点用桩，只有 builder 节点是真的。

    返回 (经过的节点顺序, 最终 state)。
    """

    ran: list[str] = []

    builder = _FakeBuilder(boom=boom)
    if candidates is not None:
        original = builder_module._module_candidates

        async def fake_candidates():
            # 名称与描述**刻意与 code 不同**（`financial-名称` 这种）：
            # "提示词里带上了名称与描述"这条断言只有在两者不同的时候
            # 才有意义 —— 相同的话光看 code 也能通过。
            return [
                builder_module._ModuleCandidate(
                    code=code,
                    name=f"{code}-名称",
                    description=f"{code}-描述",
                )
                for code in candidates
            ], []

        builder_module._module_candidates = fake_candidates

    async def stub_builder(state: ReactState, runtime: Any) -> dict:
        ran.append("builder")
        return await builder_module.agent_builder_node(state, runtime)

    agent_calls = {"n": 0}

    async def stub_agent(state: ReactState, runtime: Any) -> dict:
        ran.append("agent")
        agent_calls["n"] += 1
        # 第一轮可以模拟模型调用控制工具；之后正常收尾，避免图循环。
        if agent_tool_calls and agent_calls["n"] == 1:
            return {"messages": [AIMessage("", tool_calls=agent_tool_calls)]}
        return {"messages": [AIMessage("答复")]}

    async def stub_planner(state: ReactState, runtime: Any) -> dict:
        ran.append("planner")
        return {}

    # task_listener 跑**真实实现**：它只从 state 的 tasks 里 claim。
    # planner 桩返回 {}，所以这里通常 claim 不到任务；记一笔顺序即可。
    from app.agent.graph.react.task_listener_node import (
        task_listener_node as real_task_listener,
    )

    async def stub_task_listener(
            state: ReactState,
            runtime: Any,
    ) -> dict:
        ran.append("task_listener")
        return await real_task_listener(state, runtime)

    # tools 跑**真实实现**，只额外记一笔节点顺序 —— 控制工具的状态结算
    # 就在这个节点里，这一节要验的正是它有没有写对。
    from app.agent.graph.react.tools import tools_node as real_tools

    async def stub_tools(
            state: Any,
            runtime: Any,
            config: RunnableConfig,
    ) -> dict:
        ran.append("tools")
        return await real_tools(state, runtime, config)

    try:
        # ⚠️ 必须在 build_react_graph() **之前**替换：
        # 装配时按 spec.node_id 从这个字典里取函数。
        NODE_FUNCTIONS["builder"] = stub_builder
        NODE_FUNCTIONS["agent"] = stub_agent
        NODE_FUNCTIONS["planner"] = stub_planner
        NODE_FUNCTIONS["task_listener"] = stub_task_listener
        NODE_FUNCTIONS["tools"] = stub_tools

        graph = build_react_graph().compile(
            checkpointer=InMemorySaver()
        )

        context = ReactContext(
            pipeline_run_id=uuid4(),
            recorder=_FakeRecorder(),
            model_registry=_FakeRegistry(payload),
            builder=_FakeBuilder(),
            classifier_builder=builder,
        )

        # 初始 state 与 `_initial_state` 同形：messages + 那几个通道。
        # duty_key 只在调用方显式给的时候才写 —— 不给就是"没指定"那种
        # 情况（存量运行 / 脚本），builder 节点应当回落默认职责。
        initial: dict[str, Any] = {"messages": [HumanMessage(USER_TEXT)]}

        if duty_key is not None:
            initial["duty_key"] = duty_key

        if initial_extra:
            initial.update(initial_extra)

        state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": f"smoke-{uuid4()}"}},
            context=context,
        )

    finally:
        if candidates is not None:
            builder_module._module_candidates = original

    # 分类那次调用的 ContextBuildRequest —— 拼好的 system_prompt 就在它的
    # graph_state 里。放在模块级槽位而不是塞进返回值：`_run_graph` 的返回值
    # 被十来个用例解包，为一个断言改签名不值当（本文件是冒烟脚本）。
    if builder.requests:
        _LAST_CLASSIFIER_REQUEST.clear()
        _LAST_CLASSIFIER_REQUEST.append(builder.requests[-1])

    return ran, state  # type: ignore[return-value]


# ============================================================
# 用例
# ============================================================


async def main() -> int:

    # 控制工具是**真工具**：它们要经 tools 节点执行，而允许集与执行器
    # 都从代码注册表取。脚本里没有 lifespan，这里手动扫一遍。
    from app.services.tool.registry import get_tool_registry

    get_tool_registry().scan()

    # --------------------------------------------------------
    section("专用 Context Builder：窄输入")

    builder = create_context_builder(ContextBuilderType.CLASSIFIER)

    check(
        "classifier 能解析出实现",
        type(builder).__name__ == "ClassifierProcessorContextBuilder",
    )
    check(
        "声明四个阶段（含只读摘要注入）",
        {s.name for s in type(builder).enabled_stages} == {
            "CONTEXT_RESOLVER", "SUMMARY_LOADER",
            "HISTORY_SELECTOR", "PROMPT_ASSEMBLER",
        },
    )
    check(
        "入口与出口是 required",
        {s.name for s in type(builder).required_stages} == {
            "CONTEXT_RESOLVER", "PROMPT_ASSEMBLER",
        },
    )
    check(
        "不带 compaction（整条流水线里唯一还会再调 LLM 的处理器）",
        StageId.CONTEXT_COMPACTOR not in type(builder).enabled_stages,
    )

    # --------------------------------------------------------
    # ⚠️ 这条是整个特性里最容易静默失效的地方：CLASSIFIER 的策略基线
    # 把 history 关掉了，而用户这句话只有 history 处理器会送进 prompt。
    section("分类调用的 prompt 里必须有用户原话")

    state: ReactState = {
        "messages": [HumanMessage(USER_TEXT)],
        "conversation_id": "c1",
        "user_id": "u1",
    }  # type: ignore[typeddict-item]

    graph_state = builder_module._graph_state(
        state,
        [
            builder_module._ModuleCandidate(
                code=code, name=f"{code}-名称"
            )
            for code in ("financial", "document")
        ],
    )

    model_request = await builder.build(
        ContextBuildRequest(
            builder_type=ContextBuilderType.CLASSIFIER,
            thread_id="t1",
            messages=tuple(state["messages"]),
            graph_state=graph_state,
            agent_type="react",
            model_role="classification",
            dependencies=ContextDependencies(),
        )
    )

    joined = " ".join(
        m.content for m in model_request.messages
        if isinstance(m.content, str)
    )

    check("用户原话进了 prompt", USER_TEXT in joined)
    check("分类提示词进了 prompt", "分诊器" in joined)
    check("候选模块进了提示词", "financial" in joined)
    check("没带工具", len(model_request.tools) == 0)
    check("带上了结构化输出要求", model_request.response_format is not None)

    # 反证：退回 CLASSIFIER 基线（不覆盖 include_history）就该丢掉用户原话。
    # 固定住这个反证，避免以后有人"简化"掉那个覆盖项。
    baseline_state = dict(graph_state)
    baseline_state["context_policy"] = {
        key: value
        for key, value in graph_state["context_policy"].items()
        if key not in (
            "include_history", "min_recent_turns", "max_recent_turns"
        )
    }

    baseline_request = await builder.build(
        ContextBuildRequest(
            builder_type=ContextBuilderType.CLASSIFIER,
            thread_id="t1",
            messages=tuple(state["messages"]),
            graph_state=baseline_state,
            agent_type="react",
            model_role="classification",
            dependencies=ContextDependencies(),
        )
    )

    baseline_joined = " ".join(
        m.content for m in baseline_request.messages
        if isinstance(m.content, str)
    )

    check(
        "反证：不覆盖 include_history 就丢掉用户原话（这就是覆盖的理由）",
        USER_TEXT not in baseline_joined,
    )

    # --------------------------------------------------------
    section("路由：plan_status 决定去向（分类只负责提议）")

    ran, result = await _run_graph(payload=_payload(
        ExecutionPlan.REACT.value
    ))
    check("react -> 直接进 agent", ran == ["builder", "agent"])
    check("react 不进入计划模式",
          (result.get("plan_status") or "off") == "off")

    ran, result = await _run_graph(payload=_payload(
        ExecutionPlan.PLAN.value
    ))
    check(
        "plan -> builder 合成 EnterPlanTool -> tools -> planner -> listener",
        ran == [
            "builder", "tools", "planner", "task_listener", "agent",
        ],
    )
    check("plan 写进 state（而不是只留在 classification）",
          result["plan_status"] == "plan")
    check("进入计划模式时清空旧 plan（交给 planner 覆盖）",
          result["plan"] == {})
    check("来源记成 classifier（来自合成消息的 additional_kwargs）",
          result["plan_origin"] == "classifier")
    check(
        "合成了一条 enter_plan_mode 工具调用（不是直接改 state）",
        any(
            isinstance(m, AIMessage)
            and any(
                call.get("name") == "agent_enter_plan_mode"
                for call in m.tool_calls
            )
            for m in result["messages"]
        ),
    )

    ran, result = await _run_graph(payload=_payload(
        ExecutionPlan.OTHER.value
    ))
    check("other -> 走 agent（兜底回到改动前的行为）",
          ran == ["builder", "agent"])

    # 没有分类结果也走 agent：那一轮可能压根没跑 builder（比如从旧
    # checkpoint 续跑），路由不能因此炸掉。
    from app.agent.graph.react.routing import (
        route_after_agent,
        route_after_builder,
    )

    check(
        "state 里没有 classification -> 走 agent",
        route_after_builder({"messages": []}) == "agent",  # type: ignore[typeddict-item]
    )

    check(
        "plan 且任务未完成 -> agent（继续执行，不重置计划）",
        route_after_builder({  # type: ignore[typeddict-item]
            "messages": [],
            "plan_status": "plan",
            "plan": {"tasks": [{"id": "t1", "status": "pending"}]},
        }) == "agent",
    )

    check(
        "plan 且任务全部完成 -> planner（生成新任务）",
        route_after_builder({  # type: ignore[typeddict-item]
            "messages": [],
            "plan_status": "plan",
            "plan": {"tasks": [{"id": "t1", "status": "done"}]},
        }) == "planner",
    )

    check(
        "plan 且没有任务 -> planner",
        route_after_builder({  # type: ignore[typeddict-item]
            "messages": [],
            "plan_status": "plan",
        }) == "planner",
    )

    check(
        "off -> agent（不论旧计划长什么样）",
        route_after_builder({  # type: ignore[typeddict-item]
            "messages": [],
            "plan_status": "off",
            "plan": {"tasks": [{"id": "t1", "status": "done"}]},
        }) == "agent",
    )

    control_call = AIMessage(
        "",
        tool_calls=[{
            "name": "agent_enter_plan_mode",
            "args": {},
            "id": "call_plan_1",
        }],
    )

    sends = route_after_agent(
        {"messages": [control_call]}, None,  # type: ignore[typeddict-item]
    )
    check(
        "enter_plan_mode 也走 Send 扇出（是真工具）",
        isinstance(sends, list)
        and len(sends) == 1
        and getattr(sends[0], "node", None) == "tools",
    )

    sends = route_after_builder({  # type: ignore[typeddict-item]
        "messages": [control_call],
        "plan_status": "off",
    })
    check(
        "builder 合成的控制调用同样 Send 到 tools",
        isinstance(sends, list)
        and len(sends) == 1
        and getattr(sends[0], "node", None) == "tools",
    )

    # 工具自己返回 Command，并从注入的 ToolRuntime.state 读当前状态。
    from types import SimpleNamespace

    from langgraph.types import Command

    from app.agent.tools.plan_tool import enter_plan_mode, exit_plan_mode

    def rt(state: dict) -> Any:
        return SimpleNamespace(state=state)

    enter_cmd = await enter_plan_mode.coroutine(runtime=rt({}), reason="x")
    exit_cmd = await exit_plan_mode.coroutine(runtime=rt({}), reason="")

    check(
        "enter 工具返回 Command(goto=planner)",
        isinstance(enter_cmd, Command) and enter_cmd.goto == "planner",
    )
    check(
        "enter 在 Command.update 里直接写 plan_status",
        isinstance(enter_cmd.update, dict)
        and enter_cmd.update.get("plan_status") == "plan",
    )
    check(
        "exit 工具返回 Command(goto=agent)",
        isinstance(exit_cmd, Command) and exit_cmd.goto == "agent",
    )

    idle_enter = await enter_plan_mode.coroutine(runtime=rt({}), reason="")
    busy_enter = await enter_plan_mode.coroutine(
        runtime=rt({
            "plan_status": "plan",
            "plan": {"tasks": [{"id": "t1", "status": "pending"}]},
        }),
        reason="",
    )
    done_enter = await enter_plan_mode.coroutine(
        runtime=rt({
            "plan_status": "plan",
            "plan": {"tasks": [{"id": "t1", "status": "done"}]},
        }),
        reason="",
    )
    plan_exit = await exit_plan_mode.coroutine(
        runtime=rt({"plan_status": "plan"}), reason="",
    )

    check("enter 且当前 off -> planner", idle_enter.goto == "planner")
    check(
        "enter 且 plan 任务未完成 -> agent（幂等，不重置计划）",
        busy_enter.goto == "agent"
        and not (busy_enter.update or {}).get("plan"),
    )
    check(
        "enter 且 plan 任务全完成 -> planner（重新规划）",
        done_enter.goto == "planner",
    )
    check(
        "exit 且当前 plan -> agent，并写回 off",
        plan_exit.goto == "agent"
        and (plan_exit.update or {}).get("plan_status") == "off",
    )

    # --------------------------------------------------------
    section("主循环默认开启思考（真实 agent 节点）")

    from app.agent.graph.react.agent_node import agent_node

    async def stub_plain_builder(state: ReactState, runtime: Any) -> dict:
        return {"context_policy": {"tool_keys": []}}

    async def stub_plain_planner(state: ReactState, runtime: Any) -> dict:
        return {}

    NODE_FUNCTIONS["agent"] = agent_node
    NODE_FUNCTIONS["builder"] = stub_plain_builder
    NODE_FUNCTIONS["planner"] = stub_plain_planner

    thinking_graph = build_react_graph().compile(
        checkpointer=InMemorySaver()
    )

    thinking_context = ReactContext(
        pipeline_run_id=uuid4(),
        recorder=_FakeRecorder(),
        model_registry=_FakeRegistry("答复"),
        builder=_FakeBuilder(),
        classifier_builder=_FakeBuilder(),
    )

    _FakeRuntime.last_request = None

    await thinking_graph.ainvoke(
        {"messages": [HumanMessage(USER_TEXT)]},
        {"configurable": {"thread_id": f"thinking-{uuid4()}"}},
        context=thinking_context,
    )

    check(
        "react 主循环的 ModelRequest.enable_thinking=True",
        _FakeRuntime.last_request is not None
        and _FakeRuntime.last_request.enable_thinking is True,
    )
    check(
        "主循环没有 response_format（JSON schema 才会关思考）",
        _FakeRuntime.last_request.response_format is None,
    )

    # --------------------------------------------------------
    section("计划模式工具：enter / exit 真的改变图状态")

    ran, result = await _run_graph(
        payload=_payload(ExecutionPlan.REACT.value),
        agent_tool_calls=[{
            "name": "agent_enter_plan_mode",
            "args": {"reason": "需要先拆解"},
            "id": "call_plan_enter",
        }],
    )
    check(
        "enter -> tools -> planner -> listener -> agent",
        ran == [
            "builder", "agent", "tools", "planner",
            "task_listener", "agent",
        ],
    )
    check("plan_status 被写成 plan", result["plan_status"] == "plan")
    check("plan_origin 记成 tool", result["plan_origin"] == "tool")
    check("进入时清空旧 plan（交给 planner 覆盖）",
          result["plan"] == {})
    check(
        "控制工具真的执行了，并留下 ToolMessage（不留悬空 id）",
        any(
            isinstance(m, ToolMessage)
            and "已进入计划模式" in str(m.content)
            and m.tool_call_id == "call_plan_enter"
            for m in result["messages"]
        ),
    )
    check(
        "控制工具常驻（exit 在进入后立即可见）",
        "agent.exit_plan_mode"
        in result["context_policy"]["tool_keys"],
    )

    ran, result = await _run_graph(
        payload=_payload(ExecutionPlan.REACT.value),
        initial_extra={
            "plan_status": "plan",
            "plan": {"tasks": [{"id": "t1", "status": "pending"}]},
        },
        agent_tool_calls=[{
            "name": "agent_exit_plan_mode",
            "args": {},
            "id": "call_plan_exit",
        }],
    )
    check(
        "exit -> tools -> agent（不进 planner）",
        ran == ["builder", "agent", "tools", "agent"],
    )
    check("plan_status 被写回 off", result["plan_status"] == "off")
    check(
        "退出是直接退出，不经过审批 / 中断",
        any(
            isinstance(m, ToolMessage)
            and "已退出计划模式" in str(m.content)
            for m in result["messages"]
        ),
    )
    check(
        "控制工具常驻（exit 不因模式切换从清单里消失）",
        "agent.exit_plan_mode"
        in result["context_policy"]["tool_keys"],
    )

    ran, result = await _run_graph(
        payload=_payload(ExecutionPlan.REACT.value),
        initial_extra={
            "plan_status": "plan",
            "plan": {"tasks": [{"id": "t1", "status": "pending"}]},
        },
        agent_tool_calls=[{
            "name": "agent_enter_plan_mode",
            "args": {},
            "id": "call_plan_enter_again",
        }],
    )
    check(
        "重复 enter 幂等：仍在 plan，直接回 agent",
        ran == ["builder", "agent", "tools", "agent"],
    )
    check(
        "不重置正在执行的计划",
        result["plan"]["tasks"][0]["status"] == "pending",
    )

    # 多个控制调用：两个分支都执行，但只有**控制器**返回 Command，
    # 所以 planner 只跳一次。
    ran, result = await _run_graph(
        payload=_payload(ExecutionPlan.REACT.value),
        agent_tool_calls=[
            {
                "name": "agent_enter_plan_mode",
                "args": {},
                "id": "call_enter_a",
            },
            {
                "name": "agent_enter_plan_mode",
                "args": {},
                "id": "call_enter_b",
            },
        ],
    )
    check(
        "两个控制分支都执行，但只跳一次 planner",
        ran.count("tools") == 2
        and ran[-3:] == ["planner", "task_listener", "agent"],
    )
    check(
        "每个 tool_call 都有对应的 ToolMessage",
        sum(
            1
            for m in result["messages"]
            if isinstance(m, ToolMessage)
            and m.tool_call_id in {"call_enter_a", "call_enter_b"}
        ) == 2,
    )
    check("多分支并发写 plan_status 不冲突", result["plan_status"] == "plan")

    # --------------------------------------------------------
    section("分类结果写回 state，且不进对话历史")

    ran, result = await _run_graph(
        payload=_payload(
            ExecutionPlan.REACT.value,
            behavior=BehaviorCategory.ANALYSIS.value,
            capability_needs=["document"],
            capability_config={"window": "quarter"},
        ),
        candidates=["financial", "document"],
    )

    classification = result["classification"]

    check("behavior 落库", classification["behavior"] == "analysis")
    # ⚠️ 这个取值**必须**在候选集里：候选外的取值会落 other（见下面的用例）。
    check("capability_needs 落库",
          classification["capability_needs"] == ["document"])
    # ⚠️ modules 恒空：能力 code 只有 capability_needs 一个出口。这条断言
    # 不是"没填"，而是**钉住那个决定** —— 见 agent_builder_node 里那段
    # "能力 code 只有一个出口"。哪天有人又把分桶塞回模型那边，这条会红。
    check("modules 恒为空（能力 code 不再走这条路）",
          classification["modules"] == [])
    check("capability_config 落库",
          classification["capability_config"] == {"window": "quarter"})
    check("没被标记降级", classification["degraded"] is False)

    # ⚠️ 只断言 messages 不够：如果 classification 通道压根没生效，
    # "消息里没有分类问答"会因为**通道被静默丢弃**而同样成立。
    # 必须与"classification 确实有值"一起断言。
    check(
        "分类那次 LLM 问答没进 messages",
        [type(m).__name__ for m in result["messages"]]
        == ["HumanMessage", "AIMessage"],
    )

    # --------------------------------------------------------
    section("候选能力进提示词：标识 + 名称 + 描述")

    # 候选只给 code 的话，模型是在做字符串相似度匹配；名称与描述才是
    # "这句话与哪条能力对得上"的判断依据。这几条断言盯的就是它们真的
    # 拼进去了。
    prompt = _LAST_CLASSIFIER_REQUEST[-1].graph_state[
        "context_policy"
    ]["system_prompt"]

    check(
        "每项都带上标识与名称",
        "financial：financial-名称" in prompt
        and "document：document-名称" in prompt,
    )
    check("描述也拼进去了", "financial-描述" in prompt)
    check(
        "明确要求回填的是**标识**（不是名称）",
        "capability_needs 只填上面每项**开头的那个标识（code）**" in prompt,
    )
    # ⚠️ 提示词里**只该有 capability_needs 一个出口**。以前它同时讲
    # modules / capability_needs 两个桶，实测模型分不对（把 CAPABILITY
    # 填进了 modules）。哪天有人把两个桶写回提示词，这条会红。
    check(
        "提示词里不再提 modules（只有一个出口）",
        "modules" not in prompt,
    )
    check(
        "候选为空时给的是「没有清单」而不是一个空列表",
        "当前没有可用的能力清单"
        in get_prompt_provider().node_prompt(
            builder_module.CLASSIFIER_MODEL_ROLE, candidates=[]
        ),
    )

    # --------------------------------------------------------
    section("主 agent 的职责提示词：经 context_policy 进主循环")

    # 主循环的 system 消息只有一个出口（ContextBuilder 的 system_message()
    # = policy.system_prompt + 注入分节），而 policy 由 builder 节点写进
    # state —— 这一节盯的就是那条线真的接通了：**agent_node 一行没改**，
    # 但主模型从此有系统提示词。
    fake = _FakeProvider("你是冒烟用的职责提示词。")

    set_prompt_provider(fake)

    try:
        ran, result = await _run_graph(
            payload=_payload(ExecutionPlan.REACT.value),
            duty_key="stock_research",
        )
    finally:
        reset_prompt_provider()

    # ⚠️ 这一条盯的是**取值链路**：选职责在 `create_run` 那一步（落
    # `agent_run.duty_key`），派发时进 state，builder 节点只是照它取文本。
    # 链路断了的话，运行行上写着 stock_research，模型拿到的却还是通用职责 ——
    # 而库里两处都对得上，只有提示词是错的。
    check(
        "state 里的 duty_key 被原样传给了 Provider",
        fake.requested == ["stock_research"],
    )

    check(
        "职责提示词写进了 context_policy.system_prompt",
        result["context_policy"]["system_prompt"]
        == "你是冒烟用的职责提示词。",
    )
    # ⚠️ 分类那次调用必须仍然拿**分类**提示词。两条路共用 context_policy
    # 这一个键，顺序写反了（先 system_prompt 再 spread existing）就会让
    # 分类模型收到主 agent 的人设，而症状只是"分类偶尔判得奇怪"。
    check(
        "分类调用拿到的仍是分类提示词（职责提示词没串过去）",
        "分诊器"
        in _LAST_CLASSIFIER_REQUEST[-1].graph_state[
            "context_policy"
        ]["system_prompt"],
    )

    # state 里**没有** duty_key 时（存量运行、或者没跑过建运行那条路的
    # 脚本），传下去的是 None —— Provider 据此回落默认职责。空串走的是
    # 同一条（`state.get(...) or None`），这里锚住"没指定"这个约定。
    bare = _FakeProvider("你是冒烟用的职责提示词。")

    set_prompt_provider(bare)

    try:
        await _run_graph(payload=_payload(ExecutionPlan.REACT.value))
    finally:
        reset_prompt_provider()

    check(
        "state 里没有 duty_key -> 传 None（= 没指定，交回默认职责）",
        bare.requested == [None],
    )

    # --------------------------------------------------------
    section("职责提示词取不到时降级（不写、但看得见）")

    # 冒烟里没有 ServiceFactory（没跑过 lifespan），所以默认 Provider 走
    # 的正是这条降级路 —— 与 _tool_keys_for_capabilities 同一条纪律：
    # 主循环照常跑，降级写进 notes。
    ran, result = await _run_graph(
        payload=_payload(ExecutionPlan.REACT.value)
    )

    check(
        "取不到就不写 system_prompt（退回本轮改动前的行为）",
        "system_prompt" not in (result["context_policy"] or {}),
    )
    check(
        "降级看得见：notes 里写明原因",
        any(
            "职责提示词不可用" in note
            for note in result["classification"]["notes"]
        ),
    )
    check(
        "降级不算「分类失败」（degraded 仍为 False）",
        result["classification"]["degraded"] is False,
    )

    # --------------------------------------------------------
    section("纠偏与降级")

    ran, result = await _run_graph(
        payload=_payload("PLAN", behavior="  qA  "),
    )
    classification = result["classification"]
    check("大小写与空白被纠偏", classification["behavior"] == "qa")
    check("大写枚举值也算数（PLAN -> plan）",
          classification["execution_plan"] == "plan")

    ran, result = await _run_graph(
        payload=_payload("execute_directly"),
    )
    check("枚举外的执行方案 -> other",
          result["classification"]["execution_plan"] == "other")

    ran, result = await _run_graph(
        payload=_payload(
            ExecutionPlan.REACT.value,
            capability_needs=["financial", "查无此能力", "financial"],
        ),
        candidates=["financial", "document"],
    )
    check(
        "候选外的能力 -> other，重复项去掉",
        result["classification"]["capability_needs"]
        == ["financial", "other"],
    )

    # 模型若仍然回填 modules（老提示词、或它自作主张），装配层**不采纳**：
    # 能力 code 只有一个出口。这条盯的是"分桶不再由模型决定"。
    ran, result = await _run_graph(
        payload=_payload(
            ExecutionPlan.REACT.value,
            modules=["financial"],
            capability_needs=["document"],
        ),
        candidates=["financial", "document"],
    )
    check(
        "模型回填的 modules 被忽略（不采纳）",
        result["classification"]["modules"] == []
        and result["classification"]["capability_needs"] == ["document"],
    )

    # --------------------------------------------------------
    section("结构化输出是 pydantic 实例时不被降级（真模型路径）")

    # ⚠️ 这一条盯的是一个**已经发生过的** bug：真模型路径下
    # `with_structured_output` 回的是解析好的 `_ClassificationOutput`
    # 实例，而 `_parse` 曾经只认 dict 与 JSON 字符串，于是**每一次分类都被
    # 降级成"全 other"**（note 写着"结构化输出类型意外：_ClassificationOutput"）——
    # 症状是模型一个工具都看不到，而 state 里只有一句 notes。
    #
    # 必须**直接调 `_parse`**：本文件的 `_FakeRuntime` 永远把 payload 包在
    # 一个带 `.content` 的对象里（见上面的类定义），走的正是当年能通过的
    # 那条分支 —— 用图跑永远复现不出这个 bug。
    _CANDIDATES = [
        builder_module._ModuleCandidate(
            code="financial", name="financial-名称",
        ),
        builder_module._ModuleCandidate(
            code="document", name="document-名称",
        ),
    ]

    structured = builder_module._ClassificationOutput(
        behavior=BehaviorCategory.QA.value,
        execution_plan=ExecutionPlan.REACT.value,
        capability_needs=["financial", "document"],
        capability_config={"window": "quarter"},
        # ⚠️ 两个 flag 是**必填**（与上面四个字段一样）：缺了 pydantic 会让
        # 整条结构化输出失败，而那是"每一次分类都降级"的又一条路径。
        # 保守值是 `"false"`（不召回），但它必须由模型**明确回答**。
        retrieve_l0="true",
        retrieve_l1="false",
    )

    parsed = builder_module._parse(
        structured, candidates=_CANDIDATES, notes=[],
    )

    check("pydantic 实例不被降级", parsed["degraded"] is False)
    check("notes 里没有「类型意外」", parsed["notes"] == [])
    check("behavior 从实例里收进来", parsed["behavior"] == "qa")
    check("capability_needs 从实例里收进来",
          parsed["capability_needs"] == ["financial", "document"])
    check("一个 code 都没丢（实例路径不经过分桶）", parsed["modules"] == [])
    check("execution_plan 从实例里收进来",
          parsed["execution_plan"] == "react")
    check("capability_config 从实例里收进来",
          parsed["capability_config"] == {"window": "quarter"})
    check("retrieve_l0 从实例里收进来（宽松 str -> bool）",
          parsed["retrieve_l0"] is True)
    check("retrieve_l1 同样", parsed["retrieve_l1"] is False)

    # --------------------------------------------------------
    section("记忆召回的决策：classification -> context_policy")

    # ⚠️ 这一条盯的是"决策到底传到流水线没有"：分类判出来只留在
    # `classification` 里是**没有用**的 —— 记忆处理器读的是
    # `context_policy.memory_retrieve`（与 `tool_keys` 同一条路，见
    # agent_builder_node 的 `_memory_retrieve`）。
    ran, result = await _run_graph(
        payload=_payload(
            ExecutionPlan.REACT.value,
            retrieve_l0="true",
            retrieve_l1="no",
        ),
        duty_key="general",
    )

    check(
        "决策写进了 context_policy.memory_retrieve",
        (result.get("context_policy") or {}).get("memory_retrieve")
        == {"l0": True, "l1": False, "agent_id": "general"},
    )
    check(
        "分类结果里也留着这次判断（供观测读）",
        result["classification"]["retrieve_l0"] is True
        and result["classification"]["retrieve_l1"] is False,
    )

    # 认不出的字面量 -> 落保守值，而**不是**让整条分类降级。
    ran, result = await _run_graph(
        payload=_payload(ExecutionPlan.REACT.value, retrieve_l0="嗯这个"),
        duty_key="general",
    )

    check(
        "认不出的 flag -> 保守 False，且分类没降级",
        result["classification"]["degraded"] is False
        and (result.get("context_policy") or {})
        .get("memory_retrieve", {}).get("l0") is False,
    )

    # duty_key 为空 -> 干脆不给决策：拿一个空身份去拼 RetrievalTarget 会
    # 当场 ValidationError，而"这个 agent 是谁"判不出来时不召回是对的。
    ran, result = await _run_graph(
        payload=_payload(ExecutionPlan.REACT.value, retrieve_l0="true"),
        duty_key=None,
    )

    check(
        "没有 duty_key -> 不给决策（宁可这一轮不召回）",
        (result.get("context_policy") or {}).get("memory_retrieve") == {},
    )

    # --------------------------------------------------------
    section("分类判出的 code 怎么变成工具查询")

    # 这一节验的是"code -> tool_keys"那一段，而它**只发生在
    # `_tool_keys_for_capabilities` 里** —— 冒烟默认没有 ServiceFactory，
    # 那条路会降级成"工具目录不可用"，于是"到底拿什么 code 去查了"断言
    # 不到。这里塞一个假的 ServiceFactory 把传进 service 的 codes 抓下来。
    #
    # 注意 `_run_graph` 已经把 `_module_candidates` 换成了桩（见 `candidates=`
    # 那个参数），所以下面这个假 factory 只会被工具查找用到。
    captured: list[list[str]] = []
    override_queries: list[list[str]] = []

    class _FakeToolQuery:
        async def tool_keys_for_capabilities(self, codes: list[str]):
            captured.append(list(codes))
            return ["t.financial", "t.document"]

        async def tool_documentation_overrides(
                self, tool_keys: list[str]
        ):
            # 说明覆盖的预载与 tool_keys 是**同一轮、同一条降级路**，
            # 所以假实现也必须在这里 —— 缺了它，那一格会静默降级成
            # "读取失败"，而本节验的是正常路径。
            override_queries.append(list(tool_keys))
            return {"t.document": {"description": "人工写的说明"}}

    class _FakeScope:
        async def __aenter__(self) -> _FakeToolQuery:
            return _FakeToolQuery()

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    class _FakeFactory:
        def scope(self, service_cls: Any, **kwargs: Any) -> _FakeScope:
            return _FakeScope()

    original_factory = builder_module.get_service_factory
    builder_module.get_service_factory = lambda: _FakeFactory()

    try:
        ran, result = await _run_graph(
            payload=_payload(
                ExecutionPlan.REACT.value,
                modules=["financial"],   # 模型回填的，应被忽略
                capability_needs=["document", MODULE_OTHER],
            ),
            candidates=["financial", "document"],
        )
    finally:
        builder_module.get_service_factory = original_factory

    check(
        "查的是 capability_needs 判出的 code，且过滤掉 other",
        captured == [["document"]],
    )
    check(
        "只在**一次**查询里查完（不是每个维度查一次）",
        len(captured) == 1,
    )
    check(
        "查回来的 tool_key 写进了 context_policy，并并入了控制工具",
        result["context_policy"]["tool_keys"]
        == [
            "t.financial",
            "t.document",
            "agent.enter_plan_mode",
            "agent.exit_plan_mode",
            "agent.ask_user_question",
        ],
    )
    check(
        "人工说明覆盖按同一批 key 预载，并写进了 context_policy",
        override_queries == [["t.financial", "t.document"]]
        and result["context_policy"]["tool_overrides"]
        == {"t.document": {"description": "人工写的说明"}},
    )
    check(
        "取覆盖失败不会把这一轮标成降级（它是可降级的旁路）",
        not any(
            "工具说明" in note
            for note in result["classification"]["notes"]
        ),
    )

    ran, result = await _run_graph(payload="这不是 JSON")
    check("结构化输出解析不了 -> 全 other",
          result["classification"]["execution_plan"] == "other")
    check("并标记降级", result["classification"]["degraded"] is True)

    ran, result = await _run_graph(boom=True)
    check("分类调用失败 -> 不抛异常，仍走 agent",
          ran == ["builder", "agent"])
    check("标记降级", result["classification"]["degraded"] is True)
    check("notes 写明原因",
          any("假装分类模型挂了" in n
              for n in result["classification"]["notes"]))
    check("user 仍拿到答复",
          result["messages"][-1].content == "答复")

    # --------------------------------------------------------
    section("没有用户消息时不炸")

    empty_state: ReactState = {"messages": []}  # type: ignore[typeddict-item]

    empty_result = await builder_module.agent_builder_node(
        empty_state,
        _FakeRuntimeContext(),
    )

    check("落一个标记过的空结果，而不是抛异常",
          empty_result["classification"]["degraded"] is True)
    check("没有 messages 键（不往对话里写东西）",
          "messages" not in empty_result)

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
