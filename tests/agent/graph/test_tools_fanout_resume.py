"""
`tools` 扇出的回归测试：**resume 不能重跑已完成的工具**。

这条测试守的是一个具体的历史缺陷
--------------------------------
tools 节点原来是一个节点里 `for` 循环执行全部 tool_calls。`interrupt()`
会**丢弃该节点待写的状态、并从节点开头重跑**，于是"两个调用、第二个
需要审批"这件事的实测结果是：

    中断后   副作用 ['t1']
    resume 后 副作用 ['t1', 't1', 't2']     <- t1 被执行了两遍

下单、转账、写库这类工具的副作用重复发生是不可接受的。改成 Send 扇出
（一个 tool_call 一个分支实例）之后：

    中断后   副作用 ['t1']
    resume 后 副作用 ['t1', 't2']

本文件跑的是**真的图**：`register_react_graph` + InMemorySaver + 真的
`tools` 节点 + 真的 `route_after_agent`。被换掉的只有三个东西：

    builder / agent / planner   桩（不调模型）
    get_tool_executor           桩执行器（不碰数据库、不造 service）
    _resolve_tool_ids           桩（返回 {} = 查过了、这些工具没有链）

执行账本已经搬到处理器链里的 AROUND 拦截器
（`processor/interceptor/execution_guard.py`）。本文件不接数据库，
也不挂处理器链，因此**不覆盖**账本语义；守卫自己的单测在
`tests/agent/tools/test_execution_guard.py`。本文件守的是**扇出**
本身：resume 不重跑已完成的分支、每个 tool_call 都有 ToolMessage。

⚠️ 桩执行器必须**先 interrupt、后记副作用**
-----------------------------------------
审批发生在工具跑**之前** —— 这是"工具还没执行就被拦下"的真实顺序。
如果反过来（先记再问），两种结构都会得到"只跑一次"的假象：重跑时那个
工具也只是被重新"记录"了一次，而不是重新"执行"，测试就失去了分辨力。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from fakes import FakeAdapter, make_definition, make_registry

from app.agent.graph.react import graph as graph_module
from app.agent.graph.react import tools as tools_module
from app.agent.graph.react.catalog import AGENT_TYPE, MODEL_ROLE
from app.agent.graph.react.runtime import register_react_graph
from app.agent.model import (
    ModelCapabilities,
    ModelProfile,
    ModelRoleBinding,
    ModelRuntimePolicy,
)
from app.agent.tools.processor.context import ToolOutcome
from app.infra.graph import GraphRegistry
from app.services.tool.registry.decorator import tool

THREAD_ID = "fanout-thread"

#: 两个工具调用。**刻意让第二个需要审批**：中断发生在它身上，而第一个
#: 已经跑完 —— 那正是"重跑会不会连累已完成的分支"要验的位置。
#
#: ⚠️ 这里的 `name` 必须是**线上名**（下划线），不是 tool_key（点号）
#: -----------------------------------------------------------------
#: 模型看到、回传的都是 `for_wire` 改过名的那个字符串，而 tools 节点
#: 用 `tool_key_of` 翻回 tool_key 再判允许集（见 tools.py 模块
#: docstring 的"允许集里的 key 与模型回传的名字不是同一个字符串"）。
#:
#: 写成 `fanout.one` 的症状是**静默的**：节点判它"未暴露"，回一条
#: error ToolMessage，工具一次都不执行 —— 而断言里只看 `executor.ran`
#: 的话会以为是别的原因。这个文件的前身就是这么错的。
CALLS = [
    {"name": "fanout_one", "args": {}, "id": "call_1", "type": "tool_call"},
    {"name": "fanout_two", "args": {}, "id": "call_2", "type": "tool_call"},
]


# ============================================================
# 两个真的 @tool（只为让 select_specs 命中，执行被桩掉了）
# ============================================================


class FanoutProbe(ABC):
    """契约 —— `spec.contract` 要它来推实现。"""

    @abstractmethod
    async def one(self) -> str: ...

    @abstractmethod
    async def two(self) -> str: ...


class FanoutProbeImpl(FanoutProbe):

    @tool(key="fanout.one")
    async def one(self) -> str:
        """第一个探针工具（本测试不真的执行它）。"""
        return "one"

    @tool(key="fanout.two")
    async def two(self) -> str:
        """第二个探针工具（本测试不真的执行它）。"""
        return "two"


# ============================================================
# 桩
# ============================================================


class _StubExecutor:
    """
    记录"哪些工具**真的跑了**"。

    ⚠️ `interrupt` 在记录之前 —— 见模块 docstring。
    """

    def __init__(self) -> None:
        self.ran: list[str] = []
        #: 每次调用收到的 caller_identity，用来验"身份真的走到了执行器"。
        self.seen_caller: list[Any] = []

    async def execute(
            self,
            tool_key: str,
            arguments: dict[str, Any] | None,
            *,
            tool_id: Any = None,
            tool_call_id: str | None = None,
            caller_identity: Any = None,
            runtime: Any = None,
    ) -> ToolOutcome:

        self.seen_caller.append(caller_identity)

        if tool_key == "fanout.two":
            # 审批在工具执行**之前**。返回什么值不重要 —— 本测试不接
            # HITL 的审批解析，只关心"这次中断之后谁被重跑了"。
            interrupt({"ask": f"批准 {tool_key} 吗？"})

        self.ran.append(tool_key)

        return ToolOutcome(value=f"done:{tool_key}")


class _Recorder:
    """收 node_span 写下来的观测行。"""

    def __init__(self) -> None:
        self.updates: list = []

    async def upsert_node_run(self, update) -> None:
        self.updates.append(update)

    def by_node(self, node_id: str) -> list:
        return [item for item in self.updates if item.node_id == node_id]


class _RunContext:
    def __init__(self, recorder, caller_identity=None) -> None:
        from uuid import uuid4

        self.pipeline_run_id = uuid4()
        self.thread_id = THREAD_ID
        self.trigger_source = "test"
        self.recorder = recorder
        # RunContext 契约的一部分（见 infra/graph/registry.py）。
        self.caller_identity = caller_identity
        # 工具执行账本的判重键第一段（agent_session.session_id）。
        # ⚠️ 它必须**按运行固定**：同一份 context 复用两次 = 同一个会话
        # 同一个 run，那正是"重复调用"要验的场景。
        self.session_id = uuid4()
        # 这一轮的职责。它进 ToolCaller.agent_id（需要 agent 那一半 scope
        # 的工具靠它定位），本文件给一个固定值 —— 与 session_id 同一条
        # 理由：按运行固定，不按轮次变。
        self.duty_key = "test_duty"
        # Trace 聚合键（RunContext 契约的一部分）。本文件不测 trace。
        self.trace_id = None




# ============================================================


def _has_tool_round(state: dict) -> bool:
    """state 里是否已经出现过"要求调工具"的 AIMessage。"""

    return any(
        isinstance(message, AIMessage) and message.tool_calls
        for message in state.get("messages") or ()
    )


def _build(
        monkeypatch,
        executor: _StubExecutor,
        *,
        caller_identity: dict[str, Any] | None = None,
        tool_keys: list[str] | None = None,
        calls: list[dict[str, Any]] | None = None,
) -> tuple:

    registry = make_registry(
        models=[
            make_definition(
                "primary",
                capabilities=ModelCapabilities(
                    tool_calling=True, structured_output=True
                ),
            )
        ],
        profiles=[
            ModelProfile(
                id="reasoning_default",
                primary="fake:primary",
                policy=ModelRuntimePolicy(
                    timeout=5.0, max_retries=0, retry_backoff=0.0
                ),
            )
        ],
        bindings=[
            ModelRoleBinding(
                agent_type=AGENT_TYPE,
                model_role=MODEL_ROLE,
                profile="reasoning_default",
            )
        ],
        adapter=FakeAdapter(script={}),
    )

    recorder = _Recorder()

    exposed = (
        tool_keys
        if tool_keys is not None
        else ["fanout.one", "fanout.two"]
    )

    async def stub_builder(state, runtime) -> dict:
        # 只做一件事：把本次暴露的工具 key 写进 context_policy ——
        # 路由会把它们原文带进每个 Send payload。
        #
        # ⚠️ 每次都会**覆盖**，所以用例没法靠预置 state 里那份来控制
        # 暴露集 —— 要控制就传 tool_keys=（见 _build 的参数）。
        return {"context_policy": {"tool_keys": list(exposed)}}

    async def stub_agent(state, runtime) -> dict:

        # ⚠️ 只在**还没有过工具轮**时要求调工具：本图是
        # `agent ->(Send)-> tools -> agent` 的循环，每次都要调工具的桩
        # 会一路转到 recursion_limit，那验的是循环上限而不是本用例。
        if not _has_tool_round(state):
            return {
                "messages": [
                    AIMessage(
                        content="我来跑两个工具",
                        tool_calls=list(calls or CALLS),
                    )
                ]
            }

        return {"messages": [AIMessage(content="答复")]}

    async def stub_planner(state, runtime) -> dict:
        return {}

    monkeypatch.setitem(graph_module.NODE_FUNCTIONS, "builder", stub_builder)
    monkeypatch.setitem(graph_module.NODE_FUNCTIONS, "agent", stub_agent)
    monkeypatch.setitem(graph_module.NODE_FUNCTIONS, "planner", stub_planner)
    # tools 刻意**不换** —— 本文件要的就是真的那个扇出节点。

    monkeypatch.setattr(
        tools_module, "get_tool_executor", lambda: executor
    )

    async def stub_tool_ids(tool_keys):
        # {} = 查过了、这些工具没有处理器链 -> 直接执行。
        # （None 是另一回事：查不了，那会拒绝执行。）
        return {}

    monkeypatch.setattr(tools_module, "_resolve_tool_ids", stub_tool_ids)

    definition = register_react_graph(
        registry=GraphRegistry(),
        checkpointer=InMemorySaver(),
        recorder=recorder,
        model_registry=registry,
    )

    context = definition.context_factory(
        _RunContext(recorder, caller_identity)
    )

    return definition, context, recorder


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="跑一下两个工具", id="u1")],
        "user_id": "u1",
        "conversation_id": "c1",
    }


async def _invoke(definition, context, payload) -> dict:

    return await definition.compiled.ainvoke(
        payload,
        config={"configurable": {"thread_id": THREAD_ID}},
        context=context,
    )


# ============================================================
# 用例
# ============================================================


async def test_resume_does_not_rerun_finished_tool(monkeypatch) -> None:
    """
    核心：中断发生在第二个工具上，resume 之后**第一个工具只跑过一次**。

    这一条在单节点循环的旧实现下会失败（ran 会是
    `['fanout.one', 'fanout.one', 'fanout.two']`）。
    """

    executor = _StubExecutor()

    definition, context, _ = _build(monkeypatch, executor)

    first = await _invoke(definition, context, _state())

    # 图停下来等人 —— 这是 interrupt 的正常返回，不是异常。
    assert "__interrupt__" in first

    # 第一个工具跑完了；第二个在审批之前就被拦下，还没执行。
    assert executor.ran == ["fanout.one"]

    final = await _invoke(definition, context, Command(resume=True))

    # ⚠️ 这条就是本文件存在的理由。
    assert executor.ran == ["fanout.one", "fanout.two"]
    assert executor.ran.count("fanout.one") == 1

    # 收尾：模型拿着两条工具结果给出了答复。
    assert final["messages"][-1].content == "答复"


async def test_each_call_gets_its_own_tool_message(monkeypatch) -> None:
    """
    扇出 N 个分支 = N 条 ToolMessage，且 tool_call_id 一一对应。

    少一条会让下一轮 agent 看到悬空的 tool_call（部分 provider 直接
    拒绝这次请求），所以这条比"内容对不对"更硬。
    """

    executor = _StubExecutor()

    definition, context, _ = _build(monkeypatch, executor)

    await _invoke(definition, context, _state())

    final = await _invoke(definition, context, Command(resume=True))

    tool_messages = [
        message for message in final["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 2
    assert [message.tool_call_id for message in tool_messages] == [
        "call_1", "call_2"
    ]
    contents = [message.content for message in tool_messages]
    assert all("<ref:tool" in content for content in contents)
    assert "done:fanout.one" in contents[0]
    assert "done:fanout.two" in contents[1]
    assert all(message.status == "success" for message in tool_messages)


async def test_branch_identity_is_stable_across_resume(monkeypatch) -> None:
    """
    两个分支各写一行、且 resume 后**不会多出一行**。

    观测行的唯一键是 `(pipeline_run_id, node_id, branch_key, attempt)`。
    branch_key 若在 resume 时算出一个不同的值，被中断的那个分支就会
    多占一行（三行而不是两行）—— 那正是不确定性 scope 的症状。
    """

    executor = _StubExecutor()

    definition, context, recorder = _build(monkeypatch, executor)

    await _invoke(definition, context, _state())
    await _invoke(definition, context, Command(resume=True))

    rows = recorder.by_node("tools")

    # ⚠️ 这里数的是 **distinct branch_key**，不是行数：node_span 在开始
    # 与结束时各写一次（RUNNING -> SUCCEEDED），所以两个分支会留下多于
    # 两条更新 —— 真正做去重的是仓储那条 upsert 的唯一键
    # `(pipeline_run_id, node_id, branch_key, attempt)`。
    #
    # 因此"resume 会不会多出一行"等价于"distinct branch_key 是不是 2"：
    # 被中断的那个分支若在 resume 时算出另一把键，这里就是 3。
    keys = {row.branch_key for row in rows}

    assert len(keys) == 2
    assert all(key for key in keys)
    assert all(row.branch_scope_id is not None for row in rows)


async def test_caller_identity_reaches_the_executor(monkeypatch) -> None:
    """
    调用者身份一路走到执行器。

    这条断的是**那根线**，不是判定逻辑：RunContext -> ReactContext ->
    tools 分支 -> ToolExecutor.execute(caller_identity=...)。判定逻辑
    在 tool_processor_smoke 的 checkauth 那一节。

    ⚠️ 它守的是一个**静默**的失效：这根线断掉之后，checkauth 看到的是
    "没有身份"，于是**拒绝**——现象是"受保护的工具突然都调不动了"，
    而原因（少传一个参数）离现场很远。
    """

    executor = _StubExecutor()
    identity = {
        "user_id": "u1",
        "groups": ["Y-Trader-User"],
        "roles": ["trader_admin"],
    }

    definition, context, _ = _build(
        monkeypatch, executor, caller_identity=identity
    )

    await _invoke(definition, context, _state())
    await _invoke(definition, context, Command(resume=True))

    # ⚠️ 不数条数：被中断的那个分支**尝试过两次**（一次撞上 interrupt，
    # resume 后又一次），而"执行器收到了什么"在这两次上都要成立。
    # 断言的是"每一次拿到的都是同一份快照" —— 它按运行固定，不按轮次变。
    assert executor.seen_caller
    assert all(seen == identity for seen in executor.seen_caller)
