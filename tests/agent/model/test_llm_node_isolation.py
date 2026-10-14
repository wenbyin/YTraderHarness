"""
LLM Node 与 Model Runtime 的隔离（规格 §22 最后两条 + §9、§15）。

这一份是**端到端**的：真的跑编译过的 LangGraph 图（InMemorySaver +
register_react_graph + agent_node），只是把 provider 换成了假适配器。

它验证三件事，每一件都是规格里点名的：

    1. 一次 LLM Node execution 对应**一次** runtime.invoke，而它内部
       可以有多达 N 次 provider attempt（retry / fallback）
    2. fallback **不会**重跑节点、不会重建上下文
    3. LLM Node 里不创建 provider、不解析模型、不执行 fallback

第 3 条用两种方式测：行为断言（计数）+ 源码断言（禁止出现的符号）。
源码断言不是"洁癖"：这类边界一旦被某次"顺手加个兜底"破坏，行为断言
往往还能过（比如节点里 `except ModelError: 用别的模型再试一次`，
在没有失败场景时完全看不出来），而符号会立刻出现。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from fakes import (
    FakeAdapter,
    FakeRecorder,
    Outcome,
    StubContextBuilder,
    make_definition,
    make_registry,
    make_run_context,
)

from app.agent.graph.react.catalog import AGENT_TYPE, MODEL_ROLE
from app.agent.graph.react.runtime import register_react_graph
from app.agent.model import (
    ModelCapabilities,
    ModelCapabilityError,
    ModelProfile,
    ModelRoleBinding,
    ModelRuntimePolicy,
    ModelUnavailableError,
)
from app.infra.graph import GraphRegistry, NodeRunStatus


TOOL_CAPABLE = ModelCapabilities(tool_calling=True, structured_output=True)


class ToolBuilder(StubContextBuilder):
    """一个"这次有工具可用"的 builder（用来触发能力门禁）。"""

    display_name = "带工具的测试 builder"
    tools = (object(),)


@pytest.fixture(autouse=True)
def _reset_build_counter():
    """`build_count` 是类属性，用例之间必须清零。"""

    StubContextBuilder.build_count = 0
    yield
    StubContextBuilder.build_count = 0


def _graph(
        adapter,
        *,
        primary_caps=TOOL_CAPABLE,
        fallback_caps=TOOL_CAPABLE,
        fallbacks=("fake:backup",),
        builder=None,
        max_retries=2,
):
    registry = make_registry(
        models=[
            make_definition("primary", capabilities=primary_caps),
            make_definition("backup", capabilities=fallback_caps),
        ],
        profiles=[
            ModelProfile(
                id="reasoning_default",
                primary="fake:primary",
                fallbacks=tuple(fallbacks),
                policy=ModelRuntimePolicy(
                    timeout=1.0,
                    max_retries=max_retries,
                    retry_backoff=0.0,
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
        adapter=adapter,
    )

    recorder = FakeRecorder()
    graph_registry = GraphRegistry()

    definition = register_react_graph(
        registry=graph_registry,
        checkpointer=InMemorySaver(),
        recorder=recorder,
        model_registry=registry,
        builder=builder or StubContextBuilder(),
    )

    context = definition.context_factory(make_run_context(recorder))

    return definition, context, registry, recorder


async def _run(definition, context, text: str = "你好"):
    return await definition.compiled.ainvoke(
        {"messages": [HumanMessage(text)]},
        config={"configurable": {"thread_id": "t-1"}},
        context=context,
    )


# ============================================================
# 基础：节点 -> Registry -> Runtime -> Provider
# ============================================================


async def test_node_calls_model_through_registry() -> None:
    """节点只通过 Registry + Runtime 触达模型。"""

    adapter = FakeAdapter(
        script={"fake:primary": [Outcome(message=AIMessage(content="答完了"))]}
    )

    definition, context, registry, recorder = _graph(adapter)

    result = await _run(definition, context)

    assert result["messages"][-1].content == "答完了"

    # 上下文构建了一次、provider 打了一次
    assert StubContextBuilder.build_count == 1
    assert adapter.called_models() == ["fake:primary"]

    # 观测里带上 provider 级信息（由节点从 ModelResponse 接进 node_span）
    agent_runs = recorder.by_node("agent")

    assert agent_runs, "agent 节点没有写观测行"

    completed = recorder.last("agent")

    assert completed.status is NodeRunStatus.COMPLETED
    assert completed.metrics["model_provider"] == "fake"
    assert completed.metrics["model_name"] == "primary"
    assert completed.metrics["model_attempts"] == 1
    assert completed.metrics["model_degraded"] is False


# ============================================================
# fallback 不重跑节点（规格 §9）
# ============================================================


async def test_fallback_does_not_rerun_node_or_context() -> None:
    """
    Primary -> unavailable，Fallback -> success。

        最终：LLM Node executed once（上下文只构建一次）
              Runtime invoked once
              Provider attempts = 2
    """

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelUnavailableError("503")),
            ],
            "fake:backup": [Outcome(message=AIMessage(content="备份答的"))],
        }
    )

    definition, context, registry, recorder = _graph(adapter, max_retries=0)

    result = await _run(definition, context)

    # 1. 结果来自 fallback
    assert result["messages"][-1].content == "备份答的"

    # 2. 节点只执行了一次 —— 上下文**只构建了一次**（没有被 fallback 重建）
    assert StubContextBuilder.build_count == 1

    # 3. 图里只写了一轮 agent 观测（RUNNING + COMPLETED = 2 行）
    agent_runs = recorder.by_node("agent")

    assert len(agent_runs) == 2
    assert sum(
        1 for item in agent_runs if item.status is NodeRunStatus.COMPLETED
    ) == 1

    # 4. 两次 provider attempt，都在**同一次** runtime.ainvoke 里
    assert adapter.called_models() == ["fake:primary", "fake:backup"]

    completed = recorder.last("agent")

    assert completed.metrics["model_attempts"] == 2
    assert completed.metrics["model_degraded"] is True
    assert completed.metrics["model_name"] == "backup"

    # 逐次 attempt 的明细进了观测
    detail = completed.metrics["model_attempt_detail"]

    assert [item["model"] for item in detail] == ["primary", "backup"]
    assert [item["success"] for item in detail] == [False, True]


async def test_retry_does_not_rerun_node() -> None:
    """attempt 1 失败、attempt 2 成功 —— 节点仍然只跑一次。"""

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=ModelUnavailableError("暂时不可用")),
                Outcome(message=AIMessage(content="重试之后答的")),
            ]
        }
    )

    definition, context, registry, recorder = _graph(adapter)

    result = await _run(definition, context)

    assert result["messages"][-1].content == "重试之后答的"
    assert StubContextBuilder.build_count == 1
    assert adapter.called_models() == ["fake:primary", "fake:primary"]

    completed = recorder.last("agent")

    assert completed.metrics["model_attempts"] == 2
    assert completed.status is NodeRunStatus.COMPLETED


async def test_exhausted_chain_fails_the_node_loudly() -> None:
    """
    所有候选都失败 -> 异常**不被节点吞掉**，运行记 failed。

    返回一句"模型暂时不可用"会把一次失败的运行记成 completed，而模型
    调用其实根本没发生 —— 与 tools.py 里"不返回假 ToolMessage"是同一条
    立场。
    """

    adapter = FakeAdapter(
        default=Outcome(error=ModelUnavailableError("503"))
    )

    definition, context, registry, recorder = _graph(adapter, max_retries=0)

    with pytest.raises(ModelUnavailableError):
        await _run(definition, context)

    assert recorder.last("agent").status is NodeRunStatus.FAILED


# ============================================================
# 能力门禁在节点这一层的表现
# ============================================================


async def test_capability_mismatch_fails_before_provider_call() -> None:
    """primary 不支持工具调用、而这次请求带工具 -> 调用前就失败。"""

    adapter = FakeAdapter()

    definition, context, registry, recorder = _graph(
        adapter,
        primary_caps=ModelCapabilities(structured_output=True),
        fallbacks=(),
        builder=ToolBuilder(),
    )

    with pytest.raises(ModelCapabilityError):
        await _run(definition, context)

    assert adapter.called_models() == []
    assert recorder.last("agent").status is NodeRunStatus.FAILED


async def test_incompatible_fallback_is_skipped_inside_the_node() -> None:
    """
    带工具的请求不允许降级到不支持工具的模型 —— 节点只看到"失败"，
    而"为什么没降级"写在观测里（incompatible_candidates）。
    """

    adapter = FakeAdapter(
        default=Outcome(error=ModelUnavailableError("503"))
    )

    definition, context, registry, recorder = _graph(
        adapter,
        primary_caps=TOOL_CAPABLE,
        fallback_caps=ModelCapabilities(structured_output=True),
        builder=ToolBuilder(),
        max_retries=0,
    )

    with pytest.raises(ModelUnavailableError):
        await _run(definition, context)

    # backup 一次都没被调用
    assert adapter.called_models() == ["fake:primary"]


# ============================================================
# 静态边界：节点里不许出现的东西
# ============================================================


def _code_only(module) -> str:
    """
    模块的**代码**文本：去掉 docstring 与注释。

    为什么不能直接搜源码文本：本仓库的 docstring 会**逐条列出**被禁止
    的写法（"Runtime 不允许 `graph.invoke` / `agent.run`"）—— 直接搜
    源码会把那些说明当成违规。用 AST 重建一遍，剩下来的才是真正会执行
    的东西。
    """

    source = Path(module.__file__).read_text(encoding="utf-8")

    tree = ast.parse(source)

    for node in ast.walk(tree):

        if not isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        body = getattr(node, "body", None)

        if not body:
            continue

        first = body[0]

        if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
        ):
            body[0] = ast.Pass()

    return ast.unparse(tree)


#: LLM Node 里**禁止出现**的符号（规格 §15、§19）。
#:
#: 它们各自对应一条反模式：
#:     ChatXxx / OpenAI( / Anthropic(   节点自己创建模型
#:     factory.get                     越过 Registry 拿客户端
#:     resolver.resolve                自己解析模型策略
#:     bind_tools                      直接操作模型实例
#:     model.ainvoke                   绕过 Runtime 直接调用
#:     import openai / anthropic       判断 provider 异常的前置条件
#:     RateLimitError / APIError       判断 provider 异常
FORBIDDEN_IN_NODE = (
    "ChatOpenAI",
    "ChatDeepSeek",
    "ChatAnthropic",
    "ChatMoonshot",
    "init_chat_model",
    "get_chat_model",
    "factory.get",
    "resolver.resolve",
    "bind_tools",
    "with_structured_output",
    "usage_metadata",
    "import openai",
    "import anthropic",
    "RateLimitError",
    "APIError",
    "wait_for_human",
    "graph.invoke",
)


def test_agent_node_source_has_no_forbidden_symbols() -> None:
    """源码级护栏：反模式一出现就红，不管行为测试过不过。"""

    from app.agent.graph.react import agent_node

    code = _code_only(agent_node)

    found = [item for item in FORBIDDEN_IN_NODE if item in code]

    assert found == [], (
        f"agent_node.py 里出现了不该属于节点的东西：{found}。\n"
        "节点只该做：ContextBuilder -> registry.get_runtime -> "
        "runtime.ainvoke -> 回写 / 路由；模型的一切都在 Model 模块里。"
    )

    assert "runtime.ainvoke" in code


def test_agent_node_does_not_hold_a_model_instance() -> None:
    """
    `register_react_graph` 收的是 ModelRegistry，节点从 context 里拿
    Registry —— 而不是一个 `model` 参数 / 字段（规格 §16）。
    """

    from app.agent.graph.react import agent_node
    from app.agent.graph.react.context import ReactContext

    signature = inspect.signature(register_react_graph)

    assert "model_registry" in signature.parameters
    assert "model" not in signature.parameters

    fields = set(ReactContext.__dataclass_fields__)

    assert "model_registry" in fields
    assert "model" not in fields

    # 节点的函数签名里没有模型参数（它从 runtime.context 取）
    node_signature = inspect.signature(agent_node.agent_node)

    assert "model" not in node_signature.parameters


def test_model_runtime_has_no_agent_or_graph_dependency() -> None:
    """
    反向护栏：Model Runtime 里不许出现 Graph / Agent / Tool / HITL。

    （规格 §2 的禁止反向依赖清单 + §19 的前四条反模式。）
    """

    from app.agent.model import runtime as runtime_module

    code = _code_only(runtime_module)

    forbidden = (
        "import langgraph",
        "from langgraph",
        "from app.agent.graph",
        "from app.agent.context",
        "from app.infra.graph",
        "graph.ainvoke",
        "graph.invoke",
        "agent.run",
        "tool.invoke",
        "tool.ainvoke",
        "wait_for_human",
        "GraphState",
        "ReactState",
    )

    found = [item for item in forbidden if item in code]

    assert found == [], (
        f"model/runtime.py 里出现了 Agent / Graph / HITL 相关的东西：{found}。\n"
        "Runtime 只理解 Model / Provider / Invocation —— 需要理解 "
        "Agent / Graph / Business Context 的逻辑属于 LLM Node。"
    )


def test_resolver_does_not_call_a_model() -> None:
    """Resolver 不调 LLM（规格 §19 第 5 条反模式）。"""

    from app.agent.model import resolver as resolver_module

    code = _code_only(resolver_module)

    for item in ("ainvoke", "chat_model", "LLM"):
        assert item not in code, f"resolver.py 里出现了 {item!r}"

    # 它是一个纯函数式的查表：resolve 不是 async
    from app.agent.model import CatalogModelResolver

    assert not inspect.iscoroutinefunction(CatalogModelResolver.resolve)


def test_factory_does_not_decide_business_policy() -> None:
    """
    Factory 只负责实例创建，不判断"react 该用哪个模型"（§19 第 6 条）。
    """

    from app.agent.model import factory as factory_module

    code = _code_only(factory_module)

    for item in ("agent_type", "model_role", "resolver", "fallback", "retry("):
        assert item not in code, f"factory.py 里出现了 {item!r}"
