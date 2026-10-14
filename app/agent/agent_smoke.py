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
import sys
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.context.registry import create_context_builder
from app.agent.graph.react import agent_builder_node as builder_module
from app.agent.graph.react.context import ReactContext
from app.agent.graph.react.graph import NODE_FUNCTIONS, build_react_graph
from app.agent.graph.react.state import ReactState
from app.agent.model import ModelRequest
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

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def ainvoke(self, request: Any, **kwargs: Any) -> Any:
        return type(
            "_R", (), {"message": type("_M", (), {"content": self._payload})()}
        )()


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

    async def stub_agent(state: ReactState, runtime: Any) -> dict:
        ran.append("agent")
        return {"messages": [AIMessage("答复")]}

    async def stub_planner(state: ReactState, runtime: Any) -> dict:
        ran.append("planner")
        return {}

    try:
        # ⚠️ 必须在 build_react_graph() **之前**替换：
        # 装配时按 spec.node_id 从这个字典里取函数。
        NODE_FUNCTIONS["builder"] = stub_builder
        NODE_FUNCTIONS["agent"] = stub_agent
        NODE_FUNCTIONS["planner"] = stub_planner

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

    # --------------------------------------------------------
    section("专用 Context Builder：窄输入")

    builder = create_context_builder(ContextBuilderType.CLASSIFIER)

    check(
        "classifier 能解析出实现",
        type(builder).__name__ == "ClassifierProcessorContextBuilder",
    )
    check(
        "只声明三个阶段",
        {s.name for s in type(builder).enabled_stages} == {
            "CONTEXT_RESOLVER", "HISTORY_SELECTOR", "PROMPT_ASSEMBLER",
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
    section("路由：execution_plan 决定去向")

    ran, result = await _run_graph(payload=_payload(
        ExecutionPlan.REACT.value
    ))
    check("react -> 直接进 agent", ran == ["builder", "agent"])

    ran, result = await _run_graph(payload=_payload(
        ExecutionPlan.DIRECT.value
    ))
    check("direct -> 直接进 agent（direct/react 拓扑上没有分岔）",
          ran == ["builder", "agent"])

    ran, result = await _run_graph(payload=_payload(
        ExecutionPlan.PLAN.value
    ))
    check("plan -> 先过 planner 再进 agent",
          ran == ["builder", "planner", "agent"])

    ran, result = await _run_graph(payload=_payload(
        ExecutionPlan.OTHER.value
    ))
    check("other -> 走 agent（兜底回到改动前的行为）",
          ran == ["builder", "agent"])

    # 没有分类结果也走 agent：那一轮可能压根没跑 builder（比如从旧
    # checkpoint 续跑），路由不能因此炸掉。
    from app.agent.graph.react.routing import route_after_builder

    check(
        "state 里没有 classification -> 走 agent",
        route_after_builder({"messages": []}) == "agent",  # type: ignore[typeddict-item]
    )

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
        "查回来的 tool_key 写进了 context_policy",
        result["context_policy"]["tool_keys"]
        == ["t.financial", "t.document"],
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
