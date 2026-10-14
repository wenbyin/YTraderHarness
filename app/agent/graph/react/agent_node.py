"""
AgentNode：模型调用的前后置处理。

它的位置（对照 langgraph 的 ToolNode）
--------------------------------------
    ToolNode 包住**工具调用**（执行工具、把结果包成 ToolMessage）
    AgentNode 包住**模型调用**（构建上下文、调模型、回写上下文）

三步：

    before   ContextBuilder.build(request) -> ModelRequest
             （上下文由 Processor 流水线构建，见 context/processor/）
    call     model_registry.get_runtime(...) -> runtime.ainvoke(request)
             （模型调用唯一入口；retry / fallback / timeout 在 Runtime 里）
    after    ContextUpdater.update(...)   <- 本次空置，见 context/updater.py

⚠️ 每一次 LLM 调用都经过 ContextBuilder（规格 §廿四）
------------------------------------------------
本节点是图里**唯一**的模型调用点，而它在每次执行时都先跑一遍
ContextBuilder：

    第一次调用（用户刚提交消息）      -> phase=INITIAL
    Tool Result 之后回到本节点        -> phase=CONTINUATION

后者的上下文由**当前 Graph State** 重算（`messages` 里已经有工具结果），
而不是回库把整段会话重新加载一遍 —— 见 processor/history.py 与
processor/pipeline.py 的 `_phase`。

（因此"Initial Context Builder"不是图里的一个独立节点：它是同一个
节点在 phase=INITIAL 时的行为。少一个节点，也就少一处"两条 LLM 路径
其中一条忘了构建上下文"的可能。）

⚠️ 后置处理**还没有实现**，这是刻意的
-------------------------------------
`ContextUpdater` 目前只有一个 Protocol，`ReactContext.context_updater`
现在必然是 None，`_apply_context_update` 因此什么都不做。

不要为了"让后置处理看起来存在"而塞一个空实现进来 —— 见
app/agent/context/updater.py 里为什么没有 NoopContextUpdater 的说明。

为什么是**函数**而不是类
-----------------------
`ToolNode` 是类，因为它持有构图期的工具列表。这里没有任何
per-instance 状态：builder / registry / updater 全都从
`runtime.context` 取 —— 全都是**这次运行**的东西。

（仓库的节点也都是普通 async 函数，graph.py 按 NODE_SPECS 装配。
把它们做成类会在"节点函数的签名约定"之外多出一套约定。）

⚠️ 参数名必须是 `runtime` 且带注解
---------------------------------
LangGraph 按**参数名**注入（`Runtime` 那个位置参数）。名字写错
（比如 `rt`）不会报错，而是收到一个普通 dict —— 然后 `.context`
取到 None，报错离原因很远。document 的节点们是同一条约定。

本节点**不碰模型实例**（规格 §3、§15）
------------------------------------
节点里不会、也不该出现：

    ChatOpenAI(...) / ChatDeepSeek(...)     自己建模型
    model.bind_tools(...) / model.ainvoke() 直接操作模型
    factory.get(...)                        越过 Registry 拿客户端
    resolver.resolve(...)                   自己解析模型策略
    try/except openai.RateLimitError        判断 provider 异常

节点只做两件与模型有关的事：

    runtime = model_registry.get_runtime(agent_type=..., model_role=...)
    response = await runtime.ainvoke(request)

provider / model name / API key / fallback chain / retry policy 全在
Model 模块内部，节点一个都不知道。

⚠️ retry 与 fallback 都不在节点（也不在图）里
-------------------------------------------
    Model Runtime 内部:  timeout / retry / fallback / 异常归一化
    节点:                构建上下文 -> 一次 invoke -> 回写 / 路由

因此**刻意不给**节点挂 langgraph 的 `retry_policy`：

    模型失败的重试归 Runtime（只重试可重试的错误，且不会重跑节点、
    不重建上下文、不重放 Graph State —— 规格 §9）；
    工具重试意味着副作用执行两遍（下单、转账），那是业务必须自己
    决定的事，不是基础设施顺手加上的。

（旧实现是"客户端内部重试 2 次 + 节点上没有 retry_policy"，理由是
避免 3 × 3 次可计费调用。现在客户端内部重试被关掉了（见
model/factory.py），重试的**唯一**所有者是 Model Runtime —— 那条
叠加风险从"靠约定避免"变成了结构上不存在。）

⚠️ 异常**不做**兜底捕获
---------------------
`runtime.ainvoke` 抛出的必然是 `app.agent.model` 里的异常
（provider 异常在 Runtime 里就被归一化掉了）。节点不捕获它们：

    捕获了能做什么？重试（Runtime 已经做过）、换模型（Resolver 的
    决定）、还是返回一句"模型暂时不可用"？最后一项会让一次失败的
    运行在观测里记成 completed，而模型调用其实**根本没发生** ——
    与 tools.py 里"不返回假 ToolMessage"是同一条立场。

因此让它抛：观测行记 failed，人能一眼看到。

⚠️ 本节点**不再**生产运行事件
---------------------------
以前它往 Redis Stream 发四类事件（node.started / message.delta /
message.completed / node.completed），现在全部由
`app/agent/event/run_event_projector.py` 从 langgraph 协议事件统一投影。

两条规矩仍然有效，只是**执行者换了地方**（记在这里是因为它们约束的是
事件本身，不是某一段代码）：

    1. 事件是**旁路**。投递失败不上抛、不影响返回 —— 一次 Redis 抖动
       不该让整个运行失败，更不该让用户看到一句"模型不可用"。
       projector 用 try/except 吞掉自己的异常来兑现这一条。
    2. `message.completed` 带全文，它是 delta 流的**对账点**：订阅端
       拼出来的东西与它不一致时以它为准。
    3. `message.delta` **只有可见正文** —— 不含思维链，也不含
       tool-call 参数的碎片。judgment 现在在 projector 里（它只认协议
       的 `text-delta`）。

⚠️ 本节点**不传** on_delta —— 这是逐 token 事件能不能产生的前提
----------------------------------------------------------------
`on_delta` 一旦传入，`model/adapters.py` 就走 `runnable.astream(...)`，
而 v3 的协议事件只在 `ainvoke` 那条路上产生（判据是
`_should_use_protocol_streaming`，它不被 `astream` 调用）。实测同一节点
同一模型，只因这一个参数，逐 token 事件从 107 条塌成 1 条整消息。

代价见调用点那段注释（增量不再"提交即定"）。

⚠️ HITL 不在本节点（规格 §10）
----------------------------
本节点不知道用户是谁、要不要人工补充信息、工具是否高风险。HITL 属于
**Graph 控制流**，两个触发点：

    After LLM Node     模型需要用户补充信息 -> 人类输入触发
                       （判据在路由里，见 routing.route_after_agent）
    Before Tool Node   Tool 执行需要人工审批 -> 审批触发
                       （工具侧的拦截器在
                         app/agent/tools/processor/interceptor/approval.py）

本节点只负责"把模型这一调用做完、把结果交回 state"。要挂人类输入，
那是在**路由/控制流**上加（LangGraph 的 interrupt），不是在这里
等一个人。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.agent.context.builder.base import BaseContextBuilder
from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.model import (
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelRuntime,
    required_capabilities_for,
)
from app.infra.graph import execution_ids, node_span

from .catalog import AGENT_TYPE, CONTEXT_BUILDER_TYPE, MODEL_ROLE, NODE_SPECS
from .context import ReactContext
from .state import ReactState

if TYPE_CHECKING:
    from app.agent.context.updater import ContextUpdater

_SPEC = next(s for s in NODE_SPECS if s.node_id == "agent")

__all__ = ["agent_node"]


async def agent_node(
        state: ReactState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """
    构建上下文 → 调模型 → 回写上下文。

    Returns:
        {"messages": [AIMessage]} —— 交给 add_messages 归约合并。
    """

    context = runtime.context

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "message_count": len(state.get("messages") or ()),
                "builder_type": CONTEXT_BUILDER_TYPE.value,
                "agent_type": AGENT_TYPE,
                "model_role": MODEL_ROLE,
            },
    ) as span:

        # ⚠️ node.started 的事件**不在这里发**了：由
        # app/agent/event/run_event_projector.py 从协议的 `tasks` 事件统一
        # 投影（`tasks` 只在 transformer 通道里 —— 默认注册的
        # LifecycleTransformer / SubgraphTransformer 会把它从主日志抑制掉，
        # 见该模块的说明）。`agent_type` / `model_role` 这两个本节点独有的
        # 领域字段由图侧注入 projector，不丢。
        #
        # 观测（下面的 node_span）留在这里 —— 它走的是 infra/graph，
        # 与运行事件流是两条链路，不该跟着一起搬。

        request = ContextBuildRequest(
            builder_type=CONTEXT_BUILDER_TYPE,
            thread_id=_thread_id(runtime),
            messages=tuple(state.get("messages") or ()),
            run_id=context.pipeline_run_id,
            # 图 state 的只读快照 —— Context Processor 的绑定根
            # （`context_policy` / `user_context` / `summary` / ... 都从
            # 它解析）。节点只是把它交出去，**不**替 ContextBuilder 决定
            # 该看哪些通道（那是 Processor 的声明式绑定，见
            # app/agent/context/processor/spec.py）。
            graph_state=_graph_state(state),
            agent_type=AGENT_TYPE,
            model_role=MODEL_ROLE,
            # 依赖随请求进：Builder 实例是共享的，构造期注入无处存放
            # （见 context/builder/contracts.py 的说明）。
            dependencies=ContextDependencies(
                model_registry=context.model_registry,
                # capability 阶段靠它把 context_policy.tool_keys 解析成
                # 可 bind 的工具对象（见 infra/agent/tool_exposure.py）。
                # 它是进程级无状态的，与 model_registry 同性质。
                tool_source=context.tool_source,
            ),
        )

        # ContextBuilder 的产出直接就是 Model 模块的 ModelRequest ——
        # 全系统只有那一个请求类型，中间**没有转换层**（规格 §二十五）。
        model_request = await _build_model_request(context.builder, request)

        model_runtime = context.model_registry.get_runtime(
            agent_type=AGENT_TYPE,
            model_role=MODEL_ROLE,
            context=ModelResolveContext(
                # 这次请求本身要求的能力（有 tools -> 需要 tool_calling，
                # 有 response_format -> 需要 structured_output）。
                # 节点**不**在这里猜 vision / reasoning —— 推测会出错，
                # 见 model/types.py 的 required_capabilities_for。
                required_capabilities=required_capabilities_for(
                    model_request
                ),
            ),
        )

        # ⚠️ 这里**刻意不再传 on_delta**
        # ------------------------------
        # 传了它，adapters.py 就走 `runnable.astream(...)`，而
        # `_should_use_protocol_streaming` 只被 `_generate_with_cache` /
        # `_agenerate_with_cache` 调用（chat_models.py:1942/:2097）——
        # **`astream` 不在其中**。实测同一节点同一模型，只因这一个参数：
        #
        #     不带 on_delta（走 ainvoke）  改道命中   messages 事件 107 条（逐 token）
        #     带   on_delta（走 astream）  改道未命中 messages 事件   1 条（整条）
        #
        # 也就是说 on_delta 一开，v3 的强制 messages 模式就对这个节点失效，
        # 逐 token 的 message.delta **一个都不会产生**。运行事件的投影现在
        # 由 app/agent/event/run_event_projector.py 从协议事件统一做，
        # 这条节点内的旁路已经不需要了。
        #
        # 代价（已知并接受）：增量不再"提交即定" —— 旧规则是"一旦有增量
        # 发出去，这次调用就不再重试、不再降级"（见 model/runtime.py 的
        # on_delta 说明）。现在重试/降级会重发一条 message-start 与一串
        # delta，订阅端可能看到两段。projector 按 LLM run id 做了去重，
        # 但那只保证 message.completed 不重复，不保证 delta 不重发。
        response = await _invoke_model(
            model_runtime,
            model_request,
        )

        await _apply_context_update(
            context.context_updater, request, response.message
        )

        # ContextBuilder 的观测（规格 §十九 / §廿七）：节点把快照接进既有
        # 的 node_span，于是"这一轮模型看到了什么"随图观测落库 ——
        # ContextBuilder 自己**不认识** infra/graph（它不该认识）。
        snapshot = _context_snapshot(model_request)

        span.set_metrics({
            "message_count": len(model_request.messages),
            "tool_count": len(model_request.tools),
            **_model_metrics(response),
            **(_snapshot_metrics(snapshot) if snapshot else {}),
        })

        span.set_output_summary({
            "tool_calls": [
                call.get("name")
                for call in getattr(response.message, "tool_calls", ())
            ],
            "has_text": bool(getattr(response.message, "content", "")),
            "degraded": response.degraded,
            "context_hash": (snapshot or {}).get("context_hash"),
        })

        # 回写 state：快照（观测/Replay）+ 上下文构建产出里需要跨轮/跨迭代
        # 保活的那几样（目前是摘要的新版本、以及 L0/L1 的召回命中）。
        # **只写"变了的东西"** —— 摘要没被压过就不重复写同一个值（见 state.py
        # 里"其余通道是最后写入者赢"那条说明）；白名单在
        # `_context_state_updates` 里。
        result: dict[str, Any] = {"messages": [response.message]}

        if snapshot:
            result["context_result"] = snapshot

        result.update(_context_state_updates(model_request))

        # ⚠️ node.completed 也搬去 projector 了。可以从 `tasks` 还原的字段：
        #     node_id        data.name
        #     message_count  start 事件的 input.messages 长度（result 里没有 input）
        #     model_provider / model_name   结果消息的 response_metadata
        #     latency_ms     start/result 两条事件的 params.timestamp 之差
        #     tool_calls     结果消息上的 tool_calls
        #
        # ⚠️ 唯一**还原不了**的是 `degraded` —— 它是 ModelResponse 的概念
        # （"这次重试过/降级过"），协议事件里没有对应物。它没有彻底消失：
        # 同一份事实仍在 span metrics 的 `model_degraded` 里（见下面
        # _token_metrics），且 observer.py 会为它打 WARNING。消失的只是
        # "事件流里的那一份"。
        #
        # ⚠️ latency_ms 的含义变了：旧值是 response.latency_ms（只算 provider
        # 调用），新值是两条协议事件的时间差（含上下文构建、写库等）。
        # 新的数**更大**。字段名保持不变（它是契约），差别记在这里。

        return result


# ============================================================
# 三步，各自成函数
# ============================================================
#
# 拆开而不是写成一整段：将来别的图（planner / sub_agent）要复用
# "调模型"这一件事时，拿的是这个函数，而不是把整个 agent_node 抄一遍
# 再删掉用不上的部分。


async def _build_model_request(
        builder: BaseContextBuilder,
        request: ContextBuildRequest,
) -> ModelRequest:
    """
    前置：交给 ContextBuilder。

    builder 是**实例**（已经从 context 里取好），这里不再做任何
    "按类型查表" —— 那一步在组合根启动时做过一次了，见 context.py
    的说明。

    ⚠️ 产出就是 `app/agent/model/types.py::ModelRequest`
    -------------------------------------------------
    "上下文构建"到"模型调用"之间**没有转换层**：Context Builder 按
    `context_builder.yaml` 的 final_output 契约产出那个类型，Model Runtime
    直接消费它（规格 §二十五：所有模型调用统一经过 ModelRegistry）。
    """

    return await builder.build(request)


async def _invoke_model(
        model_runtime: ModelRuntime,
        model_request: ModelRequest,
) -> ModelResponse:
    """
    调用：把请求交给 Model Runtime，拿回统一的响应。

    这里是**整个节点里唯一与模型有关的一行**。retry / timeout /
    fallback / provider 异常归一化 / usage 全在 Runtime 内部
    （见 model/runtime.py），节点既不参与也不感知：

        runtime.ainvoke 内部可能试了 3 次、换过 2 家 provider，
        但对本节点来说那始终只是**一次**调用 —— 上下文没有重建、
        state 没有重放、工具没有重执行（规格 §9）。

    不捕获异常的理由见模块 docstring。

    ⚠️ 刻意**不传** `on_delta`
    -------------------------
    见调用点那段说明：传了它就走 `runnable.astream(...)`，而 v3 的逐 token
    协议事件只在 `ainvoke` 那条路上产生。运行事件现在由
    app/agent/event/run_event_projector.py 统一投影，节点内的旁路已废弃。

    （`on_delta` 本身仍是 model 层的合法能力，只是本节点不再用 —— 别的
    调用方若需要"给出去一点就不重试"的提交语义，仍然可以传。）
    """

    return await model_runtime.ainvoke(model_request)


async def _apply_context_update(
        updater: "ContextUpdater | None",
        request: ContextBuildRequest,
        response: AIMessage,
) -> None:
    """
    后置：把这一轮交回给 ContextUpdater。

    ⚠️ updater 为 None 就是**不回写**，不是"去调一个什么都不做的实现"。
    当前它必然是 None —— 后置处理还没有设计（见
    app/agent/context/updater.py）。
    """

    if updater is None:
        return

    await updater.update(request=request, response=response)


# ============================================================
# 运行事件流
# ============================================================
#
# ⚠️ 本节点**不再生产任何运行事件**
# ---------------------------------
# node.* / message.* / tool.* 现在都由
# app/agent/event/run_event_projector.py 从 langgraph 协议事件统一投影。
# 这里原本有四处 `_publish_event`（node.started / node.completed /
# message.completed / message.delta）与配套的 `_publish_event` /
# `_delta_sink` / `_message_text` / `_tool_call_names`，一并删掉了。
#
# 为什么搬走：每个要发事件的地方都得 `import publish_run_event`，生产者
# 散在图里各个节点上。搬到一个投影层后，"谁发什么"只有一处可读，
# 而 tool.* 这种当时要在每个工具处理器里埋一遍的，也变成一条投影规则。
#
# 事件的**契约**（事件名、信封形状）没变，订阅端不用改；变的只是生产者。


# ============================================================
# internals
# ============================================================


def _thread_id(runtime: Runtime[ReactContext]) -> str:
    """
    本次运行所属的会话。

    会话标识只能从 `execution_info` 取（`Runtime` 上没有 config 字段，
    见 langgraph 的 Runtime 定义）。它在带 checkpointer 的正常运行里
    一定存在；取不到说明不是在一条正常的图执行路径上跑。

    取不到就**抛错**而不是回落到空串：ContextBuilder 靠它去取历史、
    画像与记忆，一个空 thread_id 会让检索查到一个不存在的会话 ——
    于是上下文里静默地少了历史，而模型照常回答。
    """

    thread_id = execution_ids(runtime).thread_id

    if thread_id:
        return thread_id

    raise RuntimeError(
        "agent 节点取不到 thread_id（execution_info 里没有）。"
        "它要交给 ContextBuilder 去取历史 / 画像 / 记忆。\n"
        "常见原因：图不是在 runner 里跑的（没带 configurable.thread_id "
        "的 config），或者 langgraph 版本低于 1.1.5。"
    )


def _model_metrics(response: ModelResponse) -> dict[str, Any]:
    """
    把这次模型调用的观测写进 node_span 的 metrics。

    这是 provider 级信息**落库的唯一路径**（规格 §8.6）：Model 模块
    自己不认识 infra/graph（它不该认识），因此由节点把 Runtime 给出的
    标准 metadata 接进既有的观测机制。

    字段名的口径与 `ModelCallSummary.describe()` 一致：
    provider / model / attempt 数 / 降级 / 耗时 / token / 成本。
    多一个 `model_attempts` —— 逐次尝试的**紧凑**明细（谁、第几次、
    多久、成没成），它是"这次为什么慢 / 为什么换了一家"的唯一答案。

    缺的就不填：各家 usage 完整度不同，编一个 0 出来会让"这家没报"
    与"真的是 0"分不清（与旧实现里 _token_metrics 的取舍一致）。
    """

    meta = response.metadata

    metrics: dict[str, Any] = {
        "model_provider": response.provider,
        "model_name": response.model,
        "model_id": response.model_id,
        "model_attempts": response.attempts,
        "model_latency_ms": response.latency_ms,
        "model_degraded": response.degraded,
    }

    model_id = meta.get("model_id")
    if isinstance(model_id, str):
        metrics["model_definition"] = model_id

    if response.usage is not None:
        metrics.update({
            key: value
            for key, value in response.usage.as_dict().items()
            if isinstance(value, int)
        })

    estimated_cost = meta.get("estimated_cost")
    if isinstance(estimated_cost, (int, float)):
        metrics["model_estimated_cost"] = estimated_cost

    # 逐次 attempt 的紧凑明细。只取白名单字段：attempt.metadata 是我们
    # 自己造的（str / int / bool / None），可以直接进 JSONB。
    attempts = meta.get("attempts")
    if isinstance(attempts, (list, tuple)):
        metrics["model_attempt_detail"] = [
            {
                "provider": item.get("provider"),
                "model": item.get("model"),
                "candidate_index": item.get("candidate_index"),
                "attempt": item.get("attempt"),
                "latency_ms": item.get("latency_ms"),
                "success": item.get("success"),
                "error_code": item.get("error_code"),
            }
            for item in attempts
            if isinstance(item, dict)
        ]

    return metrics


# ============================================================
# ContextBuilder 的接线（规格 §十九 / §廿三 / §廿七）
# ============================================================

#: 进 state 的通道白名单。
#:
#: 只搬运 ContextBuilder 明确声明要读的那些通道 —— 把整个图 state
#: 原样交出去会让"上下文里为什么出现了某个字段"无从追溯，而这份白名单
#: 就是"ContextBuilder 能看到什么"的**唯一**答案（与 ReactState 里那几个
#: ContextBuilder 通道一一对应）。
#:
#: ⚠️ messages 一定要带上：历史处理器绑的就是它（`messages`），
#: 而它同时是"用户 Conversation 是唯一原始事实来源"这条原则的落点 ——
#: 会话的持久化由 LangGraph 的 checkpointer 承担，不是 ContextBuilder。
#:
#: ⚠️ `duty_key` / `classification` **刻意不在**这里
#: ----------------------------------------------
#: 它们不由 ContextBuilder 读，读它们的是节点（builder 节点）——
#: 白名单是"ContextBuilder 能看到什么"，不是"state 里有什么"。
#: 职责键更具体地说：它在 builder 节点里被换成了**文本**，进流水线的是
#: `context_policy.system_prompt`（`classification` 同理：它只驱动路由与
#: 工具选择，不参与上下文构建）。把键本身塞进来等于让 ContextProcessor
#: 也看得见一份它用不上的东西，而"谁在读这个通道"就说不清了。
_CONTEXT_STATE_CHANNELS: tuple[str, ...] = (
    "messages",
    "conversation_id",
    "user_id",
    "context_policy",
    "user_context",
    "memory",
    "retrieved_context",
    "exposed_tools",
    "summary",
)


def _graph_state(state: ReactState) -> dict[str, Any]:
    """
    挑出交给 ContextBuilder 的图 state 通道（只读快照）。

    缺的通道**不补空值**：`context_policy.include_memory` 这类 gate 的
    MISSING 语义是"上游漏配 -> 构建失败"（规格 §九），而一张预先填满
    None 的快照会让它变成"存在但类型不对" —— 报错信息一样，但更难指到
    "谁忘了写"。让缺的真的缺。
    """

    return {
        name: state[name]
        for name in _CONTEXT_STATE_CHANNELS
        if name in state
    }


def _context_snapshot(
        model_request: ModelRequest,
) -> dict[str, Any] | None:
    """
    从请求的 metadata 里取 ContextBuilder 的快照（规格 §十九/§廿七）。

    这是"上下文构建"与"图观测"之间的**唯一**通道：ContextBuilder 不认识
    infra/graph，节点也不认识 ContextProcessor 的内部结构 —— 快照以
    JSON 安全的 dict 形式随请求 metadata 过来（见
    context/processor/context.py 的 ContextSnapshot.as_metadata）。
    """

    snapshot = (model_request.metadata or {}).get("context_snapshot")

    return dict(snapshot) if isinstance(snapshot, dict) else None


def _snapshot_metrics(
        snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """
    快照里进 node_span 的那几个数（**精简版**）。

    `ContextSnapshot.compact_metrics()` 已经是精简的，但它是 ContextBuilder
    侧的产物；这里再筛一遍是为了让节点写进观测的字段名**在这里**可读
    （观测列是给人扫一眼的，字段名不该需要跳三个文件才能读懂）。
    """

    def _count(key: str) -> int:
        value = snapshot.get(key)
        return len(value) if isinstance(value, (list, tuple)) else 0

    processors = snapshot.get("processors")

    executed = skipped = failed = 0

    if isinstance(processors, (list, tuple)):

        for item in processors:

            if not isinstance(item, Mapping):
                continue

            status = item.get("status")

            executed += status == "executed"
            skipped += status == "skipped"
            failed += status == "failed"

    return {
        "context_phase": snapshot.get("phase"),
        "context_hash": snapshot.get("context_hash"),
        "context_window": snapshot.get("model_context_window"),
        "context_window_source": snapshot.get("context_window_source"),
        "context_input_tokens": snapshot.get("input_token_count"),
        "context_history_messages": _count("history_message_ids"),
        "context_memory_items": _count("memory_ids"),
        "context_retrieval_items": _count("retrieved_context_ids"),
        "context_tool_names": _count("tool_names"),
        "context_summary_revision": snapshot.get("summary_revision"),
        "context_processors_executed": executed,
        "context_processors_skipped": skipped,
        "context_processors_failed": failed,
        "context_warnings": list(snapshot.get("warnings") or ()),
    }


#: 允许从上下文构建回写进图 state 的键。
#:
#: ⚠️ 这是**白名单**，不是照单全收。那条通道
#: （`ModelRequest.metadata["context_state_updates"]`）由**处理器**写入，
#: 而处理器能借它碰到图 state —— 不设白名单的话，一个拼错了的处理器就能
#: 改 `messages` 或 `context_result`，也就是让"组装上下文"变成一条能篡改
#: 对话的路径。
#:
#: 加键时在这里加，并想清楚"它为什么必须跨轮或跨迭代保活"。
_STATE_UPDATE_KEYS: tuple[str, ...] = ("summary", "memory")


def _context_state_updates(
        model_request: ModelRequest,
) -> dict[str, Any]:
    """
    取"需要回写图 state 的上下文构建产出"。

    两个来源（都在 builder/agent_invoke.py 的 `_delegate` 里汇总）：

        summary     compaction **真的压过一轮**（`summary_dirty`）。没压过
                    就不写 —— 否则每一轮都会把同一份摘要重新写一遍，而
                    state 的 channel 会看起来一直在变。
        memory      L0/L1 的召回命中（见 processors/memory.py）。它们每个
                    **运行**只召回一次（INITIAL），而 system 每次 ReAct
                    迭代都重建 —— 不保活的话，模型在第二次迭代就看不到
                    刚刚还在的那两段。

    返回空 dict = "没有要写的"，于是调用方那句 `result.update(...)` 什么
    都不改（不会把通道写成 None）。

    值必须是 Mapping：三个 state 通道（`summary` / `memory`）都是
    `dict[str, Any]`，标量或列表在这里就该被丢掉而不是写进去。
    """

    updates = (model_request.metadata or {}).get("context_state_updates")

    if not isinstance(updates, Mapping):
        return {}

    return {
        key: dict(value)
        for key, value in updates.items()
        if key in _STATE_UPDATE_KEYS and isinstance(value, Mapping)
    }
