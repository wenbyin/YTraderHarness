"""
Agent Builder 节点：前置在主循环之前，对用户输入做**初次理解与分类**。

它做什么
--------
一次**便宜的快模型**调用，把用户这一句话归到五个维度上（见
app/agent/state/agent_state.py 的 AgentClassification）：

    ① 行为分类      干什么（chat / qa / analysis / ...）
    ② modules       业务域分类 —— **本次恒空**，见下
    ③ 执行方案      怎么执行（direct / react / plan）—— **驱动路由**
    ④ capability_needs
                    完成它要用到哪些能力（候选来自数据库的能力 code）
                    —— **驱动工具选择**，见下
    ⑤ 能力配置选择  只落字段

⚠️ 这次 LLM 对话**不进对话历史**
--------------------------------
返回值里**没有 `messages` 键**，这就是实现方式：`add_messages` 只在返回
字典里出现该键时才归约（见 graph/react/state.py）。分类是内部步骤，
它的问题是"给这句话分个类"，与用户看到的对话无关 —— 把它塞进 messages
会让主模型、以及下一轮的历史选择器都看见一段莫名其妙的问答。

⚠️ 失败一律降级，不抛异常 —— 与 agent_node 的立场**刻意不同**
------------------------------------------------------------
`agent_node` 让模型异常往上抛（观测记 failed，人能一眼看到）。本节点不
这样做，因为两者的爆炸半径不同：

    agent_node 失败  -> 这一轮**根本没有答复**，抛出去是对的
    本节点失败       -> 只是没有分类，主循环照常能答

分类是**增强**，不是对话的必要条件。而且降级结果是安全的：执行方案落
`other`，路由把 `other` 当作"没有特别指示"（= 走 agent），也就是这次
改动之前的行为。所以失败路径是"退回原样"，不是"带病继续"。

代价是异常被吞掉了，因此降级必须**看得见**：state 里 `degraded=True`、
`notes` 写明原因、span 的 metrics 记 degraded、并打一条 WARNING 日志。
观测里看不到降级的说法是不成立的。

候选能力读不到（应用没装配 ServiceFactory、或数据库连不上）走的也是这
条路：候选为空，其余维度照常由模型判断。

④ 驱动工具选择
-------------
本节点另外产出一个 `context_policy` 覆盖项：把 ④ 判出的能力 code 翻成
`tool_keys` 写进去。capability 处理器（上下文流水线第 7 阶段）读它，经
ToolExposureSource 解析成可 bind 的工具对象 —— 这就是"这次给主模型看哪些
工具"的**唯一**来源（见 app/agent/graph/react/tools.py 里"工具集不从
state 读"的说明）。

⚠️ ②（业务域）为什么恒空
---------------------
它是**刻意留着的空位**，不是忘了填。原因：候选清单里业务域与能力域接口
是混在一起的，而 `capability` 表只有 code / parent_id / is_active
（见 services/capability/models.sql）——**没有任何字段能区分两者**。

因此"这条该算业务域还是能力域"在数据上无从判断。曾经让模型按名称与描述
去分（提示词里写清两者的差别），实测它分不对：把 `CAPABILITY`（能力域）
填进了 `modules`。既然分桶的依据不存在，就不该把它留在模型那边 —— 现在
只问一个字段，code 全落 ④。

`_tool_keys_for_capabilities` 仍按两个字段的**并集**查工具，所以将来分类
体系加上区分字段、② 真的开始有值时，那条路不用改。

⚠️ 不能覆盖 `context_policy`，只能并进去：那个通道没有 reducer，
里面还装着调用方放进来的覆盖项（见 `_merged_policy`）。

⑤（capability_config）仍然只落字段 —— 它还没有消费者。

⑥ 主 agent 的**职责**提示词
-------------------------
本节点还往 `context_policy.system_prompt` 写一样东西：主 agent 搭载的模型
**是干什么的**（`stock_research` / `python_programmer` 这类职责提示词，
经 PromptProvider 取）。主循环的 system 消息只有一个出口
（`ContextBuilder` 的 `system_message()` = policy.system_prompt + 注入分节），
而本节点是 context_policy 的唯一写者 —— 所以"给主模型一段什么样的提示词"
这件事落在**这里**，`agent_node` 一行都不用改。

它与 ④ 是两回事：④ 说"这一轮用户想干什么"，职责说"这个模型是干什么的"。
职责的选择依据**当前还没有**（没有 duty 通道，agent_session.agent_id 也没
接线），因此现在恒取默认职责 —— 见 app/services/prompt/catalog.py。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.agent.context.builder.contracts import ContextBuildRequest
from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.model import (
    ModelResolveContext,
    required_capabilities_for,
)
from app.agent.prompt import get_prompt_provider
from app.agent.prompt.prompts.classification import BEHAVIORS, PLANS
from app.agent.state import (
    MODULE_OTHER,
    AgentClassification,
    coerce_behavior,
    coerce_execution_plan,
    empty_classification,
)
from app.common.factories import get_service_factory
from app.infra.graph import execution_ids, node_span
from app.services.capability.application import CapabilityService
from app.services.tool.application import ToolQueryService

from .catalog import (
    AGENT_TYPE,
    CLASSIFIER_BUILDER_TYPE,
    CLASSIFIER_MODEL_ROLE,
    NODE_SPECS,
)
from .context import ReactContext
from .state import ReactState

__all__ = ["agent_builder_node"]

logger = logging.getLogger(__name__)

_SPEC = next(s for s in NODE_SPECS if s.node_id == "builder")


# ============================================================
# 分类契约（结构化输出模型）
# ============================================================
#
# 放在本模块而不是 ContextBuilder 里：ContextBuilder 只管"上下文该长什么
# 样"（与 agent_node 的 builder 一样，它不知道提示词），"这次调用要模型
# 回答什么结构"是**本节点**的业务。
#
# ⚠️ 提示词**不在这里**（曾经在这里）
# ---------------------------------
# 分类提示词的正文与它的排版规则搬去了
# app/agent/prompt/prompts/classification.py，取值域 BEHAVIORS / PLANS 也
# 一起搬走（仍由 app/agent/state 的枚举派生），这里 import 回来给下面的
# 字段说明用 —— 取值域只有一处定义，改枚举两处一起变。
#
# 本模块从现在起只说"模型要回答什么**结构**"，不说"怎么问它"。


class _ClassificationOutput(BaseModel):
    """
    分类调用的结构化输出。

    字段一律是**宽松的 str / list**，**不**用 Literal 卡死取值：
    取值域已经写在提示词里，而 Literal 一旦被模型违反，不同 provider 的
    表现不一致（有的报错、有的静默截断）。收进来之后再统一纠偏
    （coerce_*），归不了类的落 `other` —— 那是**一次正常结果**，不是错误。

    能力候选集是**动态**的（来自数据库），所以它也进不了 Literal：
    候选由提示词给出（见 app/agent/prompt/prompts/classification.py）。

    ⚠️ 能力 code 只有**一个**出口：`capability_needs`
    -------------------------------------------------
    这里曾经同时问 `modules`（业务域）与 `capability_needs`（能力域接口），
    指望模型按语义分桶。实测它分不对 —— 候选清单是混在一起的，而库里
    那张 capability 表**没有**字段能区分两者（只有 code / parent_id /
    is_active），模型只能靠名称与描述猜，实测它把 `CAPABILITY`（能力域）
    填进了 `modules`。

    既然分桶在数据上无从判断，就不该让模型去猜：现在只问一个字段，
    code 全落在 `capability_needs`。业务域的隔离等分类体系加上那个字段
    之后再做（那时 `AgentClassification.modules` 才会被填起来）。
    """

    behavior: str = Field(
        description=f"行为分类，取值之一：{BEHAVIORS}。判不出来用 other。",
    )

    execution_plan: str = Field(
        description=f"执行方案，取值之一：{PLANS}。判不出来用 other。",
    )

    capability_needs: list[str] = Field(
        description=(
            "完成这件事要用到哪些能力，从候选清单里按 code 选（可多选、"
            "可为空）。**只填每项开头的那个标识（code）**，不是名称；"
            f"候选里没有合适的就用 {MODULE_OTHER}。"
        ),
    )

    capability_config: dict[str, Any] = Field(
        description="能力配置的初步建议（可为空对象）。",
    )

    # ⚠️ 两个 flag 声明成**宽松的 str**（"true" / "false"），不是 bool
    # ------------------------------------------------------------------
    # 与上面三个字段同一条立场（见本类的 docstring）：provider 对严格类型
    # 的违约表现不一致 —— 有的报错、有的静默截断整条结构化输出。声明成
    # bool 的话，模型回一个 `"yes"` 就让**整次分类**降级成"全 other"，
    # 而它本来只想说"要召回"。所以收进来之后再纠偏（`_coerce_flag`）。
    retrieve_l0: str = Field(
        description=(
            "这一轮要不要预先召回 L0 原始对话。填 true 或 false，"
            "判据见提示词。拿不准填 false。"
        ),
    )

    retrieve_l1: str = Field(
        description=(
            "这一轮要不要预先召回 L1 原子记忆。填 true 或 false，"
            "判据见提示词。拿不准填 false。"
        ),
    )


@dataclass(frozen=True)
class _ModuleCandidate:
    """
    候选能力（一条启用中的能力）。

    为什么不止带 code：`modules` 与 `capability_needs` 的取值都必须是
    code（下游合起来按它查工具，见 `_tool_keys_for_capabilities`），
    但**模型判断该选哪个**靠的是名称与描述。只给 code 等于让它按字符串
    相似度猜 —— 而这两个字段的区别（业务域 / 能力域接口）恰恰只能从
    名称与描述里读出来。
    """

    code: str

    name: str

    description: str | None = None


# ⚠️ 提示词的正文与排版（`_render_candidate` / `_system_prompt`，以及
# `_DESCRIPTION_LIMIT`）从这里**搬走**了，现在在
# app/agent/prompt/prompts/classification.py —— 取用一律经
# `get_prompt_provider().node_prompt(CLASSIFIER_MODEL_ROLE, ...)`
# （那个常量就是 `ModelRole.CLASSIFICATION`）。
#
# `_ModuleCandidate` 留在这里：它是"从能力表读回来的形状"（本节点自己去
# 查库得到），而提示词那边只按 code / name / description 三项排版，两边
# 各自演化。


# ============================================================
# 节点
# ============================================================


async def agent_builder_node(
        state: ReactState,
        runtime: Runtime[ReactContext],
) -> dict[str, Any]:
    """
    分类用户输入，写回 `classification` 通道。

    Returns:
        {"classification": AgentClassification,
         "context_policy": {...含本次的 tool_keys...}}
        —— **没有 `messages`**，见模块 docstring。
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
                "builder_type": CLASSIFIER_BUILDER_TYPE.value,
                "agent_type": AGENT_TYPE,
                "model_role": CLASSIFIER_MODEL_ROLE,
            },
    ) as span:

        user_input = _latest_user_text(state)

        if user_input is None:
            # 没有用户消息就没什么可分类的。这不是错误（图可能被别处
            # 用别的入口驱动），给一个标记过的空结果即可。
            classification = empty_classification(
                note="没有用户消息可供分类。"
            )
            span.set_metrics({"degraded": True})
            return {"classification": classification}

        candidates, notes = await _module_candidates()

        classification = await _classify(
            context=context,
            runtime=runtime,
            state=state,
            user_input=user_input,
            candidates=candidates,
            notes=notes,
        )

        # 能力 code -> 本次可用的工具。判出分类之后才做：这些 code 是
        # 模型给的，顺序反了我们就是拿上一轮的结果在查工具。
        #
        # ⚠️ 仍然按 ② 与 ④ 的**并集**查（一次查询、服务内部去重），
        # 尽管 ② 今天恒为空（原因见模块 docstring）。留着它是因为分类
        # 体系加上区分字段之后 ② 就会开始有值，那时这条路不用改；
        # 而现在多传一个空列表的代价是零。
        tool_keys, tool_notes = await _tool_keys_for_capabilities(
            [
                *classification["modules"],
                *classification["capability_needs"],
            ]
        )

        # 人工写的工具说明覆盖（预载）。**必须与 tool_keys 同一轮取**：
        # 覆盖是按 tool_key 索引的，而这一轮选中的 key 刚刚才定下来。
        #
        # 放在这里而不是执行期（每次模型调用那一层）是因为那是热路径：
        # 主循环每经过一次工具结果就重建一次上下文，而覆盖是慢变数据。
        # 本节点是图入口、每次运行只跑一次 —— 一次查询管一整轮。
        tool_overrides, override_notes = await _tool_overrides_for_keys(
            tool_keys
        )

        # 主 agent 的**职责**提示词（"这个模型是干什么的"）。它经
        # context_policy.system_prompt 进主循环的 system 消息 —— 本节点是
        # context_policy 的唯一写者，所以由它落笔（见 _merged_policy）。
        system_prompt, prompt_notes = await _duty_prompt(state)

        # 本次记忆召回的指令（搜哪几层 + 以谁的身份搜）。**不查库**，
        # 只把分类的决策与 duty_key 拼在一起 —— 真正的召回在上下文流水线
        # 第 5 阶段（memory 处理器）里发生，那里才是它的执行期。
        memory_retrieve = _memory_retrieve(classification, state)

        degraded = [*tool_notes, *override_notes, *prompt_notes]

        if degraded:
            # 降级信息并进 notes，让"这轮为什么没有工具 / 没有职责提示词"
            # 在 state 里读得出来，而不是只留在日志里。
            classification["notes"] = [
                *classification["notes"], *degraded
            ]

        span.set_metrics({
            "degraded": classification["degraded"],
            "module_candidates": len(candidates),
            "tool_keys": len(tool_keys),
            # 这一轮有几个工具带人工说明覆盖。它是"人在 BE 改的说明
            # 到底有没有进这一轮"的唯一观测信号 —— 没有它，"改完没生效"
            # 只能靠对着请求体看。
            "tool_overrides": len(tool_overrides),
            # 这一轮主模型有没有 system 提示词（职责提示词取不到时是 False）
            # —— 降级要看得见，观测里这一格就是它。
            "system_prompt": bool(system_prompt),
            # 这一轮判出来的记忆召回开关。**从 memory_retrieve 里读**，
            # 不从 classification 读：前者才是下游真正消费的那份（
            # agent_id 为空时它是空 dict）—— 两处分开读会在"有决策但身份
            # 缺失"时给出一个下游并没有执行的数。
            "retrieve_l0": bool(memory_retrieve.get("l0")),
            "retrieve_l1": bool(memory_retrieve.get("l1")),
        })
        span.set_output_summary({
            "behavior": classification["behavior"],
            # 两个字段都留着（② 今天是空的）：观测面板按固定键取值，
            # 键"今天没有"与"这个指标消失了"必须分得开。
            "modules": classification["modules"],
            "capability_needs": classification["capability_needs"],
            "execution_plan": classification["execution_plan"],
            "tool_keys": tool_keys,
            "memory_retrieve": memory_retrieve,
        })

        return {
            "classification": classification,
            "context_policy": _merged_policy(
                state, tool_keys, system_prompt, tool_overrides,
                memory_retrieve,
            ),
        }


# ============================================================
# 一次分类调用
# ============================================================


async def _classify(
        *,
        context: ReactContext,
        runtime: Runtime[ReactContext],
        state: ReactState,
        user_input: str,
        candidates: list[_ModuleCandidate],
        notes: list[str],
) -> AgentClassification:
    """
    构建窄上下文 -> 调快模型 -> 解析。

    整段包在 try 里：任何一步出问题都退化成"全 other + degraded"，
    理由见模块 docstring。
    """

    request = ContextBuildRequest(
        builder_type=CLASSIFIER_BUILDER_TYPE,
        thread_id=execution_ids(runtime).thread_id,
        messages=tuple(state.get("messages") or ()),
        run_id=context.pipeline_run_id,
        graph_state=_graph_state(state, candidates),
        agent_type=AGENT_TYPE,
        model_role=CLASSIFIER_MODEL_ROLE,
        dependencies=ContextDependencies(
            model_registry=context.model_registry,
        ),
    )

    try:
        # classifier_builder 是启动期解析好的实例（见 context.py）——
        # 与 agent 节点用 context.builder 是同一种形状。
        model_request = await context.classifier_builder.build(
            request
        )

        model_runtime = context.model_registry.get_runtime(
            agent_type=AGENT_TYPE,
            model_role=CLASSIFIER_MODEL_ROLE,
            context=ModelResolveContext(
                required_capabilities=required_capabilities_for(
                    model_request
                ),
            ),
        )

        response = await model_runtime.ainvoke(model_request)

        return _parse(
            response.message,
            candidates=candidates,
            notes=notes,
        )

    except Exception as exc:  # noqa: BLE001 - 见模块 docstring 的降级立场

        logger.warning(
            "Agent Builder 分类降级（%s: %s），本轮按未分类处理。",
            type(exc).__name__,
            exc,
        )

        return empty_classification(
            note=f"分类失败：{type(exc).__name__}: {exc}"
        )


def _parse(
        message: Any,
        *,
        candidates: list[_ModuleCandidate],
        notes: list[str],
) -> AgentClassification:
    """
    把模型的结构化输出收进 AgentClassification。

    三种形态都认：

        BaseModel     `with_structured_output` 回**解析好的 pydantic 实例**
                      （见 model/adapters.py 的 `_bind`）—— 真模型路径走这条
        dict          有的 provider 直接给字典
        str           有的给一段 JSON 文本

    ⚠️ 第一种必须是**第一条分支**
    ---------------------------
    这里曾经只认后两种，于是**每一次分类都被降级**：实例既没有
    `.content`、也不是 dict，直接落到下面那个"类型意外"分支，模型答对了
    也全被丢掉 —— 症状是 `modules=[]`、`tool_keys=[]`、模型一个工具都
    看不到，而 state 里只留一句 notes。所以 `BaseModel` 判在最前：
    顺序反过来时，将来 `_ClassificationOutput` 多一个叫 `content` 的字段
    就会被上面那句 `.content` 悄悄吞掉。

    ⚠️ 这里**不抛异常**，也不试图"修好"模型的话：它给了什么就收什么，
    收不进取值域的落 other。一段以解析失败告终的分类，其价值与
    "全 other" 相同，但前者会中断对话。

    不给 list / 标量加挽救路径：配 `BaseModel` schema 的结构化输出不可能
    合法地产出它们，那属于 provider 违约，就该降级。
    """

    if isinstance(message, BaseModel):
        payload = message.model_dump()

    else:
        payload = (
            message.content if hasattr(message, "content") else message
        )

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return empty_classification(
                note="结构化输出不是合法 JSON。"
            )

    if not isinstance(payload, dict):
        return empty_classification(
            note=f"结构化输出类型意外：{type(payload).__name__}。"
        )

    # 允许集只认 **code**：提示词里给的是 "code：名称 —— 描述" 那种行，
    # 模型偶尔会把名称也回填进来（尤其名称看起来更像"类目"时）。这里只
    # 认 code，认不出的（含名称）落 MODULE_OTHER —— 与提示词里那句
    # "只填每项开头的那个标识"是同一条约定。
    allowed = {candidate.code for candidate in candidates}

    codes = [
        str(item) if str(item) in allowed else MODULE_OTHER
        for item in _as_list(payload.get("capability_needs"))
    ]

    return AgentClassification(
        behavior=coerce_behavior(payload.get("behavior")),
        # ⚠️ 能力 code **只有 capability_needs 一个出口**，这里恒为空。
        #
        # 以前模型分两个字段填（业务域 / 能力域接口），实测分不对：候选
        # 清单是混在一起的，库里也没有能区分两者的字段，模型只能猜，于是
        # 把 `CAPABILITY`（能力域）填进了 modules。既然分桶在数据上无从
        # 判断，装配层就不该把它留在模型那边。
        #
        # 字段本身留着：它是对外契约与 state 的一部分，而业务域的隔离等
        # 分类体系加上区分字段之后才会启用（那时这里才有东西可填）。
        modules=[],
        execution_plan=coerce_execution_plan(
            payload.get("execution_plan")
        ),
        # 空 + 候选非空时不补 other：模型说"不属于任何一项"与"没给"是
        # 两回事，而 [] 恰好能表达前者。
        capability_needs=_dedupe(codes),
        capability_config=(
            dict(payload["capability_config"])
            if isinstance(
                payload.get("capability_config"), dict
            )
            else {}
        ),
        # 记忆召回的决策。认不出**一律 False**（见 _coerce_flag）——
        # 它是保守的一侧：多召回一次要付一次向量检索，漏召回的这一轮
        # 主模型还能自己调工具补。
        retrieve_l0=_coerce_flag(payload.get("retrieve_l0")),
        retrieve_l1=_coerce_flag(payload.get("retrieve_l1")),
        degraded=False,
        notes=list(notes),
    )


# ============================================================
# 能力候选（数据库）
# ============================================================


async def _module_candidates() -> tuple[list[_ModuleCandidate], list[str]]:
    """
    从能力表读候选，返回 (候选, 说明)。

    **只读、可降级**：拿不到就回空列表 + 一条说明，让分类继续（模型会
    把 modules 与 capability_needs 都落成 other），而不是让整轮对话失败。

    读法照抄 app/agent/tools/catalog_tool.py —— 没有 Request 的地方走
    应用级 ServiceFactory + `scope(begin=False)`（只读，不开写事务）。

    ⚠️ `_ModuleCandidate` 这个名字是历史命名（最初只有 modules 一个维度
    用它）。返回的候选是**一份共用的清单**：② 业务域与 ④ 能力域接口都从
    它里面选，库表上没有字段能把两者分开。

    名称与描述一并带上：它们要进提示词（见
    app/agent/prompt/prompts/classification.py），而
    code 单独一个对模型几乎没有语义 —— 在"要判这条属于业务域还是能力域
    接口"这件事上，名称与描述是**唯一**的依据。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        # 应用没跑过 lifespan。测试与脚本里是常态，不是故障。
        return [], [f"能力目录不可用（{exc}）"]

    try:
        async with factory.scope(
                CapabilityService, begin=False
        ) as service:

            capabilities = await service.list_capabilities(
                is_active=True
            )

    except Exception as exc:  # noqa: BLE001 - 降级，见模块 docstring

        logger.warning(
            "读取能力目录失败（%s: %s），两个维度的分类都将全部落 other。",
            type(exc).__name__,
            exc,
        )

        return [], [f"读取能力目录失败：{type(exc).__name__}"]

    return [
        _ModuleCandidate(
            code=c.code,
            name=c.name,
            description=c.description,
        )
        for c in capabilities
    ], []


# ============================================================
# 能力 code -> 工具（数据库）
# ============================================================


async def _tool_keys_for_capabilities(
        codes: list[str],
) -> tuple[list[str], list[str]]:
    """
    能力 code（② 业务域 + ④ 能力域接口的并集）-> 本次要暴露给主模型的
    工具 key，返回 (keys, 说明)。

    调用方把两个维度**合起来**传进来（见 `agent_builder_node` 里那次
    合并）：两个维度地位相同，分两次查再自己合会让"哪个维度贡献了什么"
    在代码里变成一个需要推理的问题。这里也照单全收 —— 本函数不认识
    业务域与能力域接口的区别。

    ⚠️ 精确匹配，不含子孙能力：`tool_keys_for_capabilities` 按
    `capability.code IN (...)` 查，不沿 `parent_id` 递归（见
    ToolDefinitionRepository 的契约）。因此传进来的必须是候选清单里
    那个**原样的 code** —— 这一点由 `_parse` 的允许集保证。

    与 `_module_candidates` 同一条降级纪律：**只读、可降级**，任何一步
    失败都回空列表 + 一条说明，让主循环照常跑。

    为什么失败可以安全地降级成"没有工具"
    ------------------------------------
    没有工具是**安全的中间态**：模型只是少了几只手，它仍然能照着
    已有上下文回答；而"带着一半工具跑"或"抛异常中断整轮"都更糟。
    所以这条路上宁可少给。但降级必须看得见 —— 说明写进 notes，
    span 记 tool_key 数，日志里也有 WARNING。

    为什么过滤 MODULE_OTHER
    ----------------------
    `MODULE_OTHER` 是"落不进候选集"的**桶**，不是一条真能力 ——
    库里不会有挂到它下面的工具。拿它去查只会白跑一次，
    而"查不到"与"这条 code 是桶"两件事混在一起会让 notes 读不懂。

    两个维度共用这一个哨兵值，所以这里过滤一次就够；不需要按维度分开。
    """

    usable = [
        str(code) for code in codes
        if code and str(code) != MODULE_OTHER
    ]

    if not usable:
        # 两个维度都没判出真能力（降级时就是这种）。不是错误，
        # 也不用去碰数据库。
        return [], []

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        # 应用没跑过 lifespan。测试与脚本里是常态，不是故障。
        return [], [f"工具目录不可用（{exc}）"]

    try:
        async with factory.scope(
                ToolQueryService, begin=False
        ) as service:

            keys = await service.tool_keys_for_capabilities(usable)

    except Exception as exc:  # noqa: BLE001 - 降级，见模块 docstring

        logger.warning(
            "按能力 code 读取工具失败（%s: %s），本轮不暴露工具。",
            type(exc).__name__,
            exc,
        )

        return [], [f"读取工具目录失败：{type(exc).__name__}"]

    return keys, []


async def _tool_overrides_for_keys(
        tool_keys: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    本次选中的 tool_key -> 人工写的说明覆盖，返回 (覆盖, 说明)。

    形状与 `ToolQueryService.tool_documentation_overrides` 的返回一致：

        {tool_key: {"description": str | None,
                    "parameters": {"/parent_id": str | None, ...}}}

    它进 `context_policy.tool_overrides`，由 capability 处理器透传给
    `ToolExposureSource.resolve`，最终由 `for_wire` 叠进 bind 给模型的
    工具。**预载在这里、消费在热路径**是刻意的：主循环每经过一次工具
    结果就重建一次上下文，而人工说明是慢变数据。本节点是图入口、
    每次运行只跑一次，一次查询管一整轮。

    与 `_tool_keys_for_capabilities` 同一条降级纪律：**只读、可降级**。
    任何一步失败都回空 dict + 一条说明，让主循环照常跑。

    为什么失败可以安全地降级成"没有覆盖"
    ------------------------------------
    没有覆盖 = 工具说明回落到**代码侧事实**（docstring / 声明），
    那正是这条路出现之前的行为。也就是说降级的后果是"退回到今天"，
    不会更差；而为此中断整轮（模型一个工具都拿不到）显然更糟。
    降级必须看得见：说明写进 notes，span 记覆盖数，日志里也有 WARNING。
    """

    if not tool_keys:
        # 两个维度都没判出工具（降级时就是这种）。不是错误，
        # 也不用去碰数据库。
        return {}, []

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        # 应用没跑过 lifespan。测试与脚本里是常态，不是故障。
        return {}, [f"工具说明不可用（{exc}）"]

    try:
        async with factory.scope(
                ToolQueryService, begin=False
        ) as service:

            overrides = await service.tool_documentation_overrides(
                tool_keys
            )

    except Exception as exc:  # noqa: BLE001 - 降级，见 docstring

        logger.warning(
            "读取工具说明覆盖失败（%s: %s），本轮用代码侧的说明。",
            type(exc).__name__,
            exc,
        )

        return {}, [f"读取工具说明失败：{type(exc).__name__}"]

    return overrides, []


def _merged_policy(
        state: ReactState,
        tool_keys: list[str],
        system_prompt: str | None,
        tool_overrides: dict[str, Any] | None = None,
        memory_retrieve: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    把本次选中的工具、它们的人工说明覆盖，与主 agent 的职责提示词
    并进 `context_policy` 覆盖项。

    ⚠️ **必须合并，不能覆盖**：`context_policy` 通道没有 reducer
    （后写者赢），而它同时承载调用方放进来的覆盖项。直接返回一个
    新 dict 会把它们静默抹掉 —— 症状是"某个部署配的策略项偶尔不
    生效"，取决于这一轮分类有没有判出模块。

    `tool_keys` 走 context_policy 而不是新开一个 state 通道，是因为
    capability 处理器早就绑定了 `context_policy.tool_keys`
    （见 processors/capability.py 的 StateBinding），而且
    `merge_policy_state` 保留未知键 —— 本节点已经在用同一个机制
    透传 `response_format` / `model_metadata`（见 _graph_state）。

    ⚠️ `system_prompt` 是**覆盖**写，不是 `setdefault`
    ----------------------------------------------
    本节点每一轮都会重写它。用 `setdefault` 的话，第一轮写进去的文本会被
    checkpoint 永久留住（`existing` 里已经有这个键，于是永不更新），
    结果是"改了提示词重新部署"对**长会话**无效 —— 而提示词正文恰恰是随
    部署变的东西。

    取不到（None / 空串）时**不写**：保留既有值，而不是拿空串把已有的
    提示词抹掉。谁覆盖谁在这里是明确的 —— 主循环的 system 提示词由
    PromptProvider 说了算（这也是"取提示词的地方一律走 Provider"的落点）。

    ⚠️ `tool_overrides` 与 `tool_keys` 一样是**每轮无条件重写**
    ------------------------------------------------------
    不能用 `setdefault`，理由与上面 `system_prompt` 那条是同一个：
    `context_policy` 是会被 checkpoint 的普通通道，本节点每次运行只
    跑一次、而通道里留着上一轮的值 —— `setdefault` 会让第一轮写进去的
    覆盖被永久留住，此后在 BE 改了说明也对**长会话**无效。

    `None`（这一次没取到覆盖）要写成空 dict 而不是跳过：跳过就等于
    "保留上一轮那份"，而那正是上面这条要避免的。空 dict 的表达是
    "这一轮没有覆盖"，语义正确。
    """

    existing = state.get("context_policy") or {}

    merged: dict[str, Any] = {
        **existing,
        "tool_keys": tool_keys,
        "tool_overrides": tool_overrides or {},
        # 本次记忆召回的指令（搜哪几层 + 以谁的身份搜）。
        #
        # ⚠️ 与上面两项同一条纪律：**每轮无条件重写**，`None` 写成空 dict。
        # 理由与 `tool_overrides` 那段逐字相同 —— `context_policy` 是会被
        # checkpoint 的通道，`setdefault` 会把第一轮的决策永久留住。
        # 而这里比说明覆盖更要紧：上一轮的"要召回"漏进这一轮，等于让一个
        # 本该不查记忆的运行去查记忆。
        "memory_retrieve": memory_retrieve or {},
    }

    if system_prompt:
        merged["system_prompt"] = system_prompt

    return merged


def _memory_retrieve(
        classification: AgentClassification,
        state: ReactState,
) -> dict[str, Any]:
    """
    本次记忆召回的指令：搜哪几层 + 以谁的身份搜。

    它进 `context_policy.memory_retrieve`，由 memory 处理器（上下文流水线
    第 5 阶段）读走并执行召回 —— 与 `tool_keys` 是**同一条路**（本节点是
    context_policy 的唯一写者，处理器只按声明式绑定读）。

    两个 flag 来自分类（模型判"这一轮要不要预先召回"）；`agent_id` 来自
    `duty_key` —— `RetrievalTarget` 的 schema 写着它"数据库里对应
    `business_agent_id` / `duty_key`"，记忆的 scope 就是按这个键分组的。

    ⚠️ `agent_id` 为空就回一个**空 dict**（= 不召回），不全给两个 flag
    ------------------------------------------------------------------
    两件事：空的 agent_id 拼进 `RetrievalTarget` 会当场 `ValidationError`
    （它有 `min_length=1`），而"这个 agent 是谁"判不出来时**不召回**是
    对的 —— 拿一个空的 scope 去查，要么报错，要么（更糟）查到别人的。
    返回空 dict 而不是省略这个键，是因为调用方**无条件重写**它：省略
    等于把上一轮的决策留在通道里。

    这是**只读、可降级**的：它不查库，只读 state。
    """

    agent_id = str(state.get("duty_key") or "").strip()

    if not agent_id:
        return {}

    return {
        "l0": bool(classification.get("retrieve_l0")),
        "l1": bool(classification.get("retrieve_l1")),
        "agent_id": agent_id,
    }


async def _duty_prompt(state: ReactState) -> tuple[str | None, list[str]]:
    """
    取主 agent 的职责提示词，返回 (提示词, 说明)。

    用哪个职责**不是这里选的**：`state["duty_key"]` 是建运行那一刻定下的
    （见 agent_run_service_impl.create_run 与 app/services/prompt 的
    default_duty_key），这里只是照它取文本。空串（存量运行、或解析失败
    之外没有指定）等价于"没指定"，交给 Provider 回落默认职责。

    **只读、可降级** —— 与 `_module_candidates` / `_tool_keys_for_capabilities`
    同一条纪律：任何一步失败都回 (None, 一条说明)，让主循环照常跑。

    为什么这里降级而不是抛
    --------------------
    取不到时，主循环的行为**就是本次改动之前的行为**：没有 system 提示词，
    只剩注入分节（画像 / 记忆 / 资料 / 摘要），对话完全成立。而抛出去会把
    "提示词表读不到"升级成"整轮运行失败" —— 那是拿一个小问题换一个更大的
    问题。

    脚本与测试里没有 lifespan 是常态（`get_service_factory()` 会抛
    RuntimeError），降级让它们照常跑。但降级必须**看得见**：说明进
    `classification.notes`、span 记 `system_prompt=False`、provider 那边
    还有一条 WARNING 日志。
    """

    try:
        return await get_prompt_provider().duty_prompt(
            state.get("duty_key") or None
        ), []

    except Exception as exc:  # noqa: BLE001 - 降级，理由见上面那段

        logger.warning(
            "读取职责提示词失败（%s: %s），本轮主模型没有 system 提示词。",
            type(exc).__name__,
            exc,
        )

        return None, [f"职责提示词不可用：{type(exc).__name__}"]


# ============================================================
# internals
# ============================================================


def _graph_state(
        state: ReactState,
        candidates: list[_ModuleCandidate],
) -> dict[str, Any]:
    """
    交给 ContextBuilder 的图 state 快照 + 本次的策略覆盖。

    ⚠️ `context_policy` 里那两项是**必须**的，不是调优：
    CLASSIFIER 的策略基线把 history 关掉了（include_history=False），
    而用户这句话只有 history 处理器会送进 prompt（见
    context/builder/classifier.py 的说明）。不覆盖它，模型看到的提示词
    里就没有问题本身。

    `min/max_recent_turns=1` = 只留当前这一轮（"最后一轮永不裁"）——
    窗口宽度是 `clamp(未覆盖轮数, min, 上限)`，两者都钉成 1 就没有余地。
    """

    snapshot: dict[str, Any] = {}

    for name in ("conversation_id", "user_id", "summary"):
        if name in state:
            snapshot[name] = state[name]

    existing = state.get("context_policy") or {}

    snapshot["context_policy"] = {
        **existing,
        "include_history": True,
        "min_recent_turns": 1,
        "max_recent_turns": 1,
        "max_history_tokens": 0,
        # 分类提示词：**经 PromptProvider 取**（正文在
        # app/agent/prompt/prompts/classification.py）。键用的就是本次调用
        # 的 model_role（CLASSIFIER_MODEL_ROLE = ModelRole.CLASSIFICATION）
        # —— "用哪个模型"与"给它什么提示词"由同一个常量决定。
        "system_prompt": get_prompt_provider().node_prompt(
            CLASSIFIER_MODEL_ROLE, candidates=candidates
        ),
        # 结构化输出：merge_policy_state 保留未知键，prompt 组装处理器
        # 正是从 context_policy.response_format 读它（prompt.py 的
        # StateBinding），再经 ModelRequest.response_format 交给
        # with_structured_output。
        "response_format": _ClassificationOutput,
        "model_metadata": {"agent_builder": True},
    }

    return snapshot


def _latest_user_text(state: ReactState) -> str | None:
    """最后一条用户消息的文本。没有则 None。"""

    for message in reversed(list(state.get("messages") or ())):

        if isinstance(message, HumanMessage):
            content = getattr(message, "content", "")

            return content if isinstance(content, str) else str(content)

    return None


def _as_list(value: Any) -> list[Any]:
    """模型给的可能是单个值、列表或 None —— 一律收成列表。"""

    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _dedupe(items: list[str]) -> list[str]:
    """去重且保序 —— 模型重复列同一个模块是常事。"""

    return list(dict.fromkeys(items))


#: `_coerce_flag` 认得的字面量。
#:
#: 中英各收一份：模型偶尔会用中文回"是/否"，而那是**同一个意思** ——
#: 为这个把它判成"认不出"（落 False）等于让一次正常的判断悄悄失效。
_TRUTHY = frozenset({"true", "yes", "y", "1", "是", "要", "需要"})
_FALSY = frozenset({"false", "no", "n", "0", "否", "不要", "不需要"})


def _coerce_flag(raw: Any) -> bool:
    """
    模型给的任意值 -> bool。

    ⚠️ 认不出**一律 False**，不是"猜一个"、也不是"当作缺失"。
    两边的代价不对等：多召回一次要付一次向量检索，而漏召回只是这一轮
    少一段记忆 —— 主 agent 还挂着 L0/L1 的检索工具，能自己补。
    不确定时选便宜的那边。

    与 `coerce_behavior` 同一条思路（见 app/agent/state/agent_state.py）：
    宽松收进来、统一纠偏，而不是让 provider 的违约变成整条降级。
    """

    if isinstance(raw, bool):
        return raw

    if isinstance(raw, (int, float)):
        # pydantic 偶尔会把 "1" 解成 int。0/1 之外的数没有明确语义，
        # 但既然模型给了一个数，非零当 true 是唯一讲得通的理解。
        return bool(raw)

    if isinstance(raw, str):

        text = raw.strip().lower()

        if text in _TRUTHY:
            return True

        if text in _FALSY:
            return False

    return False
