"""
工具执行节点（**扇出分支**：一次调用一个实例）。

它做什么
--------
路由（`route_after_agent`）给每一个 tool_call 发一条 `Send`，本节点因此
**每跑一次只处理一个调用**：

    读 payload 里那一个 call -> 算允许集 -> 执行或拒绝 -> 回一条 ToolMessage
    控制器分支再把工具的 Command（或默认 goto=agent）交回 LangGraph

执行本身**不在这里** —— 链怎么驱动、service 怎么造、参数怎么包，都在
`app/infra/agent/tool_executor.py`。本节点只管图语义：读 payload、
写 messages、记观测。

分两层：图语义 / 单次调用
------------------------
    tools_node        图语义：把节点的 `Runtime` 变成处理器契约要的
                      `ToolRuntime`；把结果包成 state 更新；控制器
                      分支返回 `Command` 决定下一跳
    _wrap_tool_call   一次调用的**全部**逻辑。入参与 langgraph
                      `ToolNode` 的 `awrap_tool_call` 同形，返回
                      `_ToolCallResult`（ToolMessage + 可选 Command）

为什么不再严格照 `awrap_tool_call` 的返回类型
    **任何**工具都可以返回 `Command.goto` 做 handoff（不只计划工具），
    单返回 ToolMessage 表达不了下一跳。`_ToolCallResult` 把"给模型
    看的消息"和"给图看的跳转"分开；将来若换成预构建 `ToolNode`，
    需要把控制器 / 计划状态结算这段适配过去（它自己支持 Command，
    但不知道我们的 plan_* 语义）。

为什么今天**不**直接用预构建的那个
    它一次节点执行会跑完一条 AIMessage 里的**全部** tool_call
    （`tool_node.py` 的 `asyncio.gather`），而中断会丢弃该节点待写的
    状态并**从头重跑** —— 那正是本图改扇出之前实测过的重复执行
    （见 routing.route_after_agent 的说明）。扇出 + 自己当节点，
    才能保住"一个调用一个 task"。

⚠️ 为什么必须是扇出（一个调用一个节点实例）
------------------------------------------
与 `interrupt()` 的语义有关：它会**丢弃该节点待写的状态、并从节点开头
重跑**。如果这里是一个 `for` 循环，中断发生在第 3 个工具时，前两个工具
在 resume 后会被**真实地再执行一遍**（实测：中断后 `['t1']`，resume 后
`['t1','t1','t2']`；扇出后是 `['t1','t2']`）。下单、转账这类工具不能
这么跑。

已完成的兄弟分支不受影响：它们各自是独立 task，结果已落进 checkpoint。

⚠️ 本节点**只**从 payload 读输入
-------------------------------
`Send` 的 payload **取代** state（已实测），父状态的任何字段这里都看不到。
因此"它需要什么"全部列在 `ToolBranchPayload` 里 —— 拿不到就抛，
见 `_branch_span_kwargs`。

⚠️ 本图的不变量：`exposed_tools` 恒空、`tool_source` 恒在
--------------------------------------------------------
capability 处理器有三条路：有 tool_source 且 tool_keys 非空 -> 按 key
解析；只有 tool_keys 没 source -> 抛 LookupError；否则读 state 的
`exposed_tools`。本图恒注入 tool_source、且没有任何节点写
exposed_tools —— 于是"没有 tool_keys"就等于"没有工具"，两条路给出同一个
答案。

将来若允许 `exposed_tools` 那条路生效，**本节点必须一起改**：那时
"允许执行什么"就不再只由 tool_keys 决定了。

⚠️ 允许集里的 key 与模型回传的名字**不是同一个字符串**
----------------------------------------------------
`tool_key` 带点号（`capability.list_capabilities`），而端点只接受
`^[a-zA-Z0-9_-]+$`，所以出站时转成了 `capability_list_capabilities`
（见 infra/agent/tool_exposure.py）。模型回传的是转过的那个，本节点用
`tool_key_of` 翻回 tool_key 再判允许集 —— 两端用的是同一个 `wire_name`。

翻不回来一律按"未暴露"拒绝，**不**回退成"原文碰巧在允许集里就放行"：
那会让模型能成功调用的名字变成安全名与 tool_key 的并集，而后者从没
发给它。

为什么不返回一条假的 ToolMessage
--------------------------------
曾经（执行入口还没接通时）本节点直接抛 NotImplementedError：把
`ToolMessage(content="工具执行尚未接通")` 塞回 state 会让模型基于
"工具说没接通"继续编，整轮还会被记成 completed —— 一个必然错误的结果
呈现为一次成功的运行。

**现在这条立场仍然约束着每一处失败**：只有真的执行过（或真的被拒绝 /
失败）才写 ToolMessage，且失败的要带 `status="error"`。

⚠️ 指标是**逐分支**的
--------------------
本节点每行只对应一次工具调用，所以 `tool_call_count` 恒为 1、其余键是
0/1。按 `node_id` 汇总时可以对计数求和，但 `catalog_unavailable` 是
"这个分支遇到了目录不可用"，求和得到的是**有多少分支**遇到，不是
"目录是否不可用"。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, NamedTuple
from uuid import UUID

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.prebuilt.tool_node import (
    ToolCallRequest,
    ToolRuntime,
)
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agent.context.processor.reference import wrap_reference
from app.agent.tools.hitl_tool import (
    ASK_USER_QUESTION_TOOL_KEY,
)
from app.agent.tools.plan_tool import PLAN_CONTROL_TOOL_KEYS
from app.agent.tools.processor.context import (
    ToolCallDenied,
    ToolOutcome,
)
from app.common.event import TOOL_STREAM_KIND, RunEventType
from app.common.factories import get_service_factory
from app.common.tool_caller import ToolCaller, tool_caller_scope
from app.infra.agent.tool_executor import get_tool_executor
from app.infra.agent.tool_exposure import for_wire
from app.infra.graph import NodeSpec, branch_scope_id, node_span
from app.services.tool.application import ToolQueryService
from app.services.tool.application._execution import execution_result
from app.services.tool.registry import select_specs, tool_key_of

from .catalog import (
    AGENT_NODE,
    NODE_SPECS,
    REACT_TOOL_SCOPE_NAMESPACE,
)
from .context import ReactContext
from .state import ToolBranchPayload

_SPEC = next(s for s in NODE_SPECS if s.node_id == "tools")

#: 不走处理器链的工具。
#:
#: 计划控制（enter/exit）与 HITL 提问都是进程内的控制工具：没有副作用，
#: 不需要审批 / 缓存 / 重试，而且必须在**还没同步进 tool_definition** 时
#: 也能执行。`tool_id=None` 在 ToolExecutor 里就是"空链"。
_CHAINLESS_TOOL_KEYS = PLAN_CONTROL_TOOL_KEYS | {
    ASK_USER_QUESTION_TOOL_KEY,
}

__all__ = ["command_update", "execute_tool_branch", "tools_node"]

#: ToolNode 规范里那个 `execute` 回调的形状：`(request) -> ToolMessage`。
#:
#: 本仓库从不调用它（见 `_wrap_tool_call` 的说明），只用来标住签名 ——
#: 将来换成预构建的 `ToolNode` 时，它拿到的就是这个形状的真回调。
DirectToolCall = Callable[[ToolCallRequest], Awaitable[ToolMessage]]

logger = logging.getLogger(__name__)


def _branch_span_kwargs(
        state: ToolBranchPayload,
        *,
        spec: NodeSpec = _SPEC,
        scope_namespace: UUID = REACT_TOOL_SCOPE_NAMESPACE,
) -> dict[str, Any]:
    """
    组装 `node_span` 的关键字参数，顺便守卫 payload。

    `branch_scope_id` 是**分支身份的唯一来源**：扇出的 N 个分支在
    LangGraph 的 callback metadata 里共享同一个节点名（都是 "tools"），
    没有它就分不出哪个分支是哪次调用，观测行会互相覆盖成一行。
    这里从 payload 显式取出来交给 node_span。

    ⚠️ 守卫失败要抛而不是兜底：拿不到 payload 说明**图被装配错了**
    （给扇出节点加了普通边，它会拿到父状态）。那种错误往下走会变成
    "静默地对父状态做操作"，比当场炸难查得多。

    ⚠️ 本函数在 `node_span(...)` 的**入参求值**阶段调用，在上下文
    管理器之外，因此这里抛的错**不会**在节点行里留记录，只会记到
    run 行的 error_message 上（与 document 的 `_branch.py` 同一条）。
    """

    payload = state if isinstance(state, dict) else None

    call = payload.get("call") if payload else None

    if not isinstance(call, dict) or not call:
        raise ValueError(
            "tools: 扇出节点没有收到 Send payload。"
            "它拿到的 state 里没有 'call' 键，说明上游是用普通边"
            "（而非条件边 + Send）进来的 —— 检查 graph.py 的边装配。"
        )

    call_scope = str(payload.get("call_scope") or "")

    if not call_scope:
        # 兜底也抛：call_scope 决定观测身份，缺了会让所有分支撞成
        # 同一行。它由路由算好放进 payload，缺了同样是装配错误。
        raise ValueError(
            "tools: Send payload 里没有 'call_scope'，无法给这个分支"
            "一个稳定的观测身份（见 routing._call_scope）。"
        )

    return {
        "node_id": spec.node_id,
        "label": spec.label,
        "group": spec.group,
        "sequence": spec.sequence,
        "branch_scope_id": branch_scope_id(
            scope_namespace, call_scope
        ),
    }


def _render(value: Any) -> str:
    """
    工具返回值 -> ToolMessage 的正文。

    str 直用（最常见的返回就是一段文本）；其余 JSON 化。
    `default=str` 兜住不可序列化的对象 —— 抛在这里会让"工具明明跑成功
    了"变成一条错误消息，而正文退化一点是能接受的代价。

    不做长度截断：太长该由压缩 / 预算那条路解决（见 context/processor
    的 compaction），在这里悄悄砍一刀会让模型看到一段被截断的结果
    却不知道它被截断了。
    """

    if value is None:
        # value=None 是**合法的成功结果**（见 context.py），不是失败。
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False, default=str)


def _reference_content(tool_key: str, value: Any) -> str:
    """
    成功工具结果 -> 带引用数据定界的 ToolMessage 正文。

    工具返回的可能是网页正文、检索片段、外部 API 响应 —— 这些内容由
    外部控制，统一包成 `<ref:tool>` 并附上引用数据规则。失败 / 拒绝 /
    未暴露工具的消息是**本进程写的**，不走这里。
    """

    content = _render(value)

    if not content.strip():
        return content

    return wrap_reference("tool", tool_key, content)


#: 计划控制工具在返回值里携带的指令。
_PLAN_TRANSITIONS = frozenset({"enter", "exit"})


class _ToolCallResult(NamedTuple):
    """
    一次工具调用的结果。

    `message` 是**必须**写回对话的 ToolMessage（每个 tool_call 一条）。
    `command` 只在工具返回 `Command` 时存在 —— 它携带工具自己定义的
    `goto`（计划控制工具的跳转逻辑就写在这里）。
    """

    message: ToolMessage

    command: Command | None = None


def _plan_update(value: Any) -> Mapping[str, Any] | None:
    """`Command.update` / dict 返回值里的计划指令载荷。"""

    if isinstance(value, Command):
        return value.update if isinstance(value.update, Mapping) else None

    if isinstance(value, Mapping):
        return value

    return None


def _plan_directive(value: Any) -> str | None:
    """
    工具返回值里的计划指令；不是控制工具就返回 None。

    ⚠️ 认两种形状
    -------------
        计划工具   `Command(update={"plan_transition": ...})`
        旧形状     `{"plan_transition": ...}`（纯 dict，不触发跳转）

    普通工具可能返回任意值（含字符串），这里不能把它们误判成控制工具。
    """

    payload = _plan_update(value)

    if payload is None:
        return None

    raw = payload.get("plan_transition")

    if isinstance(raw, str) and raw in _PLAN_TRANSITIONS:
        return raw

    return None


def _plan_message_text(value: Any) -> str:
    """控制工具返回的 `message`（附带可选 reason）；没有就留空。"""

    payload = _plan_update(value)

    if payload is None:
        return ""

    text = str(payload.get("message") or "")

    reason = str(payload.get("reason") or "").strip()

    if reason:
        text = f"{text}（理由：{reason}）" if text else f"理由：{reason}"

    return text


def _outcome_message(
        tool_call_id: str | None,
        tool_key: str,
        outcome: ToolOutcome,
) -> ToolMessage:
    """一次执行结果 -> ToolMessage。成功与失败共用这里，只有 status 不同。"""

    if outcome.ok:

        # 工具返回 Command：正文优先取 update.message；普通 Command
        # 没有 message 时退化为 update 的 JSON 渲染（与原版一样，工具
        # 的返回值就是这次调用的结果）。计划指令只在这条返回里**额外**
        # 进 additional_kwargs，不影响其他工具。
        if isinstance(outcome.value, Command):

            directive = _plan_directive(outcome.value)

            extra: dict[str, Any] = {}

            if directive is not None:
                extra["plan_transition"] = directive

            content = _plan_message_text(outcome.value)

            if not content and outcome.value.update:
                content = _render(outcome.value.update)

            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id or "",
                name=tool_key,
                status="success",
                additional_kwargs=extra,
            )

        return ToolMessage(
            content=_reference_content(tool_key, outcome.value),
            tool_call_id=tool_call_id or "",
            name=tool_key,
            status="success",
        )

    error = outcome.error

    # 拒绝与失败分开写：拒绝意味着**工具没有跑**（审批没过、参数不
    # 合规），模型看到之后该换个做法；而失败是"跑了但坏了"，
    # 两者对模型下一步的含义不同。
    if isinstance(error, ToolCallDenied):
        content = f"调用被拒绝（工具未执行）：{error}"
    else:
        content = f"{type(error).__name__}: {error}"

    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id or "",
        name=tool_key,
        status="error",
    )


def _unexposed_message(
        tool_call_id: str | None,
        tool_key: str,
) -> ToolMessage:
    """
    模型要求调用一个**本次没有暴露给它**的工具。

    不抛错：这是模型的一次幻觉，不该让整轮对话 failed —— 但必须留痕
    （WARNING + span metric），因为它同时也是一个信号："模型以为它
    有这个工具"，可能意味着提示词或能力映射有问题。
    """

    return ToolMessage(
        content=(
            f"工具 {tool_key!r} 不在本次可用的工具集合中，未执行。"
            "请只使用当前可用的工具。"
        ),
        tool_call_id=tool_call_id or "",
        name=tool_key,
        status="error",
    )


def _publish_tool_event(
        event_type: RunEventType,
        *,
        tool_call_id: Any,
        tool_name: str,
        **data: Any,
) -> None:
    """
    把工具生命周期写进图的**自定义流**，由 RunEventProjector 投影成前端事件。

    为什么不直接 `publish_run_event`
    -------------------------------
    运行事件的**唯一投影层**是 `RunEventProjector`（见它的模块 docstring）。
    节点自己写 Redis 会让同一次执行出现两个来源：前端拿到的 tool.called
    与 trace 的 tool span 各自独立，id 口径也对不上（实测 langgraph 的
    `tools` 通道会给直接 `ainvoke` 的工具发一对假 id 的事件）。

    写自定义流之后，工具事件与 trace 的 `fanout.*` 走同一条路：

        tools 节点 --custom stream--> RunEventProjector --> Redis（SSE）

    `get_stream_writer()` 在图外（直接调节点函数、单测）会抛，吞掉即可
    —— 那种场合本来也没有订阅端。writer 本身在未请求 `custom` 模式时
    是 no-op，不会阻塞工具执行。
    """

    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - 图外运行没有 writer，不是错误
        return

    writer({
        "kind": TOOL_STREAM_KIND,
        "event": event_type.value,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        **data,
    })


def _tool_event_result(outcome: ToolOutcome) -> dict[str, Any]:
    """ToolOutcome -> tool.completed 的 output / error 字段。"""

    if outcome.ok:

        # Command 不是可 JSON 化的结果；观测里给跳转目标与更新载荷，
        # 而不是把它 str() 成一段看不懂的 repr。
        if isinstance(outcome.value, Command):

            update = (
                dict(outcome.value.update)
                if isinstance(outcome.value.update, Mapping)
                else {}
            )

            return {
                "output": {
                    "goto": str(outcome.value.goto),
                    "update": update,
                }
            }

        return {"output": _render(outcome.value)}

    error = outcome.error

    if isinstance(error, ToolCallDenied):
        return {"error": f"调用被拒绝（工具未执行）：{error}"}

    return {"error": f"{type(error).__name__}: {error}"}


def _bucket(outcome: ToolOutcome) -> str:
    """
    一次执行结果该记进哪个计数桶。

    拒绝与失败分开：拒绝意味着**工具没有跑**，失败是"跑了但坏了"。
    两者的比值是运维真正会看的那个信号（拒绝率突然上升 = 有人在
    链上挂了审批，或模型在乱调工具），混在一个桶里就看不出来。
    """

    if outcome.ok:
        return "executed"

    if isinstance(outcome.error, ToolCallDenied):
        return "denied"

    return "failed"


async def _resolve_tool_ids(
        tool_keys: list[str],
) -> dict[str, Any] | None:
    """
    `tool_key -> tool_id`。处理器链按 tool_id 挂，执行前要它。

    ⚠️ **None 与 {} 是两件事**：

        {}      查过了，这些工具还没同步进库 -> 它们没有任何链
                （链与 tool_id 同在库里），直接执行是对的
        None    查不了（数据库断了、应用没装配）-> 不知道有没有链
                -> 一律不执行（绕过链比失败更糟）

    这两种情形必须分开：把"查不了"当成"没有链"，就会在一个挂着
    审批工具的环境里、数据库一断就**静默放行**。

    ⚠️ 扇出后这里是**逐分支**的一次单 key 查询（原来是整轮一次批查）。
    分支读不到父状态，无法在父侧预解析，这是扇出的必然代价。真成了
    瓶颈再上请求级缓存，而不是把查询挪到路由里（那会让一个自称纯函数
    的模块碰数据库）。
    """

    if not tool_keys:
        return {}

    try:
        factory = get_service_factory()

    except RuntimeError as exc:

        logger.warning("工具目录不可用（%s），本次不执行工具。", exc)

        return None

    try:
        async with factory.scope(
                ToolQueryService, begin=False
        ) as service:

            return await service.tool_ids_for_keys(tool_keys)

    except Exception:  # noqa: BLE001 - 见上面的 None/{}

        logger.exception("读取工具 id 失败，本次不执行工具。")

        return None


# ============================================================
# 执行守卫
# ============================================================
#
# 守卫已经搬到处理器链里的 AROUND 拦截器
# （app/agent/tools/processor/interceptor/execution_guard.py）。
# 本节点不再自己记账：它只负责把工具交给执行器、把结果转成
# ToolMessage。账本的预占 / 计数 / 落终态都在守卫里。
#
# 为什么搬走：守卫要按 scope（会话 / 运行 / 用户 / 全局）与
# execution（方法 / 方法 + 参数）两个维度计数，而这些配置挂在
# tool_processor 链上 —— 放在节点里就只能写死一种语义。
#
# ⚠️ 代价：没有挂守卫的工具没有执行记录（链不存在时 AROUND 不跑）。
# 上线时给需要保护的工具挂上 execution_guard_interceptor。


def _ledger_result(value: Any) -> Any:
    """
    兼容入口：落库规范化已移到 application/_execution.execution_result。

    保留这个名字是因为既有测试直接从本模块 import 它；实现只有一份。
    """

    return execution_result(value)


async def tools_node(
        state: ToolBranchPayload,
        runtime: Runtime[ReactContext],
        config: RunnableConfig,
) -> dict[str, Any] | Command:
    """
    执行**一个**工具调用，把结果作为 ToolMessage 放回 messages。

    ⚠️ 恒只有一条消息、且每个 tool_call 都必须有
    -----------------------------------------
    少一条 ToolMessage 会让下一轮 agent 看到悬空的 tool_call，部分
    provider 直接拒绝这次请求。写回靠父状态 `messages` 上的
    `add_messages` reducer 汇总（扇出的 N 个分支因此不会互相覆盖，
    且顺序与 Send 顺序一致）。

    ⚠️ 返回值有两种：dict 与 Command
    -----------------------------
    非控制器分支返回 `{"messages": [...], ...}`；控制器分支返回
    `Command(goto=..., update=...)` —— 下一跳由工具返回的 `goto`
    （计划工具）或 payload 的 `return_to`（普通工具，默认 agent）决定，
    没有 after 路由。

    计划控制工具通过注入的 `ToolRuntime.state` 读到调用前的
    `plan_status` / `plan`，在自己的 `Command.update` 里直接写
    `plan_status` / `plan_origin` / `plan`。本节点只做两件事：

        `_command_update` 过滤掉 plan_transition / message / reason
        （它们由 ToolMessage 承载，不是 state 通道）
        控制器分支把 Command 原样交给 LangGraph

    那几个状态通道带 `_last_value` reducer，并发写不会炸。

    ⚠️ 第三个参数 `config` 是 **langgraph 注入的**，不是可选的
    ------------------------------------------------------
    `Runtime` 自己**不含** config（langgraph 自己的文档写明了这一点），
    而 `ToolRuntime` 要 config 才拿得到 `configurable.thread_id` ——
    审批的批准记录按 thread/run 归档，缺了它整条审批就退化成
    fail-closed。节点函数多声明一个 `config: RunnableConfig` 参数是
    langgraph 官方推荐的取法。

    其余见模块 docstring 的"分两层"。
    """

    result = await _wrap_tool_call(
        _tool_request(state, runtime, config),
        _no_direct_execute,
    )

    update: dict[str, Any] = {"messages": [result.message]}

    # 工具在 Command.update 里直接写状态。这里**不按工具类型过滤**：
    # 任何工具返回的 Command.update 都按原样并入 state（与原版
    # LangGraph 一致）。未知键由 LangGraph 按 state schema 处理。
    if result.command is not None:
        update.update(command_update(result.command))

    # 非控制器分支只写消息 / 状态，不负责跳转 —— 一个 super-step
    # 只能有一个 Command 决定下一跳，否则多个工具同时要去的节点都会
    # 被调度。控制器是**本批第一个调用**，与工具类型无关。
    # 缺这个键（本次改动之前 checkpoint 下来的 pending task）按控制器
    # 处理：旧 payload 里没有控制工具，所有分支都会去 agent，同一个
    # 目标重复出现没有副作用；反过来（当非控制器）会让图没有下一跳。
    if not state.get("is_controller", True):
        return update

    if result.command is not None:

        # 原样透传工具的 goto：字符串 / 节点列表 / Send 列表都合法
        # （与原版 ToolNode 一致）。只有"没有目标"（None / 空 tuple）
        # 才回默认的 agent —— 本图的 tools 没有静态出边，必须给一个
        # 下一跳。
        goto = result.command.goto or AGENT_NODE

        return Command(goto=goto, update=update)

    # 普通工具（或控制工具执行失败）：回到发起这次调用的节点。
    # 主 agent 是 "agent"；planner 的工具循环传 "planner"。
    return Command(
        goto=str(state.get("return_to") or AGENT_NODE),
        update=update,
    )


async def execute_tool_branch(
        state: Mapping[str, Any],
        runtime: Runtime[ReactContext],
        config: RunnableConfig,
        *,
        spec: NodeSpec,
        scope_namespace: UUID,
) -> _ToolCallResult:
    """
    执行**一个**工具分支，供不同图共用。

    `tools_node` 与子图的 `tool_node` 都调它 —— 执行 / 审批 / 记账仍然
    只走 `ToolExecutor` + processor chain，这里不复制任何一份。

    Parameters
    ----------
    spec / scope_namespace
        观测身份。子图传自己的 `tool_node` spec 与分支 scope 命名空间，
        于是 worker 的工具行不会与主图的 `tools` 扇出撞在一起。

    ⚠️ state 必须是**分支 payload**（含 `call` / `tool_keys`），不是完整
    图 state —— `_tool_request` 按 ToolNode 规范把它原样交给
    `ToolRuntime.state`。
    """

    return await _wrap_tool_call(
        _tool_request(state, runtime, config),
        _no_direct_execute,
        span_kwargs=_branch_span_kwargs(
            state, spec=spec, scope_namespace=scope_namespace,
        ),
    )


def command_update(command: Command) -> dict[str, Any]:
    """
    `Command.update` -> state 更新。

    **原样返回**，不按工具类型过滤 —— 与原版 LangGraph 的
    `ToolNode` 一致：工具在 `Command.update` 里写什么，就按 state
    schema 合并什么。计划工具额外带的 `plan_transition` / `message` /
    `reason` 也在其中；`ReactState` 没有声明它们时由 LangGraph 丢弃。
    """

    if not isinstance(command.update, Mapping):
        return {}

    return {str(key): value for key, value in command.update.items()}


#: 兼容旧名字：既有测试与调用方按 `_command_update` 引用它。实现只有
#: 一份（见 `command_update`）。
_command_update = command_update


async def _wrap_tool_call(
        request: ToolCallRequest,
        execute: DirectToolCall,
        *,
        span_kwargs: dict[str, Any] | None = None,
) -> _ToolCallResult:
    """
    一次工具调用的**全部**逻辑 —— 校验允许集、驱动处理器链、记观测。

    入参与 langgraph `ToolNode` 的 `awrap_tool_call` 同形，但返回
    `_ToolCallResult`（ToolMessage + 可选的 Command）—— 计划工具需要
    把 `Command.goto` 带到节点外，单返回 ToolMessage 表达不了。

    观测的 `node_span` 也在这里：它是逐分支的，而本 hook 一次只服务
    一个调用，位置正好。

    ⚠️ `execute` **从不被调用**
    -------------------------
    它是"执行工具本体"的那个回调，而本仓库的工具本体**不是可执行
    对象**：`spec.descriptor` 的 `_run` / `_arun` 是故意抛
    `NotImplementedError` 的（见 registry/decorator.py 的说明）—— 直接
    执行它就绕过了处理器链，而审批、脱敏、缓存、鉴权全挂在链上。
    真正的执行走 `ToolExecutor.execute(...)`。

    参数保留是为了让"换成预构建 ToolNode"时形状不用大改；届时把
    下面那句 `get_tool_executor().execute(...)` 换成调它即可。
    """

    state = request.state if isinstance(request.state, dict) else {}
    call = request.tool_call

    requested = _requested_name(state)
    tool_call_id = call.get("id")
    arguments = call.get("args")

    async with node_span(
            request.runtime,
            **(
                span_kwargs
                if span_kwargs is not None
                else _branch_span_kwargs(state)
            ),
            input_summary={"requested": requested},
    ) as span:

        # 允许集：与 capability 装配同一个纯函数、同一个输入。payload
        # 里带的是 tool_keys **原文**（点分形式），判定落在这一处。
        #
        # ⚠️ 不图省事从 `runtime.tools` 的名字读回来
        # ----------------------------------------
        # 那些名字是 `for_wire` 改过的**下划线形式**，而 `tool_key_of` 要的
        # 是点分的 tool_key —— 它做的正是 `wire_name(key) == wire` 的匹配。
        # 从改名副本里读回来，会把 wire 名当成 tool_key 返回，于是下游的
        # `select_specs` / 链查找全部落空：症状是"工具明明暴露了却查不到"，
        # 而它**不会**报错，只会静默降级成"目录不可用"。
        #
        # `select_specs` 是纯内存的纯函数（不查库），多算一遍不值得为它
        # 换一个容易漂移的来源。
        allowed = [
            spec.tool_key
            for spec in select_specs(state.get("tool_keys") or [])
        ]

        # 模型回的是 provider 安全名（下划线），这里翻成 tool_key。
        # 查找**限定在 `allowed` 里**做 —— 那个列表同时是"这次暴露了什么"
        # 与"允许执行什么"，一个判断就够。
        #
        # ⚠️ 翻不出来一律**拒绝**，不要回退成"原文碰巧也在允许集里就放行"：
        # 那样模型能成功调用的名字就变成 {安全名} ∪ {tool_key}，而
        # tool_key 从没发给它 —— 表面上还是过了允许集，实际上放宽了。
        tool_key = tool_key_of(requested, allowed)
        event_name = tool_key or requested

        # 前端需要看到工具开始/结束。见 _publish_tool_event 的说明：
        # 写自定义流，由 RunEventProjector 统一投影。
        _publish_tool_event(
            RunEventType.TOOL_CALLED,
            tool_call_id=tool_call_id,
            tool_name=event_name,
            input=arguments,
        )

        if tool_key is None:

            logger.warning(
                "模型要求调用 %r，但它不在本次暴露的工具集合里"
                "（tool_call_id=%s），已拒绝执行。",
                requested,
                tool_call_id,
            )

            span.set_metrics(_metrics("unexposed", allowed=False))
            span.set_output_summary({
                "requested": requested,
                "tool_call_id": str(tool_call_id or ""),
                "status": "error",
            })

            _publish_tool_event(
                RunEventType.TOOL_COMPLETED,
                tool_call_id=tool_call_id,
                tool_name=event_name,
                error=f"工具 {requested!r} 不在本次暴露的工具集合中",
            )

            return _ToolCallResult(
                message=_unexposed_message(tool_call_id, requested)
            )

        command: Command | None = None

        if tool_key in _CHAINLESS_TOOL_KEYS:
            # 控制 / HITL 工具是进程内的，没有库里的处理器链，也不该被
            # 审批 / 缓存 / 重试包住。`tool_id=None` 在 ToolExecutor 里
            # 表示"空链、直接执行工具本体"。
            tool_ids: dict[str, Any] | None = {}
        else:
            tool_ids = await _resolve_tool_ids([tool_key])

        if tool_ids is None:
            # 查不到链就别执行（见 _resolve_tool_ids 的说明）。
            outcome = ToolOutcome(
                error=LookupError(
                    "工具目录当前不可用，无法确认这个工具的处理器链，"
                    "因此未执行。"
                )
            )
            bucket = "failed"
            unavailable = True
            message = _outcome_message(
                tool_call_id, tool_key, outcome
            )
        else:

            outcome = await _execute_tool(
                request,
                tool_key,
                tool_ids,
                arguments,
                tool_call_id,
            )
            bucket = _bucket(outcome)
            unavailable = False
            message = _outcome_message(
                tool_call_id, tool_key, outcome
            )

            if isinstance(outcome.value, Command):
                command = outcome.value

        _publish_tool_event(
            RunEventType.TOOL_COMPLETED,
            tool_call_id=tool_call_id,
            tool_name=event_name,
            **_tool_event_result(outcome),
        )

        span.set_metrics(
            _metrics(
                bucket,
                allowed=True,
                unavailable=unavailable,
            )
        )
        span.set_output_summary({
            "tool_key": tool_key,
            "tool_call_id": str(tool_call_id or ""),
            "status": message.status,
            # 工具自带的跳转目标（没有就是空串）—— 观测里能直接看出
            # "这次调用把图带去了哪"。
            "handoff": str(command.goto) if command is not None else "",
        })

        return _ToolCallResult(message=message, command=command)


def _tool_caller(context: Any) -> ToolCaller:
    """
    从 ReactContext 取"这次调用是谁发起的"。

    三个字段各有来源，都**没有**兜底猜测：

        user_id      `caller_identity["user_id"]`。那份快照的形状由派发方
                     写定（`{"user_id","groups","roles"}`，见
                     tool/processor/context.py 的 build_caller_context）。
                     形状不认识（空 dict、None、缺键）就是 None ——
                     JSONB 列里历史上确实有空对象，猜一个值比没有更糟
        session_id   `ReactContext.session_id`（派发方从 run 交进来）
        agent_id     `ReactContext.duty_key`（同上）。记忆那两层的 scope 是
                     (agent_id, user|org)，工具要按与**注入时同一个** agent_id
                     定位 —— 见 ToolCaller 里那段说明

    ⚠️ 拿不到 user_id 时**不要**回落到"用会话反查用户"
    -------------------------------------------------
    多一次回表、多一条可能忘的路径，而语义上"这次调用没有身份"本来就
    该 fail-closed。要身份的服务自己判断（见 memory 的包装方法）。
    """

    identity = getattr(context, "caller_identity", None)

    user_id: str | None = None

    if isinstance(identity, Mapping):
        raw = identity.get("user_id")
        user_id = str(raw) if raw else None

    # 职责：与 session_id 同一条取法（运行期常量，派发方交进来）。
    # 空串归一成 None —— "没有职责"与"职责叫空串"是同一件事，而需要它的
    # 服务方法按 None 判断"定位不了 scope"。
    duty_key = str(getattr(context, "duty_key", None) or "").strip() or None

    return ToolCaller(
        user_id=user_id,
        session_id=getattr(context, "session_id", None),
        agent_id=duty_key,
    )


async def _execute_tool(
        request: ToolCallRequest,
        tool_key: str,
        tool_ids: dict[str, Any],
        arguments: Any,
        tool_call_id: Any,
) -> ToolOutcome:
    """
    驱动执行器跑一次工具。

    账本（守卫）不再在这里：它是链上的 AROUND 处理器，由
    ToolExecutor 在 compile_chain 时包住工具调用。中断 / 取消 /
    拒绝的记账也在守卫里 —— 本节点只管把结果翻成 ToolMessage。

    ⚠️ 调用者身份仍在执行之前放进 ContextVar：工具的入参 schema 是
    给模型看的，`user_id` 绝不能走那里（模型能填一个别人的 id）。
    """

    caller = _tool_caller(request.runtime.context)

    with tool_caller_scope(caller):
        return await get_tool_executor().execute(
            tool_key,
            arguments if isinstance(arguments, dict) else {},
            tool_id=tool_ids.get(tool_key),
            tool_call_id=(
                str(tool_call_id) if tool_call_id else None
            ),
            caller_identity=getattr(
                request.runtime.context, "caller_identity", None
            ),
            # 真的 `ToolRuntime` —— 审批拦截器靠它拿
            # `tool_call_id` / `config.configurable.thread_id` /
            # `store`（见 processor/interceptor/approval.py 的
            # `_identifiers` 与 `_ledger`）。传 None 会让整条审批
            # 退化成 fail-closed 的"通道不可用"分支。
            runtime=request.runtime,
        )


def _tool_request(
        state: ToolBranchPayload,
        runtime: Runtime[ReactContext],
        config: RunnableConfig,
) -> ToolCallRequest:
    """
    分支 payload + 节点 Runtime -> ToolNode 规范里的 `ToolCallRequest`。

    ⚠️ 为什么要现造一个 `ToolRuntime`
    -------------------------------
    节点函数拿到的是 langgraph 的 `Runtime[ReactContext]`，而处理器契约
    要的是 **`ToolRuntime`** —— 两者不是同一个东西：`Runtime` 按 langgraph
    自己的文档**不含 config**，也没有 `tool_call_id`（那是每次调用一个的，
    节点级的东西表达不了）。而 `ToolRuntime` 的必填字段各有来源：

        state           分支 payload —— 与"本节点只从 payload 读输入"
                        同一条立场（见模块 docstring）
        tool_call_id    模型给的调用 id，在 payload 的 call 里
        config          节点函数的第三个参数（langgraph 注入），
                        `configurable.thread_id` 从这来
        context         ReactContext（caller_identity / pipeline_run_id）
        store           审批账本的落点；不传就只剩进程内兜底账本
        stream_writer   节点那个
        tools           允许集的**改名副本**，见下
        execution_info  节点观测标识（node_span 要用）

    ⚠️ `tools` 必须用 `for_wire` 改名
    -------------------------------
    `descriptor.name` 本来是点分形式的 `tool_key`（见 decorator.py 的
    `__set_name__`），而模型看到、回传的是下划线形式。不改名的话
    `request.tool` 对**每一个**调用都是 None —— 一个"工具全都没暴露"的
    假象。改名复用 `infra/agent/tool_exposure.py` 的 `for_wire`，与出站
    给模型看的是同一份规则（出站入站各写一份，早晚会漂移）。

    `request.tool` 的填法与 `ToolNode` 一致（按名字查表）：将来把本 hook
    交给预构建的 `ToolNode` 时，它做的正是同一件事。
    """

    specs = select_specs(state.get("tool_keys") or [])

    tools = [for_wire(spec) for spec in specs]

    requested = _requested_name(state)

    return ToolCallRequest(
        tool_call=state["call"],
        tool=next(
            (t for t in tools if t.name == requested), None
        ),
        state=state,
        runtime=ToolRuntime(
            state=state,
            tool_call_id=state["call"].get("id"),
            config=config,
            context=runtime.context,
            store=runtime.store,
            stream_writer=runtime.stream_writer,
            tools=tools,
            execution_info=runtime.execution_info,
            server_info=getattr(runtime, "server_info", None),
        ),
    )


async def _no_direct_execute(
        request: ToolCallRequest,
) -> ToolMessage:
    """
    占位的 `execute`（见 `_wrap_tool_call` 的说明）。

    本仓库的工具本体不是可执行对象，所以 `_wrap_tool_call` 从不调它。
    真被调到，说明有人改了分工却忘了改这里 —— 那时要**响亮**，不能
    悄悄返回一个空 ToolMessage：那会让模型收到一条"成功但没内容"的
    工具结果，而工具其实一次都没跑。
    """

    raise NotImplementedError(
        "_wrap_tool_call 不该直接执行工具本体：执行必须走 "
        "ToolExecutor.execute(...)，否则会绕过处理器链"
        "（审批 / 脱敏 / 缓存 / 鉴权都挂在链上）。"
    )


def _requested_name(state: ToolBranchPayload) -> str:
    """
    payload 里那个调用**请求的名字**，原样。

    ⚠️ 它是 **provider 安全的函数名**，**不是** `tool_key`
    ---------------------------------------------------
    `tool_key` 带点号（`capability.list_capabilities`），而端点只接受
    `^[a-zA-Z0-9_-]+$`，所以出站时转成了下划线形式
    （见 infra/agent/tool_exposure.py 的 `for_wire`）。模型看到的、
    回传的都是转过的名字，翻译在 `_wrap_tool_call` 里用 `tool_key_of` 做。

    本函数只负责取原文：它进 `input_summary`（"模型要什么"）与拒绝时的
    ToolMessage 正文（"被拒的是哪个名字"）—— 两处都该显示模型实际写的
    那个名字，而不是翻译后的内部键。

    空名不是崩溃点（下面会按"未暴露"处理）。
    """

    call = state.get("call") if isinstance(state, dict) else None

    return str((call or {}).get("name") or "")


def _metrics(
        bucket: str,
        *,
        allowed: bool,
        unavailable: bool = False,
) -> dict[str, Any]:
    """
    本分支的指标。见模块 docstring 的"指标是逐分支的"。

    键名是**稳定的**（观测面板按它取值），因此这里用固定集合而不是
    直接把桶名铺开 —— 后者会让"某个键今天没有"变成"这个指标消失了"。

    ⚠️ 执行账本相关的两个键已经移除：
        already_executed / ledger_unavailable

    它们属于守卫（processor/interceptor/execution_guard.py）的判定，
    不再由本节点产生。新守卫把
    `execution_guard.decision` / `execution_guard.status` 写进
    `ToolInvocation.context`，观测面板应改从那里取值。
    """

    return {
        "tool_call_count": 1,
        "allowed_count": 1 if allowed else 0,
        "executed": 1 if bucket == "executed" else 0,
        "denied": 1 if bucket == "denied" else 0,
        "failed": 1 if bucket == "failed" else 0,
        "unexposed": 1 if bucket == "unexposed" else 0,
        "catalog_unavailable": unavailable,
    }
