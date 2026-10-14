"""
工具执行节点（**扇出分支**：一次调用一个实例）。

它做什么
--------
路由（`route_after_agent`）给每一个 tool_call 发一条 `Send`，本节点因此
**每跑一次只处理一个调用**：

    读 payload 里那一个 call -> 算允许集 -> 执行或拒绝 -> 回一条 ToolMessage

执行本身**不在这里** —— 链怎么驱动、service 怎么造、参数怎么包，都在
`app/infra/agent/tool_executor.py`。本节点只管图语义：读 payload、
写 messages、记观测。

分两层：图语义 / ToolNode 规范
------------------------------
    tools_node        图语义。只做两件 `_wrap_tool_call` 做不了的事：
                      把节点的 `Runtime` 变成处理器契约要的
                      `ToolRuntime`；把返回的那条 ToolMessage 包成
                      state 更新
    _wrap_tool_call   一次调用的**全部**逻辑。签名就是 langgraph
                      `ToolNode` 的 `awrap_tool_call`：
                      `(ToolCallRequest, execute) -> ToolMessage`

为什么照 ToolNode 的规范分这一层
    将来换成 langgraph 预构建的 `ToolNode` 是**一行**
    （`ToolNode(tools=..., awrap_tool_call=_wrap_tool_call)`），
    而不是把这段再重写一遍。观测的 `node_span` 也因此落在 hook 里 ——
    那个位置在两种形态下都成立。

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

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.prebuilt.tool_node import (
    ToolCallRequest,
    ToolRuntime,
)
from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.agent.tools.processor.context import (
    ToolCallDenied,
    ToolOutcome,
)
from app.common.factories import get_service_factory
from app.common.tool_caller import ToolCaller, tool_caller_scope
from app.infra.agent.tool_executor import get_tool_executor
from app.infra.agent.tool_exposure import for_wire
from app.infra.graph import branch_scope_id, node_span
from app.services.tool.application import (
    ToolExecutionService,
    ToolQueryService,
)
from app.services.tool.domain.entities import ToolExecution
from app.services.tool.domain.schemas.enum import (
    ToolExecutionStatus,
)
from app.services.tool.registry import select_specs, tool_key_of

from .catalog import NODE_SPECS, REACT_TOOL_SCOPE_NAMESPACE
from .context import ReactContext
from .state import ToolBranchPayload

_SPEC = next(s for s in NODE_SPECS if s.node_id == "tools")

__all__ = ["tools_node"]

#: ToolNode 规范里那个 `execute` 回调的形状：`(request) -> ToolMessage`。
#:
#: 本仓库从不调用它（见 `_wrap_tool_call` 的说明），只用来标住签名 ——
#: 将来换成预构建的 `ToolNode` 时，它拿到的就是这个形状的真回调。
DirectToolCall = Callable[[ToolCallRequest], Awaitable[ToolMessage]]

logger = logging.getLogger(__name__)


def _branch_span_kwargs(
        state: ToolBranchPayload,
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
        "node_id": _SPEC.node_id,
        "label": _SPEC.label,
        "group": _SPEC.group,
        "sequence": _SPEC.sequence,
        "branch_scope_id": branch_scope_id(
            REACT_TOOL_SCOPE_NAMESPACE, call_scope
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


def _outcome_message(
        tool_call_id: str | None,
        tool_key: str,
        outcome: ToolOutcome,
) -> ToolMessage:
    """一次执行结果 -> ToolMessage。成功与失败共用这里，只有 status 不同。"""

    if outcome.ok:

        return ToolMessage(
            content=_render(outcome.value),
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
# 执行账本（tool_execution）
# ============================================================
#
# 「先查后执行」的判定**全部**落在 claim 上（一条原子语句），
# 见 app/services/tool/repository/tool_execution.py 的说明。这里
# 只做三件事：取身份、驱动 claim、按结果落终态。
#
# ⚠️ 每次本表写入都是**自己的短事务**（factory.scope(begin=True)），
# 且 claim 必须在 execute 之前提交。这不是风格问题：
#
#     写在工具那个事务里    工具回滚会把记录一起回滚，而"副作用
#                           到底发生过没有"恰恰要在那一刻还查得到
#     写在 execute 之后     崩在中间就没有任何记录，账本等于没有


class _LedgerUnavailable(Exception):
    """
    账本当前用不了（身份取不到、工厂没装配、数据库断了）。

    ⚠️ 它的语义是**不执行工具**，不是"跳过记录照常执行"。
    一次没有记录的副作用是不可追溯的，而不可追溯比失败更糟 ——
    与 `_resolve_tool_ids` 拿不到链时一律不执行同一条立场。
    """


def _ledger_identity(
        request: ToolCallRequest,
) -> tuple[UUID, UUID]:
    """
    判重键里的 `(session_id, run_id)` —— **两个都取自 graph context**。

        session_id  ReactContext.session_id（即 agent_session.session_id）
        run_id      ReactContext.pipeline_run_id

    ⚠️ 为什么两个都不从 config 取
    ----------------------------
    config 里**没有** session_id，只有 `configurable.thread_id` ——
    那是 langgraph 的内部键（`dev:react:<uuid>`），不是会话主键。
    拿它当 session_id 存进去，账本行上就是一个谁都认不出的字符串，
    排查时每次都得先反查会话表。

    config 顶层那个 `run_id` 同样不用：它是给 LangSmith trace 对齐的
    langgraph 概念。本仓"一次运行"的身份是 `context.pipeline_run_id`
    （审批归档也用它，见 interceptor/approval.py 的 `_run_id`）。两者
    今天恰好相等，但契约上只有后者是业务身份。

    ⚠️ 拿不到就抛（不执行工具），**不兜底造一个键**
    --------------------------------------------
    造出来的键会把**不同的会话**混进同一行 —— 那比拒绝执行危险得多：
    前者的症状是"A 会话的调用被 B 会话的记录挡住"，而它不会报错。

    `session_id` 由派发方从 `run.session_id` 填（agent_run_dispatch），
    它在这条路径上必然有值；为 None 只可能是图装配错了。
    """

    context = getattr(request.runtime, "context", None)

    session_id = getattr(context, "session_id", None)
    run_id = getattr(context, "pipeline_run_id", None)

    if session_id is None or run_id is None:
        raise _LedgerUnavailable(
            "拿不到 (session_id, run_id)，无法建立执行记录，"
            "因此未执行工具。两者都来自 graph context"
            "（ReactContext.session_id / pipeline_run_id）——"
            "缺任一个都说明这次调用不在正常的图装配下。"
        )

    return session_id, run_id


def _execution_scope(
        *,
        begin: bool,
) -> Any:
    """
    取一段**独立**的执行账本事务的上下文管理器。

    与 `_resolve_tool_ids` 同一套取工厂的方式（`get_service_factory`
    在应用没跑过 lifespan 时抛 RuntimeError），失败一律转成
    `_LedgerUnavailable`：调用方只认那一个信号。

    取工厂是同步的，所以这里是普通函数 —— 不要写成 `async def`：
    那样调用处得先 `await` 拿上下文管理器再 `async with`，多一层
    没有意义的跳转。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        raise _LedgerUnavailable(
            f"应用未装配，执行账本不可用（{exc}），因此未执行工具。"
        ) from exc

    return factory.scope(ToolExecutionService, begin=begin)


async def _claim_execution(
        session_id: UUID,
        run_id: UUID,
        tool_key: str,
        arguments: Mapping[str, Any],
) -> ToolExecution | None:
    """
    原子抢占。返回 None = **没抢到，不许执行**。

    抢不到只有一种含义：那一行已经存在且状态不可重跑（别人正在跑、
    已经跑成功、或不知道跑没跑）。见 repository 那边的说明。
    """

    try:
        async with _execution_scope(begin=True) as service:

            return await service.claim(
                session_id=session_id,
                run_id=run_id,
                tool_key=tool_key,
                arguments=arguments,
            )

    except _LedgerUnavailable:
        raise

    except Exception as exc:  # noqa: BLE001 - 一律转成"不执行"

        raise _LedgerUnavailable(
            f"执行账本不可用（{type(exc).__name__}: {exc}），"
            "因此未执行工具。"
        ) from exc


async def _find_execution(
        session_id: UUID,
        run_id: UUID,
        tool_key: str,
        arguments: Mapping[str, Any],
) -> ToolExecution | None:
    """
    取那一行，**只为措辞**（"它现在是什么状态"）。

    失败返回 None 而不是抛：它不参与判定，拿不到就少说一句状态，
    为此把整个拒绝过程炸掉不值得。
    """

    try:
        async with _execution_scope(begin=False) as service:

            return await service.find(
                session_id=session_id,
                run_id=run_id,
                tool_key=tool_key,
                arguments=arguments,
            )

    except Exception:  # noqa: BLE001 - 见上

        logger.exception("读取执行记录失败，拒绝文案里将不带状态。")

        return None


async def _settle_execution(
        execution_id: UUID,
        *,
        status: ToolExecutionStatus,
        result: Any = None,
        error_exc: BaseException | None = None,
) -> None:
    """
    落终态。

    ⚠️ **永不抛**。走到这里工具已经跑过了（或已经被中断/取消），
    把一次"已经发生的事"因为记账失败而变成一次失败，是拿观测去
    推翻事实。失败记 ERROR 级日志。

    ⚠️ 但它的失败是有代价的：中断与取消那两条路上，一行停在
    PENDING/RUNNING 会让 resume 或下一次相同调用被**拒绝**（策略
    是 fail-closed）。因此这里的失败必须看得见 —— 它是运维信号，
    不是噪音。
    """

    try:
        async with _execution_scope(begin=True) as service:

            await service.settle(
                execution_id,
                status=status,
                result=result,
                error_exc=error_exc,
            )

    except Exception:  # noqa: BLE001 - 见 docstring

        logger.exception(
            "执行记录落终态失败（execution_id=%s, status=%s）。"
            "⚠️ 工具本身已经跑过了，这次失败只影响账本 —— "
            "但那一行会停在旧状态，下一次相同调用会被判成"
            "「不可重跑」而拒绝。",
            execution_id,
            status.value,
        )


#: 递归规范化的深度上限。
#:
#: 防御**自引用**的容器（A 里放 B，B 里又放回 A）。`json.dumps` 遇到
#: 那种结构会自己抛 ValueError，而这里是手写递归 —— 不设上限就是
#: RecursionError，那会一路冒到节点外，把一次成功的调用变成崩图。
#:
#: 与 registry/schema.py 的 MAX_SCHEMA_DEPTH 是同一类防御。
_MAX_RESULT_DEPTH = 12

#: 递归里遇到一个"不知道该变成什么"的叶子。
#:
#: 用一个内部异常而不是返回哨兵值：容器递归的每一层都要能把"这一支
#: 认不出来"往上带，而哨兵值没法与"工具真的返回了这个值"区分。
class _NotJsonable(Exception):
    def __init__(self, type_name: str) -> None:
        super().__init__(type_name)
        self.type_name = type_name


def _jsonable(value: Any, depth: int = 0) -> Any:
    """
    递归把一个值变成 JSON 能表达的东西；认不出来就抛 `_NotJsonable`。

    ⚠️ 为什么必须递归，而不是试一次 `json.dumps`
    ------------------------------------------
    第一版是"试着 dumps 一下，失败就整份降级"，结果**本仓最常见的
    返回形状恰好是最先降级的那个**：`capability.list_capabilities`
    返回 `list[Capability]`（pydantic 实体列表），落到"容器里有异物"
    那条路，`result` 列存的是一段 repr 文本 —— 而它作为一个 JSONB 列
    的意义正是可查询。真实数据里看到的：

        result = "\\"[\\\\\\"id=UUID('8af0325a-...') code='CAPABILI..."

    递归之后同一份值存成 `[{"id": "...", "code": "..."}, ...]`，
    `result->0->>'code'` 这种查询才有意义。

    ⚠️ "能变成什么"的判据锚在 pydantic 上
    ------------------------------------
    标量那几条（datetime / date / UUID / Decimal 的形态）与
    `model_dump(mode="json")` 给出的一致 —— 同一份值不论裹不裹在
    pydantic 里，存进库的形状都一样。这不是"随手挑的几种"，是
    "pydantic 会把它变成什么，就当它是什么"。
    """

    if depth > _MAX_RESULT_DEPTH:
        raise _NotJsonable(f"嵌套超过 {_MAX_RESULT_DEPTH} 层（或自引用）")

    # JSON 原生 —— 原样存。
    #
    # ⚠️ bool 要排在 int 前面？不需要：`isinstance(True, int)` 为真，
    # 但两者都是 JSON 原生、都原样返回，所以先后无所谓。
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # 服务方法与工具最常返回的东西：实体 / DTO。
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, depth + 1)
            for key, item in value.items()
        }

    # tuple / set 也当数组 —— 与 model_dump(mode="json") 的处理一致。
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item, depth + 1) for item in value]

    # ---- 标量：与 model_dump(mode="json") 对齐的那几种 ----
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, (UUID, Decimal)):
        return str(value)

    raise _NotJsonable(type(value).__name__)


def _ledger_result(value: Any) -> Any:
    """
    工具返回值 -> `result` 列能存的形态。

    ⚠️ 与 `_render` 的分工：`_render` 产出**给模型看的正文**（一个
    字符串），本函数产出**入库的结构**（JSONB）。两者判断的"值能
    变成什么"有重叠，但目的不同 —— 不要合并。

    ⚠️ 降级**不静默**：认不出来的叶子会让**整份**值降级成
    `_render` 的文本，于是 `result` 列存的是一段字符串而不是结构，
    读的人无从分辨。所以打 WARNING 并带上类型名 —— 与
    `select_specs` 跳过不可执行工具时的"跳过要留痕"同一条立场。

    想让某个工具的结果保持结构化，就让它返回
    `"能 JSON 化"` 的形状（`model_dump(mode="json")` 之后的实体、
    或 `str(uuid)`）—— 与 cache 拦截器那条"参数请用可序列化的形态"
    是同一类要求。

    ⚠️ 不做长度截断：`_render` 的 docstring 已经说明了为什么
    （悄悄砍一刀会让读的人以为那就是全部）。大结果的组织方式是
    artifact 那条路，不是在这里截。

    ⚠️ **永不抛**
    ------------
    走到这里工具已经跑过了。为了记一笔账而把一次成功的调用变成崩节点，
    是拿观测去推翻事实 —— 与 `_settle_execution` 同一条立场。

    这不是空话：降级那条路走的是 `_render`，而 `_render` 的
    `default=str` 会调 `str()`，**一个 repr 会抛的对象能让它炸**。
    实测过，所以这里兜住。
    """

    try:
        return _jsonable(value)

    except _NotJsonable as exc:
        return _degrade(
            value, f"类型 {exc.type_name} 不知道怎么变成 JSON"
        )

    except RecursionError:
        # _MAX_RESULT_DEPTH 挡在前面，走到这里说明别的递归（例如
        # BaseModel.model_dump 自己陷进去了）。兜住，不让它出节点。
        return _degrade(value, "规范化时递归过深")


def _degrade(value: Any, reason: str) -> str:
    """
    降级成一段文本，并留下 WARNING。

    ⚠️ 这一段是"能记多少算多少"：`_render` 也可能炸（repr 会抛的
    对象），所以再兜一层。两层的存在理由是同一个 —— 记账失败绝不能
    影响工具调用本身的成败。
    """

    logger.warning(
        "工具返回值记入执行记录时降级成文本（%s），"
        "result 列里存的是这段文本，不是结构化数据。",
        reason,
    )

    try:
        return _render(value)

    except Exception:  # noqa: BLE001 - 见 docstring

        logger.warning(
            "连降级渲染也失败了，result 列里只留一个占位。",
            exc_info=True,
        )

        return f"<{type(value).__name__}：无法记入 result 列>"


def _recorded_message(
        tool_call_id: str | None,
        tool_key: str,
        existing: ToolExecution | None,
) -> ToolMessage:
    """
    这次调用在这个键上**已经有记录**，且那个状态不可重跑 ——
    因此工具没有执行。

    ⚠️ 回的是 `status="error"` 的真实结果，不是假的成功消息。
    工具确实一次都没跑，把"已执行过"说成一次成功会让模型基于一个
    不存在的结果继续编 —— 见模块 docstring 的同一段立场。

    ⚠️ 文案**带状态**：模型据此能分清"已经成功了别再问"与
    "上一次没跑成或不知道跑没跑"。运维看日志时也一样。
    """

    status = (
        existing.status.value if existing is not None else "未知"
    )

    return ToolMessage(
        content=(
            f"工具 {tool_key!r} 的这次调用（相同参数）已经有一条执行"
            f"记录，状态为 {status}，本次**未重复执行**。\n"
            "上一次的结果已经在上文里；如果确实需要重新执行，"
            "请改变参数或先处理掉那条记录。"
        ),
        tool_call_id=tool_call_id or "",
        name=tool_key,
        status="error",
    )


async def tools_node(
        state: ToolBranchPayload,
        runtime: Runtime[ReactContext],
        config: RunnableConfig,
) -> dict[str, Any]:
    """
    执行**一个**工具调用，把结果作为 ToolMessage 放回 messages。

    ⚠️ 返回值只有一个键、且恒只有一条消息 —— 这是刻意的：每个
    tool_call 都必须有一条对应的 ToolMessage，少一条会让下一轮 agent
    看到悬空的 tool_call，部分 provider 直接拒绝这次请求。写回靠父
    状态 `messages` 上的 `add_messages` reducer 汇总（扇出的 N 个分支
    因此不会互相覆盖，且顺序与 Send 顺序一致）。

    ⚠️ 第三个参数 `config` 是 **langgraph 注入的**，不是可选的
    ------------------------------------------------------
    `Runtime` 自己**不含** config（langgraph 自己的文档写明了这一点），
    而 `ToolRuntime` 要 config 才拿得到 `configurable.thread_id` ——
    审批的批准记录按 thread/run 归档，缺了它整条审批就退化成
    fail-closed。节点函数多声明一个 `config: RunnableConfig` 参数是
    langgraph 官方推荐的取法。

    其余见模块 docstring 的"分两层"。
    """

    message = await _wrap_tool_call(
        _tool_request(state, runtime, config),
        _no_direct_execute,
    )

    return {"messages": [message]}


async def _wrap_tool_call(
        request: ToolCallRequest,
        execute: DirectToolCall,
) -> ToolMessage:
    """
    一次工具调用的**全部**逻辑 —— 校验允许集、驱动处理器链、记观测。

    签名就是 langgraph `ToolNode` 的 `awrap_tool_call`（见模块 docstring
    的"分两层"）。观测的 `node_span` 也在这里：它是逐分支的，而本 hook
    一次只服务一个调用，位置正好。

    ⚠️ `execute` **从不被调用**
    -------------------------
    它是"执行工具本体"的那个回调，而本仓库的工具本体**不是可执行
    对象**：`spec.descriptor` 的 `_run` / `_arun` 是故意抛
    `NotImplementedError` 的（见 registry/decorator.py 的说明）—— 直接
    执行它就绕过了处理器链，而审批、脱敏、缓存、鉴权全挂在链上。
    真正的执行走 `ToolExecutor.execute(...)`。

    参数留着是为了**签名对齐**：将来把它交给预构建的 `ToolNode` 时，
    那里传进来的 `execute` 会是真的执行入口，届时把下面那句
    `get_tool_executor().execute(...)` 换成调它即可。
    """

    state = request.state if isinstance(request.state, dict) else {}
    call = request.tool_call

    requested = _requested_name(state)
    tool_call_id = call.get("id")
    arguments = call.get("args")

    async with node_span(
            request.runtime,
            **_branch_span_kwargs(state),
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

            return _unexposed_message(tool_call_id, requested)

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
            already_executed = False
            ledger_unavailable = False
            message = _outcome_message(
                tool_call_id, tool_key, outcome
            )
        else:

            (
                message,
                bucket,
                unavailable,
                already_executed,
                ledger_unavailable,
            ) = await _execute_with_ledger(
                request,
                tool_key,
                tool_ids,
                arguments,
                tool_call_id,
            )

        span.set_metrics(
            _metrics(
                bucket,
                allowed=True,
                unavailable=unavailable,
                already_executed=already_executed,
                ledger_unavailable=ledger_unavailable,
            )
        )
        span.set_output_summary({
            "tool_key": tool_key,
            "tool_call_id": str(tool_call_id or ""),
            "status": message.status,
        })

        return message


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


async def _execute_with_ledger(
        request: ToolCallRequest,
        tool_key: str,
        tool_ids: dict[str, Any],
        arguments: Any,
        tool_call_id: Any,
) -> tuple[ToolMessage, str, bool, bool, bool]:
    """
    走执行账本地执行一次工具。

    返回 `(ToolMessage, bucket, catalog_unavailable, already_executed,
    ledger_unavailable)`。

    顺序是**固定的四步**，每一步的位置都有理由：

        1. 取身份      拿不到就不执行（fixture 之外的正常装配下必有）
        2. claim       独立事务、**在 execute 之前提交**
        3. execute     抢到了才跑
        4. settle      按结果落终态

    ⚠️ 2 与 3 之间必须真的提交
    -------------------------
    claim 写在工具那个事务里的话，工具一回滚记录跟着回滚 ——
    而"副作用到底发生过没有"恰恰要在那一刻还查得到。这与
    `_resolve_tool_ids` 为什么单独开一段 scope 是同一件事。

    ⚠️ 为什么"已经有记录"回的是 error 而不是成功
    -----------------------------------------
    因为工具**确实一次都没跑**。把"已执行过"说成一次成功，会让模型
    基于一个不存在于本次运行的结果继续编 —— 见模块 docstring 里
    "为什么不返回一条假的 ToolMessage"。
    """

    try:
        session_id, run_id = _ledger_identity(request)

        claimed = await _claim_execution(
            session_id,
            run_id,
            tool_key,
            arguments if isinstance(arguments, dict) else {},
        )

    except _LedgerUnavailable as exc:

        # 账本用不了 -> **不执行**。一次没有记录的副作用不可追溯，
        # 而不可追溯比失败更糟（与"拿不到链就不执行"同一条立场）。
        logger.warning("执行账本不可用，%s 未执行：%s", tool_key, exc)

        return (
            _outcome_message(
                tool_call_id, tool_key, ToolOutcome(error=exc)
            ),
            "failed",
            False,
            False,
            True,
        )

    if claimed is None:

        # 抢不到 = 已经有一行、且状态不可重跑。取回来只为说清是哪种。
        existing = await _find_execution(
            session_id,
            run_id,
            tool_key,
            arguments if isinstance(arguments, dict) else {},
        )

        logger.warning(
            "工具 %s 的这次调用（相同参数）已有执行记录"
            "（status=%s），未重复执行。",
            tool_key,
            existing.status.value if existing is not None else "?",
        )

        return (
            _recorded_message(tool_call_id, tool_key, existing),
            "denied",
            False,
            True,
            False,
        )

    # ⚠️ 身份在**执行之前**放进进程内的调用者上下文
    # -------------------------------------------
    # 见 app/common/tool_caller.py：工具的入参 schema 是给**模型**看的，
    # `user_id` 绝不能走那里（模型能填一个别人的 id）。所以需要身份的服务
    # 方法从那边读，而不是从自己的参数里拿。
    #
    # 写入点只有这一处 —— 工具执行只有这一条链。
    caller = _tool_caller(request.runtime.context)

    try:
        with tool_caller_scope(caller):
            outcome = await get_tool_executor().execute(
                tool_key,
                arguments if isinstance(arguments, dict) else {},
                tool_id=tool_ids.get(tool_key),
                tool_call_id=(
                    str(tool_call_id) if tool_call_id else None
                ),
                # 调用者身份：从**每次运行**的 context 取（它由派发方
                # 交进来），不是从 state —— 分支读不到父状态，而身份在
                # 一次运行里也不变，放这里比放 state 干净（见
                # context.py 的说明）。它最终进 invocation.context，
                # 供工具级的鉴权处理器判权。
                #
                # ⚠️ 与上面那个 caller **不是同一件事**：这份快照是给
                # 工具级鉴权用的（它要 groups / roles 判权），而
                # `ToolCaller` 只带 user_id 与会话，给"要搜谁的数据"
                # 那种需要。两者同源（都出自 ReactContext），但消费者
                # 不同 —— 别把前者删了以为后者能替。
                caller_identity=getattr(
                    request.runtime.context, "caller_identity", None
                ),
                # ⚠️ 真的 `ToolRuntime` —— 审批拦截器靠它拿
                # `tool_call_id` / `config.configurable.thread_id` /
                # `store`（见 processor/interceptor/approval.py 的
                # `_identifiers` 与 `_ledger`）。传 None 会让整条审批
                # 退化成 fail-closed 的"通道不可用"分支。
                runtime=request.runtime,
            )

    except GraphInterrupt:

        # 审批中断。**工具没有执行**（审批是 BEFORE），所以这一行
        # 要记成 INTERRUPT —— 而它必须是可重跑的，否则 resume 时
        # claim 会撞上这一行、把审批通过的工具永久挡住。
        #
        # ⚠️ 原样抛回：langgraph 靠这个异常暂停图，吞掉它等于把审批
        # 变成静默的放行。实测抓了再抛不会破坏中断机制。
        await _settle_execution(
            claimed.execution_id,
            status=ToolExecutionStatus.INTERRUPT,
        )

        raise

    except asyncio.CancelledError as exc:

        # 取消。与 tool_executor 的处理一致：取消不是"工具失败了",
        # 它继续往上抛（见那里的模块 docstring）。这里只留痕。
        # 工具未跑到（取消发生在链上任意位置，可能已经跑过 —— 说不准
        # 就是 RUNNING 的语义，但取消的常规路径是"还没执行"）。
        await _settle_execution(
            claimed.execution_id,
            status=ToolExecutionStatus.CANCELLED,
            error_exc=exc,
        )

        raise

    # ---- 分类落终态 --------------------------------------------------
    #
    # ⚠️ 没有 FAILED 这一支：它背着"确认无副作用"这个承诺，而我们
    # 判不出它是否成立（工具可能已经产生外部副作用 —— 下单走外部
    # 接口那类不在 factory.scope 的事务里，回滚不了）。因此未分类的
    # 失败一律落 RUNNING（不确定），而**不是** FAILED —— 后者会错误地
    # 许可重试一个有副作用的操作。见 enum 的说明。
    if outcome.ok:

        await _settle_execution(
            claimed.execution_id,
            status=ToolExecutionStatus.SUCCESS,
            result=_ledger_result(outcome.value),
        )

    elif isinstance(outcome.error, ToolCallDenied):

        await _settle_execution(
            claimed.execution_id,
            status=ToolExecutionStatus.REJECTED,
            error_exc=outcome.error,
        )

    else:

        await _settle_execution(
            claimed.execution_id,
            status=ToolExecutionStatus.RUNNING,
            error_exc=outcome.error,
        )

    return (
        _outcome_message(tool_call_id, tool_key, outcome),
        _bucket(outcome),
        False,
        False,
        False,
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
        already_executed: bool = False,
        ledger_unavailable: bool = False,
) -> dict[str, Any]:
    """
    本分支的指标。见模块 docstring 的"指标是逐分支的"。

    键名是**稳定的**（观测面板按它取值），因此这里用固定集合而不是
    直接把桶名铺开 —— 后者会让"某个键今天没有"变成"这个指标消失了"。

    两个新增的键（执行账本）：

        already_executed    这次调用撞上了同键的已有记录、没重复执行。
                            **它上升说明模型在同一轮里反复要求同一件事**，
                            或者用户在重复提交同一个操作 —— 是个业务信号，
                            不是噪音。
        ledger_unavailable  这个分支遇到了"账本不可用"，因此没执行工具。
                            与 catalog_unavailable 同性质：求和得到的是
                            "有多少分支遇到"，**不是**"账本是不是不可用"。
    """

    return {
        "tool_call_count": 1,
        "allowed_count": 1 if allowed else 0,
        "executed": 1 if bucket == "executed" else 0,
        "denied": 1 if bucket == "denied" else 0,
        "failed": 1 if bucket == "failed" else 0,
        "unexposed": 1 if bucket == "unexposed" else 0,
        "catalog_unavailable": unavailable,
        "already_executed": already_executed,
        "ledger_unavailable": ledger_unavailable,
    }
