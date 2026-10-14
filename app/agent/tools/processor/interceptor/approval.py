"""
人工审批（Human In The Loop）：三个选项，以及"已批过"的记录放在哪。

三个选项
--------
审批人看到的是按钮，回答落在 `ApprovalDecision` 上：

    同意（approve）      放行**这一次**调用
    拒绝（reject）       短路，工具不执行
    始终允许（always）   本次运行内这个工具**不再问**

三者的**后果**不同，不要当成同一个"通过/不通过"的三种说法：

    同意 / 始终允许   返回 invocation —— `process()` 接着调 `next_call()`，
                      也就是链上后续的处理器与工具本身继续执行
    拒绝              返回 `ToolOutcome(error=ToolCallDenied(...))` ——
                      **短路**，工具一次都不会被调用

而"始终允许"比"同意"多做一件事：它写下一笔**工具级**的记录，
让本次运行里该工具的后继调用直接跳过审批。

⚠️ 第三个选项**由配置决定是否提供**（`ApprovalConfig.always_allowed`）
--------------------------------------------------------------------
默认只画两个按钮。转账、下单这类工具必须逐次确认 —— 用户批了
「转 1 元」之后，同一个运行里「转 100 万」不该因为一次点击就不再问。
所以这是**每个工具**的风险决定，写在那个工具的链配置里。

没提供这个按钮时，客户端就算提交了 always 也会被**降级成"仅这次"**
（见 `ApprovalInterceptor.before`）。让一次回答能升级作用域，等于让
审批界面绕过工具自己的风险策略。

挂在哪一段
----------
BEFORE：审批的意义就是"决定这次调用要不要发生"，那正是 BEFORE 唯一
能做而其它两段做不到的事（AFTER 时工具已经跑完了）。

只支持 SYNC：审批必须"等出结果"才有意义，ASYNC 起一个不等返回值的
任务等于没审批。这在挂载期就会被校验（DB 的 CHECK 也拦一层）。

它怎么"跳过"（四条路径，各自靠谁）
--------------------------------
用户要求的是"只有从没被批准过的才真正进入 interrupt"。落下来是四条路径：

    A. 中断恢复      靠 **LangGraph** 自己。同一个 task 重跑时，已经答过的
                     `interrupt()` 调用点直接返回 resume 值，不再阻塞。
                     本处理器**不需要做任何事**。
    B. 链内重试      靠 **位置**。重试是 AROUND 处理器（`base.py` 写明了
                     "重试、超时、熔断都挂在这里"），而 AROUND 的
                     `next_call()` 只重跑它**下游**的处理器。本拦截器若排在
                     重试处理器之前（审批本来就是第二顺位、仅次于鉴权），
                     重试根本不会回到这里 —— 不需要跳过逻辑。
                     ⚠️ 反过来（重试排在审批之前）重试会重跑本处理器，
                     那就是**同一次执行里第二次调用 `interrupt()`**，会被当成
                     一个全新的中断而再次阻塞。账本能挡住它（见下）。
    C. LLM 重发      靠**账本**。模型看到工具报错后重发是**新的
                     `tool_call_id`**，且是新的超步，LangGraph 不认识它 ——
                     只有"这个工具在这个运行里已经批过"那一笔（工具级键，
                     即"始终允许"）能跳过。
    D. 用户选"始终允许" 靠**账本**，键是工具级（不是某次调用）。

⚠️ 跳过时**不会调用** `interrupt()` —— 这有个必须知道的位置风险
-------------------------------------------------------------
`interrupt()` 按"在重跑的节点里第几次调用"**按顺序**匹配 resume 值
（`scratchpad.interrupt_counter()`）。因此**少调用一次会让后面的中断错位**。
我实测过这个后果：

    节点里两个中断，把第一个换成"不调用"，然后按 id 回答第一个 ——
    第二个中断拿到了**给第一个的答案**，而且**不报错**。

所以本实现有一条硬约束：

    ⚠️ 跳过的前提是"这个 task 里只有一个审批中断"。
    也就是要求**一个工具一个分支**（Send 扇出，每个分支独立 task）——
    一个节点里串多个挂审批的工具时，账本跳过会让后面的中断答非所问。

✅ 这条前提**已经满足**：react 图的 tools 节点是 Send 扇出的分支节点
（见 app/agent/graph/react/routing.py 与 tools.py），一次运行里每个
分支只处理一个工具调用、最多产生一个中断。

这条没法在处理器内部检查（它看不见本 task 里还有几个中断），因此：
跳过时**打一条 WARNING**，把"可能错位"写进日志。真出现答非所问时，
那条 warning 是唯一的线索。

✅ 已经接进图了：tools 节点现在会构造一个真的 `ToolRuntime` 交给
`ToolExecutor.execute`（见 graph/react/tools.py 的 `_tool_request`），
`tool_call_id` / `run_id` / `store` 都拿得到，因此 `interrupt()` 那条路
是**可达**的。

`runtime` 仍然可以是 None（冒烟脚本、图外直接调 `execute()` 的路径），
那时走的是下面那条 fail-closed 的"审批通道不可用"分支 —— 那不是常态，
是明确标注的降级。

⚠️ 但"能 interrupt"离"能审批"还差一端：图停下来之后要有人能 resume。
那个入口在 `AgentRunService.resume_run` / `POST /runs/{run_id}/resume`
（见 services/session/application/agent_run_service.py）。两端**必须一起
上线** —— 只有 interrupt 没有 resume，会让带审批的工具把运行永久卡在
`waiting_input`。

⚠️ 记录必须落在**持久**存储上
-----------------------------
审批处理器在恢复时会**重跑**（LangGraph 的 resume 是节点级重跑）。
因此"这个调用已经批过"这件事**不能**放在 `ToolInvocation.context` 里 ——
那是本次调用新造的便签区，重跑时是空的。

    持久存储    → 重启后仍然有效，`waiting_input` 跨重启也能接上
    内存存储    → 只在"同一进程、同一次运行"内有效；重启即丢

因此生产必须给出 `runtime.store`；拿不到时会回落到进程内内存实现并
**打 warning** —— 悄悄用一个重启即丢的账本，会让"已经批准过的又去问
一遍"这种现象在重启后才出现，而那时没有人会想到是存储的问题。

✅ 已经接上了：`register_react_graph(store=…)` 拿的是 checkpointer 数据源
在**同一个连接池**上另建的 `AsyncPostgresStore`（见
app/infra/graph/checkpointer.py 的模块 docstring）。重启后记录仍在，
resume 接得上。上面那条 warning 因此只在"图外直接调 execute()"或冒烟
脚本那种拿不到 runtime 的路径上才会出现。

⚠️ 服务端仍然**不判**超时：`wait_seconds` 只是下发给客户端显示
------------------------------------------------------------
`interrupt()` **没有超时机制** —— 它抛出去之后，图就一直停在那儿，直到有人
resume。超时只能由**运行生命周期的所有者**来做（`agent_run.status` 停在
`waiting_input` 超过 `wait_seconds` 就判失败/取消，那是一个清理任务的事，
不是处理器的事）。

⚠️ 但 `wait_seconds` 已经**随 `hitl.requested` 的载荷下发**了（见 `_payload`）
-----------------------------------------------------------------------
这两件事不矛盾，别把它们看成一件：**显示**和**判定**是分开的。

    下发它   审批界面要能显示"还能等多久"。配置里本来就写着这个数，
             不告诉客户端，客户端要么显示不出来，要么**自己编一个窗口**
             —— 而编出来的窗口与后端将来真正清理的时刻对不上，只会更糟。
    不判它   到点了要不要拒绝/取消，是运行生命周期所有者的决定。处理器
             在这里没这个权力（它连"现在是几点"都不该关心），所以客户端
             拿到这个数也只该用来**倒计时**，不该拿它自己提交一个拒绝。

`on_timeout` 与超时**无关**（名字有误导，见它的 Field 说明）：它管的是
"审批通道根本不可用"（拿不到 runtime），那时没有 interrupt 可抛，只能在
放行与拒绝之间选一个。真正的超时动作今天**还不存在**。

保留 `on_timeout` 是因为它仍有一条真实用途：**拿不到 runtime 时**（没有
ToolRuntime 就没有 interrupt 能力，比如离线测试、或将来某个非图的调用路径），
本处理器只能走"审批通道不可用"这条分支，那时 `on_timeout` 决定放行还是拒绝。
默认 DENY 是刻意的 fail-closed：一个挂在链上、却没有任何人真的审批过的关卡，
唯一安全的默认是不放行。
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

from ..base import ToolInterceptor
from ..context import (
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "ApprovalConfig",
    "ApprovalDecision",
    "ApprovalInterceptor",
    "ApprovalLedger",
    "InMemoryApprovalLedger",
    "StoreApprovalLedger",
    "TOOL_WIDE_KEY",
    "approval_key",
    "approval_namespace",
]

logger = logging.getLogger(__name__)


# ============================================================
# 记录放在哪（命名空间与键）
# ============================================================
#
# 命名空间是 `(前缀, run_id)` —— **run_id 才是"一次运行"的身份**。
#
# ⚠️ 这里曾经用的是 thread_id，那是个 bug
# ---------------------------------------
# 曾经的理由是"thread_id 在 Agent 运行时里就是一次运行的身份"
# （`new_thread_id(namespace, subject_id, run_id)` 把 run_id 编了进去）。
# 那条只对 **document** 那条路成立 —— agent 走的是
# `new_session_thread_id`（见 infra/graph/ids.py），它**按会话稳定**：
# 同一个会话里的多次运行共用一条 thread。
#
# 后果很具体：用户点一次「始终允许」，同一个会话里**后续每一次运行**
# 的这个工具都会被静默放行 —— 而需求是"在该次 run 内始终放行"。
# 记录还带着 TTL（最长 24 小时），所以这个泄漏能持续一整天，且没有任何
# 地方会提示。所以作用域必须按 run，不按 thread。
#
# ⚠️ `tool_key` **不能**放进命名空间
# ---------------------------------
# langgraph 的 BaseStore 校验命名空间标签：**不允许含点**
# （`InvalidNamespaceError: Namespace labels cannot contain periods`），
# 而本仓库的工具 key 恰恰是点分形式的（`finance.transfer`、
# `document.retrieve_documents`）—— 放进去会在第一次写记录时炸，而且
# 那是在审批通过之后、离配置现场很远的地方。
#
# 键（key）没有这个限制，所以工具与调用都编进 key：见 `approval_key`。

_NAMESPACE_PREFIX = "tool_approval"


#: "整个运行内这个工具都批过了"用的后缀。
#:
#: 它不是某个 tool_call_id，而是"这个工具在这个运行里"的标记 ——
#: 也就是用户点"始终允许"时记下的那一笔。
TOOL_WIDE_KEY = "*"


def approval_namespace(run_id: str) -> tuple[str, ...]:
    """一次运行的批准记录所在的命名空间。"""

    return (_NAMESPACE_PREFIX, run_id)


def approval_key(tool_key: str, suffix: str) -> str:
    """
    工具 + 调用 -> 记录键。

    用 `:` 而不是 `.` 分隔：`tool_key` 自己就是点分形式，再用点当分隔符
    会分不清"工具的两段"与"工具和调用的边界"。
    """

    return f"{tool_key}:{suffix}"


# ============================================================
# 回答的形状
# ============================================================


class ApprovalDecision(str, Enum):
    """
    审批人的回答 —— **就是 `interrupt()` 的 resume 值**。

    ⚠️ 三个值，不是"bool + 作用域"两个字段
    -------------------------------------
    两字段能编出四个组合，其中一个是无意义的（拒绝 + 始终），而审批
    界面本来就只画三个按钮。**可表达的状态数与真实的状态数一致**时，
    "这个组合是什么意思"就不再是一个需要回答的问题。

    ⚠️ 客户端**填不了**超出配置的作用域
    ---------------------------------
    工具没开 `always_allowed` 时客户端传 `always`，会在处理器里被
    **降级成"仅这次"**（见 `ApprovalInterceptor.before`）。让回答能升级
    作用域，等于让审批界面绕过工具自己的风险策略 —— 那个策略是配置里
    的决定，不该被一次回答推翻。

    ⚠️ 认不出来的一律当**拒绝**
    -------------------------
    审批这一步没有"格式不对所以放行"的选项，与 `ApprovalConfig.on_timeout`
    默认 DENY 同一条理由（fail-closed）。
    """

    APPROVE = "approve"
    REJECT = "reject"
    ALWAYS = "always"

    @property
    def allows(self) -> bool:
        """这次调用放行吗。**只有拒绝不放行**。"""

        return self is not ApprovalDecision.REJECT

    @property
    def run_wide(self) -> bool:
        """这个回答是"本次运行内这个工具都不再问"吗。"""

        return self is ApprovalDecision.ALWAYS

    @classmethod
    def parse(cls, raw: Any) -> "ApprovalDecision":
        """
        把 resume 值收成决定。

        `allow` / `deny` 是**别名**，只为少一次"格式不对所以拒绝" ——
        它们不改变任何语义，认出来的仍然是上表那三个值。
        """

        if isinstance(raw, cls):
            return raw

        if isinstance(raw, bool):  # 简写：true -> 同意 / false -> 拒绝
            return cls.APPROVE if raw else cls.REJECT

        if isinstance(raw, dict):

            # 旧形状是 {"decision": "allow", "scope": "always"} —— 作用域
            # 写在旁边而不是并进值里。这里把它折成同一个 always。
            choice = raw.get("decision", raw.get("choice"))

            if choice in ("allow", "approve") and raw.get("scope") == "always":
                return cls.ALWAYS

            raw = choice

        if isinstance(raw, str):

            normalized = raw.strip().lower()

            if normalized in ("approve", "allow"):
                return cls.APPROVE

            if normalized == "always":
                return cls.ALWAYS

            if normalized in ("reject", "deny"):
                return cls.REJECT

        logger.warning("审批回答无法识别（%r），按拒绝处理。", raw)

        return cls.REJECT


# ============================================================
# 记录放在哪（账本）
# ============================================================


class ApprovalLedger(Protocol):
    """已经批准过的记录。端口，实现可换。"""

    async def approved(
            self,
            *,
            namespace: tuple[str, ...],
            keys: tuple[str, ...],
    ) -> bool:
        """这些 key 里有任意一个存在就表示已批准。"""

        ...

    async def remember(
            self,
            *,
            namespace: tuple[str, ...],
            key: str,
            decision: ApprovalDecision,
    ) -> None:
        ...


class StoreApprovalLedger:
    """
    langgraph BaseStore 实现（生产用这条）。

    BaseStore 的 `aput` 支持 `ttl` —— 批准记录**应该**带上有效期，否则
    「始终允许」会变成永不过期的后门。这里用调用方给的 ttl，缺省由
    `ApprovalConfig` 决定（见那里的 `always_ttl_seconds`）。

    ⚠️ 两个单位不是同一个，换算收在这里
    ----------------------------------
        ApprovalConfig.always_ttl_seconds   对外契约，**秒**（前端表单在用）
        BaseStore.aput(ttl=…)               langgraph 的，**分钟**

    这不是「顺手除一下」。langgraph 的 docstring 写的是
    *"Time to live in minutes"*，PG 实现里是
    `ttl_minutes = float(op.ttl)` → `NOW() + ttl_minutes * 60 seconds`。
    把秒直接喂进去，有效期会**长 60 倍**：

        默认 3600（契约写「1 小时」）    →  实际 3600 分钟 = 60 小时
        上限 86400（契约写「24 小时」）  →  实际 86400 分钟 = 60 天

    而 `always_ttl_seconds` 存在的全部理由就是别让「始终允许」变成
    永不过期的后门 —— 60 倍正好是从背后把它拆掉。

    换算只在构造时做一次，写下去的就是分钟。
    """

    def __init__(
            self,
            store: Any,
            *,
            ttl_seconds: float | None = None,
    ) -> None:
        self._store = store

        #: 交给 `aput` 的值。**单位是分钟** —— 见类 docstring 的换算说明。
        self._ttl_minutes = (
            None if ttl_seconds is None else ttl_seconds / 60.0
        )

    async def approved(
            self,
            *,
            namespace: tuple[str, ...],
            keys: tuple[str, ...],
    ) -> bool:

        for key in keys:

            item = await self._store.aget(namespace, key)

            if item is not None:
                return True

        return False

    async def remember(
            self,
            *,
            namespace: tuple[str, ...],
            key: str,
            decision: ApprovalDecision,
    ) -> None:

        value = {"decision": decision.value}

        # 只有「工具级」（始终允许）那一笔由**这里**给有效期。
        #
        # ⚠️ 「不传 ttl」不等于「永不过期」
        # ------------------------------
        # store 层还有一个 default_ttl（见 datasource 的 StoreSettings），
        # 没传 ttl 的写入会拿到它 —— 所以这些行仍然会被清扫掉，区别只在
        # "多久"：单次批准用 store 的默认值（7 天，远超任何人工审批的
        # 等待），「始终允许」用 always_ttl_seconds。
        #
        # 这不违背"单次批准随运行结束就该失效"：它的键里带 run_id，运行
        # 一结束就不可达，default_ttl 决定的只是**什么时候物理删掉**。
        if not key.endswith(f":{TOOL_WIDE_KEY}"):

            await self._store.aput(namespace, key, value)

            return

        # ⚠️ 不是所有 store 都支持 TTL
        # --------------------------
        # 实测 `langgraph.store.memory.InMemoryStore` 会直接抛
        # `NotImplementedError: TTL is not supported`。那时仍要把记录写下去
        # （否则"始终允许"这个功能整个失效），但**必须响亮** ——
        # 一条不过期的工具级放行记录就是永不过期的后门，而这正是
        # `always_ttl_seconds` 存在的理由。
        try:
            await self._store.aput(
                namespace, key, value, ttl=self._ttl_minutes
            )

        except NotImplementedError:

            logger.warning(
                "这个 store 不支持 TTL（%s），工具级批准记录**不会过期** —— "
                "一次「始终允许」将长期有效。生产请换支持 TTL 的 store，"
                "或把该工具的 always_allowed 关掉。",
                type(self._store).__name__,
            )

            await self._store.aput(namespace, key, value)


class InMemoryApprovalLedger:
    """
    进程内账本。**只在拿不到 store 时兜底**，且必须打 warning。

    与持久实现的差别只有一个，但后果很具体：进程重启后 `waiting_input`
    的运行恢复时，之前批过的又要重问一遍。
    """

    def __init__(self) -> None:
        self._seen: set[tuple[tuple[str, ...], str]] = set()

    async def approved(
            self,
            *,
            namespace: tuple[str, ...],
            keys: tuple[str, ...],
    ) -> bool:

        return any((namespace, key) in self._seen for key in keys)

    async def remember(
            self,
            *,
            namespace: tuple[str, ...],
            key: str,
            decision: ApprovalDecision,
    ) -> None:

        self._seen.add((namespace, key))


# ============================================================
# 参数
# ============================================================


class ApprovalConfig(BaseModel):
    """审批参数。title / description 是对外契约（前端表单）。"""

    prompt: str = Field(
        default="是否允许执行该工具调用？",
        min_length=1,
        max_length=500,
        title="审批提示",
        description="展示给审批人的问题，应当说明这次调用要做什么。",
    )

    approver_role: str | None = Field(
        default=None,
        max_length=255,
        title="审批角色",
        description=(
            "谁有权审批。留空表示不限定角色。"
            "注意这只是**路由依据**，不是权限本身 —— "
            "权限由审批系统判定，这里写什么都不构成授权。"
        ),
    )

    always_allowed: bool = Field(
        default=False,
        title="提供「始终允许」选项",
        description=(
            "**这是每个工具的风险决定，不是全局策略** —— 所以它是"
            "配置项、按工具设。\n\n"
            "关闭（默认）：审批界面只画两个按钮（同意、拒绝），"
            "**每次调用都要重新问**。"
            "关闭时客户端即使提交 always 也会被降级成『仅这次』。\n"
            "打开：多一个「始终允许」按钮，点了之后**本次运行内**"
            "这个工具不再问（工具级记录带有效期，见 always_ttl_seconds）。\n\n"
            "只读工具打开它无妨；转账、下单这类**必须**保持关闭 —— "
            "否则用户批了『转 1 元』，同一个运行里『转 100 万』就不再问了。"
        ),
    )

    always_ttl_seconds: int = Field(
        default=3_600,
        ge=60,
        le=86_400,
        title="「始终允许」的有效期（秒）",
        description=(
            "工具级批准记录的有效期。**不设过期就是永不过期的后门** —— "
            "一次点击换来无限期放行，而没有任何地方会提醒这件事。\n"
            "只在 always_allowed 打开时有意义。"
        ),
    )

    wait_seconds: int = Field(
        default=300,
        ge=1,
        le=86_400,
        title="等待时长",
        description=(
            "最多等审批人多久（秒）。⚠️ **服务端不执行它**：`interrupt()` 没有"
            "超时机制，判超时要由运行生命周期的所有者（`waiting_input` 的"
            "清理任务）来做 —— 见模块 docstring。这个值留给那个消费者。\n\n"
            "它**已经**随 `hitl.requested` 的载荷下发给客户端（见 `_payload`），"
            "唯一用途是让审批界面显示**配置里声明的**等待窗口（倒计时）。"
            "客户端不该拿它自己提交一个拒绝：到点做什么是上面那个清理任务的"
            "决定，不是界面的。"
        ),
    )

    on_timeout: Literal["DENY", "ALLOW"] = Field(
        default="DENY",
        title="审批通道不可用时的策略",
        description=(
            "**拿不到 runtime 时**（没有 interrupt 能力）的处理。"
            "默认 DENY（fail-closed）：没有任何人真的审批过就放行，"
            "等于这道关卡不存在。"
        ),
    )


#: 拿不到 store 时的兜底账本。进程级一份。
#:
#: ⚠️ 它**不跨重启**。给出来只是为了让"没有 store 的环境"（测试、离线脚本）
#: 仍能跑通整条审批路径，而不是每次调用都退化成超时分支。生产必须给 store。
_FALLBACK_LEDGER = InMemoryApprovalLedger()

#: 警告过就不重复刷屏。
_WARNED_NO_STORE = False

#: 同理：拿不到 run_id、退回按会话归档时的警告。见 `_run_id`。
_WARNED_NO_RUN_ID = False


# ============================================================
# 处理器
# ============================================================


class ApprovalInterceptor(ToolInterceptor):
    """
    调用前要人点头，除非这次调用（或这个工具在这个运行里）已经批过。

    返回值遵循 BEFORE 的契约：

        同意 / 始终允许  返回 invocation —— `process()` 接着调
                        `next_call()`，链上**后续的处理器与工具本身**
                        继续执行
        拒绝             `ToolOutcome(error=ToolCallDenied(...))` ——
                        **短路**，工具一次都不会被调用

    重试拦截器据此判断"不该重试" —— 拒绝是对这次调用的判定，
    重试只会再被拒一次。
    """

    name = "approval_interceptor"

    display_name = "人工审批"

    description = (
        "调用前送人工审批：同意 / 拒绝，可选「始终允许」（本次运行内"
        "该工具不再问）。已批准过的直接放行。"
    )

    config_model: ClassVar[type[BaseModel]] = ApprovalConfig

    supported_hooks: ClassVar[frozenset[ProcessorHook]] = (
        frozenset({ProcessorHook.BEFORE})
    )

    #: 只有 SYNC —— 审批必须"等出结果"才有意义，ASYNC 起一个不等返回值的
    #: 任务等于没审批。这在挂载期就会被校验（DB 的 CHECK 也拦一层）。
    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.SYNC})

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation | ToolOutcome:

        parsed = self.config_model.model_validate(config)

        # ---- 没有 interrupt 能力：走"通道不可用"分支 --------------
        #
        # 不假装审批过。理由与默认 DENY 相同：一个没人真的批过的关卡，
        # 唯一的诚实行为是按配置拒绝（或明确知道自己在干什么时放行）。
        identifiers = _identifiers(runtime)

        if identifiers is None:
            return self._channel_unavailable(
                parsed, invocation, runtime
            )

        run_id, tool_call_id = identifiers

        ledger = self._ledger(runtime, parsed)
        namespace = approval_namespace(run_id)

        call_key = approval_key(invocation.tool_key, tool_call_id)
        wide_key = approval_key(invocation.tool_key, TOOL_WIDE_KEY)

        # ---- 已经批过就跳过：**不调用 interrupt()** ----------------
        if await ledger.approved(
                namespace=namespace,
                keys=_keys(parsed.always_allowed, call_key, wide_key),
        ):

            logger.warning(
                "%s：%s 的这次调用已批准过，跳过审批。"
                "⚠️ 跳过意味着本 task 少调用一次 interrupt()，"
                "若这个 task 里还有别的中断，它们的位置会前移并可能"
                "拿到错位的 resume 值（实测过：不报错）。"
                "安全的前提是「一个工具一个分支」（Send 扇出），"
                "见模块 docstring。",
                self.get_key(),
                invocation.tool_key,
            )

            return invocation

        # ---- 真的问 ------------------------------------------------

        decision = ApprovalDecision.parse(
            interrupt(self._payload(parsed, invocation, run_id))
        )

        # ---- 拒绝：短路，工具不执行 --------------------------------

        if not decision.allows:

            return ToolOutcome(
                error=ToolCallDenied(
                    f"{self.get_key()}：{invocation.tool_key} 的这次调用"
                    "未获批准。"
                )
            )

        # ---- 始终允许 / 同意：放行，工具继续执行 --------------------

        if decision.run_wide and parsed.always_allowed:

            # 工具级那一笔：本运行内这个工具不再问。
            await ledger.remember(
                namespace=namespace,
                key=wide_key,
                decision=decision,
            )

        else:

            # ⚠️ 走到这里有两种情形，都是"仅这次"：
            #
            #   1. 用户点了同意
            #   2. 用户点了始终允许，但这个工具**没开** always_allowed
            #
            # 第 2 种要**降级**成仅这次，不能顺着用户/客户端的意思升级
            # 作用域（见 ApprovalDecision 的说明）。它同时也说明界面与
            # 配置不一致，"降级"不能是静默的 —— 记一条 WARNING。
            if decision.run_wide:
                logger.warning(
                    "%s：%s 没开 always_allowed，却收到了 always，"
                    "已降级为「仅这次」—— 审批界面与链配置不一致，"
                    "去核对这个工具的挂载参数与前端按钮。",
                    self.get_key(),
                    invocation.tool_key,
                )

            # 单次那一笔让"同一次调用的重试"也能跳过（见模块 docstring
            # 的路径 B）。记录的是 APPROVE 而不是原回答：键是调用级的，
            # 值不该声称一个它并不具备的作用域。
            await ledger.remember(
                namespace=namespace,
                key=call_key,
                decision=ApprovalDecision.APPROVE,
            )

        return invocation

    # ========================================================
    # internals
    # ========================================================

    def _ledger(
            self,
            runtime: ToolRuntime | None,
            parsed: ApprovalConfig,
    ) -> Any:

        store = getattr(runtime, "store", None)

        if store is not None:
            return StoreApprovalLedger(
                store, ttl_seconds=float(parsed.always_ttl_seconds)
            )

        global _WARNED_NO_STORE

        if not _WARNED_NO_STORE:
            _WARNED_NO_STORE = True
            logger.warning(
                "%s：没有 runtime.store，批准记录退回**进程内**账本 —— "
                "它不跨重启。重启后 `waiting_input` 的运行恢复时，"
                "之前批过的会再问一遍。"
                "（持久化需要在 compile(store=...) 里给一个 BaseStore。）",
                self.get_key(),
            )

        return _FALLBACK_LEDGER

    def _payload(
            self,
            parsed: ApprovalConfig,
            invocation: ToolInvocation,
            run_id: str,
    ) -> dict[str, Any]:
        """
        交给审批人看的东西。

        ⚠️ **必须带参数**：审批的粒度要和参数的粒度对齐 —— 不给参数，
        审批人就只能盲批"这个工具"，那正是"始终允许"变成后门的第一步。

        `options` 是**按钮列表**，由**配置**决定，不由客户端猜：
        没开 always_allowed 的工具根本不提供那个按钮，客户端也就没有
        机会提交一个超范围的作用域。客户端**只应**在 options 里选，
        选了别的一律按拒绝处理（见 `ApprovalDecision.parse`），
        选了 always 而它不在 options 里则降级成仅这次。

        ⚠️ `wait_seconds` 是**显示用**的，不是判定用的
        ------------------------------------------
        它是配置里声明的等待窗口，客户端拿它画倒计时用的 —— 让审批界面
        显示的是**真实声明的**那个数，而不是自己编一个。**判定**超时是运行
        生命周期所有者的事，不在这里，也不在客户端（见模块 docstring）。
        """

        return {
            "ask": parsed.prompt,
            "tool_key": invocation.tool_key,
            "arguments": dict(invocation.arguments),
            "approver_role": parsed.approver_role,
            "options": _options(parsed),
            "run_id": run_id,
            # ⚠️ 下发等待时长：审批界面要显示"还能等多久"。
            #   没有它，客户端只能自己定一个窗口 —— 那个窗口与后端将来真正
            #   清理的时刻对不上，等于编了一个不存在的 deadline。
            "wait_seconds": parsed.wait_seconds,
        }

    def _channel_unavailable(
            self,
            parsed: ApprovalConfig,
            invocation: ToolInvocation,
            runtime: ToolRuntime | None,
    ) -> ToolInvocation | ToolOutcome:

        logger.warning(
            "%s：拿不到 interrupt 所需的上下文（runtime=%s），"
            "%s 的调用按 on_timeout=%s 处理。",
            self.get_key(),
            "None" if runtime is None else "有但缺字段",
            invocation.tool_key,
            parsed.on_timeout,
        )

        if parsed.on_timeout == "ALLOW":
            return invocation

        return ToolOutcome(
            error=ToolCallDenied(
                f"{self.get_key()} 未获批准（审批通道不可用，"
                f"按 on_timeout={parsed.on_timeout} 处理）。"
            )
        )


# ============================================================
# internals
# ============================================================


#: 总是提供的两个按钮。第三个（始终允许）由配置决定，见 `_options`。
_BASE_OPTIONS: tuple[str, ...] = (
    ApprovalDecision.APPROVE.value,
    ApprovalDecision.REJECT.value,
)


def _options(parsed: ApprovalConfig) -> list[str]:
    """这次审批画哪几个按钮 —— 也就是"这个工具有几个选项"。"""

    if parsed.always_allowed:
        return [*_BASE_OPTIONS, ApprovalDecision.ALWAYS.value]

    return list(_BASE_OPTIONS)


def _keys(
        always_allowed: bool,
        call_key: str,
        wide_key: str,
) -> tuple[str, ...]:
    """
    查账本时要看哪些键。

    开了 always_allowed 才看工具级那个键。这不是省一次查询 ——
    它是**第二个闸**：`always_allowed` 关掉之后，一个早先落下的
    工具级记录会立刻失效。否则"关掉这个开关"对那些已经点过
    「始终允许」的运行不生效，症状是"配置改了但还在放行"。
    """

    if always_allowed:
        return (wide_key, call_key)

    return (call_key,)


def _identifiers(
        runtime: ToolRuntime | None,
) -> tuple[str, str] | None:
    """
    取出 `(run_id, tool_call_id)`。取不到返回 None。

    `tool_call_id` 来自 **runtime 而不是 ToolInvocation** —— 后者没有这个
    字段（见 processor/context.py 的 ToolInvocation）。模型给的
    tool_call_id 是**跨恢复稳定**的（它在被 checkpoint 的 messages 里），
    因此它是"这一次调用"唯一可用的稳定标识。

    ⚠️ run_id 来自 `runtime.context`，**不是** `config` 里的 thread_id
    -----------------------------------------------------------------
    agent 那条路上的 thread 是**按会话稳定**的（`new_session_thread_id`），
    拿它当"一次运行"会让「始终允许」泄漏到同一会话的后续运行里 —— 见本
    模块顶部命名空间那段的说明。

    真 run_id 在 `ReactContext.pipeline_run_id` 上，它随 `runtime.context`
    一路带到这里。这里用 getattr 鸭子类型读，**不** import ReactContext：
    处理器包不该反向依赖 graph 包（那会把依赖方向拧反）。
    """

    if runtime is None:
        return None

    tool_call_id = getattr(runtime, "tool_call_id", None)

    if not tool_call_id:
        return None

    run_id = _run_id(runtime)

    if not run_id:
        return None

    return run_id, str(tool_call_id)


def _run_id(runtime: ToolRuntime) -> str | None:
    """
    这次运行的身份：`runtime.context.pipeline_run_id`。

    读不到时**退回 thread_id 并打 warning** —— 退回去等于把作用域从"本次
    运行"放宽到"整个会话"，是个安全相关的降级，不能静默（见
    `_identifiers`）。两条都没有时返回 None，审批因此走不可用分支
    （fail-closed）。
    """

    context = getattr(runtime, "context", None)

    run_id = getattr(context, "pipeline_run_id", None)

    if run_id:
        return str(run_id)

    config = getattr(runtime, "config", None) or {}
    configurable = (
        config.get("configurable", {})
        if isinstance(config, dict)
        else {}
    )
    thread_id = configurable.get("thread_id")

    if not thread_id:
        return None

    global _WARNED_NO_RUN_ID

    if not _WARNED_NO_RUN_ID:
        _WARNED_NO_RUN_ID = True
        logger.warning(
            "approval_interceptor：拿不到 runtime.context.pipeline_run_id，"
            "批准记录退回按 **thread**（会话）归档 —— 那会让一次"
            "「始终允许」在同一个会话的后续运行里也生效，而它只该在本"
            "运行内有效。成因：调用方没把 ReactContext 交给 ToolRuntime"
            "（见 graph/react/tools.py 的 _tool_request）。"
        )

    return str(thread_id)
