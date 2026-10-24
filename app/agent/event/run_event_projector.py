"""
把 langgraph **协议事件**投影成 `RunEvent` 推 Redis Stream。

这是运行事件的**唯一投影层**：节点不再各自 `import publish_run_event`
（以前的埋点在 agent_node.py 四处 + agent_run_dispatch.py 两处），
而是由这一个 transformer 推出 `node.*` / `message.*` / `tool.*`。

`node.*` / `message.*` 来自 v3 协议事件；`tool.*` 来自工具节点写进
**自定义流**的生命周期（与 trace 的 `fanout.*` 同一条路）。

    协议事件（v3 流）──RunEventProjector──▶ publish_run_event ──▶ Redis Stream

订阅端（SSE）完全不动：事件的**信封形状**没变（`RunEvent` 的
type / topic / topic_id / node 平铺 + data），变的是谁在生产它们。

为什么必须是 StreamTransformer（而不是在节点里继续埋点）
--------------------------------------------------------
`node.*` 只走 transformer 通道 —— 见下面「tasks 只在 transformer 通道里」。
`tool.*` 由工具节点写自定义流、这里统一投影：节点负责"发生了什么"，
投影层负责"怎么对外说"，两边都不直接写 Redis。

⚠️ 前提：驱动必须是 v3
----------------------
`StreamTransformer` **只**在 v3 通道里存在。`StreamMux` 仅在
langgraph/pregel/main.py:3533(sync)/:3589(async) 构造，`ainvoke`/`invoke`
完全不经过它。所以 agent_run_dispatch.py 用的是
`astream_events(version="v3")` —— 换掉那一行，这个类就再也不会被调用，
症状是**事件流整个静默**（既不报错也没有事件），最难查的那种。
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from typing import Any, ClassVar, Final

from langgraph.stream import StreamTransformer

from app.agent.event import RunEventType, publish_run_event
from app.agent.event.trace_publisher import RedisTracePublisher
from app.agent.model.leaked_tool_call import LeakGate, strip_leaked_tool_calls
from app.common.event import (
    SUBAGENT_STREAM_KIND,
    TASK_STREAM_KIND,
    TOOL_STREAM_KIND,
    EventType as TraceEventType,
    SpanKind,
    SpanStatus,
    TraceError,
    TraceEvent,
)
from app.common.event.trace_emitter import tool_span_id as _tool_span_id

__all__ = [
    "RunEventProjector",
    "projector_factory",
]


# ============================================================
# 协议侧的字面量
# ============================================================
#
# 这些都是 langgraph / langchain_core 的协议值，不是我们的契约。集中在这里
# 是为了让"上游改了词表"只有一个地方要改。

#: `messages` 通道的子事件名（见 langchain_core 的 message_to_events）。
_EV_MESSAGE_START: Final[str] = "message-start"
_EV_BLOCK_START: Final[str] = "content-block-start"
_EV_BLOCK_DELTA: Final[str] = "content-block-delta"
_EV_BLOCK_FINISH: Final[str] = "content-block-finish"
_EV_MESSAGE_FINISH: Final[str] = "message-finish"

#: ⚠️ **只认 `text-delta`**，这一条就是契约要求的"message.delta 不含
#: 思维链、不含 tool-call 参数碎片"（app/common/event/event.py:93-95）。
#:
#: 词表在 langchain_core/language_models/chat_model_stream.py:706-754：
#:     text-delta          可见正文         ← 进 message.delta
#:     reasoning-delta     思维链           ← 进 message.reasoning.delta（另一条通道）
#:     block-delta         通用块增量       ← 排除
#:     legacy-block-delta  旧字段增量（含 tool_call 参数碎片）← 排除
#:
#: 排除逻辑写在**白名单**方向（只取这两种），不是黑名单 —— 上游将来
#: 加一种新 delta（比如 `audio-delta`）时，白名单的默认行为是"不发"，
#: 而黑名单是"发出去"，后者会把不该给用户看的东西漏到聊天窗里。
_DELTA_TEXT: Final[str] = "text-delta"

#: 思维链增量。⚠️ 它进的是**另一个事件**（`message.reasoning.delta`），
#: 不是 message.delta —— 后者在契约里明写不含思维链，理由见 event.py
#: 那两个新事件上面那段。
_DELTA_REASONING: Final[str] = "reasoning-delta"

#: content block 的类型。`text` 是可见正文，`reasoning` 是思维链。
_BLOCK_TEXT: Final[str] = "text"
_BLOCK_REASONING: Final[str] = "reasoning"

#: `tools` 通道的子事件名（ToolCallTransformer 的 docstring 列举了全部）。
_TC_STARTED: Final[str] = "tool-started"
_TC_FINISHED: Final[str] = "tool-finished"
_TC_ERROR: Final[str] = "tool-error"


# ============================================================
# 在途状态
# ============================================================


class _PendingMessage:
    """
    一次 LLM 调用在途期间累积的东西。

    ⚠️ `visible` 是**在 message-start 那一刻**定下来的
    ------------------------------------------------
    "这条消息的正文给不给用户看"由 `model_role` 决定，而 role 在那次
    调用的 metadata 里（跟着每一条协议事件一起来，包括第一条）。所以
    这个判断不需要缓冲、不需要事后撤回 —— 第一条 delta 到达之前就知道
    了。`role` 只是留着进事件载荷（订阅端要能分辨"空正文"与"模型没
    说话"，见 event.py 的 message.* 那一段）。

    不可见的消息**不累积正文**：它一个字都不发，拼出来也没有第二个
    读者。

    ⚠️ 两个累积器**不是冗余**，是分工：
        delta_parts         逐块拼出来的（= 订阅端自己也能拼出来的那份）
        authoritative_parts 来自 `content-block-finish` 的整块正文

    契约（event.py:97-99）要求 `message.completed` 是 delta 的**对账点**：
    "订阅者拼出来的东西与 message.completed 的全文不一致时，以全文为准"。
    如果 completed 用 delta 拼出来的结果，两边**永远相等** —— 那条对账
    规则就成了空转，"丢了几个 delta"再也检测不出来。

    所以优先用 authoritative；拿不到（协议没给 finish，或只是整条消息）
    才退回 delta 拼接，并在代码里承认那一刻的退让。

    ⚠️ 思维链是**平行的一套**，不是这上面的第二个字段
    ---------------------------------------------
    `reasoning_*` 与上面那三样逐条对应（同样两个累积器、同样"authoritative
    优先"、同样不可见就不累积），只是容器不同。放同一个对象里而不是另开
    一个 dict：另开就得把生命周期、`visible` 门、收尾清理各复制一遍，
    而三样都会与这里漂。

    ⚠️ 两条通道的闸**必须是两个实例**
    ------------------------------
    `LeakGate` 是跑在字符流上的**状态机**（扣住"可能是标记前缀"的几个
    字符）。把推理的分片喂进正文那个闸，等于让两条无关的字符流共用一份
    状态 —— 正文刚扣住一个 `<`，推理这边接着来的字符会被当成它的续集，
    于是正文里漏出一截或吞掉一段。开销只是一个对象，共用则是错。
    """

    __slots__ = (
        "node",
        "message_id",
        "role",
        "visible",
        "gate",
        "delta_parts",
        "authoritative_parts",
        "reasoning_gate",
        "reasoning_delta_parts",
        "reasoning_authoritative_parts",
        "tool_names",
    )

    def __init__(
            self,
            *,
            node: Any,
            message_id: Any,
            role: str | None,
            visible: bool,
    ) -> None:
        self.node = node
        self.message_id = message_id
        self.role = role
        self.visible = visible

        #: 出口的闸（一条消息一个实例）：模型把工具调用写成正文时，标记会
        #: **跨分片**出现，所以这里必须是有状态的（见 leaked_tool_call 的
        #: 模块 docstring）。收尾在 `_on_message_finish`（flush）。
        self.gate = LeakGate()

        #: 思维链的闸：**另一个实例**，理由见类 docstring（状态机不能共用）。
        #: 泄漏的成因是思考模式，也就是产出思维链的那条路，所以它同样要过。
        self.reasoning_gate = LeakGate()

        self.delta_parts: list[str] = []
        self.authoritative_parts: list[str] = []
        self.reasoning_delta_parts: list[str] = []
        self.reasoning_authoritative_parts: list[str] = []
        self.tool_names: list[str] = []

    def content(self) -> str:
        if self.authoritative_parts:
            return "".join(self.authoritative_parts)
        return "".join(self.delta_parts)

    def reasoning(self) -> str:
        """思维链全文（与 `content()` 同一条判据：authoritative 优先）。"""

        if self.reasoning_authoritative_parts:
            return "".join(self.reasoning_authoritative_parts)
        return "".join(self.reasoning_delta_parts)


class _PendingTask:
    """
    一次节点执行在途期间记住的东西。

    `tasks` 的两条事件**信息不对称**：start 带 `input` 但没有结果，
    result 带结果但**没有 input**。所以 `message_count` 只能在 start 时
    抓下来存着 —— 到 result 那一刻已经拿不到了。
    """

    __slots__ = ("name", "message_count", "started_at")

    def __init__(self, *, name: Any, message_count: int | None, started_at: int | None) -> None:
        self.name = name
        self.message_count = message_count
        self.started_at = started_at


# ============================================================
# Trace 小工具
# ============================================================


def _ms_to_datetime(value: Any) -> datetime | None:
    """协议事件的 timestamp 是毫秒整数；转成带时区的 UTC。"""

    if not isinstance(value, int):
        return None

    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _span_elapsed_ms(started: Any, ended: Any) -> int | None:
    if not (isinstance(started, int) and isinstance(ended, int)):
        return None
    return max(0, ended - started)


def _trace_error(value: Any) -> TraceError | None:
    """
    把任务结果里的 error 压成结构化错误。

    `data["error"]` 可能是异常对象，也可能是字符串。只留类型名与截断后的
    消息 —— 完整堆栈与业务参数不进 Trace。
    """

    if not value:
        return None

    if isinstance(value, BaseException):
        message = str(value)
        kind = type(value).__name__
        code = getattr(value, "code", None)
    else:
        message = str(value)
        kind = "Error"
        code = None

    if len(message) > 500:
        message = message[:497] + "..."

    return TraceError(
        type=kind,
        code=code if isinstance(code, str) else None,
        message=message,
    )


def _output_size(output: Any) -> int | None:
    """工具结果的大小（UTF-8 字节）。拿不到就是 None，不编 0。"""

    try:
        return len(
            json.dumps(output, ensure_ascii=False, default=str).encode("utf-8")
        )
    except Exception:  # noqa: BLE001
        try:
            return len(str(output).encode("utf-8"))
        except Exception:  # noqa: BLE001
            return None


# ============================================================
# 投影层
# ============================================================


class RunEventProjector(StreamTransformer):
    """
    协议事件 -> `RunEvent` -> Redis Stream。

    ⚠️ 三个形状约束（违反会在**注册期**炸，不是运行期）
    ---------------------------------------------------
    1. 必须是**类或 `factory(scope)` 形状**，不能直接传实例 ——
       `_normalize_stream_transformer_factories` 对实例直接 `TypeError`
       （langgraph/pregel/main.py:430-436）。用 `projector_factory()` 造。
    2. 构造必须收 `scope`（mux 按 `factory(scope)` 调，_mux.py:134-138）。
    3. `init()` 是抽象方法且必须返回 dict（_mux.py:243-247）。这里返回 `{}`：
       不贡献投影键，于是不跟内置 transformer 抢键名、也不参与 channel 转发。

    ⚠️ 全方法**绝不外抛**
    --------------------
    `StreamMux.apush` 是**内联** await `aprocess` 的（_mux.py:371-378）。
    这里抛出的异常会一路上抛到 `AsyncGraphRunStream._apump_next` 的 except
    （run_stream.py:475-478）-> `afail(e)` -> `run.output()` 里 raise ->
    agent_run_dispatch 的 `except BaseException` -> **这次运行被记成 FAILED**。

    一个观测/投影层**不该有把成功运行变失败的能力**。所以每个钩子都包
    try/except 并吞掉：投影失败就是"这条事件没发出去"，由 publisher 自己
    的留痕去暴露，而不是把整次运行拖下水。

    ⚠️ `aprocess` 里**不许出现真正的 await**（除了下面这一处）
    ---------------------------------------------------------
    `aprocess` 在 mux 里是**串行**的：它 await 完才轮到下一个 transformer，
    也才把事件放进主日志。而这个 await 点正好在**调用方驱动的泵**上
    （`GraphRunStream` 没有后台线程，"调用方的 for 循环就是泵"）——
    所以这里每多等一毫秒，都是**整个图**多停一毫秒。

    之所以现在安全：`publish_run_event` 内部按设计**不含 await**
    （只是一次 `put_nowait` 进有界队列，见 app/agent/event/publisher.py 的
    「publish() 体内不许出现 await」）。写入由那条常驻写线程去做。
    **那条纪律一破，这里就会变成图的停顿点** —— 这是两个模块之间一条
    看不见的耦合，写在这里免得将来有人拆掉它。
    """

    #: 要额外请求的 stream mode。
    #:
    #: `custom` **始终**要：工具节点不直接写 Redis，它把
    #: tool.called / tool.completed 写进图的自定义流，由这里投影 ——
    #: 与 trace 的 fanout.* 同一条路。不请求 custom，writer 就是 no-op，
    #: 前端会一条工具事件都收不到（静默，不报错）。
    #:
    #: `tools` 只在开了 trace 时加（见 __init__）：它是 trace 工具 span
    #: 的来源。**不声明就根本不会产生**该通道事件，不是"产生了但不发"：
    #: 底层 `astream(stream_mode=...)` 取的是所有已注册 transformer 的
    #: `required_stream_modes` 并集（pregel/main.py:398-414），而
    #: `ToolCallTransformer` **不是**默认内置（它会声明 "tools"）。
    #:
    #: `values` / `messages` / `tasks` 不用在这里列 —— 三个默认内置
    #: （ValuesTransformer / MessagesTransformer / LifecycleTransformer）
    #: 已经把它们的模式请求过了。
    required_stream_modes: ClassVar[tuple[str, ...]] = ("custom",)

    def __init__(
            self,
            scope: tuple[str, ...] = (),
            *,
            run_id: str,
            message_nodes: Collection[str] = (),
            message_roles: Collection[str] = (),
            node_facts: Mapping[str, Mapping[str, Any]] | None = None,
            trace_id: str | None = None,
            session_id: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        run_id
            本次运行的 id，也就是 Redis Stream 的 `topic_id`。与
            `agent_node._publish_event` 同一口径：`RunContext.pipeline_run_id`。
        message_roles
            **哪些 model_role 的消息正文算"给用户看的"**。

            ⚠️ 它和 `message_nodes` 是**两道门，缺一不可**
            -------------------------------------------
            `message_nodes` 拦的是"别的节点"；拦不住**同一个节点里的另一次
            模型调用**。实测：上下文压缩（`model_role=summarization`）就在
            agent 节点**里面**跑，它的 `langgraph_node` 也是 `"agent"` ——
            按节点过滤永远拦不住，压缩结果会以 2k+ 字的形式当助手正文流进
            聊天窗（真事，见 Redis 里的运行流）。

            所以正文的放行判据是角色，而角色来自：
                调用时挂上去的 run metadata（`ModelRuntime._dispatch`
                的 `run_metadata` -> adapters 的 `config={"metadata": …}`）
                -> langchain 的 `on_chat_model_start` -> 每条 messages 事件
                的 metadata。

            传空集合 = **一条正文都不发**（`message.delta` 完全不出现，
            `message.completed` 的 content 是空串）。与 `message_nodes`
            同一口径：要说清楚"什么算给用户看的"，这里是唯一的地方。

            ⚠️ 判据是**白名单**，方向与 `_DELTA_TEXT` 一致
            -------------------------------------------------
            缺 key（老代码、没走 Runtime 的调用、注入断掉）一律按"不可见"
            处理：默认行为是**不发**。这是刻意的 —— 反过来（缺 key 就放行）
            会让"注入在某条路径上没生效"表现成"内部消息又漏出去了"，而
            那种 bug 只有用户看得见。

            代价必须认：**注入一旦整体失效，message.* 的正文会整体静默**
            （边界事件还在，所以前端只会一直转圈）。所以注入那一环
            （runtime.py / adapters.py）与这里是一件事的两半，改一边
            必须动另一边。
        message_nodes
            **哪些节点的 messages 事件算"给用户看的助手正文"**。

            ⚠️ 这个参数是必需的，不是可选的洁癖
            ----------------------------------
            协议里的 `messages` 事件**不只来自 agent 节点**。实测一份真实
            日志：`builder` 节点产生了 19 条（`message-start` ->
            `content-block-start` -> 15× delta -> `content-block-finish` ->
            `message-finish`），正文是**结构化输出的 JSON**：

                {"behavior": "qa", "execution_plan": "react", ...}

            照单全收地把它们投影成 `message.delta`，用户的聊天窗里就会
            流出这段路由 JSON —— 而它在改造前**从来不是** message 事件
            （旧实现只有 agent 节点发 message.*）。

            所以"哪些节点算"必须由调用方显式说出来。传空集合 = 一条
            `message.*` 都不发（tool.* 与 node.* 不受影响）。
        node_facts
            节点名 -> 附加字段，合并进 `node.started` 的 data。

            协议只给得出节点名（`tasks.data.name`），给不出 `agent_type`
            / `model_role` 这类**我们自己的领域概念**。它们是图的编译期
            常量，由调用方（agent_run_dispatch）注入 —— 这样本模块不必
            import 任何 graph 侧的东西，保持与图无关。
        """

        super().__init__(scope)

        self._run_id = run_id

        # ---- Trace ----
        #
        # trace_id 来自客户端、已落 agent_run 行；None = 这次运行不采
        # trace。custom 模式仍然要请求（工具事件走它），只是额外再开
        # tools 通道来收工具 span。
        self._trace_id = trace_id or None
        self._session_id = None if session_id is None else str(session_id)
        self._trace_enabled = bool(self._trace_id)
        self._trace_publisher = RedisTracePublisher()

        #: 同一 node_name 在一整次运行里的执行序号（从 0 开始）。
        #: 投影层按 scope 实例化；Send 分支也在同一个根 scope，因此
        #: 这里的计数与"同一 Run 内"的口径一致。
        self._trace_node_index: dict[str, int] = {}

        #: tasks 里 tools 分支的映射：tool_call_id -> 分支信息。
        #: 分支信息来自 `tasks.data.input`（Send payload，业务数据，不是
        #: trace 数据），用于给 tool.* 找真实的父 span。
        self._trace_tool_tasks: dict[str, dict[str, Any]] = {}

        #: fanout 分支：branch_id -> {fanout_id, branch_span_id, branch_index}
        self._trace_fanout_branches: dict[str, dict[str, Any]] = {}

        #: fanout 父：fanout span_id -> 计数状态
        self._trace_fanouts: dict[str, dict[str, Any]] = {}

        #: 已经发过 first_output 的标记（每个投影实例一次）。
        self._trace_first_output = False

        #: LLM run_id -> message-start 时刻 / model span 关联。
        self._trace_llm: dict[str, dict[str, Any]] = {}

        #: 已经发过 first_token 的 LLM run_id。
        self._trace_first_token: set[str] = set()

        #: tool_call_id -> 开始时刻 / span 信息。
        self._trace_tools_inflight: dict[str, dict[str, Any]] = {}

        #: task_id -> 该次执行的 trace 信息（span / index / 分支）。
        self._trace_tasks: dict[str, dict[str, Any]] = {}
        self._message_nodes = frozenset(message_nodes)
        #: 正文放行的角色集合（见 __init__ 的 message_roles）。
        self._message_roles = frozenset(str(role) for role in message_roles)
        self._node_facts = {k: dict(v) for k, v in (node_facts or {}).items()}

        #: 在途消息：LLM run id -> _PendingMessage。
        #: 用 LLM run id 当键而不是节点名 —— 并发节点各占一个键，天然隔离；
        #: 同一节点重试时是新 run id，也会正确地开一条新的。
        self._messages: dict[str, _PendingMessage] = {}

        #: 在途任务：task id -> _PendingTask（见 _PendingTask 的说明）。
        self._tasks: dict[str, _PendingTask] = {}

        #: 已经发过 `message.completed` 的 LLM run id。
        #: 见 `_on_messages` 里对"整条消息"去重的说明。
        self._completed: set[str] = set()

        # ⚠️ 这是**实例属性**，不是类属性：langgraph 的
        # `_collect_stream_modes` 是对 mux 里的实例 `getattr`。
        # custom 是基类就有的（工具事件），tools 只在采 trace 时才多要。
        if self._trace_enabled:
            self.required_stream_modes = ("custom", "tools")

    def init(self) -> dict[str, Any]:
        return {}

    # ----------------------------------------------------------
    # 入口
    # ----------------------------------------------------------

    async def aprocess(self, event: Any) -> bool:
        """
        每条协议事件都过这里。**始终 `return True`**。

        ⚠️ 返回 `False` 会把事件从**主日志**里抑制掉（_mux.py:289-292），
        而那会连带影响别的消费者（比如探针、SDK 客户端）。投影层只读不拦。

        ⚠️ 只读不改：转发事件的对象在子 mux 之间是**共享**的
        （`_assign_seq=False` 的子 mux 不复制，见 _mux.py:277-283），
        就地改它会污染别的消费者。
        """

        try:
            await self._dispatch(event)
        except Exception:  # noqa: BLE001
            # 刻意吞掉，理由见类 docstring（观测层不该把运行拖成 FAILED）。
            # 不用 logger.exception：这个模块刻意不 import logging —— 见下。
            pass

        return True

    async def _dispatch(self, event: Any) -> None:
        if not isinstance(event, dict):
            return

        params = event.get("params")
        if not isinstance(params, dict):
            return

        # ⚠️ scope 过滤：transformer 按 scope **逐个子图**实例化
        # （_mux.py:193-225），不过滤就会把子图的事件也当成根图的发一遍。
        # 照抄 TasksTransformer.process 的判据（transformers.py:1032-1037）。
        if tuple(params.get("namespace") or ()) != tuple(self.scope):
            return

        method = event.get("method")
        data = params.get("data")

        timestamp = params.get("timestamp")

        if method == "tasks":
            await self._on_tasks(data, timestamp)
        elif method == "messages":
            await self._on_messages(data, timestamp)
        elif method == "tools":
            await self._on_tool(data, timestamp)
        elif method == "custom":
            await self._on_custom(data, timestamp)

    # ----------------------------------------------------------
    # Trace 出口
    # ----------------------------------------------------------

    async def _emit_trace(
            self,
            event_type: TraceEventType,
            *,
            span_id: str,
            span_kind: SpanKind,
            span_name: str,
            parent_span_id: str | None = None,
            status: SpanStatus | None = None,
            attempt: int = 1,
            duration_ms: int | None = None,
            payload: Mapping[str, Any] | None = None,
            error: TraceError | None = None,
            occurred_at: datetime | None = None,
            fanout_id: str | None = None,
            branch_id: str | None = None,
            branch_index: int | None = None,
    ) -> None:
        """把一条投影出来的事件写进 trace topic。未开启时直接返回。"""

        if not self._trace_enabled:
            return

        await self._trace_publisher.emit(
            event_type,
            trace_id=self._trace_id,
            run_id=self._run_id,
            thread_id=self._session_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_kind=span_kind,
            span_name=span_name,
            status=status,
            attempt=attempt,
            duration_ms=duration_ms,
            fanout_id=fanout_id,
            branch_id=branch_id,
            branch_index=branch_index,
            payload=payload,
            error=error,
            occurred_at=occurred_at,
        )

    async def _on_custom(self, data: Any, timestamp: Any) -> None:
        """
        节点 / 自有 runtime 通过 custom stream 发来的东西，有两种：

            1. 工具生命周期（`kind == TOOL_STREAM_KIND`）
               工具节点写的，这里投影成 `tool.called` / `tool.completed`。
               **与 trace 无关**，未采 trace 时也照常发。

            2. TraceEvent
               节点侧已经决定了 span_id / parent_span_id / attempt，
               这里只做二次校验与转发。丢弃条件：

                   - 不是完整 TraceEvent（schema_version 不对 / 校验失败）
                   - trace_id / run_id 与当前投影实例不一致（跨运行串线的保险）
        """

        if not isinstance(data, dict):
            return

        if data.get("kind") == TOOL_STREAM_KIND:
            await self._on_tool_stream_event(data)
            return

        if data.get("kind") == TASK_STREAM_KIND:
            await self._on_task_stream_event(data)
            return

        if data.get("kind") == SUBAGENT_STREAM_KIND:
            await self._on_subagent_stream_event(data)
            return

        if not self._trace_enabled:
            return

        try:
            event = TraceEvent.model_validate(data)

        except Exception:  # noqa: BLE001 —— 非法事件只丢这一条
            return

        if event.trace_id != self._trace_id or event.run_id != self._run_id:
            return

        try:
            await self._trace_publisher.publish(event)
        except Exception:  # noqa: BLE001
            return

        # fanout 记账：branch_created 之后，tools 分支的 task 才能把
        # branch_id 映射到真实的 branch_span_id / fanout_id。
        if event.event_type is TraceEventType.FANOUT_STARTED:
            self._trace_fanouts[event.span_id] = {
                "expected": event.payload.get("expected_branch_count"),
                "parent_span_id": event.parent_span_id,
                "created": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "branches": set(),
                "done": set(),
            }

        elif event.event_type is TraceEventType.FANOUT_BRANCH_CREATED:
            self._trace_fanout_branches[str(event.payload["branch_id"])] = {
                "fanout_id": event.span_id,
                "branch_span_id": event.payload["branch_span_id"],
                "branch_index": event.payload["branch_index"],
            }

            state = self._trace_fanouts.get(event.span_id)
            if state is not None:
                state["created"] += 1
                state["branches"].add(str(event.payload["branch_id"]))

    # ----------------------------------------------------------
    # custom(tool) -> tool.called / tool.completed
    # ----------------------------------------------------------
    #
    # 工具节点不直接写 Redis（见 graph/react/tools.py 的
    # `_publish_tool_event`）：它把下面这两条写进自定义流，与 trace 的
    # fanout.* 同一条路。这里只做形状转换，不改字段。
    #
    # ⚠️ `node` 只能是 `None`
    # ---------------------------
    # 工具事件没有节点名这个字段（与原来的契约一致）。写死 "tools" 会让
    # `RunEvent.node` 从事实变成猜测，订阅端可能拿它做分组。

    async def _on_tool_stream_event(self, data: Mapping[str, Any]) -> None:
        event = data.get("event")

        if event == RunEventType.TOOL_CALLED.value:
            await self._emit(
                RunEventType.TOOL_CALLED,
                node=None,
                data={
                    "tool_call_id": data.get("tool_call_id"),
                    "tool_name": data.get("tool_name"),
                    "input": data.get("input"),
                },
            )
            return

        if event == RunEventType.TOOL_COMPLETED.value:
            payload: dict[str, Any] = {
                "tool_call_id": data.get("tool_call_id"),
                "tool_name": data.get("tool_name"),
                "output": data.get("output"),
            }

            # 成功与失败共用一条事件（契约里没有 tool.error）：失败带
            # `error`，成功不带 —— 不要给成功补一个 `error: None`。
            if "error" in data:
                payload["error"] = data.get("error")

            await self._emit(
                RunEventType.TOOL_COMPLETED,
                node=None,
                data=payload,
            )

    async def _on_task_stream_event(self, data: Mapping[str, Any]) -> None:
        """custom kind=task -> plan.published / task.updated / plan.cleared。"""

        mapping = {
            RunEventType.PLAN_PUBLISHED.value: (
                RunEventType.PLAN_PUBLISHED
            ),
            RunEventType.TASK_UPDATED.value: RunEventType.TASK_UPDATED,
            RunEventType.PLAN_CLEARED.value: RunEventType.PLAN_CLEARED,
        }

        event_type = mapping.get(str(data.get("event") or ""))

        if event_type is None:
            return

        payload = {
            key: value
            for key, value in data.items()
            if key not in {"kind", "event", "node"}
        }

        await self._emit(
            event_type,
            node=(
                str(data["node"])
                if data.get("node") is not None
                else None
            ),
            data=payload,
        )

    async def _on_subagent_stream_event(
            self,
            data: Mapping[str, Any],
    ) -> None:
        """
        custom kind=subagent -> subagent.created / claimed / task.updated /
        completed。

        与任务板那组同形：只做 kind/event 的形状转换，载荷原样保留
        （`worker_id` / `task` / `previous_state` / `status` 都是前端
        需要的字段，不在投影层筛选）。
        """

        mapping = {
            RunEventType.SUBAGENT_CREATED.value: (
                RunEventType.SUBAGENT_CREATED
            ),
            RunEventType.SUBAGENT_CLAIMED.value: (
                RunEventType.SUBAGENT_CLAIMED
            ),
            RunEventType.SUBAGENT_TASK_UPDATED.value: (
                RunEventType.SUBAGENT_TASK_UPDATED
            ),
            RunEventType.SUBAGENT_COMPLETED.value: (
                RunEventType.SUBAGENT_COMPLETED
            ),
        }

        event_type = mapping.get(str(data.get("event") or ""))

        if event_type is None:
            return

        payload = {
            key: value
            for key, value in data.items()
            if key not in {"kind", "event", "node"}
        }

        await self._emit(
            event_type,
            node=(
                str(data["node"])
                if data.get("node") is not None
                else None
            ),
            data=payload,
        )

    # ----------------------------------------------------------
    # tasks -> node.started / node.completed
    # ----------------------------------------------------------
    #
    # ⚠️ `tasks` **只走 transformer 通道**，主日志里永远没有它：
    # `_TasksLifecycleBase.process`（LifecycleTransformer 与 SubgraphTransformer
    # 的基类，两者都默认注册）对 `tasks` 无条件 `return False`
    # （transformers.py:458-471，注释原文 "suppress from the main event log
    # so iterators don't double-see the same information in two shapes"）。
    # 而 `transformers=` 只能加不能减，没法把内置的摘掉。
    #
    # 也就是说 node.* **只能**由 transformer 产出 —— 这反而是这条路线
    # 独有的一块能力，不是妥协。

    async def _on_tasks(self, data: Any, timestamp: Any) -> None:
        if not isinstance(data, dict):
            return

        task_id = str(data.get("id"))
        name = data.get("name")

        if "result" not in data:
            # ---- 开跑 ----
            self._tasks[task_id] = _PendingTask(
                name=name,
                message_count=_message_count(data.get("input")),
                started_at=timestamp if isinstance(timestamp, int) else None,
            )

            payload: dict[str, Any] = {
                "node_id": name,
                # 图侧注入的领域字段（agent_type / model_role）。协议给不出。
                **self._node_facts.get(str(name), {}),
            }

            await self._emit(RunEventType.NODE_STARTED, node=name, data=payload)

            await self._trace_task_started(
                task_id=task_id,
                name=name,
                data=data,
                timestamp=timestamp,
            )
            return

        # ---- 结束 ----
        pending = self._tasks.pop(task_id, None)
        result = data.get("result")

        # ⚠️ 节点结束 = 它名下的模型调用不可能还在跑
        # ----------------------------------------
        # 兜住"最后一次调用失败、没有下一次 message-start 来触发
        # _close_superseded"的情况。放在 node.completed **之前**：订阅端
        # 先收到 message.completed（收起 loading）、再收到 node.completed，
        # 顺序与正常结束时一致。
        await self._close_node_pending(node=name)

        completed: dict[str, Any] = {
            "node_id": name,
            # start 时抓下来的，result 事件里没有 input（见 _PendingTask）。
            "message_count": pending.message_count if pending else None,
            "tool_calls": _tool_call_names(result),
            "model_provider": _result_provider(result),
            "model_name": _result_model_name(result),
            "latency_ms": _elapsed_ms(pending, timestamp),
        }

        # ⚠️ `degraded` **没有**了
        # ------------------------
        # 旧实现在这里发 `degraded: response.degraded`，那是 ModelResponse 的
        # 概念（"这次调用重试过 / 降级过"），**协议事件里没有对应物**。
        #
        # 它不是彻底消失：同一份事实还写在 span metrics 的 `model_degraded`
        # （agent_node 的 _token_metrics）里，并且 observer.py 会为它打
        # WARNING。消失的只是"事件流里的那一份"。
        #
        # 若前端确实在消费它，唯一的补法是给 agent 节点留一条最小埋点 ——
        # 那也意味着 message.completed 的 degraded 同样要留。别顺手加回
        # 一个恒为 False 的字段：那会让"这次真的没降级"与"这个字段没人填"
        # 变得分不清，比缺字段更糟。

        # 节点报错时补一个 error（只在非空时出现，是**追加**字段，
        # 不动既有形状）。
        error = data.get("error")
        if error:
            completed["error"] = str(error)

        await self._emit(RunEventType.NODE_COMPLETED, node=name, data=completed)

        await self._trace_task_finished(
            task_id=task_id,
            name=name,
            data=data,
            timestamp=timestamp,
        )

    # ----------------------------------------------------------
    # Trace: node.* / fanout 收口
    # ----------------------------------------------------------

    async def _trace_task_started(
            self,
            *,
            task_id: str,
            name: Any,
            data: Mapping[str, Any],
            timestamp: Any,
    ) -> None:
        if not self._trace_enabled:
            return

        node_name = str(name)
        facts = self._node_facts.get(node_name, {})

        # 同一 node_name 在本次运行里的执行序号（从 0 开始）。
        index = self._trace_node_index.get(node_name, 0)
        self._trace_node_index[node_name] = index + 1

        span_id = task_id
        parent_span_id = self._run_id
        fanout_id: str | None = None
        branch_id: str | None = None
        branch_index: int | None = None

        if node_name == "tools" or facts.get("group") == "fanout":
            tip = self._record_tool_branch(task_id, data)
            if tip is not None:
                # tools 节点的 span 是分支节点（task_id）；fanout 是它的
                # 父，tool 执行是它的子 span（见 _tool_refs）。
                parent_span_id = tip["fanout_id"] or self._run_id
                fanout_id = tip["fanout_id"]
                branch_id = tip["branch_id"]
                branch_index = tip["branch_index"]

        self._trace_tasks[task_id] = {
            "name": node_name,
            "node_type": facts.get("group") or facts.get("node_type"),
            "index": index,
            "started_ms": timestamp if isinstance(timestamp, int) else None,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "fanout_id": fanout_id,
            "branch_id": branch_id,
            "branch_index": branch_index,
        }

        await self._emit_trace(
            TraceEventType.NODE_STARTED,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_kind=SpanKind.NODE,
            span_name=node_name,
            status=SpanStatus.RUNNING,
            occurred_at=_ms_to_datetime(timestamp),
            fanout_id=fanout_id,
            branch_id=branch_id,
            branch_index=branch_index,
            payload={
                "node_name": node_name,
                "node_type": facts.get("group") or facts.get("node_type"),
                "node_execution_index": index,
            },
        )

    def _record_tool_branch(
            self,
            task_id: str,
            data: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """
        从 tools 分支的 Send payload 里读出 tool_call_id / call_scope。

        Send payload 是**业务数据**（ToolBranchPayload），不是 trace 数据；
        这里只是用它做关联，不往 state 里写任何 trace 字段。
        """

        raw_input = data.get("input")

        if not isinstance(raw_input, dict):
            return None

        call = raw_input.get("call")
        call_id = call.get("id") if isinstance(call, dict) else None
        call_scope = raw_input.get("call_scope")

        branch = (
            self._trace_fanout_branches.get(str(call_scope))
            if call_scope is not None
            else None
        )

        tip: dict[str, Any] = {
            "task_id": task_id,
            "call_scope": None if call_scope is None else str(call_scope),
            "branch_span_id": branch["branch_span_id"] if branch else None,
            "fanout_id": branch["fanout_id"] if branch else None,
            "branch_index": branch["branch_index"] if branch else None,
            "branch_id": None if call_scope is None else str(call_scope),
        }

        if call_id is not None:
            self._trace_tool_tasks[str(call_id)] = tip

        return tip

    async def _trace_task_finished(
            self,
            *,
            task_id: str,
            name: Any,
            data: Mapping[str, Any],
            timestamp: Any,
    ) -> None:
        if not self._trace_enabled:
            return

        info = self._trace_tasks.pop(task_id, None)

        if info is None:
            return

        duration_ms = _span_elapsed_ms(info.get("started_ms"), timestamp)
        error = data.get("error")
        interrupts = data.get("interrupts") or ()

        common = {
            "span_id": info["span_id"],
            "parent_span_id": info["parent_span_id"],
            "span_kind": SpanKind.NODE,
            "span_name": str(name),
            "duration_ms": duration_ms,
            "occurred_at": _ms_to_datetime(timestamp),
            "fanout_id": info.get("fanout_id"),
            "branch_id": info.get("branch_id"),
            "branch_index": info.get("branch_index"),
        }

        outcome = "completed"

        if error:
            outcome = "failed"
            await self._emit_trace(
                TraceEventType.NODE_FAILED,
                status=SpanStatus.FAILED,
                payload={
                    "node_name": str(name),
                    "node_type": info.get("node_type"),
                },
                error=(
                    _trace_error(error)
                    or TraceError(type="NodeError", message="")
                ),
                **common,
            )

        elif interrupts:
            outcome = "interrupted"
            for interrupt in interrupts:
                interrupt_id = (
                    interrupt.get("id")
                    if isinstance(interrupt, dict)
                    else getattr(interrupt, "id", None)
                )
                await self._emit_trace(
                    TraceEventType.NODE_INTERRUPTED,
                    status=SpanStatus.INTERRUPTED,
                    payload={
                        "node_name": str(name),
                        "interrupt_id": str(interrupt_id),
                        "interrupt_type": "hitl",
                    },
                    **common,
                )

        else:
            await self._emit_trace(
                TraceEventType.NODE_COMPLETED,
                status=SpanStatus.COMPLETED,
                payload={
                    "node_name": str(name),
                    "node_type": info.get("node_type"),
                },
                **common,
            )

        await self._trace_fanout_progress(info, outcome)

    async def _trace_fanout_progress(
            self,
            info: Mapping[str, Any],
            outcome: str,
    ) -> None:
        """一个分支结束时记账；收齐后发 fanout.completed。"""

        fanout_id = info.get("fanout_id")
        branch_id = info.get("branch_id")

        if not fanout_id or not branch_id:
            return

        state = self._trace_fanouts.get(str(fanout_id))

        if state is None or str(branch_id) in state["done"]:
            return

        if outcome == "interrupted":
            # 暂停不是结束：既不算 completed 也不算 failed，等 resume
            # 后同一个 branch_id 再交结果。不能在这里计入 done。
            return

        state["done"].add(str(branch_id))

        if outcome == "failed":
            state["failed"] += 1
        else:
            state["completed"] += 1

        expected = state.get("expected")

        if expected is None or len(state["done"]) < int(expected):
            return

        self._trace_fanouts.pop(str(fanout_id), None)

        await self._emit_trace(
            TraceEventType.FANOUT_COMPLETED,
            span_id=str(fanout_id),
            parent_span_id=state.get("parent_span_id"),
            span_kind=SpanKind.FANOUT,
            span_name="tool_fanout",
            status=SpanStatus.COMPLETED,
            payload={
                "created_branch_count": state["created"],
                "completed_branch_count": state["completed"],
                "failed_branch_count": state["failed"],
                "cancelled_branch_count": state["cancelled"],
            },
        )

    # ----------------------------------------------------------
    # messages -> message.started / message.delta / message.completed
    # ----------------------------------------------------------

    async def _on_messages(self, data: Any, timestamp: Any = None) -> None:
        # ⚠️ 是 **tuple**，不是 list
        # --------------------------
        # `MessagesTransformer` 的 docstring 写的是
        # `params["data"] = (payload, metadata)` —— 圆括号，进程内真的是
        # tuple。只判 list 会让这里**每条都提前返回**，症状是 message.*
        # 一条都不发，而 node.* 照常（它走 tasks，形状不同）—— 于是看起来
        # "部分工作正常"，最难定位。
        #
        # ⚠️ 这个坑是靠**进程内端到端**跑出来的，日志回放**测不出来**：
        # 事件进了 JSON 之后 tuple 就变成 list 了，回放永远走对分支。
        if not (isinstance(data, (list, tuple)) and len(data) == 2):
            return

        payload, meta = data[0], data[1]

        if not isinstance(meta, dict):
            return

        node = meta.get("langgraph_node")

        # trace 的 first_token / first_output 必须在 **message_nodes 门之前**
        # 处理：它们要覆盖内部调用（压缩 / 分类）产生的模型请求，而门只
        # 决定"哪些消息算给用户看的正文"。
        await self._trace_on_llm_event(
            payload=payload,
            meta=meta,
            node=node,
            timestamp=timestamp,
        )

        if str(node) not in self._message_nodes:
            # 不属于"给用户看的助手正文"的节点（比如 builder 的结构化输出）。
            # 见 __init__ 的 message_nodes 说明 —— 不过滤会把路由 JSON
            # 漏进聊天窗。
            return

        # ---- 整条消息（来自 on_chain_end，节点返回了拼好的消息）----
        if not isinstance(payload, dict):
            await self._emit_completed_from_message(meta, node, payload)
            return

        #: 这次调用是**谁**（`ModelRuntime._dispatch` 挂上去的），以及
        #: 它的正文给不给用户看。两道门见 __init__ 的 message_roles。
        #:
        #: ⚠️ 取不到 role 就是**不可见**，不是"当作主循环"。注入断了的时候
        #: 宁可少发正文（前端只是没有字），也不要让压缩结果漏进聊天窗
        #: （那是用户看得见的错）——`message.started` 仍会发，所以"这里
        #: 静默了"与"模型没说话"在事件流上分得开。
        role = meta.get("model_role")
        role = str(role) if role is not None else None
        visible = role in self._message_roles

        run_key = str(meta.get("run_id"))

        # ---- 协议生命周期 ----
        sub = payload.get("event")

        if sub == _EV_MESSAGE_START:
            # ⚠️ 同一节点、同一角色的上一条还开着，就先把它收尾
            # ------------------------------------------------
            # 重试 / 降级会开一条新的 LLM run，而**失败的那次不会发
            # message-finish**（协议的终态只由成功路径产出）。若不在这里
            # 关掉，事件流上会留下一条永远 `started` 的消息 —— 订阅端把
            # 这一条的 delta 接在它后面，就是"一步的消息跨到下一步"。
            #
            # 判据是"节点 + 角色"而不是只有节点：同一节点的模型调用目前
            # 是串行的（压缩 -> 主循环），但将来若出现同节点并发调用，
            # 不同角色各有各的在途消息，不该被彼此误关。
            await self._close_superseded(
                node=node,
                role=role,
                run_key=run_key,
            )

            self._messages[run_key] = _PendingMessage(
                node=node,
                message_id=payload.get("id"),
                role=role,
                visible=visible,
            )

            # ⚠️ 边界事件**无条件发**，正文才受 visible 管
            # ------------------------------------------
            # 不可见的消息（压缩 / 分类）也是"这会儿有模型在工作"：没有
            # message.started，前端在那几秒里什么都收不到，看起来像卡住。
            # 发它不发正文，前端就能一直转圈到 completed 为止。
            await self._emit(
                RunEventType.MESSAGE_STARTED,
                node=node,
                data={"role": role, "visible": visible},
            )
            return

        if sub == _EV_BLOCK_DELTA:
            await self._on_block_delta(run_key, payload)
            return

        if sub == _EV_BLOCK_FINISH:
            pending = self._messages.get(run_key)
            if pending is not None:
                block = payload.get("content")
                if isinstance(block, dict) and block.get("type") == _BLOCK_TEXT:
                    # 不可见的消息**不攒正文**（与 _on_block_delta 同一条）：
                    # 没有读者的正文不该占内存，也不该让"其实拼得出来"看起来
                    # 像一条被实现了一半的路径。
                    if pending.visible:
                        # ⚠️ 整块文本走**无状态**的版本，不要喂给那个流式的
                        # 闸：两者是同一段字的两种形状，先后喂进去会让状态机
                        # 把本该丢的区段当成新文本再判一次（见 leak 模块）。
                        text, _ = strip_leaked_tool_calls(
                            str(block.get("text") or "")
                        )
                        pending.authoritative_parts.append(text)
                elif isinstance(block, dict) and block.get("type") == _BLOCK_REASONING:
                    # 思维链整块：与上面那条对称（无状态过滤 + 只有可见才攒）。
                    # 这是 `message.reasoning.completed` 的**对账来源** ——
                    # 有它，那个事件的全文才不等于"delta 拼出来的东西"。
                    if pending.visible:
                        text, _ = strip_leaked_tool_calls(
                            _block_reasoning(block)
                        )
                        if text:
                            pending.reasoning_authoritative_parts.append(text)
                elif isinstance(block, dict):
                    # 工具名照收：它进的是 completed 的载荷（边界事件对内部
                    # 消息也发），与正文是不是可见无关。
                    name = _block_tool_name(block)
                    if name:
                        pending.tool_names.append(name)
            return

        if sub == _EV_MESSAGE_FINISH:
            await self._on_message_finish(run_key, payload)

    async def _trace_on_llm_event(
            self,
            *,
            payload: Any,
            meta: Mapping[str, Any],
            node: Any,
            timestamp: Any,
    ) -> None:
        """
        Trace 侧的模型级事件：

            model.first_token   首条 token（按 LLM run 分别计算 ttft）
            run.first_output    第一条**可见**正文
        """

        if not self._trace_enabled:
            return

        run_key = str(meta.get("run_id"))
        role = meta.get("model_role")
        role = str(role) if role is not None else None

        # 与 run 事件的两道门同口径：节点 + 角色。
        visible = (
            str(node) in self._message_nodes
            and role in self._message_roles
        )

        if not isinstance(payload, dict):
            # 整条消息（on_chain_end）：没有 delta，不可能是流式
            # first_token；只有可见正文才可能补 first_output。
            if visible and not self._trace_first_output:
                text = _message_text(payload)

                if text:
                    await self._emit_first_output(timestamp)
            return

        sub = payload.get("event")

        if sub == _EV_MESSAGE_START:
            self._trace_llm[run_key] = {
                "started_ms": timestamp if isinstance(timestamp, int) else None,
                # 由 ModelRuntime 写进 run metadata（见 model/runtime.py）。
                "span_id": meta.get("trace_span_id"),
                "parent_span_id": (
                    meta.get("trace_parent_span_id") or self._run_id
                ),
                "streaming": bool(meta.get("trace_streaming", True)),
            }
            return

        if sub == _EV_BLOCK_DELTA:
            state = self._trace_llm.get(run_key)
            delta = payload.get("delta")
            kind = delta.get("type") if isinstance(delta, dict) else None

            if (
                    kind in (_DELTA_TEXT, _DELTA_REASONING)
                    and state is not None
                    and run_key not in self._trace_first_token
            ):
                started_ms = state.get("started_ms")
                span_id = state.get("span_id")

                if (
                        isinstance(started_ms, int)
                        and isinstance(timestamp, int)
                        and span_id
                        and state.get("streaming", True)
                ):
                    await self._emit_trace(
                        TraceEventType.MODEL_FIRST_TOKEN,
                        span_id=str(span_id),
                        parent_span_id=state.get("parent_span_id"),
                        span_kind=SpanKind.MODEL,
                        span_name="model.invoke",
                        occurred_at=_ms_to_datetime(timestamp),
                        payload={
                            "ttft_ms": max(0, timestamp - started_ms),
                        },
                    )
                    self._trace_first_token.add(run_key)

            if (
                    kind == _DELTA_TEXT
                    and visible
                    and not self._trace_first_output
            ):
                await self._emit_first_output(timestamp)

            return

        if sub == _EV_MESSAGE_FINISH:
            self._trace_llm.pop(run_key, None)

    async def _emit_first_output(self, timestamp: Any) -> None:
        """run.first_output：整个运行只发一次（每个投影实例）。"""

        if self._trace_first_output:
            return

        self._trace_first_output = True

        await self._emit_trace(
            TraceEventType.RUN_FIRST_OUTPUT,
            span_id=self._run_id,
            span_kind=SpanKind.RUN,
            span_name="run",
            occurred_at=_ms_to_datetime(timestamp),
            payload={"output_channel": "message"},
        )

    async def _on_block_delta(self, run_key: str, payload: dict[str, Any]) -> None:
        pending = self._messages.get(run_key)

        # ⚠️ 不可见的消息**在这里断掉整条路**
        # ----------------------------------
        # 从"这条调用是谁"到"这个字发不发"只隔了这一次查找，而判据早在
        # message-start 就定了 —— 这就是为什么不需要缓冲：压缩结果的
        # 第一个字到达时，我们已经知道它不该出去。
        #
        # 也不累积（`pending.delta_parts`）：一个没有读者的正文不该占内存，
        # 而且它会诱使后来的人以为"其实拼得出来，只是没发"。
        if pending is not None and not pending.visible:
            return

        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return

        kind = delta.get("type")

        if kind == _DELTA_REASONING:
            await self._on_reasoning_delta(pending, delta)
            return

        # 白名单，理由见模块顶部的说明。
        if kind != _DELTA_TEXT:
            return

        text = delta.get("text")
        if not isinstance(text, str) or not text:
            return

        # ⚠️ 出口过滤：模型把工具调用写成正文时，标记是**跨分片**来的
        # ------------------------------------------------------
        # 实测那次泄漏的分片是 1~10 个字（先是尖括号、再是两个竖线加 DSML、
        # 再是空格加标签名……），逐块做替换一块都匹配不上。所以这里有状态：
        # `feed` 只扣住"可能是标记前缀"的几个字符，其余立刻放行 —— 逐 token
        # 推流的代价因此只有几个字符的延迟。收尾在 `_on_message_finish`。
        #
        # 拿不到 pending（中途才开始收）时退化成整段过滤：那一段是完整的
        # 一块文本，不需要状态机。
        if pending is not None:
            text = pending.gate.feed(text)
        else:
            text, _ = strip_leaked_tool_calls(text)

        if not text:
            # 整块都是标记（或还扣在闸里）：这一块没有可发的正文。
            # `message.delta` 一条都不发 —— 而边界事件（started / completed）
            # 照发，前端因此不会卡在"以为还有下文"。
            return

        if pending is not None:
            pending.delta_parts.append(text)

        node = pending.node if pending is not None else None

        await self._emit(
            RunEventType.MESSAGE_DELTA,
            node=node,
            # 形状与旧实现逐字一致（agent_node._delta_sink 发的是
            # {"delta": text}），订阅端不用改。
            data={"delta": text},
        )

    async def _on_reasoning_delta(
            self,
            pending: "_PendingMessage | None",
            delta: Mapping[str, Any],
    ) -> None:
        """
        一条思维链增量 -> `message.reasoning.delta`。

        ⚠️ 可见性**不在这里判** —— `_on_block_delta` 的入口已经拦掉了不可见
        的消息（摘要 / 分类的那些），走到这里的一定是给用户看的。这条分工
        与正文那条通道逐字相同，所以两边的"漏了会怎样"也相同：多一层判断
        不会更安全，只会让两处判据有机会漂。

        ⚠️ 取的是 `delta["reasoning"]`，不是 `delta["text"]`
        ------------------------------------------------
        词表在 langchain_core/language_models/chat_model_stream.py:715-716：
        `{"type": "reasoning-delta", "reasoning": …}`。取错键的症状是
        **一个推理字都不发**，而事件名照发 -- 看起来像"这个模型没有思维链"。
        """

        text = delta.get("reasoning")

        if not isinstance(text, str) or not text:
            return

        # 与正文同一条出口规则、**另一个**闸实例（见 _PendingMessage）。
        if pending is not None:
            text = pending.reasoning_gate.feed(text)
        else:
            text, _ = strip_leaked_tool_calls(text)

        if not text:
            return

        if pending is not None:
            pending.reasoning_delta_parts.append(text)

        await self._emit(
            RunEventType.MESSAGE_REASONING_DELTA,
            node=pending.node if pending is not None else None,
            # 字段名与 message.delta 一样是 delta：订阅端的追加逻辑可以照抄。
            data={"delta": text},
        )

    async def _on_message_finish(self, run_key: str, payload: dict[str, Any]) -> None:
        pending = self._messages.pop(run_key, None)

        if run_key in self._completed:
            # 这条已经通过"整条消息"发过了。实测两种形态**不会**同时出现
            # （走协议生命周期就只发生命周期，走 on_chain_end 就只发整条），
            # 但这条防线很便宜，而重复的 message.completed 会让订阅端把
            # 同一段正文渲染两遍 —— 那种 bug 很难从日志上看出来。
            return

        self._completed.add(run_key)

        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}

        await self._complete_message(pending, metadata)

    async def _close_superseded(
            self,
            *,
            node: Any,
            role: str | None,
            run_key: str,
    ) -> None:
        """
        关闭同节点、同角色、更早的那条在途消息。

        调用点：新的一条 `message-start` 到达时（见 `_on_messages`）。
        那时旧的那条只可能是**已经死掉但没发终态**的那次调用（重试 /
        降级换掉了它）—— 同节点同角色同时在跑的两次调用在本仓库不存在。
        """

        for other_key, pending in list(self._messages.items()):

            if other_key == run_key:
                continue

            if str(pending.node) != str(node) or pending.role != role:
                continue

            await self._close_one(other_key, pending)

    async def _close_node_pending(self, *, node: Any) -> None:
        """关闭一个节点名下所有在途消息。节点结束时调用（见 `_on_tasks`）。"""

        for other_key, pending in list(self._messages.items()):

            if str(pending.node) != str(node):
                continue

            await self._close_one(other_key, pending)

    async def _close_one(self, run_key: str, pending: _PendingMessage) -> None:
        """
        给一条没等到 `message-finish` 的消息补上终态。

        ⚠️ 正文用**已经累积的那一份**（不重发 delta）
        --------------------------------------------
        订阅端此前收到的 delta 就是它拼出来的东西；这里再发一遍全文会让
        正文出现两份。completed 的正文与 delta 的拼接**对得上**（本来就
        是同一份），对不上也以 completed 为准 —— 与正常结束同一条规则。

        `finish_reason` 传空：这次调用没有正常的终态，不能编一个。
        """

        self._messages.pop(run_key, None)

        if run_key in self._completed:
            return

        self._completed.add(run_key)

        await self._complete_message(pending, {})

    async def _complete_message(
            self,
            pending: _PendingMessage | None,
            metadata: Mapping[str, Any],
    ) -> None:
        """收尾一条消息：flush 两个闸、发 reasoning.completed（如有）与 completed。"""

        #: 这条消息的正文给不给用户看（判据在 message-start 那一刻就定了）。
        #: pending 拿不到（没见过 start 就收到 finish）时按**不可见**处理：
        #: 与 `_on_block_delta` 同一条方向 —— 缺证据时不发正文。
        visible = pending.visible if pending is not None else False

        if pending is not None and visible:
            # 关闸：还扣着的那点字符要么是**正常正文**（看着像标记前缀、
            # 其实不是 —— 比如回复正好以 `<` 结尾），要么属于一个没闭合的
            # 泄漏区段（整段丢掉）。两种都由 flush 判，这里只负责发出去。
            tail = pending.gate.flush()

            if tail:
                pending.delta_parts.append(tail)

                await self._emit(
                    RunEventType.MESSAGE_DELTA,
                    node=pending.node,
                    data={"delta": tail},
                )

            # 思维链的闸同理（另一个实例，见 _PendingMessage）。
            reasoning_tail = pending.reasoning_gate.flush()

            if reasoning_tail:
                pending.reasoning_delta_parts.append(reasoning_tail)

                await self._emit(
                    RunEventType.MESSAGE_REASONING_DELTA,
                    node=pending.node,
                    data={"delta": reasoning_tail},
                )

            # ⚠️ 思维链的全量**先于** message.completed 发
            # ------------------------------------------
            # 顺序是"模型先想、后说"的投影：订阅端按到达顺序渲染时，推理块
            # 因此落在正文上方。反过来发的话，前端得自己把已经收尾的消息
            # 再"往上插"一段 —— 那是它最容易做错的一步。
            #
            # 只在真的有思维链时发（非推理模型一个空壳都不该收到）：这与
            # message.started / message.completed **照发边界**不同 ——
            # 那两条是"有模型在工作"，而"有没有推理"是内容。
            #
            # ⚠️ 载荷里**没有** `visible`
            # ------------------------
            # 推理只在可见时发（上面那道 if 就是判据），所以那个字段恒为
            # True —— 一个永远不变的字段比缺字段更糟：订阅端会以为
            # "False 也是可能的"，从而写一条永远走不到的分支。这与
            # `_on_tasks` 里拒绝把 `degraded` 加回来是同一条立场。
            # `role` 留着：白名单可以是多个角色，它是有信息的。
            reasoning = pending.reasoning()

            if reasoning:
                await self._emit(
                    RunEventType.MESSAGE_REASONING_COMPLETED,
                    node=pending.node,
                    data={
                        "content": reasoning,
                        "role": pending.role,
                    },
                )

        await self._emit(
            RunEventType.MESSAGE_COMPLETED,
            node=pending.node if pending else None,
            data={
                # ⚠️ 空 content **不等于**"模型没说话"：不可见的消息也是
                # 空串（见 event.py 的 message.* 那一段）。要区分就读 role。
                "content": pending.content() if (pending is not None and visible) else "",
                "tool_calls": list(pending.tool_names) if pending else [],
                "finish_reason": metadata.get("finish_reason"),
                "role": pending.role if pending is not None else None,
                "visible": visible,
                # `degraded` 见 _on_tasks 里那段说明 —— 协议里没有它。
            },
        )

    async def _emit_completed_from_message(self, meta: dict[str, Any], node: Any, message: Any) -> None:
        """
        整条 `AIMessage`（`on_chain_end` 触发）-> `message.completed`。

        ⚠️ 去重键**不能**用 `meta["run_id"]`
        -----------------------------------
        实测这个形状的 meta 里 `run_id` 键在、但值是 **None**（它是链的
        metadata，不是那次 LLM 调用的 metadata）。拿它当键会让所有整条消息
        都归到 `"None"` 底下 —— 第二条起全被当成重复丢掉。

        症状极隐蔽：**事件静默少了一条**，日志上什么异常都没有，而
        "最后一条回复没出现"会被当成前端或存储的问题去查。

        所以这里分两层：
          1. 若 meta 里**真的**有 run_id，用它做跨形状去重（防同一次调用
             既发 message-finish 又发整条消息）；
          2. 否则退回用 AIMessage 自己的 `id`（langchain 一定给，
             形如 `lc_run--...`），并加上前缀避免与 LLM run id 撞。
        两者都拿不到就**直接发** —— 宁可重复，不可漏。
        """

        raw_run_id = meta.get("run_id")

        if raw_run_id is not None:
            key = str(raw_run_id)
            if key in self._completed:
                return
            self._completed.add(key)
        else:
            message_id = getattr(message, "id", None)
            if message_id is not None:
                key = f"msg:{message_id}"
                if key in self._completed:
                    return
                self._completed.add(key)

        metadata = getattr(message, "response_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}

        # ⚠️ 这条路径上**拿不到 role**，所以不按角色过滤
        # -------------------------------------------
        # 这里的 metadata 是**节点那次 run** 的（`on_chain_start` 存下来的
        # 那份），而 role 挂在**模型那次调用**的 run 上 —— 两者不是一个
        # run，模型层的 metadata 传不上来。硬要过滤只能猜，而猜错的代价是
        # 一段真正的答复被静默吞掉（比漏一条压缩结果更糟）。
        #
        # 之所以可以接受：这条路径只在模型**没有**走协议流式时才触发
        # （走流式时 on_chain_end 里那条会被 message-start 记下的 id 去重
        # 掉），而 v3 会把所有 chat model 调用改道到协议事件 —— 也就是说
        # 在本仓库当前的驱动方式下它基本不会被走到。真走到时，它的行为
        # 与改造前一致（节点门已经过了）。
        #
        # ⚠️ 思维链**刻意不在这里发**（这是个已知缺口，不是漏写）
        # ---------------------------------------------------
        # 与上面同一条理由：这条路上拿不到 `model_role`，而推理的放行**必须**
        # 过角色门 —— 摘要调用的思维链里有内部判据与用户原文，漏给用户看
        # 比漏一段正文糟得多。在这里"顺手"发一条 reasoning.completed 就是
        # 全仓库唯一一处绕过两道门的推理出口。
        #
        # 代价：模型不走协议流式的那些调用，思维链只在库里（历史接口有），
        # 实时流上没有。要补就得先解决"role 传不到这一层"，而不是在这里
        # 加一行。
        await self._emit(
            RunEventType.MESSAGE_COMPLETED,
            node=node,
            data={
                # 这条路只有整条消息、没有分片，所以用无状态的版本（与
                # `_on_block_delta` 的流式闸是同一套规则，见 leak 模块）。
                "content": strip_leaked_tool_calls(_message_text(message))[0],
                "tool_calls": _tool_call_names_from_message(message),
                "finish_reason": metadata.get("finish_reason"),
                "role": None,
                "visible": True,
            },
        )

    # ----------------------------------------------------------
    # tools -> trace 的 tool.*（**不**发 RunEvent）
    # ----------------------------------------------------------
    #
    # ⚠️ 这里**只给 trace 用**，不再发 RunEvent
    # -------------------------------------
    # `tools` 通道的事件由 langchain 的 `on_tool_*` 回调产生
    # （langgraph/pregel/_tools.py）。本仓的 tools 节点是自定义扇出
    # 节点，执行时直接 `spec.tool.ainvoke(...)`，没带 tool_call_id：
    # 回调里 langgraph 会回退成 `str(run_id)`（一个 UUID）、名字取
    # 工具函数名（`search` / `current_time`）。
    #
    # 前端的 tool.called / tool.completed 走**另一条**：工具节点写
    # 自定义流（`_on_tool_stream_event`），带模型的 tool_call_id 与
    # tool_key。这里若再发一遍，订阅端会看到两次调用 —— 而两次的 id
    # 还对不上（上一版就是这么漏成 4 条的）。
    #
    # trace 仍走这里：工具 span 本来就是挂在这条回调链上的，去掉就没了。

    async def _on_tool(self, data: Any, timestamp: Any = None) -> None:
        if not isinstance(data, dict):
            return

        sub = data.get("event")
        call_id = data.get("tool_call_id")

        tip = (
            self._trace_tool_tasks.get(str(call_id))
            if call_id is not None
            else None
        )

        if sub == _TC_STARTED:
            await self._trace_tool_started(
                call_id=call_id,
                name=data.get("tool_name"),
                tip=tip,
                timestamp=timestamp,
            )
            return

        if sub in (_TC_FINISHED, _TC_ERROR):
            # `tool-finished` 的载荷里没有名字（见 __init__），取不到
            # 就退化成 None，由 trace 那边用在途记录里的名字兜底。
            await self._trace_tool_finished(
                call_id=call_id,
                name=data.get("tool_name"),
                tip=tip,
                data=data,
                failed=sub == _TC_ERROR,
                timestamp=timestamp,
            )

        # `tool-output-delta` 刻意不发：它是工具产出的**流式**分片，
        # 频率与 message.delta 同量级，而契约里没有对应的工具增量事件。
        # 不做比编一个更诚实。

    # ----------------------------------------------------------
    # Trace: tool.*
    # ----------------------------------------------------------

    def _tool_refs(
            self,
            *,
            call_id: Any,
            name: Any,
            tip: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """tool 的 span / parent / 分支信息（拿不到的字段留 None）。"""

        # ⚠️ 用 tool_call_id 而不是 call_scope：工具执行链（幂等 / 重试
        # 埋点）只知道 tool_call_id，两边必须算出同一个 span_id。
        tool_key = (
            str(call_id)
            if call_id is not None
            else ((tip or {}).get("call_scope") or str(name or "tool"))
        )

        # 工具 Span 是 tools 分支节点（task_id）的子 Span；它的 id 优先
        # 用 route 预先算好的 branch_span_id（与执行链的确定性 id 一致）。
        node_span_id = None
        branch_span_id = None
        if tip is not None:
            node_span_id = tip.get("task_id")
            branch_span_id = tip.get("branch_span_id")

        return {
            "span_id": (
                branch_span_id
                or _tool_span_id(self._trace_id or "", str(tool_key))
            ),
            "parent_span_id": node_span_id or self._run_id,
            "fanout_id": (tip or {}).get("fanout_id"),
            "branch_id": (tip or {}).get("branch_id"),
            "branch_index": (tip or {}).get("branch_index"),
        }

    async def _trace_tool_started(
            self,
            *,
            call_id: Any,
            name: Any,
            tip: Mapping[str, Any] | None,
            timestamp: Any,
    ) -> None:
        if not self._trace_enabled or call_id is None:
            return

        refs = self._tool_refs(call_id=call_id, name=name, tip=tip)

        self._trace_tools_inflight[str(call_id)] = {
            **refs,
            "started_ms": timestamp if isinstance(timestamp, int) else None,
            "tool_name": None if name is None else str(name),
        }

        await self._emit_trace(
            TraceEventType.TOOL_STARTED,
            span_id=refs["span_id"],
            parent_span_id=refs["parent_span_id"],
            span_kind=SpanKind.TOOL,
            span_name=str(name or "tool"),
            status=SpanStatus.RUNNING,
            occurred_at=_ms_to_datetime(timestamp),
            fanout_id=refs["fanout_id"],
            branch_id=refs["branch_id"],
            branch_index=refs["branch_index"],
            payload={
                "tool_name": str(name or ""),
                "tool_call_id": str(call_id),
            },
        )

    async def _trace_tool_finished(
            self,
            *,
            call_id: Any,
            name: Any,
            tip: Mapping[str, Any] | None,
            data: Mapping[str, Any],
            failed: bool,
            timestamp: Any,
    ) -> None:
        if not self._trace_enabled or call_id is None:
            return

        info = self._trace_tools_inflight.pop(str(call_id), None)

        if info is None:
            refs = self._tool_refs(call_id=call_id, name=name, tip=tip)
            started_ms = None
        else:
            refs = {
                "span_id": info["span_id"],
                "parent_span_id": info["parent_span_id"],
                "fanout_id": info.get("fanout_id"),
                "branch_id": info.get("branch_id"),
                "branch_index": info.get("branch_index"),
            }
            started_ms = info.get("started_ms")

        tool_name = str(name or (info or {}).get("tool_name") or "")

        common = {
            "span_id": refs["span_id"],
            "parent_span_id": refs["parent_span_id"],
            "span_kind": SpanKind.TOOL,
            "span_name": tool_name or "tool",
            "duration_ms": _span_elapsed_ms(started_ms, timestamp),
            "occurred_at": _ms_to_datetime(timestamp),
            "fanout_id": refs["fanout_id"],
            "branch_id": refs["branch_id"],
            "branch_index": refs["branch_index"],
        }

        if failed:
            await self._emit_trace(
                TraceEventType.TOOL_FAILED,
                status=SpanStatus.FAILED,
                payload={
                    "tool_name": tool_name,
                    "tool_call_id": str(call_id),
                },
                error=(
                    _trace_error(data.get("error"))
                    or TraceError(type="ToolError", message="")
                ),
                **common,
            )
            return

        output = data.get("output")

        await self._emit_trace(
            TraceEventType.TOOL_COMPLETED,
            status=SpanStatus.COMPLETED,
            payload={
                "tool_name": tool_name,
                "tool_call_id": str(call_id),
                "result_size_bytes": _output_size(output),
            },
            **common,
        )

    # ----------------------------------------------------------
    # 出口
    # ----------------------------------------------------------

    async def _emit(
            self,
            type: RunEventType,
            *,
            node: Any,
            data: Mapping[str, Any],
    ) -> None:
        # 所有出站事件（run / trace）都走同一个 EventPublisher 抽象；
        # 写线程池在 publisher.py 内部按 stream key 分片维护，投影层只
        # 负责入队，绝不直接碰 Redis。
        await publish_run_event(
            type,
            self._run_id,
            data=data,
            node=None if node is None else str(node),
        )

    # ----------------------------------------------------------
    # 收尾钩子（都是空实现，但必须存在且不抛）
    # ----------------------------------------------------------

    def finalize(self) -> None:
        """mux 关闭时调用。**不外抛**：`StreamMux.close` 会把第一个
        finalize 异常在清理完成后重新抛出（_mux.py:300-310），那会被记成
        一次运行失败。"""

    def fail(self, err: BaseException) -> None:
        """运行失败时调用。同样不外抛。"""

        # 刻意不做任何事：run.failed 由 agent_run_dispatch 的 _finish 发
        # （那是图外的事，本层只负责图内事件）。在这里再发一条会重复。


# ============================================================
# 工厂
# ============================================================


def projector_factory(
        *,
        run_id: str,
        message_nodes: Collection[str],
        message_roles: Collection[str] = (),
        node_facts: Mapping[str, Mapping[str, Any]] | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
) -> Any:
    """
    `astream_events(..., transformers=[...])` 要的是 `factory(scope) -> StreamTransformer`，
    **不能**直接传实例（main.py:430-436）。这个函数把运行期的 `run_id`
    等事实闭包进去，产出一个合形的工厂。

    用法：

        run_stream = await compiled.astream_events(
            state, config, version="v3",
            transformers=[projector_factory(
                run_id=str(run_id),
                message_nodes={"agent"},
                message_roles={ModelRole.REASONING},   # 只有它算给用户看的正文
            )],
        )
    """

    def _factory(scope: tuple[str, ...]) -> RunEventProjector:
        return RunEventProjector(
            scope,
            run_id=run_id,
            message_nodes=message_nodes,
            message_roles=message_roles,
            node_facts=node_facts,
            trace_id=trace_id,
            session_id=session_id,
        )

    return _factory


# ============================================================
# 从 payload 里掏字段的小工具
# ============================================================
#
# 全部**防御式**：协议事件的 data 是外部形状，任何一处 key 缺失或类型不对
# 都直接退化成"这个字段没有"，而不是抛。抛的后果见类 docstring。


def _message_count(node_input: Any) -> int | None:
    """`tasks` start 事件的 `input.messages` 长度。拿不到就是 None。"""

    if not isinstance(node_input, dict):
        return None

    messages = node_input.get("messages")
    if isinstance(messages, (list, tuple)):
        return len(messages)

    return None


def _elapsed_ms(pending: _PendingTask | None, finished_at: Any) -> int | None:
    """start/result 两条 `tasks` 事件的 `params.timestamp` 之差。

    ⚠️ 这是**墙钟**之差，不是旧实现那个 `response.latency_ms`（后者只算
    provider 调用本身，不含上下文构建 / 写库）。两者不等价 —— 这里的数
    会更大。改名不改字段名是刻意的：字段名是契约的一部分，而"这个数
    现在含得更多了"这件事写在注释里比塞进字段名里更有用。
    """

    if pending is None or pending.started_at is None:
        return None
    if not isinstance(finished_at, int):
        return None

    delta = finished_at - pending.started_at
    return delta if delta >= 0 else None


def _block_tool_name(block: dict[str, Any]) -> str | None:
    """content block 里的工具名（tool_call 族）。不是就返回 None。"""

    if "tool_call" not in str(block.get("type") or ""):
        return None

    name = block.get("name")
    return str(name) if name else None


def _block_reasoning(block: dict[str, Any]) -> str:
    """
    推理块里的原文。

    取 `reasoning` 键，退回 `text` —— 与 `processor/messages.py` 的
    `message_reasoning` 是**同一条判据**（同一个原因：两种键都存在过，
    只认一种的症状是整段静默取不到）。两处都写是因为这一层刻意不 import
    agent 侧的上下文模块（它只管把协议事件翻成事件，见模块 docstring）。

    取不到返回空串：宁可这一段不进 authoritative，也不要往流里发一个
    `None` 的载荷 —— 那会在订阅端变成一个字符串 "None"。
    """

    text = block.get("reasoning")

    if not isinstance(text, str) or not text:
        text = block.get("text")

    return text if isinstance(text, str) else ""


def _tool_call_names(result: Any) -> list[str]:
    """`tasks` result 里最后一条消息上的 tool_calls 名字。"""

    if not isinstance(result, dict):
        return []

    messages = result.get("messages")
    if not isinstance(messages, (list, tuple)):
        return []

    names: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or ():
            name = call.get("name") if isinstance(call, dict) else None
            if name:
                names.append(str(name))

    return names


def _tool_call_names_from_message(message: Any) -> list[str]:
    """整条消息上的 tool_calls 名字。"""

    names: list[str] = []
    for call in getattr(message, "tool_calls", None) or ():
        name = call.get("name") if isinstance(call, dict) else None
        if name:
            names.append(str(name))

    return names


def _message_text(message: Any) -> str:
    """
    助手消息的**可见正文**。

    与 agent_node._message_text / model/adapters._chunk_text 是同一条判据
    （`.text` 排除思维链与 tool_call 参数）。langchain 1.x 里 `.text` 是
    属性，更早是方法，两种都兜住。

    取不到退回空串 —— `message.completed` 的首要价值是"这条消息结束了"
    这个**边界**，正文取不到不该让事件整个不发。
    """

    text = getattr(message, "text", None)
    if callable(text):
        try:
            text = text()
        except Exception:  # noqa: BLE001
            return ""
    if isinstance(text, str):
        return text
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _result_metadata(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    messages = result.get("messages")
    if not isinstance(messages, (list, tuple)) or not messages:
        return {}

    metadata = getattr(messages[-1], "response_metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _result_provider(result: Any) -> str | None:
    value = _result_metadata(result).get("model_provider")
    return str(value) if value else None


def _result_model_name(result: Any) -> str | None:
    value = _result_metadata(result).get("model_name")
    return str(value) if value else None
