"""
上下文的**工作集、执行记录与产物**（规格 §六、§十九、§二十六、§二十七）。

三个对象，三件事
--------------
    ContextBuildContext    一次构建的**工作集**（流水线里被 Processor 逐步改写）
    ContextSnapshot        一次构建的**可观测产物**（"为什么这一轮看到了这些"）
    ContextBuildResult     对外返回的**结果**（含 ModelRequest）

`ContextBuildContext` **不是** Graph State
---------------------------------------
与既有契约里的 `ContextBuildState` 的关系：

    ContextBuildState       既有契约的每阶段容器（request / resolution / artifacts / draft）
    ContextBuildContext     本流水线的工作集，放在 `ContextBuildState.processor_context` 里

它只在一次 `build()` 里存活，不进 checkpoint、不跨节点。凡是"这次调用的
东西"都在这里，`self`（共享的 builder / processor 实例）上一个字节都不放
（与 `ContextBuildState`、`BaseToolProcessor` 对 runtime 的警告同一条纪律）。

⚠️ 累积轨迹必须能被路径读到
-------------------------
Processor 的输入只从 State Path 来（规格 §八），因此"前一个 Processor
产出的东西"也必须以 State 结构的形式暴露给后一个 —— 这就是 `overlay`：

    state_root() = base_state ⊕ overlay        （overlay 赢）

    例：resolver 阶段把 policy / budget 写进 overlay
        -> history 处理器绑 `context_budget.history_budget` 就取得到

优先级是 overlay 赢：执行期算出来的有效值（policy 合并后、budget 实测后）
比调用方给的输入更具体。

⚠️ 产物必须可 JSON 序列化
----------------------
`ContextSnapshot.as_metadata()` 会被塞进 `ModelRequest.metadata`，随后进
Model Runtime 的响应 metadata、再进 node_span 的 JSONB 列。因此这个结构里
**只允许** str / int / float / bool / None / list / dict —— 没有任何消息
正文（正文在 checkpoint 里，按 id 能查到）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage

from .budget import TokenBudget
from .messages import (
    message_identifier,
    message_identity,
    message_text,
    tool_calls,
    tool_name,
)
from .policy import ContextPolicy
from .values import ContextSection, SummarySnapshot

if TYPE_CHECKING:  # pragma: no cover - 只为类型
    from app.agent.model import ModelRequest

    from .dependencies import ContextDependencies

__all__ = [
    "BuildPhase",
    "ContextBuildContext",
    "ContextBuildResult",
    "ContextSection",
    "ContextSnapshot",
    "ProcessorExecution",
    "ProcessorStatus",
    "SummarySnapshot",
]


# ============================================================
# 执行记录
# ============================================================


class ProcessorStatus(str, Enum):
    """
    规格 §二十六 的三种状态。

    刻意只有三种：`EXECUTED` / `SKIPPED` / `FAILED`。
    "因为 gate 为 MISSING 而整体失败"不是一个 Processor 的状态 ——
    那是**构建失败**，运行里没有它的位置（流水线直接抛）。
    """

    EXECUTED = "executed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProcessorExecution:
    """
    一个 Processor 这一次执行的结果（规格 §二十六）。

    它是 Agent Replay 的基础：把这一组记录按 order 排开，就能回答
    "这一轮上下文是怎么长出来的、哪一步被跳过、花了多久"。
    """

    processor: str
    order: int
    stage: str

    #: gate 的 State 路径；None = 无条件执行的阶段。
    when_path: str | None

    #: gate 的取值：True / False。
    when_value: Any

    status: ProcessorStatus

    duration_ms: int = 0

    error: str | None = None

    #: 这次缺失的可选输入（必需缺失会直接让构建失败，不留记录）。
    missing_inputs: tuple[str, ...] = ()

    #: 这次执行往上下文里注入了什么（计数 + id），便于 replay 对账。
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:

        return {
            "processor": self.processor,
            "order": self.order,
            "stage": self.stage,
            "when": self.when_path,
            "when_value": self.when_value,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "missing_inputs": list(self.missing_inputs),
            "detail": dict(self.detail),
        }


# ============================================================
# 快照
# ============================================================


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """
    一次 LLM Invocation 实际使用了哪些上下文（规格 §二十七）。

    它回答的是"为什么这一轮 LLM 看到了这些内容"，而不是"把 Prompt 存下来"。
    """

    conversation_id: str | None = None
    run_id: str | None = None

    builder_type: str = ""
    agent_type: str = ""
    phase: str = ""

    summary_revision: int | None = None

    history_message_ids: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    retrieved_context_ids: tuple[str, ...] = ()
    injected_sections: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()

    policy: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)

    model_role: str = ""
    model_id: str | None = None
    model_context_window: int | None = None
    context_window_source: str = ""

    input_token_count: int = 0
    estimated_output_tokens: int = 0

    context_hash: str = ""

    processors: tuple[Mapping[str, Any], ...] = ()

    warnings: tuple[str, ...] = ()

    def as_metadata(self) -> dict[str, Any]:
        """进 `ModelRequest.metadata` 的那一份（JSON 安全）。"""

        return {
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
            "builder_type": self.builder_type,
            "agent_type": self.agent_type,
            "phase": self.phase,
            "summary_revision": self.summary_revision,
            "history_message_ids": list(self.history_message_ids),
            "memory_ids": list(self.memory_ids),
            "retrieved_context_ids": list(self.retrieved_context_ids),
            "injected_sections": list(self.injected_sections),
            "tool_names": list(self.tool_names),
            "policy": dict(self.policy),
            "budget": dict(self.budget),
            "model_role": self.model_role,
            "model_id": self.model_id,
            "model_context_window": self.model_context_window,
            "context_window_source": self.context_window_source,
            "input_token_count": self.input_token_count,
            "estimated_output_tokens": self.estimated_output_tokens,
            "context_hash": self.context_hash,
            "processors": [dict(item) for item in self.processors],
            "warnings": list(self.warnings),
        }

    def compact_metrics(self) -> dict[str, Any]:
        """
        给 node_span metrics 用的**精简**版。

        完整快照随 `ModelRequest.metadata` 一路进 Model Runtime 的观测；
        而节点那一列是给人扫一眼的，塞十几个 processor 记录只会让表格
        难读。这里只留结论性的几个数。
        """

        counts = {
            status.value: sum(
                1
                for item in self.processors
                if item.get("status") == status.value
            )
            for status in ProcessorStatus
        }

        return {
            "context_phase": self.phase,
            "context_hash": self.context_hash,
            "context_window": self.model_context_window,
            "context_window_source": self.context_window_source,
            "context_input_tokens": self.input_token_count,
            "context_estimated_output_tokens": self.estimated_output_tokens,
            "context_history_messages": len(self.history_message_ids),
            "context_memory_items": len(self.memory_ids),
            "context_retrieval_items": len(self.retrieved_context_ids),
            "context_tool_count": len(self.tool_names),
            "context_summary_revision": self.summary_revision,
            "context_processors_executed": counts["executed"],
            "context_processors_skipped": counts["skipped"],
            "context_processors_failed": counts["failed"],
            "context_warnings": list(self.warnings),
        }

    @classmethod
    def from_metadata(
            cls,
            raw: Mapping[str, Any] | None,
    ) -> "ContextSnapshot | None":
        """
        从 metadata 里还原一份快照。

        用于"手上只有一个 ModelRequest"的场景（`build_with_result` 的
        默认实现、SSE / 审计回看）。缺字段就按缺省值填 —— 快照是**观测**，
        不该因为多一个少一个键而抛错。
        """

        if not raw:
            return None

        def _tuple(key: str) -> tuple[Any, ...]:

            value = raw.get(key)

            return tuple(value) if isinstance(value, (list, tuple)) else ()

        return cls(
            conversation_id=_optional_str(raw.get("conversation_id")),
            run_id=_optional_str(raw.get("run_id")),
            builder_type=str(raw.get("builder_type") or ""),
            agent_type=str(raw.get("agent_type") or ""),
            phase=str(raw.get("phase") or ""),
            summary_revision=_optional_int(raw.get("summary_revision")),
            history_message_ids=_tuple("history_message_ids"),
            memory_ids=_tuple("memory_ids"),
            retrieved_context_ids=_tuple("retrieved_context_ids"),
            injected_sections=_tuple("injected_sections"),
            tool_names=_tuple("tool_names"),
            policy=dict(raw.get("policy") or {}),
            budget=dict(raw.get("budget") or {}),
            model_role=str(raw.get("model_role") or ""),
            model_id=_optional_str(raw.get("model_id")),
            model_context_window=_optional_int(raw.get("model_context_window")),
            context_window_source=str(raw.get("context_window_source") or ""),
            input_token_count=int(raw.get("input_token_count") or 0),
            estimated_output_tokens=int(
                raw.get("estimated_output_tokens") or 0
            ),
            context_hash=str(raw.get("context_hash") or ""),
            processors=tuple(
                item
                for item in (raw.get("processors") or ())
                if isinstance(item, Mapping)
            ),
            warnings=tuple(str(item) for item in (raw.get("warnings") or ())),
        )


# ============================================================
# 工作集
# ============================================================


class BuildPhase(str, Enum):
    """
    这一次构建处在哪个执行层级（规格 §四）。

        INITIAL        用户提交新消息后的**第一次**构建
                       （从会话历史 + 摘要 + 画像 + 记忆 + 检索里长出来）
        CONTINUATION   ReAct 循环里的后续构建（Tool Result 之后）
                       （基于当前 Graph State 里已有的上下文重新计算，
                         **不重新从库里加载整个会话**）

    为什么放在 State 路径上（`run.phase`）而不是只做 Python 参数：
    Processor 可能想按阶段改变行为（比如"后续轮次不要重复注入记忆"），
    而它只能通过声明式绑定读输入（规格 §六）。
    """

    INITIAL = "initial"
    CONTINUATION = "continuation"


@dataclass(slots=True)
class ContextBuildContext:
    """
    一次上下文构建的工作集。

    Processor 通过 `process(inputs, context)` 拿到它，并就地累积：

        add_section()            注入一段上下文（画像 / 记忆 / 检索 / 摘要）
        add_history()            交出选定的历史消息
        add_tools()              交出本次暴露的工具
        set_summary()            更新摘要快照
        set(key, value)          往 overlay 写一个可被后续 Processor 按路径读到的值
        warn()                   记一条非致命提示
        note_usage()             累加嵌套模型调用的用量
    """

    #: 绑定根（调用方/图给的 State 结构）。overlay 的底。
    base_state: Mapping[str, Any] = field(default_factory=dict)

    #: 本次构建的外部依赖（模型注册表 / 外部上下文源 / 计数器…）。
    #:
    #: 由调用方在 `ContextBuildRequest.dependencies` 里给，缺省时是
    #: 进程级默认（见 dependencies.py）。Processor 从这里取依赖而**不**
    #: 去摸全局变量，测试因此可以注入假的注册表（规格 §二十九 集成测试
    #: 要求"ModelRegistry 是唯一的模型入口"）。
    dependencies: "ContextDependencies | None" = None

    agent_type: str = ""
    builder_type: str = ""
    model_role: str = ""

    phase: BuildPhase = BuildPhase.INITIAL

    conversation_id: str | None = None
    run_id: str | None = None

    policy: ContextPolicy | None = None
    budget: TokenBudget | None = None

    #: 供后续 Processor 用 State Path 读取的执行期结果（**overlay 优先**）。
    overlay: dict[str, Any] = field(default_factory=dict)

    sections: list[ContextSection] = field(default_factory=list)
    history: list[Any] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)

    #: **必须**出现在最终上下文里的消息（典型：当前用户消息）。
    #:
    #: 与 `history` 分开是因为裁剪语义不同：history 是可以被预算裁掉的
    #: 老消息，这些是"少了这次调用就不成立"的（规格 §十五：当前用户
    #: 消息不能被裁剪）。
    required_messages: list[Any] = field(default_factory=list)

    summary: SummarySnapshot | None = None

    #: `True` 表示摘要**这一次真的生成过**（revision 变了）。
    #:
    #: 为什么需要这个位：摘要处理器**每一次**都会把 state 里那份摘要
    #: "设"一遍（set_summary），那是**读** —— 只有触发轮次窗口、真的调了
    #: 一次摘要模型，才是**写**。节点靠这一位决定要不要把摘要回写进图
    #: state（见 agent_node 的 `_context_state_updates`）—— 没有它，
    #: 每一轮都会把同一份摘要重复写一遍（并把 state 的 channel 变成
    #: "看起来每次都在变"）。
    summary_dirty: bool = False

    #: 装配出来的请求（由 PROMPT_ASSEMBLER 写，后续阶段可以改写）。
    #:
    #: 用 Model 模块的 `ModelRequest`（`app/agent/model/types.py`）——
    #: 全系统只有那一个请求类型，Builder 在每阶段之后把它同步进既有的
    #: `ContextBuildState.draft`（那是既有契约的产出槽）。
    draft: Any = None

    executions: list[ProcessorExecution] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    #: 构建过程中**嵌套**发生的模型调用用量（目前只有 summary 阶段）。
    token_usage: dict[str, int] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    # ========================================================
    # 路径根
    # ========================================================

    def state_root(self) -> dict[str, Any]:
        """
        State Path 的解析根 = base_state ⊕ overlay（overlay 赢）。

        见模块 docstring：执行期算出来的有效值比调用方给的输入更具体。
        """

        return {**dict(self.base_state), **self.overlay}

    # ========================================================
    # 写入
    # ========================================================

    def set(self, key: str, value: Any) -> None:
        """
        往 overlay 写一个值（点分路径也可以，会按段建 dict）。

        它让"前一个 Processor 的产出"成为"后一个 Processor 的输入"，
        路径与其它输入完全一致（规格 §十）。
        """

        parts = key.split(".")
        cursor = self.overlay

        for part in parts[:-1]:

            existing = cursor.get(part)

            if not isinstance(existing, dict):
                existing = {}
                cursor[part] = existing

            cursor = existing

        cursor[parts[-1]] = value

    def add_section(
            self,
            name: str,
            label: str,
            text: str,
            *,
            source_ids: Sequence[str] = (),
            token_estimate: int = 0,
    ) -> ContextSection:
        """注入一段上下文。空文本**不**产生 section（不留空壳）。"""

        section = ContextSection(
            name=name,
            label=label,
            text=text,
            source_ids=tuple(str(item) for item in source_ids),
            token_estimate=token_estimate,
        )

        if text.strip():
            self.sections.append(section)

        return section

    def add_history(self, messages: Sequence[Any]) -> None:
        """交出选定的历史（**替换**而不是追加：选择器给的是完整结果）。"""

        self.history = list(messages)

    def add_tools(self, tools: Sequence[Any]) -> None:

        self.tools = list(tools)

    def add_required_message(self, message: Any) -> None:

        self.required_messages.append(message)

    def set_summary(self, summary: SummarySnapshot | None) -> None:
        """
        更新摘要，并同步进 overlay。

        同步是必须的：后续 Processor（compaction / prompt / validation）
        与快照都通过 `summary.*` 路径读它，两处不一致会让"摘要注入了但
        快照说没有"这种事发生。
        """

        self.summary = summary

        self.set("summary", summary.as_state() if summary else None)

    def warn(self, message: str) -> None:
        """记一条非致命提示（进快照的 warnings）。"""

        text = str(message)

        if text not in self.warnings:
            self.warnings.append(text)

    def note_usage(self, usage: Any, *, calls: int = 1) -> None:
        """
        累加一次嵌套模型调用的用量。

        认 `ModelUsage`（模型模块）与 mapping 两种形态。
        """

        def _value(name: str) -> int | None:

            if usage is None:
                return None

            if isinstance(usage, Mapping):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)

            return value if isinstance(value, int) else None

        for key in ("input_tokens", "output_tokens", "total_tokens"):

            value = _value(key)

            if value is None:
                continue

            self.token_usage[key] = self.token_usage.get(key, 0) + value

        self.token_usage["calls"] = self.token_usage.get("calls", 0) + calls

    def record(self, execution: ProcessorExecution) -> None:

        self.executions.append(execution)

    # ========================================================
    # 产物
    # ========================================================

    def system_message(self) -> Any:
        """
        组装 system 消息（计数与组装共用**同一条**路径）。

        分节顺序是**固定**的：策略里的 system_prompt → 各注入段按插入
        顺序。固定顺序让 context_hash 稳定（同样的输入 -> 同样的哈希），
        而哈希是 Replay / 去重的基础。

        没有内容时返回 None（不是一条空消息 —— 空 system 消息既浪费
        token，也会让某些 provider 报错）。
        """

        blocks: list[str] = []

        if self.policy is not None and self.policy.system_prompt:
            if self.policy.system_prompt.strip():
                blocks.append(self.policy.system_prompt.strip())

        for section in self.sections:

            if section.text.strip():
                blocks.append(f"## {section.label}\n{section.text.strip()}")

        if not blocks:
            return None

        return SystemMessage(content="\n\n".join(blocks))

    def assembled_messages(self) -> list[Any]:
        """
        这次真正要发给模型的消息列表。

            一条 system（有内容时）
            + history（选择器选出来的）
            + required_messages（必须在的，按标识去重）

        去重按**标识**（id，没有 id 时退化成对象身份）：同一个当前用户
        消息既在 history 里又被声明为 required 时只发一条 —— 否则模型
        会看到两遍同一句话，而 token 是照付的。
        """

        messages: list[Any] = []

        system = self.system_message()

        if system is not None:
            messages.append(system)

        seen: set[str] = set()

        for message in self.history:

            seen.add(message_identity(message))
            messages.append(message)

        for message in self.required_messages:

            key = message_identity(message)

            if key in seen:
                continue

            seen.add(key)
            messages.append(message)

        return messages

    def estimate_input_tokens(self, counter: Any) -> int:
        """估算这次调用实际发出去的 token（system + 历史 + 工具）。"""

        total = counter.count_messages(self.assembled_messages())
        total += counter.count_tools(self.tools)

        return total

    def policy_state(self) -> dict[str, Any]:
        """
        这次构建**有效**的策略结构（gate 实际解析的那一份）。

        优先用 overlay 里的合并结果（基线 ⊕ 覆盖，含未知键），而不是
        工作集上那份**类型正确**的 `ContextPolicy`：快照要回答的是
        "这一次的策略到底长什么样"，包括"某个键被写成了非法值"这种
        事实 —— 那正是 gate 判定 INVALID 的依据。
        """

        merged = self.overlay.get("context_policy")

        if isinstance(merged, Mapping):
            return dict(merged)

        return self.policy.as_state() if self.policy is not None else {}

    def context_hash(self) -> str:
        """
        本次上下文的一份稳定指纹（规格 §十九）。

        覆盖：system 分节文本、历史消息的（类型 + 文本 + tool_call id）、
        必须在的消息、工具名、策略、phase。**不覆盖**时间戳之类的
        东西 —— 哈希的用途是"两次调用看到的是不是同一份上下文"，
        抖动就失去意义了。
        """

        payload = {
            "system_prompt": (
                self.policy.system_prompt if self.policy else None
            ),
            "sections": [
                {"name": section.name, "text": section.text}
                for section in self.sections
            ],
            "history": [
                {
                    "type": type(message).__name__,
                    "text": message_text(message),
                    "tool_calls": [
                        call.get("id") for call in tool_calls(message)
                    ],
                    "tool_call_id": getattr(message, "tool_call_id", None),
                }
                for message in self.history
            ],
            "required": [
                {
                    "type": type(message).__name__,
                    "text": message_text(message),
                }
                for message in self.required_messages
            ],
            "tools": sorted(tool_name(tool) for tool in self.tools),
            "policy": self.policy_state(),
            "phase": self.phase.value,
        }

        digest = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()

        return digest[:32]

    def snapshot(self, *, counter: Any = None) -> ContextSnapshot:
        """
        产出这一次构建的快照（规格 §二十七）。

        `counter` 给了就顺便算 input token（预算阶段之后调用）；
        没给就留 0，由 validation 阶段或调用方补。
        """

        budget_state = (
            self.budget.as_state() if self.budget is not None else {}
        )

        return ContextSnapshot(
            conversation_id=self.conversation_id,
            run_id=self.run_id,
            builder_type=self.builder_type,
            agent_type=self.agent_type,
            phase=self.phase.value,
            summary_revision=(
                self.summary.revision if self.summary is not None else None
            ),
            history_message_ids=tuple(
                message_identifier(message, index)
                for index, message in enumerate(self.history)
            ),
            memory_ids=_section_ids(self.sections, "memory"),
            retrieved_context_ids=_section_ids(self.sections, "retrieval"),
            injected_sections=tuple(
                section.name for section in self.sections
            ),
            tool_names=tuple(tool_name(tool) for tool in self.tools),
            policy=self.policy_state(),
            budget=budget_state,
            model_role=self.model_role,
            model_id=budget_state.get("model_id"),
            model_context_window=budget_state.get("context_window"),
            context_window_source=str(
                budget_state.get("window_source") or ""
            ),
            input_token_count=(
                self.estimate_input_tokens(counter)
                if counter is not None
                else 0
            ),
            estimated_output_tokens=(
                self.policy.max_output_tokens if self.policy else 0
            ),
            context_hash=self.context_hash(),
            processors=tuple(
                execution.as_metadata() for execution in self.executions
            ),
            warnings=tuple(self.warnings),
        )


# ============================================================
# 对外结果
# ============================================================


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """
    Context Builder 的对外结果（规格 §十九）。

    它回答的问题只有一个，但很关键：

        这一次 LLM 调用到底使用了哪些上下文

    因此它不只是 messages —— 还有选定/注入了哪些 id、摘要版本、预算、
    哈希，以及每个 Processor 的执行状态。Debug、Replay、Evaluation、
    Audit 都建立在这上面。

    `model_request` 用的是 **Model 模块的 `ModelRequest`**
    （`app/agent/model/types.py`）—— 全系统只有那一个请求类型，
    Context Builder 的产出直接交给 Model Runtime，中间不做转换。
    """

    model_request: "ModelRequest"

    snapshot: ContextSnapshot

    #: 构建过程中嵌套模型调用的用量（compaction）。
    token_usage: Mapping[str, int] = field(default_factory=dict)

    #: 这次构建额外想告诉调用方的东西（不含正文）。
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> tuple[Any, ...]:
        return tuple(self.model_request.messages)

    @property
    def tools(self) -> tuple[Any, ...]:
        return tuple(self.model_request.tools)

    @property
    def summary_revision(self) -> int | None:
        return self.snapshot.summary_revision

    @property
    def selected_history_message_ids(self) -> tuple[str, ...]:
        return self.snapshot.history_message_ids

    @property
    def injected_memory_ids(self) -> tuple[str, ...]:
        return self.snapshot.memory_ids

    @property
    def injected_context_ids(self) -> tuple[str, ...]:
        return self.snapshot.retrieved_context_ids

    @property
    def context_hash(self) -> str:
        return self.snapshot.context_hash

    @property
    def processor_executions(self) -> tuple[Mapping[str, Any], ...]:
        return self.snapshot.processors

    def as_metadata(self) -> dict[str, Any]:
        """一次构建的**精简**摘要（不含正文），给 SSE / 日志用。"""

        return {
            **self.snapshot.compact_metrics(),
            "tokens": dict(self.token_usage),
            **{str(key): value for key, value in self.metadata.items()},
        }

    @classmethod
    def from_model_request(
            cls,
            model_request: "ModelRequest",
    ) -> "ContextBuildResult":
        """
        从一个 ModelRequest 还原结果。

        这是"手上只有请求对象"时的通道（`BaseContextBuilder.build()` 的
        契约只返回 ModelRequest）：快照由请求的 metadata 带出来，因此
        信息不丢 —— 只是没有 Python 侧的富对象而已。
        """

        raw = dict(model_request.metadata or {})

        snapshot = ContextSnapshot.from_metadata(
            raw.get("context_snapshot")
        )

        return cls(
            model_request=model_request,
            snapshot=snapshot or ContextSnapshot(),
            token_usage={
                str(key): int(value)
                for key, value in (raw.get("context_token_usage") or {}).items()
                if isinstance(value, int)
            },
            # ⚠️ `context_state_updates` 与上面两项一样**排除掉**：它装的是
            # 要回写图 state 的**内容**（记忆分节的正文），而
            # `as_metadata()` 的契约是"精简摘要、不含正文"（它给 SSE / 日志
            # 用）。不排除的话，整篇画像会随每一条 SSE 事件发出去。
            metadata={
                str(key): value
                for key, value in raw.items()
                if key not in (
                    "context_snapshot",
                    "context_token_usage",
                    "context_state_updates",
                )
            },
        )


# ============================================================
# internals
# ============================================================


def _section_ids(
        sections: Sequence[ContextSection],
        name: str,
) -> tuple[str, ...]:

    return tuple(
        source_id
        for section in sections
        if section.name == name
        for source_id in section.source_ids
    )


def _optional_str(value: Any) -> str | None:

    if value is None:
        return None

    text = str(value)

    return text or None


def _optional_int(value: Any) -> int | None:

    return value if isinstance(value, int) and not isinstance(value, bool) else None
