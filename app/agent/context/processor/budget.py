"""
Token 预算（规格 §二十一）。

公式（必须按这个顺序扣）：

    Model Context Window
    - System Prompt
    - Tool Definitions
    - User Profile
    - Memory
    - Retrieved Context
    - Expected Output
    - Safety Reserve
    = History Budget

两阶段计算，因为两端的已知量不同
------------------------------
    resolver 阶段（CONTEXT_RESOLVER）  只知道窗口与各段的**上限**
                                       -> 给出一份**保守**的
                                          history_budget（供历史选择器用）
    budget 阶段（TOKEN_BUDGETER）      组装之后各段的**实际**大小才已知
                                       -> 用真实值重算，并据此裁剪

因此 `TokenBudget.allocate()` 被调用两次：第一次用 policy 的上限，
第二次用测量值。两次都写进 overlay（`context_budget.*`），快照里
记录的是**最终**那一份。

不要简单 len(messages)
--------------------
    ❌ len(messages)                  消息条数不是 token
    ❌ len(text)                      字符数不是 token

本模块用 `TokenCounter` 端口 + 默认的启发式计数（按字符数折算，
中英混排取保守系数），并**预留** tiktoken 实现（它需要下载词表，
不能作为默认 —— 见 requirements.txt 里那条警告）。

模型窗口从 ModelRegistry 来（规格 §二十一：必须优先用 Registry，
而不是硬编码模型名）
----------------------------------------------------------
`ContextResolveProcessor` 通过 `ModelRegistry.get_runtime(...)` 拿到
这次要用的 profile，再 `definition(profile.primary).context_window`
读窗口。拿不到定义 / 没声明窗口时回落到
`ContextPolicy.default_context_window`，并在快照里注明
（`context_window_source`），让"用的是兜底值"这件事**可见**。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .errors import ContextBudgetError
from .messages import message_text, tool_calls
from .policy import ContextPolicy

__all__ = [
    "HeuristicTokenCounter",
    "TokenBudget",
    "TokenCounter",
]


# ============================================================
# 计数
# ============================================================


@runtime_checkable
class TokenCounter(Protocol):
    """
    估算 token 数。

    刻意是**估算**而不是"精确"：真正的 tokenizer 由模型决定
    （各家不同），而预算要的是"不会超"的下界估计 + 一个安全余量。
    精确但需要联网下载词表的方案不适合放在请求路径上。
    """

    def count_text(self, text: str) -> int:
        ...

    def count_messages(self, messages: Sequence[Any]) -> int:
        ...

    def count_tools(self, tools: Sequence[Any]) -> int:
        ...


class HeuristicTokenCounter:
    """
    按字符数折算的启发式计数（默认实现，无依赖、无网络）。

    `chars_per_token=3.0` 是**保守**取值：英文约 4 字符/token，中文
    约 1–1.5 字符/token，混排时 3 更接近上界。保守的方向是对的 ——
    低估会让请求撞上 provider 的硬上限（一次**整个窗口**的失败），
    高估只是少放几条历史。

    ⚠️ 每条消息额外算 `per_message_overhead` 个 token：provider 会为
    角色标记、分隔符、tool_call 结构付费，这部分与正文长度无关。
    """

    def __init__(
            self,
            *,
            chars_per_token: float = 3.0,
            per_message_overhead: int = 4,
    ) -> None:

        if chars_per_token <= 0:
            raise ValueError("chars_per_token 必须为正。")

        self._chars_per_token = float(chars_per_token)
        self._per_message_overhead = int(per_message_overhead)

    def count_text(self, text: str) -> int:

        if not text:
            return 0

        return int(len(text) / self._chars_per_token) + 1

    def count_messages(self, messages: Sequence[Any]) -> int:

        total = 0

        for message in messages:

            total += self._per_message_overhead
            total += self.count_text(message_text(message))

            # tool_calls 的 JSON 也是请求体的一部分（名字 + 参数）。
            for call in tool_calls(message):

                total += self.count_text(json.dumps(
                    call, ensure_ascii=False, default=str
                ))

        return total

    def count_tools(self, tools: Sequence[Any]) -> int:

        total = 0

        for tool in tools:

            schema = _tool_schema(tool)

            total += self.count_text(json.dumps(
                schema, ensure_ascii=False, default=str
            ))

        return total


# ============================================================
# 预算
# ============================================================


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """
    一次 LLM 调用的预算分配（规格 §二十一 的那组字段）。

    不变式：`history_budget >= 0`，且
    `allocated + history_budget <= context_window`
    （`allocated` 是各**专属**段之和，不含历史）。
    """

    context_window: int

    system_budget: int
    profile_budget: int
    memory_budget: int
    retrieval_budget: int
    tool_budget: int
    output_budget: int
    reserve_budget: int

    history_budget: int

    #: 这份预算从哪来：`model_registry` / `policy_default`。
    #: 它让"这次用的是兜底窗口"在快照里看得见。
    window_source: str = "policy_default"

    #: 模型定义 id（拿得到时）。
    model_id: str | None = None

    @property
    def allocated(self) -> int:
        """各专属段之和（不含历史）。"""

        return (
            self.system_budget
            + self.profile_budget
            + self.memory_budget
            + self.retrieval_budget
            + self.tool_budget
            + self.output_budget
            + self.reserve_budget
        )

    def as_state(self) -> dict[str, Any]:

        return {
            "context_window": self.context_window,
            "system_budget": self.system_budget,
            "profile_budget": self.profile_budget,
            "memory_budget": self.memory_budget,
            "retrieval_budget": self.retrieval_budget,
            "tool_budget": self.tool_budget,
            "output_budget": self.output_budget,
            "reserve_budget": self.reserve_budget,
            "history_budget": self.history_budget,
            "allocated": self.allocated,
            "window_source": self.window_source,
            "model_id": self.model_id,
        }

    # --------------------------------------------------------

    @classmethod
    def allocate(
            cls,
            *,
            context_window: int,
            policy: ContextPolicy,
            system_budget: int | None = None,
            tool_budget: int | None = None,
            window_source: str = "policy_default",
            model_id: str | None = None,
    ) -> "TokenBudget":
        """
        分配预算。

        `system_budget` / `tool_budget` 缺省用 policy 里的**上限**；
        传实际测量值时（budget 阶段）用真实占用。

        Raises
        ------
        ContextBudgetError
            扣完专属段之后历史预算为负 —— 说明窗口本身就不够
            （或 output + reserve 已经吃掉整个窗口）。这是 Fatal：
            没有任何裁剪能让这次调用成立。
        """

        if context_window <= 0:
            raise ContextBudgetError(
                f"模型上下文窗口必须是正数，得到 {context_window}。"
            )

        system = (
            policy.system_budget if system_budget is None else system_budget
        )
        tools = policy.tool_budget if tool_budget is None else tool_budget

        fixed = (
            system
            + policy.profile_budget
            + policy.memory_budget
            + policy.retrieval_budget
            + tools
            + policy.max_output_tokens
            + policy.reserve_tokens
        )

        history = context_window - fixed

        if history < 0:
            raise ContextBudgetError(
                "上下文窗口不足以容纳 system / 工具 / 各注入段 / 输出 / 余量："
                f"窗口 {context_window} < 固定占用 {fixed}"
                f"（system={system}, profile={policy.profile_budget}, "
                f"memory={policy.memory_budget}, "
                f"retrieval={policy.retrieval_budget}, tools={tools}, "
                f"output={policy.max_output_tokens}, "
                f"reserve={policy.reserve_tokens}）。\n"
                "两条路：调小 policy 里各段的上限 / max_output_tokens，"
                "或用窗口更大的模型 —— 这不是可以裁剪出来的空间。",
                details={
                    "context_window": context_window,
                    "fixed": fixed,
                    "history_budget": history,
                },
            )

        # `max_history_tokens` 是**更紧**的一层上限（规格 §十五要求
        # 它与轮次窗口同时生效），因此取两者的小值。
        if policy.max_history_tokens is not None:
            history = min(history, policy.max_history_tokens)

        return cls(
            context_window=context_window,
            system_budget=system,
            profile_budget=policy.profile_budget,
            memory_budget=policy.memory_budget,
            retrieval_budget=policy.retrieval_budget,
            tool_budget=tools,
            output_budget=policy.max_output_tokens,
            reserve_budget=policy.reserve_tokens,
            history_budget=max(history, 0),
            window_source=window_source,
            model_id=model_id,
        )


# ============================================================
# internals
# ============================================================


def _tool_schema(tool: Any) -> Any:
    """取一个工具的 JSON schema（bind_tools 认的东西）。"""

    for attribute in ("args_schema", "args", "tool_call_schema"):

        schema = getattr(tool, attribute, None)

        if schema is None:
            continue

        dump = getattr(schema, "model_json_schema", None)

        if callable(dump):
            return dump()

        if isinstance(schema, dict):
            return schema

    for attribute in ("name", "description"):

        value = getattr(tool, attribute, None)

        if value:
            return {attribute: value}

    return {"tool": str(tool)}
