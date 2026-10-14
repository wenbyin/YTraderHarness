"""
Context Policy：**执行策略**，不是执行逻辑（规格 §二十）。

分工
----
    Agent Builder / Policy 层   决定"这次用什么策略"
    ContextBuilder              按策略执行（取多少历史、读不读 memory…）

因此：

    ❌ ContextBuilder 里写 `if agent_type == "react": ... elif ...`
    ✅ 策略由外面给进来，ContextBuilder 只读它

规格 §二十 的字段在这里落地，并**全部**出现在 `as_state()` 里 ——
因为 Processor 的 gate 与输入都通过 State Path 读取它们
（`context_policy.include_memory`），所以"策略里有哪些位"必须是一份
**完整的、可被路径解析的**结构。

⚠️ `merge` 刻意**不做类型转换**
-----------------------------
如果调用方（Agent Builder / 图 state）把 `include_memory` 写成 `"yes"`，
合并之后它仍然是 `"yes"` —— 于是 gate 解析会判 INVALID 并让
ContextBuilder FAILED（规格 §九）。

反向的做法（`bool(value)` 静默转一下）看起来"健壮"，实际把
"上游写错了"变成了"这台机器上恰好跑通了"，而那正是最难查的一类问题。

⚠️ Policy 从哪来
--------------
    ContextPolicyResolver（端口）  按 (agent_type, context_builder_type) 给出基线策略
    图 state 的 `context_policy`   调用方/Agent Builder 的覆盖项（优先）

`DefaultContextPolicyResolver` 只提供**默认基线**：本仓库还没有
Agent Builder，因此"哪类 Agent 用什么策略"目前是一个常量表 —— 等
Agent Builder 落地时替换 resolver 即可，ContextBuilder 一行都不用改
（这正是把策略抽出来的意义）。
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from app.common.context import ContextBuilderType

__all__ = [
    "ContextPolicy",
    "ContextPolicyResolver",
    "DefaultContextPolicyResolver",
]


#: 策略展开成 state 结构时用的键名（供 gate / 输入绑定引用）。
#:
#: 写成常量而不是散落的字符串字面量：Processor 的 `when` 与这里的键
#: 必须逐字一致，而"gate 路径拼错了"的表现是**启动后第一次调用就
#: FAILED**（MISSING），离原因很远。
POLICY_GATE_SUMMARY = "include_summary"
POLICY_GATE_HISTORY = "include_history"
POLICY_GATE_USER_PROFILE = "include_user_profile"
POLICY_GATE_MEMORY = "include_memory"
POLICY_GATE_RETRIEVAL = "include_retrieved_context"
POLICY_GATE_CAPABILITIES = "include_capabilities"
POLICY_GATE_BUDGET = "enforce_token_budget"
POLICY_GATE_COMPACTION = "allow_compaction"
POLICY_GATE_VALIDATION = "validate"


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """
    一次上下文构建的**执行策略**。

    规格 §二十 的字段全在这里，另加几项本仓库需要的：

        system_prompt           系统提示（业务能力走 skills 注入，见 context/__init__.py）
        reserve_tokens          安全余量
        default_context_window  拿不到模型定义时的兜底窗口
        min_recent_turns        轮次窗口的下限（也是唯一的可压缩重叠量）
        max_recent_turns        轮次窗口的上限（同时决定摘要的折叠节奏）
        summary_output_tokens   摘要模型一次最多写多少 token
        profile_budget / memory_budget / retrieval_budget / tool_budget
                                各段的**上限**（预算分配用，见 budget.py）

    ⚠️ 所有开关都是 `bool` —— 它们会被 State Path 解析并**严格**判
    bool（规格 §九）。这里不写 `bool | None`：策略一旦给出就必须给出
    一个明确的决定，含糊的 None 会在运行期变成 FAILED。
    """

    # ---- 系统提示 -------------------------------------------

    system_prompt: str | None = None

    # ---- 历史 -----------------------------------------------

    #: 窗口**最少**保留的轮数（User -> Assistant 算一轮）。
    #:
    #: 它是两个阶段的共同下限，含义各一半：
    #:     history 阶段    预算裁剪不裁到它以下 —— "最近的现场"必须有
    #:     compaction 阶段  压重叠时的**唯一可压缩量**（重叠最多 = 它 − 1）
    #:
    #: ⚠️ 它是"无预算压力时的下限"，**不是硬保证**：真的超预算时
    #: compaction 有权把重叠压掉，窗口因此可能少于这个数（见
    #: processors/compaction.py）。
    min_recent_turns: int = 3

    #: 窗口**最多**保留的轮数（上限；实际宽度 = clamp(未覆盖轮数, min, 上限)）。
    #:
    #: 上限不是常量：它随对话长度从 `min_recent_turns` 长到它为止
    #: （`turns.window_limit`：每满 10 轮 +1）。因此"折叠间隔"也随对话
    #: 变长而从 3 轮拉到 5 轮。
    #:
    #: ⚠️ 上限决定**折叠间隔**：未覆盖轮数一旦超过它，下一次窗口往前一滑
    #: 就会露出缺口，所以 summary 阶段必须在那之前重新摘要。两者是同一个
    #: 数，不要为"多久折一次"另设一个旋钮（那样两个数迟早对不上）。
    max_recent_turns: int = 5

    #: 历史部分的 token 上限；None = 只按轮次，不按 token 收。
    max_history_tokens: int | None = 6000

    # ---- 开关（gate 的取值来源） -----------------------------

    include_summary: bool = True
    include_history: bool = True
    include_user_profile: bool = True
    include_memory: bool = True
    include_retrieved_context: bool = True
    include_capabilities: bool = True
    enforce_token_budget: bool = True
    allow_compaction: bool = True
    validate: bool = True

    # ---- 注入上限 -------------------------------------------

    max_memory_items: int = 5
    max_retrieved_items: int = 5

    #: 本次请求**暴露给模型**的工具 key 列表。
    #:
    #: 由 Agent Builder 按识别出的能力解析后写进覆盖项，capability
    #: 处理器按它调 ToolExposureSource 拿到可 bind 的工具对象。
    #:
    #: ⚠️ 与 `include_capabilities` 是**两件事**：那个是 gate（要不要
    #: 跑能力加载阶段），这个是输入（加载哪些）。gate 关掉时它整个
    #: 不生效 —— classifier 就是这种情形。
    #:
    #: 空元组 = 本次没有工具，这是完全正常的一种请求（不是"没配"）。
    #: 用元组不用 list：本类 frozen，可变默认值会破坏 __hash__；
    #: 而且它跨越多次 attempt 被反复读，本来就不该能被就地改。
    tool_keys: tuple[str, ...] = ()

    # ---- 预算 -----------------------------------------------

    max_context_tokens: int | None = None
    max_output_tokens: int = 1024
    reserve_tokens: int = 512
    system_budget: int = 1500
    profile_budget: int = 400
    memory_budget: int = 500
    retrieval_budget: int = 1500
    tool_budget: int = 1200
    default_context_window: int = 8192

    # ---- 摘要 -----------------------------------------------

    #: 摘要模型一次最多写多少 token。
    summary_output_tokens: int = 512

    # ---- 元信息 ---------------------------------------------

    #: 这份策略从哪来（日志/快照里说得出"谁决定的"）。
    source: str = ""

    # --------------------------------------------------------

    def __post_init__(self) -> None:

        if self.min_recent_turns < 0:
            raise ValueError("ContextPolicy.min_recent_turns 不能为负。")

        if self.max_recent_turns < self.min_recent_turns:
            # 上限低于下限会让 `turns.window_limit` 静默吐出一个小于下限的
            # 值，而那正是"窗口明明够得到 cursor 却还是缺了一轮"的来源。
            raise ValueError(
                "ContextPolicy.max_recent_turns 不能小于 min_recent_turns，"
                f"得到 {self.max_recent_turns!r} < {self.min_recent_turns!r}。"
            )

        if self.max_history_tokens is not None and self.max_history_tokens < 0:
            raise ValueError("ContextPolicy.max_history_tokens 不能为负。")

        if self.max_output_tokens < 0 or self.reserve_tokens < 0:
            raise ValueError("ContextPolicy 的预算项不能为负。")

    # --------------------------------------------------------

    def as_state(self) -> dict[str, Any]:
        """
        展开成可被 State Path 解析的结构。

        键名即 gate 路径的末段（`context_policy.include_memory`），
        因此**这里少一个键**的直接后果是"那个 Processor 启动后第一次
        调用就 FAILED(MISSING)" —— 这也是为什么 `merge` 用 dataclass
        的完整字段集，而不是手写一份 dict。
        """

        return {
            # 开关
            POLICY_GATE_SUMMARY: self.include_summary,
            POLICY_GATE_HISTORY: self.include_history,
            POLICY_GATE_USER_PROFILE: self.include_user_profile,
            POLICY_GATE_MEMORY: self.include_memory,
            POLICY_GATE_RETRIEVAL: self.include_retrieved_context,
            POLICY_GATE_CAPABILITIES: self.include_capabilities,
            POLICY_GATE_BUDGET: self.enforce_token_budget,
            POLICY_GATE_COMPACTION: self.allow_compaction,
            POLICY_GATE_VALIDATION: self.validate,
            # 数值
            "min_recent_turns": self.min_recent_turns,
            "max_recent_turns": self.max_recent_turns,
            "max_history_tokens": self.max_history_tokens,
            "max_memory_items": self.max_memory_items,
            "max_retrieved_items": self.max_retrieved_items,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "reserve_tokens": self.reserve_tokens,
            "system_budget": self.system_budget,
            "profile_budget": self.profile_budget,
            "memory_budget": self.memory_budget,
            "retrieval_budget": self.retrieval_budget,
            "tool_budget": self.tool_budget,
            "default_context_window": self.default_context_window,
            # 元组字段展开成 list：这个结构是给 State Path 与快照看的，
            # 而快照会进 JSON（元组会在那里变成数组，不如一开始就是）。
            "tool_keys": list(self.tool_keys),
            "summary_output_tokens": self.summary_output_tokens,
            "source": self.source,
            # system_prompt 也进结构：prompt 组装处理器通过路径读它，
            # 而不必让上下文里多一个"特殊字段"。
            "system_prompt": self.system_prompt,
        }

    def with_(self, **changes: Any) -> "ContextPolicy":
        """派生一份改了若干项的策略（用于 resolver 按类型给差异化基线）。"""

        return replace(self, **changes)

    # --------------------------------------------------------

    @classmethod
    def from_state(
            cls,
            raw: Mapping[str, Any],
            *,
            base: "ContextPolicy | None" = None,
    ) -> "ContextPolicy":
        """
        从**合并后的**策略结构里读出一份有效策略。

        只接受**类型正确**的值：

            bool 字段        只认真正的 bool（`"yes"` / 1 / None 全部忽略）
            数值字段         只认 int / float（且不认 bool）
            system_prompt    只认 str / None

        忽略而不是强转：非法值会在**gate**那一步让整次构建 FAILED
        （规格 §九），因此这里不需要（也不该）替它猜一个值。忽略的后果
        只是"有效策略里那一项保持基线"，而构建反正已经失败了。
        """

        base = base or cls()

        values: dict[str, Any] = {}

        for name in _POLICY_FIELDS:

            if name not in raw:
                continue

            value = raw[name]
            current = getattr(base, name)

            if isinstance(current, bool):

                if isinstance(value, bool):
                    values[name] = value

                continue

            if isinstance(current, int):

                if isinstance(value, int) and not isinstance(value, bool):
                    values[name] = value

                continue

            if isinstance(current, float):

                if isinstance(value, (int, float)) and not isinstance(
                        value, bool
                ):
                    values[name] = float(value)

                continue

            if isinstance(current, tuple):

                # 元组字段（今天只有 tool_keys）：只接受"元素全是 str"
                # 的 list / tuple，并统一收成 tuple。
                #
                # ⚠️ 少了这一分支它会**静默**丢掉：下面几个分支都不认
                # tuple，于是循环直接进下一轮，`effective.tool_keys`
                # 永远是基线值 —— 而输出那段又恰恰是从 raw 里读的，
                # 于是"覆盖生效了"与"effective 里没有"同时成立。
                # as_state 的 docstring 已经记着"少一个键 = 静默无效"
                # 这一类坑，这里不再添一个。
                if isinstance(value, (list, tuple)) and all(
                        isinstance(item, str) for item in value
                ):
                    values[name] = tuple(value)

                continue

            if current is None or isinstance(current, str):

                if value is None or isinstance(value, str):
                    values[name] = value

                continue

        return replace(base, **values)


#: `ContextPolicy` 的字段名。
#:
#: 从 dataclass 反射而不是手写一份：手写清单一定会与字段漂移，而漂移的
#: 后果是"某个覆盖项永远不生效" —— 静默、且极难查。
_POLICY_FIELDS: tuple[str, ...] = tuple(
    item.name for item in dataclasses.fields(ContextPolicy)
)


def merge_policy_state(
        base: ContextPolicy,
        overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    基线策略 ⊕ 调用方覆盖项 -> 可被路径解析的 `context_policy` 结构。

    **不做类型转换**（规格 §九）：覆盖项里写 `"yes"` 就是 `"yes"`，
    于是 gate 判定为 INVALID 并让构建 FAILED。

    未知键**保留**：策略结构是 Processor 与调用方之间的开放契约
    （自定义 Processor 完全可以绑一个自己的开关），因此这里不裁剪 ——
    裁剪会让一个拼错的键从"gate MISSING"变成"静默无效"，后者更难查。
    """

    merged = dict(base.as_state())

    for key, value in (overrides or {}).items():
        merged[str(key)] = value

    return merged


@runtime_checkable
class ContextPolicyResolver(Protocol):
    """
    `(agent_type, context_builder_type) -> ContextPolicy`。

    实现只允许做**确定性映射**：不查网络、不调模型。它跑在每次
    上下文构建之前，任何不确定性都会变成"同一个 Agent 有时取 8 轮
    历史有时取 2 轮"，且无法从日志复现。
    """

    def resolve(
            self,
            *,
            agent_type: str,
            builder_type: ContextBuilderType,
    ) -> ContextPolicy:
        ...


class DefaultContextPolicyResolver:
    """
    默认 resolver：按 `context_builder_type` 给一份基线。

    为什么按 builder_type 而不是 agent_type：**builder_type 是
    Context Builder 自己的类型**（规格里那六个），它说的正是"上下文
    怎么长出来的"；而 agent_type 的语义属于 Agent Builder（规格 §六：
    不要让 Context Builder 去理解 Agent 选择）。等 Agent Builder
    落地后，注入一个按 agent_type 细分的 resolver 即可。

    ⚠️ 六个类型里有五个目前还没有实现（classifier / compaction /
    memory / sub_agent / planner），它们的基线策略仍然给出来：
    "这个类型该用什么策略"是**声明**，与"有没有实现"是两件事。
    真正取用它们要等对应 builder 落地。
    """

    #: 各 builder_type 的基线覆盖项（空 = 用 ContextPolicy 的默认值）。
    #:
    #: 默认空是**刻意的**：默认值已经在 ContextPolicy 里，多一份"等于
    #: 默认值的表"只会成为第二个需要同步的地方。子类或部署要差异化
    #: 时覆写它即可。
    BASELINES: Mapping[ContextBuilderType, Mapping[str, Any]] = {}

    def resolve(
            self,
            *,
            agent_type: str,
            builder_type: ContextBuilderType,
    ) -> ContextPolicy:

        base = ContextPolicy(source=f"default:{builder_type.value}")

        overrides = self.baselines().get(builder_type)

        if overrides:
            return base.with_(**overrides)

        # classifier 那种"窄输入"的类型：默认不带历史 / 记忆 / 知识 / 工具。
        # 这与 context/__init__.py 对六个类型的说明一致 —— 它是**类型**
        # 的性质，不是某个调用的选择。
        if builder_type is ContextBuilderType.CLASSIFIER:
            return base.with_(
                include_history=False,
                include_summary=False,
                include_user_profile=False,
                include_memory=False,
                include_retrieved_context=False,
                include_capabilities=False,
                allow_compaction=False,
                min_recent_turns=0,
                max_recent_turns=0,
                max_history_tokens=0,
            )

        return base

    @classmethod
    def baselines(cls) -> Mapping[ContextBuilderType, Mapping[str, Any]]:
        """可覆写的基线表（默认空 —— 默认值已经在 ContextPolicy 里）。"""

        return cls.BASELINES
