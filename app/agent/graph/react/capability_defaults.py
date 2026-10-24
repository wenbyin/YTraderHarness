"""主 agent 的能力暴露策略：全局配置 + 模式叠加。

这个文件是策略表，不是执行逻辑。有两张表：

    MODEL_CAPABILITIES      agent / duty -> 总是暴露的能力（全局）
    DEFAULT_CAPABILITIES    执行模式      -> 系统级追加（plan 加运行时编排）

最终暴露：

    effective = MODEL_CAPABILITIES[agent_key]       # 全局 default
              ∪ DEFAULT_CAPABILITIES[mode]          # 模式 default
              ∪ classification.capability_needs     # 本次 input 识别

`agent_key` 是 `state.duty_key`（例如 `general`）；`mode` 由
`resolve_mode` 从 plan_status / plan_enter_requested / execution_plan
推出来。

⚠️ 两类 default 是“地板”，分类器识别是“增量”
-----------------------------------------
`MODEL_CAPABILITIES` 与 `DEFAULT_CAPABILITIES` 是**总是可见**的基础
集合；分类器这次识别出的 `capability_needs` 默认**并入**它们（见
`ALLOW_CLASSIFIER_CAPABILITIES`）。分类器输出在 `_parse` 里已经被
限制到候选 code（认不出的落 `MODULE_OTHER`），不会突破能力目录；
真正的权限仍在工具的 checkauth / approval 链上。

把 `ALLOW_CLASSIFIER_CAPABILITIES` 置 False 可回到“只信全局配置”
的严格模式：`requested` 只进 `clipped` 供观测。

⚠️ 这里的值是 **capability code**，不是 tool_key
-----------------------------------------------
    AGENT_RUNTIME        能力树上的一个节点
    agent.create_team    某条能力下挂着的工具

两者之间的映射由 `tool_definition_classification` 维护。这里只决定
「暴露哪些能力」，不直接决定「给模型看哪些工具」。

⚠️ 三种模式
----------
    direct   直接回答 / 单步执行（execution_plan=other 或判不出来）
    react    ReAct 主循环（execution_plan=react）
    plan     计划模式（本次请求进入，或已经在 plan 里）
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "ALLOW_CLASSIFIER_CAPABILITIES",
    "CapabilityExposure",
    "DEFAULT_CAPABILITIES",
    "DefaultCapabilities",
    "MODEL_CAPABILITIES",
    "allowed_capability_codes",
    "default_capability_codes",
    "parent_codes_of",
    "resolve_capability_exposure",
    "resolve_mode",
]


@dataclass(frozen=True, slots=True)
class DefaultCapabilities:
    """
    一种执行模式的默认能力。

    codes
        显式列出的 capability code。

    subtrees
        以这些 code 为根，**递归包含根与所有子孙**（见
        `include_subtree_root`）。适合「整棵运行时能力都可见」这种规则。

    include_subtree_root
        subtrees 展开时是否包含根节点自己。默认包含；如果只想要严格的
        「以下」（不含根），改成 False。
    """

    codes: tuple[str, ...] = ()
    subtrees: tuple[str, ...] = ()
    include_subtree_root: bool = True


#: 默认能力表 —— **要调整策略就改这里**。
#:
#: direct / react 默认留空：
#:     分类器选中的能力仍然生效，只是不给模式级的额外能力。
#:     三个常驻控制工具（enter_plan_mode / exit_plan_mode /
#:     ask_user_question）由 `plan_state.plan_tool_keys` 单独加，
#:     不需要在这里重复。
#:
#: plan 默认暴露 AGENT_RUNTIME 整棵子树：
#:     AGENT_RUNTIME
#:       ├─ AGENT_RUNTIME_ORCHESTRATION   enter / exit
#:       ├─ AGENT_INTERRUPT               ask_user_question
#:       ├─ AGENT_TASK                    task 控制工具
#:       ├─ AGENT_SUB_AGENT               create_team
#:       └─ AGENT_SKILL                   skill 相关工具
#:     新增一条挂在 AGENT_RUNTIME 下的能力，会自动进入 plan 的默认集合。
#:
#: planner 子图复用同一份规则：它与 plan 模式面对的是同一棵运行时
#: 能力树。**共用同一个不可变实例**，避免两处策略各自漂移。
_RUNTIME_DEFAULTS = DefaultCapabilities(subtrees=("AGENT_RUNTIME",))

DEFAULT_CAPABILITIES: Mapping[str, DefaultCapabilities] = {
    "direct": DefaultCapabilities(),
    "react": DefaultCapabilities(),
    "plan": _RUNTIME_DEFAULTS,
    "planner": _RUNTIME_DEFAULTS,
}


#: 每个 agent / 模型人格**暴露**哪些能力（全局配置）。
#:
#: key 是 `state.duty_key`（建运行那一刻定下的业务 agent 身份，例如
#: `general`）；`"*"` 是未配置时的兜底。
#:
#: 最终暴露 = 这里的全局能力 ∪ 模式默认能力（DEFAULT_CAPABILITIES）。
#:
#: 也就是说：
#:
#:     general 在 direct / react 下     -> TOOL + DATETIME_TOOL
#:     general 在 plan 下               -> TOOL + DATETIME_TOOL
#:                                          + AGENT_RUNTIME 整棵子树
#:
#: entry 里每一项的写法：
#:
#:     "TOOL"            -> 只要 TOOL 自己（不自动带子能力）
#:     "AGENT_RUNTIME/*" -> AGENT_RUNTIME 及其所有子孙
#:
#: 为什么要区分：TOOL 下面还挂着 DATETIME_TOOL / WEB_TOOL / TEST_TOOL；
#: 写 "TOOL" 通常只想给基础工具，而不是把整棵工具树都给出去。
#: 需要整棵子树时显式写 `/*`。
#:
#: 大小写不敏感（`tool` / `TOOL` 等价），方便手写；最终仍以数据库里
#: 的原样 code 解析工具。
#:
#: 改这里的典型场景：
#:     新 agent                 -> 加一条 "stock_research": ("STOCK", ...)
#:     收回某类工具             -> 从对应条目里删掉那个 code
#:     plan 额外可见的编排能力   -> 改下面的 DEFAULT_CAPABILITIES
MODEL_CAPABILITIES: Mapping[str, tuple[str, ...]] = {
    # general 模型默认看到：基础工具 + 时间 + 能力目录查询。
    #
    # `CAPABILITY` 是 `capability.list_capabilities` 所在的能力 code：
    # 让主 agent 能自己查“平台有哪些能力”，而不是靠系统提示词列死。
    "general": (
        "TOOL",
        "DATETIME_TOOL",
        "CAPABILITY",
    ),
    # planner 子图：默认能力来自 DEFAULT_CAPABILITIES["planner"]
    # （AGENT_RUNTIME 子树），全局只额外给一个能力目录查询入口。
    "planner": ("CAPABILITY",),
    # 没配置的 agent：fail-closed，不给任何能力工具。
    # plan 模式仍会按 DEFAULT_CAPABILITIES 追加 AGENT_RUNTIME 子树；
    # 如果不希望未配置的 agent 有编排能力，在下面给它加显式条目并把
    # plan 的子集收窄。
    "*": (),
}


#: 分类器的 `capability_needs` 是否**追加**到暴露集合。
#:
#: 默认 True：两类 default 是基础集合，本次 input 识别出的能力是增量
#: （effective = defaults ∪ requested）。分类器输出已在 `_parse` 里
#: 限制到候选 code，工具执行仍走 checkauth / approval。
#:
#: 置 False 回到严格模式：requested 只进 `clipped` 供观测，不参与暴露。
ALLOW_CLASSIFIER_CAPABILITIES = True


@dataclass(frozen=True, slots=True)
class CapabilityExposure:
    """一次能力暴露策略解析的结果（给观测与 notes 用）。"""

    agent_key: str
    allowed: tuple[str, ...]
    effective: tuple[str, ...]
    clipped: tuple[str, ...]


def resolve_mode(
        *,
        execution_plan: str | None,
        plan_status: str | None,
        plan_enter_requested: bool,
) -> str:
    """
    本次运行该用哪一档默认能力。

    优先级：

        本次请求进入 plan，或已经处于 plan  -> plan
        execution_plan == react             -> react
        其余（other / 空 / 判不出来）       -> direct

    `plan_status` 优先于分类器的 execution_plan：状态是已经生效的事实，
    分类只是提议。这与 `routing._route_for_plan` 的权威来源一致。
    """

    status = str(plan_status or "").strip().lower()

    if plan_enter_requested or status == "plan":
        return "plan"

    if str(execution_plan or "").strip().lower() == "react":
        return "react"

    return "direct"


def default_capability_codes(
        mode: str,
        *,
        codes: Sequence[str],
        parent_of: Mapping[str, str | None],
) -> list[str]:
    """
    展开某个模式的默认能力 code。

    Parameters
    ----------
    mode
        `resolve_mode` 的返回值；未知模式返回空列表（不抛 —— 策略表
        里多写一个没人用的键，不该让整轮运行失败）。
    codes
        当前目录里**可用**的 capability code（通常是 ACTIVE 的那些）。
        不在这个集合里的 code 会被忽略。
    parent_of
        `code -> 父 code`。用来递归展开 subtrees。

    返回顺序：先显式 codes，再按 subtrees 的深度优先顺序；已去重。
    """

    rule = DEFAULT_CAPABILITIES.get(mode)

    if rule is None:
        return []

    # 保持调用方给的顺序（能力目录的 sort_order），不要用 set 迭代：
    # 工具顺序会影响 bind_tools 的提示词布局，也影响调试时能不能
    # 复现一次运行。
    ordered = [str(code) for code in codes]
    available = set(ordered)

    result: list[str] = [
        code for code in rule.codes if code in available
    ]

    for root in rule.subtrees:
        result.extend(
            _subtree(
                root,
                ordered=ordered,
                available=available,
                parent_of=parent_of,
                include_root=rule.include_subtree_root,
            )
        )

    return list(dict.fromkeys(result))


def parent_codes_of(
        items: Sequence[Any],
) -> dict[str, str | None]:
    """
    `code -> 父 code`，给默认能力策略展开子树用。

    接受任何带 `code` / `id` / `parent_id` 属性的对象：主图的
    `_ModuleCandidate` 与 CapabilityService 的 schema 都满足这个形状。

    父节点不在清单里（被停用 / 被删除）时记 None —— 那棵子树到这个
    节点就断了，不会把孙节点误挂到更上层。
    """

    by_id = {
        str(item.id): str(item.code)
        for item in items
        if getattr(item, "id", None) is not None
    }

    return {
        str(item.code): (
            by_id.get(str(item.parent_id))
            if getattr(item, "parent_id", None) is not None
            else None
        )
        for item in items
        if getattr(item, "code", None)
    }


def allowed_capability_codes(
        agent_key: str | None,
        *,
        codes: Sequence[str],
        parent_of: Mapping[str, str | None],
) -> list[str]:
    """
    这个 agent / 模型人格**允许**看到的能力（已展开子树）。

    key 先精确匹配 `MODEL_CAPABILITIES`，再小写匹配，最后落到 `"*"`。
    配置里的 code 大小写不敏感；解析结果始终是目录里的原样 code。
    """

    ordered = [str(code) for code in codes]
    available = set(ordered)

    by_upper: dict[str, list[str]] = {}
    for code in ordered:
        by_upper.setdefault(code.upper(), []).append(code)

    result: list[str] = []

    for raw in _model_roots(agent_key):

        text = str(raw or "").strip()

        if not text:
            continue

        # `CODE/*` = 根 + 所有子孙；普通 code = 只要它自己。
        subtree = text.endswith("/*")
        candidate = text[:-2] if subtree else text

        code = _canonical_code(
            candidate,
            available=available,
            by_upper=by_upper,
        )

        if code is None:
            # 配置里写了一个目录里不存在的 code：跳过，不中断整轮。
            # 它会在 span 的 allowed 数量里体现为少一项。
            continue

        if subtree:
            result.extend(
                _subtree(
                    code,
                    ordered=ordered,
                    available=available,
                    parent_of=parent_of,
                    include_root=True,
                )
            )
        else:
            result.append(code)

    return list(dict.fromkeys(result))


def resolve_capability_exposure(
        agent_key: str | None,
        *,
        mode_defaults: Sequence[str],
        requested: Sequence[str],
        codes: Sequence[str],
        parent_of: Mapping[str, str | None],
) -> CapabilityExposure:
    """
    最终暴露集合 = 该 agent 的全局能力 ∪ 模式默认能力。

    `mode_defaults` 是系统级追加（目前只有 plan 的 AGENT_RUNTIME 子树），
    不受全局配置裁剪 —— 它是模式本身的性质，不是某个模型的选择。

    `requested` 是分类器选中的能力（本次 input 识别的增量）。默认
    **并入**最终集合；`ALLOW_CLASSIFIER_CAPABILITIES=False` 时才只在
    `clipped` 里报告「分类器选了、但被全局策略忽略了」。
    """

    key = str(agent_key or "").strip()

    allowed = allowed_capability_codes(
        key,
        codes=codes,
        parent_of=parent_of,
    )

    effective = list(dict.fromkeys([
        *mode_defaults,
        *allowed,
    ]))

    unique_requested = list(dict.fromkeys(
        str(code) for code in requested
    ))

    if ALLOW_CLASSIFIER_CAPABILITIES:
        effective = list(dict.fromkeys([
            *effective,
            *unique_requested,
        ]))
        clipped: list[str] = []
    else:
        effective_set = set(effective)
        clipped = [
            code for code in unique_requested
            if code not in effective_set
        ]

    return CapabilityExposure(
        agent_key=key,
        allowed=tuple(allowed),
        effective=tuple(effective),
        clipped=tuple(clipped),
    )


def _model_roots(agent_key: str | None) -> Sequence[str]:
    """取这个 agent 的配置项；精确 key -> 小写 key -> `"*"`。"""

    key = str(agent_key or "").strip()

    if key in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[key]

    lower = key.lower()

    if lower in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[lower]

    return MODEL_CAPABILITIES.get("*", ())


def _canonical_code(
        raw: object,
        *,
        available: set[str],
        by_upper: Mapping[str, Sequence[str]],
) -> str | None:
    """
    配置里的 code -> 目录里的原样 code。

    先精确匹配；再按大写找唯一匹配。大小写有歧义（目录里同时存在
    `Tool` 与 `TOOL`）时不猜，返回 None。
    """

    text = str(raw or "").strip()

    if not text:
        return None

    if text in available:
        return text

    matches = by_upper.get(text.upper(), ())

    return matches[0] if len(matches) == 1 else None


def _subtree(
        root: str,
        *,
        ordered: Sequence[str],
        available: set[str],
        parent_of: Mapping[str, str | None],
        include_root: bool,
) -> list[str]:
    """
    从 root 展开整棵子树。

    根不存在时返回空列表：能力目录没同步 / 根被停用，都不该让分类
    整轮失败；调用方会照常得到「分类器选中的能力」。
    """

    if root not in available:
        return []

    children: dict[str, list[str]] = {}

    for code in ordered:
        parent = parent_of.get(code)

        if parent:
            children.setdefault(parent, []).append(code)

    ordered: list[str] = []
    seen: set[str] = set()
    stack: list[str] = [root]

    # 显式栈而不是递归：能力树很深时也不会撞 Python 的递归上限；
    # seen 同时兜住脏数据里的环（A 的 parent 是 B，B 的 parent 是 A）。
    while stack:

        code = stack.pop()

        if code in seen:
            continue

        seen.add(code)

        if code != root or include_root:
            ordered.append(code)

        # 逆序入栈，出栈时保持目录原顺序。
        for child in reversed(children.get(code, ())):
            stack.append(child)

    return ordered
