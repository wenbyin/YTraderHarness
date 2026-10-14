"""
`model_role=classification` 的提示词：Agent Builder 的任务分诊。

这份提示词从 app/agent/graph/react/agent_builder_node.py 搬过来
--------------------------------------------------------------
文本一字未改，搬的只是**位置** —— 目的是让"提示词住在节点里"这件事
不再发生（见 app/agent/prompt/__init__.py）。搬过来时一并带走了三样：
`_BEHAVIORS` / `_PLANS` 两个取值域、候选清单的排版规则、描述截断上限 ——
它们都是提示词的一部分，留在节点里就等于提示词还散着。

取值域为什么由枚举派生
--------------------
behavior / execution_plan 的取值域写在 `app/agent/state/agent_state.py`
的枚举里，这里是**派生**出来的，不是抄一份。抄一份就会漂移：枚举加了
一个行为、提示词里没加，模型就永远判不出那个值，而症状是"某类输入总是
落 other"—— 看起来像模型能力问题。

⚠️ 提示词里出现 `$` 要写成 `$$`
------------------------------
本文件用 `string.Template`（而不是 f-string / `str.format`）拼装，因为
提示词正文里随时可能出现 JSON 示例的花括号（`str.format` 会当场炸，
且是"哪天有人往提示词里贴了个例子"才炸）。代价是 `$` 成了保留字符 ——
正文里真要写 `$` 时必须写成 `$$`。
"""

from __future__ import annotations

from collections.abc import Sequence
from string import Template
from typing import Protocol

from app.agent.model import ModelRole
from app.agent.state import MODULE_OTHER, BehaviorCategory, ExecutionPlan

from ..contracts import PromptDefinition

__all__ = [
    "BEHAVIORS",
    "CandidateLike",
    "DESCRIPTION_LIMIT",
    "PLANS",
    "PROMPT",
    "render_capabilities",
]


#: 分类提示词里的取值域。由枚举派生（见模块 docstring）。
BEHAVIORS = "、".join(m.value for m in BehaviorCategory)
PLANS = "、".join(m.value for m in ExecutionPlan)


#: 单条能力描述进提示词时的长度上限（字符）。
#:
#: 只是"读起来别太长"的取舍，不是预算保护 —— 分类链路没有预算裁剪
#: （CLASSIFIER 的策略基线把 TOKEN_BUDGETER 关掉了，见
#: context/builder/classifier.py）。截断会补一个 `…`，所以读到的人看得出
#: 它被截过，而不是以为原文就这么短。
DESCRIPTION_LIMIT = 80


class CandidateLike(Protocol):
    """
    候选能力在这份提示词眼里的样子。

    只读三项，因此**不认识**候选是从哪来的 —— 节点那边的
    `_ModuleCandidate` 是"从能力表读回来的形状"（还有别的字段的余地），
    这里是"要排版的那三项"。两边各自演化，改一边不会牵动另一边。
    """

    code: str

    name: str

    description: str | None


def render_capabilities(candidates: Sequence[CandidateLike]) -> str:
    """
    候选能力清单 -> 提示词里的那一段。

    候选为**空**时明确告诉模型"没有候选可用"，让它一律落 other ——
    比给一个空列表然后指望模型不要瞎编要好。

    ⚠️ 每项带上**名称与描述**，不只是 code
    -------------------------------------
    code 是给机器看的稳定标识（`retrieval` / `doc.retrieve` 这类），对
    模型来说几乎没有语义 —— 只给 code 时它是在做字符串相似度匹配，而
    不是"这句话属于哪块能力"。名称与描述才是判断依据。

    **对 `capability_needs` 这一步尤其如此**：模型要判"这句话与哪条能力
    对得上"，而 code（`CAPABILITY` / `doc.retrieve` 这类）本身几乎没有
    语义 —— 只给 code 时它是在做字符串相似度匹配。所以"每项带名称与描述"
    不是可读性优化，是这一步能否成立的前提。

    代价是提示词长度随能力条目线性增长，而这条分类链路**没有预算裁剪**。
    因此这里只对**单条描述**设上限（见 `DESCRIPTION_LIMIT`），**不**对
    条目数设上限：少给一条就是静默地让模型选不到它，而描述太长只是这一条
    不好读 —— 两种代价不对等。
    """

    if not candidates:
        return (
            "当前没有可用的能力清单。"
            f"capability_needs 一律留空（或填 [{MODULE_OTHER}]）。"
        )

    return (
        "可选能力清单（来自系统已登记的能力目录）：\n"
        + "\n".join(_render_candidate(c) for c in candidates)
        + "\ncapability_needs 只填上面每项**开头的那个标识（code）**，"
        "不要填名称；可多选，都不合适就填 "
        f"{MODULE_OTHER}，确实不需要任何能力时可以留空。"
    )


def _render_candidate(candidate: CandidateLike) -> str:
    """
    一条候选 -> 提示词里的一行。

         - retrieval：检索 —— 从知识库与文档里找相关内容

    code 放最前、用 `：` 与名称隔开，是为了让"要回填的是哪一段"一眼可辨
    （上面那句"只填每项开头的那个标识（code）"指着它）。描述缺失时整段
    省掉，而不是留一个空的破折号。
    """

    line = f"- {candidate.code}：{candidate.name}"

    description = (candidate.description or "").strip()

    if not description:
        return line

    if len(description) > DESCRIPTION_LIMIT:
        description = description[:DESCRIPTION_LIMIT] + "…"

    return f"{line} —— {description}"


#: 正文。`$` 为保留字符，见模块 docstring。
_TEXT = Template(
    "你是一个任务分诊器。读用户最新的一句话，判断它属于什么任务，"
    "只输出结构化结果，不要回答用户的问题本身。\n\n"
    "behavior：用户想干什么。取值：$behaviors。\n"
    "$capabilities\n"
    "execution_plan：怎么执行。取值：$plans。\n"
    "    direct 直接回答即可；react 需要调用工具；"
    "plan 需要先制定计划再执行。\n"
    "capability_config：上面这些能力的初步配置建议（可为空对象）。\n\n"
    "$memory\n\n"
    "behavior 与 execution_plan 拿不准就填 other —— "
    "猜一个近似的类目会让下游按错误前提工作。"
)


#: `retrieve_l0` / `retrieve_l1` 那一段的正文（判据）。
#:
#: ⚠️ 四条触发 + 两条排除，取自 Tencent `memory_injection.md`（仓库根目录）
#: 里给主模型看的那套规则 —— 不同之处在于**判它的地方**：那边是主模型
#: 自己看着规则现判要调哪个工具，这里是分诊这一步就先判掉，判不出 /
#: 不该判的留给主模型用工具补。
#:
#: ⚠️ 为什么不照抄原文的"线索 -> 层"映射
#: -----------------------------------
#: 原文里**同一条线索会指向不同的层**：规则①的「上次」判给
#: `tdai_conversation_search`（L0），规则④的「上次方案是啥」判给
#: `tdai_memory_search`（L1）。原文那样写不出问题，因为它讲的是"调哪个
#: 工具"，而且允许两个都调（那边有"每轮合计 ≤ 3 次"的预算）。
#:
#: 压成两个布尔位之后这个重叠就变成了**二选一**：模型看到「上次」两处
#: 都沾边，只能猜一边。实测「你上次推荐我练什么曲子」被判成 l1=true、
#: l0=false —— 而它问的正是**用户具体说过的话**（L0 的定义）。
#:
#: 因此这里把原文隐含的那条判据**说破**：不是"看到哪个词就填哪个"，
#: 而是**要的东西在哪一层**。线索只说明"历史相关"，层由"要原话还是要
#: 提炼"决定；分不清就两个都填，而不是硬猜一个。
#:
#: ⚠️ 原版规则④写着「关键词化后」再检索 —— **本仓库不关键词化**，查询串
#: 就是用户原话（见 processors/memory.py）。所以这里只说"要不要召回"，
#: 不要把那半句抄进来：抄了模型会以为本该由它改写查询串。
#:
#: ⚠️ 原版还有一条排除：「已经在 L3 画像段落里直接看到答案」。
#: **这里判不了** —— 分诊跑在上下文构建之前，画像那时还没取出来。
#: 那条留给主模型自己判断（它看得到画像段，也挂着检索工具）。
_MEMORY_RULES = (
    "retrieve_l0 / retrieve_l1：这一轮要不要**预先**召回长期记忆。\n"
    "    系统每轮已经自带用户画像与情境索引，这两项说的是要不要再往下\n"
    "    检索更细的两层：\n"
    "        retrieve_l0  L0 原始对话 —— 用户具体说过的话\n"
    "        retrieve_l1  L1 原子记忆 —— 已提炼的事实 / 偏好 / 长期要求\n"
    "    **怎么判**：先看这句话要的答案在**哪一层** —— 要「原话」填 l0，"
    "要「提炼过的事实」填 l1，两层都要就**两个都填 true**。\n"
    "    下面这些线索说明「这句话与历史有关」，命中任一条就该往下判：\n"
    "        · 提及历史 / 过去 / 之前 ——「我之前说过」「我告诉过你」「上次」"
    "「你还记不记得」「我们聊过」「之前那个」\n"
    "        · 问及自己的身份 / 偏好 / 习惯 / 约束 ——「我叫什么」"
    "「我的名字」「我喜欢」「我不喜欢」「我的团队」「我常用」「我不允许」"
    "\n"
    "        · 要求回忆或查找 ——「回忆一下」「想起」「找出」"
    "「有没有关于 X 的记录」「查我们之前」\n"
    "        · 答案强依赖历史里发生过什么 ——「那个 bug 我们怎么修的」"
    "「上次方案是啥」「我们的约定是什么」\n"
    "    ⚠️ 线索之间会重叠（「上次推荐我练什么曲子」同时像「提及历史」与"
    "「答案依赖历史事实」）—— 那不是要你二选一，**两层都可能要**："
    "问「上次说过什么」要的是**原话**（l0），问「我们的约定是什么」要的是"
    "**提炼**（l1）。真分不清就两个都填 true，漏一层会让这一轮少一半依据。\n"
    "    以下情形**不要**召回，填 false：\n"
    "        · 与用户历史无关的请求（「你是谁」「写个脚本」「帮我改代码」）\n"
    "        · 当前这句话本身已经能回答\n"
    "    两个字段一律填 true 或 false —— 不要留空、不要填别的值。"
    "只有**确实与历史无关**时才填 false：多召回一次要付一次向量检索的"
    "代价，但漏召回的这一轮主模型只能自己再调工具补。"
)


def _render(*, candidates: Sequence[CandidateLike]) -> str:
    """
    参数名 `candidates` 就是对调用方的契约（见 contracts.PromptDefinition）。

    取值域那两个变量由本模块自己填（它们是枚举的派生物，调用方不该知道
    也不该传），调用方只需要给候选清单。记忆那一段（`_MEMORY_RULES`）
    同样是本模块自己的常量 —— 它是**判据**，随提示词一起演化，不是
    调用方每次能变的输入。
    """

    return _TEXT.substitute(
        behaviors=BEHAVIORS,
        plans=PLANS,
        capabilities=render_capabilities(candidates),
        memory=_MEMORY_RULES,
    )


PROMPT = PromptDefinition(
    key=ModelRole.CLASSIFICATION,
    description=(
        "Agent Builder 的分类调用：判行为 / 执行方案 / 能力需求，"
        "只输出结构化结果。候选能力清单逐次拼进来。"
    ),
    render=_render,
)
