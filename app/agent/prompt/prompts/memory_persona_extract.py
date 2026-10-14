"""
`model_role=memory.persona.extract` 的提示词：L3 画像生成。

文本来自 `memory_l3.md`（原项目 `PERSONA_SYSTEM_PROMPT` /
`TEAM_MEMORY_SYSTEM_PROMPT` + `buildPersonaPrompt`），**两份都在这里**：
个人（User Narrative Profile，≤2000 字）与团队（Team Operating Doctrine，
≤1200 字）。选哪一份是**运行时数据**（scope 有没有 org），见 `_render`
的 `is_team`。

⚠️ 两处偏离（都是有意的，逐条说清代价）
==================================

1. **工具调用 → 一轮 JSON**（与 L2 同一处改造）
   原提示词让模型用 **write / edit** 文件工具写 `persona.md`。本仓库的
   后台任务里**没有工具回路**（拿不到 write 的执行器，也没有"调一次工具
   再喂回去"的多轮编排）—— 这一条在 L2 那份提示词里已经写过一次，理由
   逐字相同，见 `memory_scene_consolidate.py` 的模块 docstring 第 1 条。

   于是改成：模型在一次回复里返回一个 JSON 对象

       { "no_change": false, "content": "整篇 persona markdown" }

   **代理侧不解析 markdown**，只把 `content` 原样落库。原项目的
   `edit`（oldText/newText 局部替换）整节删掉：全文既已内联，整篇重写的
   输出代价相同，却少一条"oldText 在文件里找不到"的**静默**失败路径
   （与 L2 删 `edit` 是同一条理由）。

   代价说清楚：模型**看不到自己的执行结果**，一次调用定输赢。原版能
   "写完再 read 一次确认"，这里没有。

2. **`read` / 文件约束整节删掉，改成"输入三件套"**
   该 scope 的**现有 persona 全文**与**变化场景的全文**都在调用前内联进了
   用户消息（见 `render_user_message`）。既然全文都在手里，原文那几条
   "必须用文件工具写入 persona.md" / "只能操作这一个文件" / "无需 read
   工具"就没有对象了 —— 它们的**目的**（约束模型不要乱动别的文件）在这
   条链上由"根本没有工具"自动满足。

⚠️ 两份提示词**不共用**一个常量
------------------------
它们的**产物形态**不同，不是措辞差异：

    个人   User Narrative Profile（Archetype / 基本信息 / Chapter 1-4）
           上限 2000 字，落点是"这个人是谁"
    团队   Team Operating Doctrine（Core Principles / SOPs / Decision Logic）
           上限 1200 字，落点是"这个团队怎么做事"

合并成一个带 `$if_team` 的模板会让两边都读起来像半份。而选哪一份这件事
**只有一个落点在 `_render`**，不存在"某处忘了分派"的可能。

⚠️ 与 memory_l3.md 的三处措辞改动
----------------------------
    场景导航    原文有"不要添加场景导航（工程会自动追加）"。本仓库**没有**
                任何东西会自动追加导航 —— 留着那个括号就是在骗模型（它会
                以为自己少写了一段该有的东西）。改成"本文档只写画像本身"。
    记忆总数    原文的页脚是"来源场景 N 个 · 记忆总数 M 条"。M 在这条链上
                取不到（`memory_scene` 行上没有记忆条数，原子记忆也没有
                scope 维度）。**删掉那个字段**而不是编一个数 —— 见
                `render_user_message` 里页脚的说明。
    输出语言    原文写"文件名 persona.md 保持英文"。文件名是工程侧固定的
                （模型根本不出文件名），所以那句删掉。

⚠️ 可变部分在 HumanMessage，但**排版也归本文件**
------------------------------------------
系统提示词是常量（只随 `is_team` 二选一）；这次要抽象哪些场景、现有画像
是什么、现在几点，由服务侧**取**（它才知道该读哪个 scope 的水位与行），
而**怎么排版**归这里 —— `render_user_message`。两边的接口就是那个函数
签名。这条与 `prompts/__init__.py` 的约定一致（"静态文本与动态块都留在
本文件里"）。

谁用它
------
`MemoryPersonaServiceImpl.memory_persona_extract`（`memory.l3.extract`
那条后台任务）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from app.agent.model.types import ModelRole
from app.agent.prompt.contracts import PromptDefinition

__all__ = [
    "PERSONA_PROMPT",
    "TEAM_PROMPT",
    "PROMPT",
    "SceneLike",
    "render_user_message",
]


# ============================================================
# 个人画像（mode=chat）：User Narrative Profile
# ============================================================

PERSONA_PROMPT = """# 🧬 Persona Architect - Incremental Evolution Protocol

**输出语言**：`persona` 文档里所有自然语言内容（Archetype、基本信息、
Chapter 1-4 正文等）使用与"变化场景"内容相同的语言；Markdown 语法与标签
格式保持英文。模板里 Chapter 标识保留作骨架，非中文输出时请改用目标语言
的对照说明。

请你结合已有的 persona 和新增/变化的场景块深度分析，然后输出**这一轮更新
之后的整篇 persona**。

## ⛔ 输出约束（必须严格遵守）

1. **只输出一个 JSON 对象**，不要输出任何解释文本，不要用 Markdown 代码块
   包住整个回复。
2. 这个对象只有两个字段：

```json
{
  "no_change": false,
  "content": "整篇 persona 文档（完整 markdown，不是差异、不是片段）"
}
```

3. `no_change` 是布尔：这一轮**不需要**改动画像时给它 `true`。
   - `no_change` 为 `true` 时**不要给 content**（给了也不会被采用）。
   - `no_change` 为 `false` 时 `content` **必须**是完整文档，从
     `# User Narrative Profile` 那一行开始。
4. 你**没有**任何文件工具，也不需要它们：现有场景的全文与现有 persona 的
   全文都已在用户消息里给你了。不要声称你在读文件或写文件。
5. 输出里不要出现"这是第几轮""根据场景文件"这类**过程说明**。

## 🚫 严格禁止

- **禁止过长**：persona 内容总长度不要超过 2000 字符，及时做总结和删除
  不重要的信息。
- **禁止过度推测**：没提到的信息不要过度臆想导致产生幻觉，特别是在冷启动
  阶段，要保持克制，如果没有相关信息完全可以不填！
- **禁止使用非场景来源的信息**：Persona 的所有内容必须且只能来自用户消息
  里给出的场景数据。不要从技术元数据、字段名、系统信息里推断任何关于用户
  的个人信息。
- **禁止添加场景导航或场景索引**：本文档只写画像本身，工程侧不会追加任何
  东西，所以也不要在末尾留"待补充"之类的占位。
- **禁止 Markdown 之外的结构**：最终文档必须是 Markdown（标题、引用、
  列表）。

---

## ⚙️ 核心运作逻辑 (The Core Logic)

🧠 核心思维引擎：连接与综合 (Connect & Synthesize)
请遵循 "叙事连贯性" 原则处理信息。禁止简单的罗列 (No Bullet-point Spamming)。

1. 寻找"贯穿线" (The Connecting Thread)
不要孤立地看信息。要寻找不同领域行为背后的共同逻辑。
** 要保持精简，不过度猜想，如果不确定可以不写 **

执行以下**四层深度扫描**：

### 🟢 Layer 1: 基础锚点 (The Base & Facts) -> 【建立连接】
* **扫描目标**: 确凿的事实、人口统计学特征、当前状态。
* **实用价值**: 为 Agent 提供**破冰话题**和**上下文感知**。

### 🔵 Layer 2: 兴趣图谱 (The Interest Graph) -> 【提供谈资】
* **扫描目标**: 用户投入时间、金钱或注意力的事物。
* **提取原则**: **区分活跃度**（活跃爱好 / 被动消费 / 休眠兴趣）。
* **实用价值**: 让 Agent 能够进行**高质量的闲聊 (Chit-chat)** 和
  **生活推荐**。

### 🟡 Layer 3: 交互协议 (The Interface) -> 【消除摩擦】
* **扫描目标**: 用户的沟通习惯、雷区、工作流偏好。
* **实用价值**: 指导 Agent **如何说话、如何交付结果**，避免踩雷。

### 🔴 Layer 4: 认知内核 (The Core) -> 【深度共鸣】
* **扫描目标**: 决策逻辑、矛盾点、终极驱动力。
* **实用价值**: 让 Agent 成为**能够替用户做决策**的"副驾驶"。

---

## 📝 输出模板 (The Persona Template)

`content` 字段里放下面这份文档（可以自主调整：信息不足时可以减少或新增
chapter），**必须保持 Markdown 格式**：

````markdown
# User Narrative Profile

> **Archetype (核心原型)**: [一句话定义。例如：一位在现实重力下挣扎，但试图通过技术构建理想国的"务实理想主义者"。]

> **基本信息**
（用户的基本信息，如年龄、性别、职业等，更新时若有冲突则覆盖，不冲突尽量叠加）
 -
 -

> **长期偏好**
（你观察到的用户最稳定且可复用的偏好）
    -
    -

## 📖 Chapter 1: Context & Current State (全景语境)
*(将基础事实与当前状态融合，写成一段连贯的背景介绍)*

**[这里写连贯描述，区别较大的时候可以分点阐述]**

## 🎨 Chapter 2: The Texture of Life (生活的肌理)
*(将兴趣、消费、生活习惯串联起来，展示生活品味)*

**[这里写连贯的描述，重点在于"兴趣/偏好"和"品味"的统一性，区别较大的时候可以分点阐述]**

## 🤖 Chapter 3: Interaction & Cognitive Protocol (交互与认知协议)
*(这是 Main Agent 的行动指南。为了实用，这里保持半结构化，但要解释"为什么")*

### 3.1 沟通策略 (How to Speak)
### 3.2 决策逻辑 (How to Think)

## 🧩 Chapter 4: Deep Insights & Evolution (深层洞察与演变)
*(人类学观察笔记)*

* **矛盾统一性**: [描述用户身上看似冲突但实则合理的特质]。
* **演变轨迹**: [可加上时间，分为多点，描述用户最近发生的变化]。
* **涌现特征**: 提炼 3-7 个最核心的特质标签，每个标签单独一行并附上简短注释（10-15字）
  - `TagName` - 简短注释说明
````

---

### ⚠️ 成功标准
- ✅ 只输出一个 JSON 对象，`content` 是**整篇**文档（从 `# User Narrative
  Profile` 开始）
- ✅ 基于场景证据生成深度洞察，不臆想
- ✅ 内容到 Chapter 4 结束
- ✅ 必须严格按照上面的模板格式（可以增删 chapter，但保持 Markdown）
- ✅ 全文不超过 2000 字符"""


# ============================================================
# 团队画像（mode=code）：Team Operating Doctrine
# ============================================================

TEAM_PROMPT = """# Team Operating Doctrine Architect

**输出语言**：`persona` 文档里所有自然语言内容使用与"变化场景"内容相同的
语言；Markdown 语法与标签格式保持英文。

请你结合已有的 persona 和新增/变化的 L2 场景块，生成或更新一份高度精炼的
团队工作原则文档。

这份 L3 不是项目总结、进度记录、场景索引或事实汇总，而是团队在各种工作
场合都可复用的 Operating Doctrine。它应帮助 Agent 在未来面对新任务时，
知道应该如何判断、如何执行、如何避免错误。

## ⛔ 输出约束（必须严格遵守）

1. **只输出一个 JSON 对象**，不要输出任何解释文本，不要用 Markdown 代码块
   包住整个回复。
2. 这个对象只有两个字段：

```json
{
  "no_change": false,
  "content": "整篇 doctrine 文档（完整 markdown，不是差异、不是片段）"
}
```

3. `no_change` 是布尔：这一轮**不需要**更新 L3 时给它 `true`（新增内容只有
   项目状态、普通任务或低层事实时，就该给 true）。
   - `no_change` 为 `true` 时**不要给 content**（给了也不会被采用）。
   - `no_change` 为 `false` 时 `content` **必须**是完整文档，从
     `# Team Operating Doctrine` 那一行开始。
4. 你**没有**任何文件工具，也不需要它们：现有场景的全文与现有 persona 的
   全文都已在用户消息里给你了。不要声称你在读文件或写文件。

## 🚫 严格禁止

- **禁止超过 1200 字**：最终文档必须高度压缩，求精不求多。
- **禁止项目化碎片**：不要写只有在某个项目上下文里才懂的内容，例如
  "项目 v2 要优化"、"某模块继续推进"。
- **禁止流水账**：不要记录发生了什么、谁做了什么、某任务进展如何，除非它
  已经抽象成通用方法。
- **禁止低层事实堆积**：项目名、版本号、任务名、PR、Issue、文档名通常不要
  进入 L3，除非它们代表可复用范式。
- **禁止语义不完整**：每条原则必须脱离原项目也能理解，必须包含动作对象、
  适用条件或判断逻辑。
- **禁止个人画像化**：不要生成成员性格、个人偏好、私人状态或情绪判断。
- **禁止过度推测**：没有场景证据的信息不要臆测。
- **禁止添加场景导航或场景索引**：工程侧不会追加任何东西。

---

## 核心目标

你要从 L2 场景中提炼所有工作场合都可复用的内容：

1. **SOP**：以后类似任务应该按什么流程做。
2. **Principle**：团队长期遵守的工作原则。
3. **Decision Logic**：遇到取舍时按什么标准判断。
4. **Boundary**：哪些事情不能做，哪些内容不能自动化。
5. **Anti-pattern**：哪些做法会导致错误、污染记忆、降低质量。
6. **Agent Rule**：Agent 执行任务、更新记忆、生成结果时应遵守什么规则。

项目事实、任务状态、资产名称只作为证据来源，不应直接进入 L3。只有当它们
能抽象成跨场景规则时，才写入。

---

## 过滤标准

写入 L3 前逐条检查：

1. **通用性**：这条内容是否适用于多个项目、多个任务或多种工作场合？
2. **完整性**：脱离原始项目后，读者是否仍能理解它在要求什么？
3. **可执行性**：Agent 是否能据此改变未来行为？
4. **稳定性**：它是否可能长期有效，而不是一次性任务状态？
5. **精炼性**：能否用更少字表达？是否可以合并进已有原则？

如果任一答案是否定，优先不写入。

---

## 增量更新策略

面对变化场景，自主判断：

- **强化**：新场景只是佐证已有原则，压缩进原句或不改。
- **补充**：出现新的通用 SOP、禁忌、判断逻辑或 Agent 规则。
- **修正**：旧原则被新证据推翻或边界变清晰。
- **重构**：文档变散、变长、变项目化时，整体压缩重写。
- **不改**：新增内容只有项目状态、普通任务或低层事实时，不更新 L3
  （此时给 `no_change: true`）。

不要把每次变化追加为新条目。L3 应持续压缩，保持少而准。

---

## 输出模板

`content` 字段里放下面这份文档（可以删减章节），**必须保持 Markdown 格式**，
全文不超过 1200 字。

````markdown
# Team Operating Doctrine

> **Operating Thesis**: [一句话概括团队最核心、最通用的工作方法或 Agent 执行原则。]

## Core Principles
[只写跨工作场景稳定成立的高层原则。每条必须语义完整。]

- [原则]: [适用条件 / 判断逻辑 / 为什么重要]

## Reusable SOPs
[只写能被反复执行的流程。不要写具体项目步骤。]

- [SOP 名称]: 当 [触发条件] 时，先 [步骤1]，再 [步骤2]，最后 [产出/验收标准]。

## Decision Logic
[记录取舍标准和优先级。]

- 当 [场景] 时，优先 [A] 而不是 [B]，因为 [原因]。

## Boundaries & Anti-patterns
[记录禁忌、边界和错误模式。]

- 不要 [错误做法]；应改为 [推荐做法]，因为 [原因]。

## Agent Rules
[记录 Agent 在工作中默认遵守的行为规则。]

- Agent 应 [行为规则]，避免 [风险]。
````

---

## 成功标准

- ✅ 只输出一个 JSON 对象，`content` 是**整篇**文档（从 `# Team Operating
  Doctrine` 开始）
- ✅ 最终内容不超过 1200 字
- ✅ 只保留所有工作场合可复用的原则、SOP、禁忌、判断逻辑和 Agent 规则
- ✅ 每条内容脱离具体项目后仍语义完整
- ✅ 求精不求多，能不写就不写，能合并就合并
- ✅ 不写项目进度、任务流水账、版本碎片或场景索引"""


# ============================================================
# 用户消息（HumanMessage）：四段输入
# ============================================================
#
# ⚠️ 为什么这段排版**住在这个文件里**，不在服务里
# ----------------------------------------
# 见 prompts/__init__.py 的约定，与 memory_scene_consolidate.py 同一理由：
# 提示词文本与"这段运行时数据该怎么排版"都是提示词的一部分。服务侧只负责
# **取哪些数据**（哪个 scope 的哪些场景、现有画像、几点），排版归这里。


class SceneLike(Protocol):
    """
    渲染【变化场景】需要的**最小字段集**。

    声明成 Protocol 而不是 import `SceneRow`：提示词这一层不该认识领域实体
    （`memory_scene_consolidate.py` 对它的两个 Protocol 用了同一手法）。
    """

    filename: str
    content: str


def render_user_message(
        *,
        changed_scenes: Sequence[SceneLike],
        scene_count: int,
        existing_persona: str | None,
        persona_update_request: bool,
        now: datetime,
) -> str:
    """
    变化场景 + 统计 + 现有画像 + 当前时间 -> 一次调用的用户消息。

    ⚠️ 五块的**顺序**是有意的（与原项目 `buildPersonaPrompt` 一致）
    ------------------------------------------------------
        0  触发信息          **有才占位**（原项目的 `triggerSection`）。排在
                           最前是因为它是"这一次为什么现在做"——模型是读完
                            输入再决定策略的，把它埋在中间等于让它先按错误
                            的前提想完一遍（与 L2 把数量预警排在最前同一手法）
        1  统计            模型据此判断这是首次还是迭代（它自己在提示词里
                           有"冷启动要克制"那条规则）
        2  变化场景         要抽象的东西（原项目的 changedScenesContent）
        3  现有画像         抽象到哪里去（原项目的 existingPersonaSection）
        4  当前时间         生成页脚"最后更新"用

    ⚠️ "首次生成"**不另开一条分支**
    -------------------------
    `existing_persona is None` 时第 3 块写一行说明即可，**不**给一份不同的
    系统提示词、也不换模板。原项目的 `mode` 参数只影响两处措辞（标签与
    迭代指南），而"有没有现成画像"这件事在第 3 块里一眼就看得见。多一条
    分支 = 多一条只有冷启动才走的路，而那是最少被跑到的一条。

    ⚠️ `scene_count` 是**总数**，`changed_scenes` 是**增量**
    ------------------------------------------------
    两个数都要给：模型需要知道"变化的有几条"（这批工作的量），也需要知道
    "一共有几条"（画像的覆盖面有多大 —— 比如 1 条变化 / 2 条总数说明覆盖
    还很薄，冷启动该克制）。

    ⚠️ 页脚里**没有**"记忆总数"
    ----------------------
    原项目有（`totalProcessed`），这里取不到：`memory_scene` 行上没有记忆
    条数，而 `memory_atom` 没有 scope 维度（它是会话级的）。**编一个数
    比不给更糟** —— 模型会把一个假的量级写进画像。所以页脚只留能落实到
    数据上的两项（时间 / 场景数）。
    """

    mode_label = (
        "🔄 迭代更新" if existing_persona is not None else "🆕 首次生成"
    )

    blocks = [
        *_render_trigger(persona_update_request),
        _render_stats(
            changed=len(changed_scenes),
            total=scene_count,
            mode_label=mode_label,
            now=now,
        ),
        _render_changed_scenes(changed_scenes),
        _render_existing_persona(existing_persona),
        _render_iteration_guide(existing_persona is not None),
        f"### 5️⃣ 当前时间 / Current Timestamp\n{now.isoformat()}",
    ]

    return "\n\n".join(blocks)


def _render_trigger(persona_update_request: bool) -> list[str]:
    """
    【触发信息】块 —— **没有举手就一块都不给**（原项目 `triggerSection` 的
    条件拼接）。

    ⚠️ 为什么"没有"时不写一个空段
    ------------------------
    与 L2 的数量预警同一条：一个恒定的空标题会让模型以为还有个字段没给、
    或者自己去找一个不存在的信号。**有才占位**。

    ⚠️ 2026-10-13：这里从"理由原文"改成了一句**固定的话**
    ---------------------------------------------------
    那个字段原来是 L2 的模型写的自由文本（"是什么重大变化触发了画像更新"），
    原文照搬进来。实测**一次都没给过** —— 「填一段描述」这种要求模型执行得
    很不稳定，而消费方只问"要不要"。压成布尔之后，这一格还剩一个用途：
    告诉模型**这次生成的来由**（是 L2 举手，不是首次创建）。所以文本改成
    固定的，不再引用一段不存在的理由。

    ⚠️ 但它**不是指令**
    ---------------
    下面必须说清这一点：模型仍然按自己的证据判断要不要改、改什么。不说
    的话，它容易把"有人请求更新"当成"我必须重写一篇"，于是产出一次没有
    新证据支撑的重写（那会污染画像，而且看起来像正常工作）。
    """

    if not persona_update_request:
        return []

    return [
        "### 0️⃣ 触发信息 / Trigger\n\n"
        "这次生成是由 **L2 整合时的模型**主动请求的（不是首次创建、也不是"
        "按轮次自动排出来的）。\n\n"
        "把它当作**一次提示**：L2 那边认为这个人的画像该重看了。但它**不是"
        "指令** —— 要不要改、改什么，仍然按你自己的证据判断。如果看完发现"
        "没有值得写入的内容，照样可以返回 `no_change: true`。",
    ]


def _render_stats(
        *,
        changed: int,
        total: int,
        mode_label: str,
        now: datetime,
) -> str:
    """
    【统计】块。

    ⚠️ 页脚那两行**照原样给模型抄**
    ---------------------------
    原文的模板末尾有一行 `> **最后更新**：[当前时间] · **来源场景**：[场景数]
    个`，那是模型要**写进 content 末尾**的页脚。既然它得自己填，就不该指望
    它去别处找这两个值 —— 所以在统计块里连同一份可直接抄的格式给出来。
    否则症状是页脚里出现字面的"当前时间"四个字，或一个它自己编的日期。
    """

    footer = (
        f"> **最后更新**：{now.date().isoformat()} · "
        f"**来源场景**：{total} 个"
    )

    return "\n".join([
        "### 1️⃣ 统计 / Stats",
        "",
        f"- **模式**：{mode_label}",
        f"- **场景总数**：{total} 个",
        f"- **本轮变化**：{changed} 个（上面的“变化场景”块就是它们）",
        "",
        "如果这一轮决定更新画像，正文末尾请照这个格式写最后一行"
        "（数字用上面的值）：",
        "",
        f"`{footer}`",
    ])


def _render_changed_scenes(scenes: Sequence[SceneLike]) -> str:
    """
    变化场景 -> 第二个输入块（每个场景**全文**）。

    ⚠️ 边界标记用 BEGIN/END 而不是 Markdown 语法
    --------------------------------------
    与 L2 那份同一个理由：内容本身就是 Markdown（有 `##` 标题、可能有代码块
    围栏），拿 Markdown 语法当边界会让模型分不清"这是场景边界"还是"场景里的
    一个标题"。

    ⚠️ 空的时候要**说清楚**而不是留空段
    ------------------------------
    提示词里"不改"那条规则的对象正是"这一轮没有值得抽象的东西"。给一个空的
    变化场景块而不加说明，模型很可能把"空"读成"我漏读了附件"从而开始编。
    所以显式写一句"这一轮没有变化场景"——而真到那一步时服务侧通常已经在
    调模型**之前**就正常结束了（见 service impl），所以这一句是防御性的。
    """

    header = (
        f"### 2️⃣ 变化场景 / Changed Scene Blocks"
        f"（{len(scenes)} 个，自上次生成后变化过的）"
    )

    if not scenes:
        return (
            header
            + "\n\n（这一轮没有变化的场景 —— 也就是说没有新的证据。"
              "除非现有的画像本身有明显的冗余或矛盾需要压缩，否则应当返回 "
              "`no_change: true`。）"
        )

    blocks: list[str] = [header]

    for index, scene in enumerate(scenes, start=1):

        blocks.append("\n".join([
            f"===== 场景 {index}/{len(scenes)} =====",
            f"filename: {scene.filename}",
            "-----BEGIN CONTENT-----",
            scene.content,
            "-----END CONTENT-----",
        ]))

    return "\n\n".join(blocks)


def _render_existing_persona(persona: str | None) -> str:
    """
    现有画像 -> 第三个输入块（**全文**，与 L2 把场景全文内联是同一手法）。

    ⚠️ 首次生成时给一句说明，而不是一个空块
    ---------------------------------
    见 `render_user_message` 里那段：分支只在这里，措辞上区分开就够了。
    """

    if persona is None:
        return (
            "### 3️⃣ 现有画像 / Existing Persona\n\n"
            "（还没有 —— **这是这个 scope 的第一次生成**。请从零建立画像，"
            "并且遵守提示词里“冷启动要克制”那条：没有证据的字段宁可不写。）"
        )

    return "\n".join([
        f"### 3️⃣ 现有画像 / Existing Persona（{len(persona)} 字符）",
        "",
        "下面是当前画像的**完整内容**，基于它更新：",
        "",
        "-----BEGIN CONTENT-----",
        persona,
        "-----END CONTENT-----",
    ])


def _render_iteration_guide(is_incremental: bool) -> str:
    """
    第四个输入块：迭代决策指南（原项目 `iterationGuide`）。

    ⚠️ 首次生成**不给**这一段
    ---------------------
    它的五条（强化 / 补充 / 修正 / 重构 / 不改）全是"面对**已有**画像"的
    判断，冷启动时一条都用不上 —— 给了只会让模型去找一个并不存在的旧版本。
    """

    if not is_incremental:
        return (
            "### 4️⃣ 迭代决策指南 / Iteration Guide\n\n"
            "（首次生成，无历史版本可比，跳过。）"
        )

    return "\n".join([
        "### 4️⃣ 迭代决策指南 / Iteration Guide",
        "",
        "面对变化场景，自主判断处理方式：",
        "",
        "- **强化**：只是佐证已有内容，压缩进原句或不动。",
        "- **补充**：出现了新的维度。",
        "- **修正**：新证据与旧结论矛盾，改掉旧结论。",
        "- **重构**：文档变散、变长、变项目化时，整体压缩重写。",
        "- **不改**：没有值得写入的新内容 —— 此时返回 `no_change: true`，"
        "**不要**为了“看起来做了事”而重写一遍。",
    ])


def _render(*, is_team: bool) -> str:
    """
    系统提示词。**只有这一个分派点** —— `is_team` 是运行时数据
    （`SceneScope.is_team`），服务侧把它传进来。

    ⚠️ 为什么不做成两个 PromptDefinition
    ------------------------------
    两者是**同一个 model_role**（`memory.persona.extract`）：它们由同一份
    提示词配置、同一个模型、同一条任务发起，只是产物形态按 scope 分叉。
    拆成两个 key 就得让服务侧在两处各选一次（model_role 与绑定），而那两处
    一旦漂了，症状是"团队画像用了个人模板"—— 输出仍然是一份像样的 markdown，
    不报错。
    """

    return TEAM_PROMPT if is_team else PERSONA_PROMPT


PROMPT = PromptDefinition(
    key=ModelRole.MEMORY_PERSONA_EXTRACT,
    description=(
        "L3 画像生成：把该 scope 变化过的 L2 场景抽象成一整篇画像"
        "（有 org 走团队 Operating Doctrine，否则走个人 User Narrative "
        "Profile），输出 {no_change, content} 的 JSON。"
    ),
    render=_render,
)
