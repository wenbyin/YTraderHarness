"""
`model_role=memory.scene.consolidate` 的提示词：L2 场景整合。

文本来自 `memory_l2.md`（原项目 `buildSceneSystemPrompt`），**文件语义的
那套词汇一字未改**：场景仍然是"目录里的 .md 文件"，删除仍然只能表达成
"把这个文件的内容写成 `[DELETED]`"，文件名规范照旧。

⚠️ 四处偏离（都是有意的，逐条说清代价）
==================================

1. **工具调用 → 一轮 JSON**（最大的一处）
   原提示词让模型用 **read / write / edit** 文件工具操作 `scene_blocks/`
   下的文件。本仓库的后台任务里**没有工具回路**：拿不到 write 的执行器，
   也没有"调一次工具再喂回去"的多轮编排（L1 那两份提示词同样是"要求
   严格 JSON + 事后解析"，见 memory_atom_extract.py）。

   于是改成：模型在一次回复里返回一个 JSON 对象，`operations[]` 里每条
   是 `{path, content}` —— **仍然是文件语言**，只是把"调用 write"换成
   "告诉工程系统这一次要 write 什么"。工程侧负责把它翻成行的坐标
   （原项目的 `classifyPath` 干的就是这件事，本仓库在
   application/memory_scene_service_impl.py 的 `_normalize_filename`）。

   代价说清楚：模型的"错觉"（它在操作文件）保住了，但它**看不到执行
   结果**，一次调用定输赢。原版能"写完再 read 一次确认"，这里没有。

2. **`read` 整节删掉**
   该 scope 下所有场景文件的**完整内容**在调用前就全部内联进了用户消息
   （上限 $max_scenes 个、每个 ≤1500 字符，量是可控的）。既然全文都在
   手里，"用 read 工具去读"这一步就没有对象了 —— 而它恰恰是多轮循环的
   唯一理由。删掉它，整条链就退化成一问一答。

3. **`edit` 那几条删掉，统一整篇重写**
   原文允许"局部更新用 edit（oldText/newText）"。全文既已内联，整篇重写
   与局部替换的输出代价相同，却少一条"oldText 在文件里找不到"的**静默**
   失败路径（那种失败要么报错、要么改错地方，两种都不好）。所以
   `operations[]` 只有"整篇内容"这一种形状。

4. **输入三件套明确化**
   原文的"New Memory"是单条；这里是一批 L1 增量（按会话水位取出来的），
   逐条带上 `record_id / type / priority / scene_name / time`，与 L1 去重
   提示词的渲染格式一致 —— 同一份数据在两处长得一样，读日志时少一次
   换算。"Existing Blocks Map"从"文件名 + 摘要"变成"文件名 + 全文"。

⚠️ `[DELETED]` 那条约定**一个字都没有改**
--------------------------------------
原项目的 sandbox 只给 write 权限（没有 rm/delete 工具），而空字符串会被
参数校验拒绝 —— 所以"删"只能表达成"写一个标记"，由工程侧在模型跑完后
真正删掉（原项目 scene-extractor.ts 的 Phase 5：remove soft-deletes）。
**模型永远拿不到删除能力**，这是设计意图，不是权宜之计。本仓库照搬这条：
`content == "[DELETED]"` 进删除集，由 apply 阶段带 scope 校验真删。

⚠️ 为什么用 `string.Template` 而不是 `str.format`
--------------------------------------------
正文里有大量 JSON 花括号（`{` `}`）与模板示例，`str.format` 会把它们
当占位符。`Template` 只认 `$`，而这份提示词里除了 `$max_scenes` 没有别的
`$`。与 classification.py 同一取舍。

⚠️ 可变部分在 HumanMessage，但**排版也归本文件**
------------------------------------------
系统提示词是常量（只随 `max_scenes` 变，而那是个配置值）；这次要分析
哪些记忆、现有场景是什么、现在几点，由服务侧**取**（它才知道该读哪个
scope 的水位与行），而**怎么排版**归这里 —— `render_user_message`。
两边的接口就是那个函数签名。

这条与 prompts/__init__.py 的约定一致（"静态文本与动态块都留在本文件
里"）。它是本模块**从服务里搬过来**的一部分：原来散在
`MemorySceneServiceImpl._build_request` / `_render_scenes` /
`_render_atoms` 里，而"共 N 个"这类计数在两处各写一遍时，改一处另一处
就漂，漂了不报错，只是模型的行为变得说不清。META 块的解析器
（`parse_meta`）同理搬了进来 —— 那个块的**格式**由上面 TEXT 的模板定义。

谁用它
------
`MemorySceneServiceImpl.memory_scene_extract`（`memory.l2.extract`
那条后台任务）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from string import Template
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent.model.types import ModelRole
from app.agent.prompt.contracts import PromptDefinition

__all__ = [
    "PROMPT",
    "TEXT",
    "AtomLike",
    "SceneLike",
    "SceneMeta",
    "parse_meta",
    "render_user_message",
    "scene_count_warning",
]


TEXT = """# Memory Consolidation Architect

**输出语言**：所有自然语言内容（文件名、章节标题、正文）使用与"新增记忆"中记忆相同的语言；META 字段名（created/updated/summary/heat）和 `[DELETED]` 等标记保持英文。模板中给出的中文章节标题（`## 用户核心特征` 等）作为结构骨架——非中文输出时请用目标语言的等价表达替换。

## 角色定义 (Role Definition)
你是记忆整合架构师。你的目标是为用户构建一个"数字第二大脑"。你不仅仅是在记录数据，你更像是一位人类学家和心理学家，负责分析原始记忆，从中提取核心特征、捕捉隐性信号，并构建不断演变的叙事。


## 架构模型

### Layer 1 (Input): Raw Memories
- **来源**：新增记忆清单（L1 提炼出来的结构化记忆，一批若干条）
- **状态**：碎片化、无序

### Layer 2 (Processing): Scene Diaries
- **形态**：**不是清单，是连贯的叙事文档**
- **逻辑**：将 L1 碎片融合进特定场景文件
- **动作**：Create（创建）、Integrate（整合）、Rewrite（重写）
- **禁止**：简单追加列表

你主要负责 L1 到 L2 的生成任务

## 输入环境 (Input Context)
你将接收三类输入（外加一段可能出现的警告）：
0. 场景数量警告 (Scene Count Warning)：**只在需要时出现**。它由工程系统按当前场景总数算好，直接告诉你这一轮该先合并、只能重写、还是照常。**出现时以它为准** —— 不要去自己数那个数字，也不要用下面的通用规则覆盖它。
1. 新增记忆 (New Memories List)：一段原始的、非结构化的新近回忆信息。
2. 现有场景文件 (Existing Scene Blocks)：当前所有记忆块（Markdown 文件）的**文件名清单与完整内容**，以及当前的文件总数。
3. 当前时间 (Current Time)：用于生成元数据的具体时间戳。

**⚠️ 场景文件数量上限：$max_scenes 个。处理完成后场景文件数量必须严格小于此上限。**

## ⛔ 输出约束（必须严格遵守）
1. **你不会操作文件，也没有任何工具。** 你在一次回复里返回**一个 JSON 对象**，工程系统按它写入对应的 `.md` 文件。你的整条回复只能是这个 JSON 对象，不要有任何解释、前言或 Markdown 代码块修饰符。
2. **所有 `path` 使用相对文件名**（如 `技术研究-Rust学习.md`），不带目录、不带 `scene_blocks/` 前缀。
3. **写文件就是给出一条 `write` 操作**，`content` 是该文件的**完整内容**（整篇重写，不是片段、不是 diff）。
4. **该 scope 下所有场景文件的完整内容已经在"现有场景文件"里给出**。你不需要、也无法请求读取任何额外内容。
5. **删除文件的唯一方式**：给出一条 `path` = 文件名、`content` = `[DELETED]` 的 `write` 操作。系统会自动清理带此标记的文件。**禁止**写入空字符串。**禁止**用 `[ARCHIVE]`、`[CONSOLIDATED]` 等其他标记替代删除——只有 `[DELETED]` 标记会触发系统清理。
6. **禁止创建报告/整合/汇总类文件**。你的输出必须是有意义的场景叙事文件（如"技术架构与工程实践.md"、"日常生活与工作节奏.md"）。禁止创建以 BATCH、REPORT、CONSOLIDATION、INTEGRATION、ARCHIVE、SUMMARY 等为前缀的文件。
7. **没有改动的文件不要出现在 `operations` 里**。只写你真的创建、重写或删除的那些。

## 📛 文件命名规范（强制）

为保证下游工具（场景导航、健康检查、对象存储同步等）能正确解析路径引用，**新建文件**或 **MERGE 后的目标文件**必须遵守以下命名规则：

- **允许字符**：英文字母、数字、CJK 中日韩文字、短横线 `-`、下划线 `_`、点号 `.`
- **必须以 `.md` 结尾**（小写）
- **❌ 禁止包含**：空格、全角空格、引号、括号 `( ) [ ] { }`、斜杠 `/ \\`、冒号 `:`、分号 `;`、问号 `?`、感叹号 `!`、星号 `*`、竖线 `|`、其他标点
- **多词分隔**：使用 `-`（短横线）连接，不要用空格
- **更新现有文件**时，沿用清单中给出的文件名，不要改名

✅ 正确示例：
- `Daily-Rhythm-in-Shanghai.md`
- `日常生活-健康管理.md`
- `技术研究-Rust学习.md`
- `Coffee-Yirgacheffe.md`

❌ 错误示例（每次都会触发工程兜底重命名）：
- `Daily Rhythm in Shanghai.md`（含空格）
- `Coffee (Yirgacheffe).md`（含括号）
- `Q1 Milestone?.md`（含空格和问号）

> 提示：即使你没遵守，工程系统会自动归一化文件名（空格替换为短横线、删除括号等），但这会增加日志噪音和潜在冲突。请在 `path` 里直接使用合规名字。


## 工作流与逻辑 (Workflow & Logic)
在生成输出之前，你必须执行以下"思维链"过程：

### ⚠️ 阶段 0：强制检查场景总数（必须先执行）

**在处理任何记忆之前，你必须：**

1. **看当前场景总数**：**优先看【场景数量警告】**（那是工程系统按当前数量算好的分级与动作要求，出现时以它为准）；没有那一段时，看【现有场景文件】标题里标注的总数，再按下面第 3 条的三级规则自己判断
2. **最终目标**：处理完成后，场景文件数量必须 **严格小于 $max_scenes**
3. **遵守分级预警**（**判据**，不是要你重新算一遍 —— 与上面那段警告冲突时以警告为准）：
   - 红色预警（≥ $max_scenes）：**必须先通过 MERGE 减少文件数量**，将最相似的 2-4 个场景合并为 1 个，**并删除被合并的旧文件**，直到文件数 < $max_scenes 后，再处理新记忆
   - 橙色预警（= $orange_limit，即只剩 1 个名额）：**只能重写现有场景，不能新建场景**
   - 黄色预警（接近 $max_scenes）：**优先重写或主动 MERGE 相似场景**

**合并优先级**（当需要合并时，按以下顺序选择）：
1. **主题高度重叠**：如"Python后端开发"和"Go后端开发" → 合并为"后端开发技术栈"
2. **叙事弧线相同**：如"求职材料-JD匹配"和"职业发展-能力对齐" → 合并为"职业发展与求职"
3. **热度最低的场景**：如果没有明显重叠，合并或删除 heat 最低的 2-3 个场景

### 阶段 1：分析与分类
分析 新增记忆。它的核心领域是什么？（例如：编程风格、情绪状态、职业轨迹、人际关系）。
提取事实事件链（触发 -> 行动 -> 结果）以及底层的心理状态。

### 阶段 2：检索与策略选择
将新记忆与 现有场景文件 的**文件名和完整内容**逐一比对。

**核心原则：默认策略是 UPDATE，不是 CREATE。** 当犹豫于 UPDATE 和 CREATE 之间时，选择 UPDATE。

策略选择（按优先级排序）：
1. **UPDATE（更新）**【首选策略】: 如果存在相关的场景文件（基于文件名或内容的相似性），锁定该文件进行整篇重写
2. **MERGE（合并）**:
   - 合并后的新场景应该是概括性更强的场景，包含已有的多个相似场景
   - **强制合并**：当前文件总数 **≥ $max_scenes** 时，必须先将多个相似场景合并
   - **主动合并**：即使未达上限，如果两个场景属于同一叙事弧线，也应合并以增加深度
   - **⚠️ 合并后必须删除旧文件**：被合并的旧场景文件必须各自给出一条 `content` = `[DELETED]` 的操作。**仅仅打标记（如 [ARCHIVE]、[CONSOLIDATED]）不算删除，文件仍会占用配额。**
3. **CREATE（新建）**【最后手段】:
   - **前提条件**：当前场景总数 < $max_scenes
   - **CREATE 前的强制验证**：必须先在"现有场景文件"里比对至少 2 个最相似的场景的**完整内容**，确认新记忆确实无法融入后才能新建。跳过验证直接新建是被禁止的
   - 如果话题是全新的且与现有内容区分度高，可以创建新的场景文件
   - **每次批处理最多新增 1 个场景**

**示例 A：新记忆整合进已有场景（UPDATE - 原地更新）**
**具体操作步骤**：
1. 在"现有场景文件"里找到 `Python后端开发.md` 的完整内容 A
2. 分析新记忆 + 已有内容 A → 整合生成新内容 B（`heat = 旧heat + 1`）
3. 输出一条 `write`：`path` = `Python后端开发.md`，`content` = B（**整篇重写**）

**示例 B：合并多个场景（MERGE — 合并后必须删除旧文件）**
**具体操作步骤**：
1. 在"现有场景文件"里取到 `Python后端开发.md` 的内容 A 与 `Go后端开发.md` 的内容 B
2. 整合 A + B + 新记忆 → 生成新内容 C（`heat = heatA + heatB + 1`）
3. 输出 `write`：`path` = `后端开发技术栈.md`，`content` = C → 创建合并后的新文件
4. 输出 `write`：`path` = `Python后端开发.md`，`content` = `[DELETED]` → **⚠️ 删除旧文件 A**
5. 输出 `write`：`path` = `Go后端开发.md`，`content` = `[DELETED]` → **⚠️ 删除旧文件 B**
**关键**：步骤 4-5 是必须的！不删除 = 文件总数不减少 = 合并无效。

### 阶段 3：撰写与合成（核心任务）
深度整合: 严禁简单的文本追加。你必须结合上下文（基于现有场景的完整内容）重写叙事，将新信息自然地融入其中。
隐性推断: 寻找用户 没说出口 的信息。更新"隐性信号"部分。
冲突检测: 如果新记忆与旧记忆相矛盾，将其记录在"演变轨迹"或"待确认/矛盾点"中。

### 撰写准则 (严格遵守)
核心部分禁止列表: "用户核心特征"和"核心叙事"必须是连贯的段落，信息要连贯，可以分段。
叙事弧线: "核心叙事"必须遵循故事结构（情境 -> 行动 -> 结果）。

### 热度管理 (Heat Management):
新建场景: heat: 1
更新场景: heat: 旧heat + 1
合并场景: heat: sum(所有相关场景的heat) + 1

## 输出规范 (Output Specification)

### 📄 场景文件内容模板

请你参考这个模板产出每个 `.md` 的 `content`，或基于已有 md 的完整内容更新，每个 md 控制在 1500 字符内。不要把模板本身放在 Markdown 代码块中，只需把要写入文件的原始文本放进 `content` 字段。

> 模板中的中文章节标题（`## 用户核心特征` 等）和示例文本仅作为**结构骨架**参考；**实际章节标题与正文必须按上述输出语言书写**（例如英文场景：`## User Core Traits`、`## User Preferences`、`## Implicit Signals`、`## Core Narrative` 等）。

```
-----META-START-----
created: {{EXISTING_CREATED_TIME_OR_CURRENT_TIME}}
updated: {{CURRENT_TIME}}
summary: [30-40 words concise summary for indexing]
heat: [Integer]
-----META-END-----

## 用户基础信息
[可为空，如果没有可不写这节，可按照需求添加更多点，合并和更新方式尽量叠加，有冲突则覆盖]
   -姓名：
   -职业：
   -居住地：
   - ……

## 用户核心特征
[这里不是列表！是一段连贯的描述。你细心推断出来最核心的用户特征，宁缺毋滥，**控制在100字以内**]
[示例: 用户在后端开发方面表现出对 Python 的强烈偏好，特别是异步框架。近期（2026-02）开始关注 Rust 的所有权机制，这表明用户有向系统级编程转型的意图。]

## 用户偏好
[这里可以是列表！**如果没有可以写这节**，记录用户明确的偏好信息（显性偏好），注意不要重复信息，不要流水账，偏好要可复用，更新时可以动态整合甚至重写]
[示例：用户喜欢吃苹果]

## 隐性信号
[这是给人类学家看的，记录那些"没明说但很重要"的事，和显性偏好不一样，一定是你推断出来的，需要深思熟虑后再生成，可以为空，宁缺毋滥。你可以随时更新/删除/修改这里的信息]

## 核心叙事
[这里不是列表！是一段连贯的描述，**控制在400字以内**，注意不要重复信息，不要流水账，可以动态整合甚至重写]
*(这里记录连贯的故事，必须包含 Trigger -> Action -> Result)*

[ 示例：本周用户主要集中在后端重构上。初期因为旧代码的耦合度高感到沮丧（**情绪点**），但他拒绝了"打补丁"的建议，坚持进行彻底解耦（**决策点**）。他在此过程中频繁查阅架构设计模式，表现出对"代码洁癖"的执着。]


## 演变轨迹
> [注意] 可以为空，仅记录【用户偏好/性格/重大观念】转变，不记录琐碎、日常更新。当发生冲突时，不要直接覆盖，要记录变化轨迹。
- [2026-01-10]: 从 "反对加班" 转向 "接受弹性工作"，原因：创业压力（记忆ID: #987）


## 待确认/矛盾点
- [记录当前无法整合的矛盾信息，等待未来记忆澄清]

```

### ✅ 最终回复的 JSON 形状（必须严格遵守）

返回且仅返回一个合法 JSON 对象，形状如下：

```
{
  "operations": [
    {
      "path": "技术研究-Rust学习.md",
      "content": "该文件的完整内容。要删除这个文件就把整个 content 写成 [DELETED]。"
    },
    {
      "path": "日常生活-健康管理.md",
      "content": "[DELETED]"
    }
  ],
  "persona_update_request": false
}
```

字段说明：
- `operations`：本次要落盘的全部变更，一条一个文件。**没改动的文件不要出现。** 没有任何变更时给空数组。
- `path`：相对文件名，遵守上面的命名规范。
- `content`：该文件的**完整内容**。`[DELETED]` 表示删除该文件。
- `persona_update_request`：**布尔**，默认 `false`。发生了重大价值观转变、跨场景突破性洞察时填 `true`（这不是文件操作）。

不要输出任何额外的解释文本，不要用 Markdown 代码块包住整个回复。

#### 主动触发 Persona 更新（可选）

**触发条件**：重大价值观转变、跨场景突破性洞察。

**触发方式**：把上面 JSON 里的 `persona_update_request` 填成 `true`。

⚠️ 这是一个**布尔**，不是描述 —— 不要填一段文字说明"为什么要更新"。
上面的 `operations` 本身已经把"变了什么"写清楚了，下游不需要理由。"""


# ============================================================
# META 块：格式的**定义**与解析器住在一起
# ============================================================
#
# ⚠️ 为什么解析器在提示词文件里，不在领域模型里
# --------------------------------------
# META 块的**格式**由上面 TEXT 的模板定义（`created` / `updated` /
# `summary` / `heat` 四行，夹在 `-----META-START-----` 之间）。解析器放在
# 别处的代价不是理论上的：改了模板的字段名而忘了改解析器，症状是**清单里
# 每个场景的摘要都变空**，而模型从此选不准该 UPDATE 进哪一个 —— 不报错，
# 只是"它忽然开始乱建场景"。
#
# 同一个文件的另一个好处是：这里的字段与模板只隔几十行，改模板的人不可能
# 看不到它。


#: META 块的整体形状。
#:
#: ⚠️ 用 `.*?` + DOTALL 而不是逐行扫
#: ------------------------------
#: 块是”起始行 / 若干字段行 / 结束行“，逐行扫要写状态机；而这里只要能
#: 把块**切出来**，字段解析交给下面那个更简单的逐行正则。非贪婪 + DOTALL
#: 一次切到最近的 END。
_META_BLOCK = re.compile(
    r"-----\s*META-START\s*-----(?P<body>.*?)-----\s*META-END\s*-----",
    re.DOTALL,
)

#: 块里的一个字段：`key: value`。值可能带引号（模型爱加）。
_META_FIELD = re.compile(r"^\s*(?P<key>[A-Za-z_]+)\s*:\s*(?P<value>.*?)\s*$")


class SceneMeta(BaseModel):
    """
    一份场景文件 META 块里的内容。

    ⚠️ 为什么**不**做成表上的列
    ----------------------
    这些值由模型写在 `content` 里，提到列上就造出**第二个**"这个场景讲
    什么"的答案，而同步两份的成本落在每一次写入上。读场景清单时按需解析
    （一个 scope 就十几行）—— 一行只有一个真相。见 models/sqlalchemy/
    _scope.py 的 MemoryFileColumns。

    ⚠️ 全部可缺省
    ---------
    模型可能整块没写、也可能写了块但漏了某个字段。**不报错**：清单里少
    一行摘要远好于整次整合失败（内容本身还在，那才是要保的东西）。
    块在不在由 `has_meta_block` 区分 —— 那是排查"为什么清单里每个场景都
    没有摘要"时的第一站。
    """

    model_config = ConfigDict(extra="ignore")

    has_meta_block: bool = Field(
        default=False,
        description="content 里到底有没有 META 块。False 时下面几个字段一律为默认值。",
    )

    created: str = Field(default="", description="这份场景第一次建立的时间（模型写的，文本）。")
    updated: str = Field(default="", description="最后一次更新的时间（模型写的，文本）。")

    summary: str = Field(
        default="",
        description=(
            "30-40 词的摘要。**它是场景清单里唯一给模型看的识别信号**"
            "（决定新的记忆该 UPDATE 进哪一个文件），所以它空着时清单的"
            "质量会明显下降 —— 服务侧会为此打一条 warning。"
        ),
    )

    heat: int | None = Field(
        default=None,
        description=(
            "热度（提示词规定的合成规则：新建 1 / 更新 旧+1 / 合并 sum+1）。\n\n"
            "None = 没写或解析不出整数。合并时的取舍要用它，所以**取值域"
            "外的值不在这里挡**（挡了就是整次失败），解析不出就当没给。"
        ),
    )


def parse_meta(content: str) -> SceneMeta:
    """
    一份场景文件的 content -> 它的 META（摘要 / 热度 / 时间）。

    ⚠️ 从不抛
    --------
    解析不出来就是"没有元数据"，那是**允许**的状态（见 SceneMeta 的说明）。
    与 `parse_consolidation` 恰好相反 —— 那个是刚出炉的模型输出，判不清就
    该重试；这边是库里的存量内容，读不懂不该让整个 scope 的场景都读不
    出来（那会让这一轮直接失败，而内容本身是好的）。
    """

    block = _META_BLOCK.search(content or "")

    if block is None:
        return SceneMeta()

    fields: dict[str, str] = {}

    for line in block.group("body").splitlines():

        matched = _META_FIELD.match(line)

        if matched is None:
            continue

        fields[matched.group("key").lower()] = _unquote(matched.group("value"))

    return SceneMeta(
        has_meta_block=True,
        created=fields.get("created", ""),
        updated=fields.get("updated", ""),
        summary=fields.get("summary", ""),
        heat=_as_int(fields.get("heat")),
    )


def _unquote(value: str) -> str:
    """
    去掉值两端成对的引号，再去掉模板占位用的方括号。

    模型爱给值加引号（`updated: "2026-10-11T09:12Z"`），而模板里写的
    `summary: [30-40 words concise summary for indexing]` 那对方括号是
    "替换我"的标记 —— 它偶尔会被原样留下，于是清单里每个场景的摘要都
    长着一对方括号。只在**两端成对**时剥，正文里的方括号不受影响。
    """

    text = value.strip()

    for left, right in (('"', '"'), ("'", "'"), ("“", "”")):

        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
            break

    if len(text) >= 2 and text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    return text


def _as_int(value: str | None) -> int | None:
    """
    `heat` 的文本 -> 整数。解析不出返回 None。

    ⚠️ 不是 0 —— 0 是一个**有意义的取值**（"这个场景没人碰过"），拿它当
    "没给"会让合并时的取舍悄悄偏向它。
    """

    if value is None:
        return None

    try:
        return int(value.strip())

    except ValueError:
        return None


# ============================================================
# 用户消息（HumanMessage）：三段输入 + 可能出现的预警
# ============================================================
#
# ⚠️ 为什么这段排版**住在这个文件里**，不在服务里
# ----------------------------------------
# 见 prompts/__init__.py 的约定：提示词文本与"这段运行时数据该怎么排版"
# 都是提示词的一部分。散到调用方去的代价很具体 —— 同一句话在两处各写
# 一遍（比如"共 N 个"这种计数），改了一处另一处就漂，而漂了不报错，
# 只是模型的行为变得说不清。
#
# 服务侧只负责**取哪些数据**（哪个 scope 的哪些场景、哪些新记忆、几点），
# 排版归这里。两边的接口就是下面这个函数签名。


class AtomLike(Protocol):
    """
    渲染【新增记忆】需要的**最小字段集**。

    声明成 Protocol 而不是 import `MemoryAtom`：提示词这一层不该认识
    领域实体（`classification.py` 对候选清单用了同一手法）。代价是传进来
    的东西字段少了不会在 import 期报错，而是 AttributeError —— 可接受，
    因为它只有一个调用点。
    """

    record_id: Any
    type: str
    priority: int
    scene_name: str
    timestamp_str: str
    content: str


class SceneLike(Protocol):
    """渲染【现有场景文件】需要的**最小字段集**。"""

    filename: str
    content: str


def render_user_message(
        *,
        atoms: Sequence[AtomLike],
        scenes: Sequence[SceneLike],
        max_scenes: int,
        now: datetime,
) -> str:
    """
    新增记忆 + 现有场景 + 当前时间 -> 一次调用的用户消息。

    ⚠️ 四块的**顺序**是有意的（与原项目的 userPrompt 一致）
    ---------------------------------------------------
        0  场景数量警告     有预警时排在最前 —— 它是**这一轮的行动要求**
                            （先合并还是只能重写），而模型是读完输入再决定
                            策略的，把它埋在中间等于让它先按错误的前提
                            想完一遍
        1  新增记忆         要处理的东西
        2  现有场景         处理到哪里去（清单 + 全文）
        3  当前时间         生成 META 的 updated 用

    警告**没有**时不占位（不写"无警告"那种空段）—— 一个恒定的空标题会
    让模型以为还有个字段没给。
    """

    warning = scene_count_warning(len(scenes), max_scenes=max_scenes)

    blocks = [
        *(
            [f"⚠️ **场景数量警告**：{warning}"]
            if warning
            else []
        ),
        _render_atoms(atoms),
        _render_scenes(scenes, max_scenes=max_scenes),
        f"### 3️⃣ 当前时间 / Current Timestamp\n{now.isoformat()}",
    ]

    return "\n\n".join(blocks)


def scene_count_warning(count: int, *, max_scenes: int) -> str:
    """
    当前场景数 -> **这一轮该怎么做**（原项目 `sceneCountWarning` 的对应物）。

    空的返回表示"没有预警"。

    ⚠️ 为什么要有它（系统提示词里已经写了三级规则）
    ------------------------------------
    提示词里那三级是**通用规则**（"≥ 上限时先合并"），模型必须自己把当前
    数量对上去才知道这一轮落在哪一级 —— 而它**数错一次就少做一次该做的
    MERGE**，后果是场景数慢慢顶到上限、新记忆堆在清单外（不报错，只是
    "这个人的画像越来越不准"）。

    所以分级不由模型做：工程侧算好（数量是它给的），把**结论**直接写进
    用户的输入里。系统提示词里那三级仍然留着 —— 它是"为什么是这样"的
    依据，模型据此知道合并该按什么优先级挑。两者不冲突：一个是判据，
    一个是判定的结果。

    ⚠️ 阈值住在**同一个文件**里，这一点不是巧合
    ----------------------------------
    上面 TEXT 里那三级（红 / 橙 / 黄）和这里的分支必须一致。分到两个文件
    就会漂：提示词说"接近上限时优先合并"，而工程侧的红线在别处被调过了，
    症状是模型收到一段与它自己那套规则对不上的指令，然后**挑一个执行**
    （哪一个是随机的）。

    阈值（红 ≥ max / 橙 = max-1 / 黄 ≥ max-2）里的"黄"是**工程侧定的口径**，
    提示词里只说"接近"。取 2 的理由是：只剩一两个名额时才值得提醒它
    "优先合并"，更早提醒会让模型过早把不同主题的场景压成一个。

    ⚠️ `count == 0` 必须在比较之前单独返回
    --------------------------------
    "黄"那一档写的是 `count >= max_scenes - 2`，而**上限很小时这个下界会
    变成 0 甚至负数**：`max_scenes = 1` 时它就是 `count >= -1`，于是"一个
    场景都还没有"的第一次整合也会收到"接近上限，优先合并"的预警 ——
    而它没有任何东西可以合并。这不是理论情况：上限是个配置值，测试会把
    它压到 1，线上也可能被调小。
    """

    if count <= 0:
        return ""

    if count >= max_scenes:
        return (
            f"当前已有 {count} 个场景文件，**已达到上限 {max_scenes} 个**。"
            "本轮**必须先 MERGE**：把最相似的 2-4 个场景合并成一个，"
            "并用 `content` = `[DELETED]` 删掉被合并的旧文件，"
            f"直到场景数严格小于 {max_scenes}，然后再处理新增记忆。"
            "**本轮禁止新建场景文件。**"
        )

    if count == max_scenes - 1:
        return (
            f"当前已有 {count} 个场景文件，上限 {max_scenes} 个（只剩 1 个名额）。"
            "本轮**只能重写（UPDATE）现有场景，不要新建** —— "
            "新记忆必须并进已有的某个场景。"
        )

    if count >= max_scenes - 2:
        return (
            f"当前已有 {count} 个场景文件，接近上限 {max_scenes} 个。"
            "优先 UPDATE，或主动把主题重叠的场景 MERGE 起来。"
        )

    return ""


def _render_atoms(atoms: Sequence[AtomLike]) -> str:
    """
    新增记忆 -> 第一个输入块（原项目 `memoriesJson` 的位置）。

    渲染格式与 L1 的冲突检测提示词**逐字一致**（`[record_id=…] type=…
    priority=…`）—— 同一份数据在两处长得一样，出问题时读日志少一次换算。

    ⚠️ 原项目传的是一段 JSON，这里传行式文本
    ---------------------------------
    两者承载的信息相同（内容 + 类型 + 优先级 + 情境 + 时间）。选行式的
    理由只有一个：**与 L1 那两份提示词同形**。
    """

    lines = [
        f"[record_id={atom.record_id}] type={atom.type} "
        f"priority={atom.priority} scene={atom.scene_name!r} "
        f"time={atom.timestamp_str!r}\n{atom.content}"
        for atom in atoms
    ]

    return "### 1️⃣ 新增记忆 / New Memories List\n\n" + "\n\n".join(lines)


def _render_scenes(scenes: Sequence[SceneLike], *, max_scenes: int) -> str:
    """
    现有场景 -> 两个输入块：**清单**（文件名逐行，供模型指认）与每个文件的
    **全文**。

    ⚠️ 那个"共 N 个"必须在**标题里**
    ---------------------------
    提示词的阶段 0 要求模型对一下当前总数。把数量塞在正文某处等于让它自己
    去数（它会数错），所以标题里给一次、`scene_count_warning` 里再给一次
    —— 后者带的是**结论**（这一轮该合并还是只能重写）。

    ⚠️ 清单那几行不是冗余
    -----------------
    原项目的用户输入里单独有一段"已有场景文件清单（仅以下文件可 read）"，
    作用是让模型**只用真实存在的名字**。这里没有 read 工具，但同一件事
    仍然成立：模型要写回某个文件时必须沿用清单里的名字（自己编一个 =
    多出一个内容重叠的场景）。全文块里虽然有 `filename:` 那一行，但一份
    短清单让"有哪些名字"一眼可数 —— 而模型正需要这个来判断该不该新建。

    ⚠️ 边界标记用 BEGIN/END 而不是 Markdown 语法
    --------------------------------------
    内容本身就是 Markdown（有 `##` 标题、可能有代码块围栏），拿 Markdown
    语法当边界会让模型分不清"这是文件边界"还是"文件里的一个标题"。
    `-----BEGIN CONTENT-----` / `-----END CONTENT-----` 在正文里出现的
    概率极低（而且即使出现，模型还有 filename 那一行可以对上）。
    """

    header = (
        f"### 2️⃣ 现有场景文件 / Existing Scene Blocks"
        f"（共 {len(scenes)} 个，上限 {max_scenes} 个）"
    )

    if not scenes:
        return (
            header
            + "\n（当前一个都没有 —— 这是这个 scope 的第一次整合，"
              "只能新建场景文件）"
        )

    head = "\n".join([
        header,
        "",
        "**已有场景文件清单（更新时只能沿用下列文件名，不要改名）**：",
        *[f"- `{scene.filename}`" for scene in scenes],
    ])

    blocks: list[str] = [head]

    for index, scene in enumerate(scenes, start=1):

        meta = parse_meta(scene.content)

        body = [
            f"===== 文件 {index}/{len(scenes)} =====",
            f"filename: {scene.filename}",
        ]

        # 摘要与热度是模型判断 UPDATE / MERGE 的主要依据，有就渲染出来。
        # 缺了不写（而不是写一行空值）—— 空值会被读成"这个场景没有摘要"
        # 这个**事实**，而有人的地方应该是"我们不知道"。
        if meta.summary:
            body.append(f"summary: {meta.summary}")

        if meta.heat is not None:
            body.append(f"heat: {meta.heat}")

        if meta.updated:
            body.append(f"updated: {meta.updated}")

        body.append("-----BEGIN CONTENT-----")
        body.append(scene.content)
        body.append("-----END CONTENT-----")

        blocks.append("\n".join(body))

    return "\n\n".join(blocks)


def _render(*, max_scenes: int) -> str:
    """
    系统提示词。**两个变量都从 `max_scenes` 算出来** —— 场景清单与新增记忆
    都不在这里（它们走 `render_user_message`）。

    ⚠️ 模板里**不要**写算式
    ------------------
    橙档原来是 `橙色预警（= $max_scenes - 1）`，渲染出来是字面的
    **"橙色预警（= 12 - 1）"** —— `Template` 只替换变量、不理解算术，于是
    模型得自己做减法，而且那一行看起来像占位符没渲染干净。改成把算好的值
    传进去（`$orange_limit`）。

    这类错误的共同点是**只在渲染结果里看得见**：模板读起来是对的，而它真
    正发给模型的样子没人会盯一眼。改模板时请把渲染结果打出来看一眼。

    `Template.substitute` 而不是 `safe_substitute`：少给一个变量要当场
    报 `KeyError`，而不是把 `$max_scenes` 原样留在提示词里发给模型 ——
    那会变成"模型看到一个字面量占位符"这种没人查得出来的怪症状。
    """

    return Template(TEXT).substitute(
        max_scenes=max_scenes,
        # 橙档的下界。**必须**与 `scene_count_warning` 里那个分支同一个口径
        # （那里是 `count == max_scenes - 1`）—— 两处不一致的症状是：提示词
        # 说的橙档与工程侧注入的警告对不上，模型挑一个执行。
        orange_limit=max_scenes - 1,
    )


PROMPT = PromptDefinition(
    key=ModelRole.MEMORY_SCENE_CONSOLIDATE,
    description=(
        "L2 场景整合：把新增 L1 记忆融合进该 scope 的场景文件"
        "（UPDATE / MERGE / CREATE / [DELETED]），输出 JSON 的写入清单。"
    ),
    render=_render,
)
