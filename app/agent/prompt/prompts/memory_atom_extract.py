"""
`model_role=memory.atom.extract` 的提示词：L1 记忆抽取。

文本来自 `memory_l1.md`（原项目 `EXTRACT_MEMORIES_SYSTEM_PROMPT`），
除下面那一处删减之外**一字未改** —— 包括那些看起来像排版习惯的地方
（分节线、加粗、示例句式）。它是一份被调过的提示词，改标点都可能改变
模型的输出分布。

⚠️ 唯一的删减：**输出里不要情境级的 `message_ids`**（2026-10-10）
-----------------------------------------------------------
规格的"任务三"要求每个情境回显 `message_ids`（"属于该情境的消息ID列表"）。
本仓库**不落库这一项**（memory_atom 里没有放它的列 —— 情境的权威定义在
L2），而它恰恰是输出里最大的一块：一批 20 条消息就是 ~800 字符的 UUID，
模型还得逐个回显得**一字不差**。

代价是实测出来的：`max_tokens` 与思考模式共用，那一长串 UUID 加上推理
把预算吃光，输出**在 `message_ids` 中间被截断**，报
`Unterminated string starting at: line 23 column 11` —— 看起来像模型乱写，
其实是没写完（每次重试都要重付一次模型调用）。

`source_message_ids`（**每条记忆**的）保留 —— 那一项要落库（`memory_atom.
source_message_ids`，溯源用）。

谁用它
------
`MemoryAtomServiceImpl.extract`（`memory.l1.extract` 那个后台任务）。

⚠️ 为什么**一个字都不能动**（而 run 的可变部分是 HumanMessage）
------------------------------------------------------
与摘要那份提示词同一分工：系统提示词是常量（这份一个变量都没有），
可变部分（这次要分析哪些消息、上一个情境是什么）是请求里的
**HumanMessage** —— 它由本文件的 `render_user_message` 渲染，服务只
负责**取**哪些消息。理由见 app/agent/prompt/contracts.py 的模块 docstring
与 prompts/__init__.py 的约定。

⚠️ `render` 没有参数，不是漏了
--------------------------
因为这份提示词一个字都不随运行时数据变。要把它做成"带变量的模板"就得
动上面的约束（比如把消息拼进来），而那正是上面说的分工要避免的事。

⚠️ 提示词里的 JSON 花括号**不是**占位符
--------------------------------
契约刻意不用 `str.format`，就是因为它会吃花括号而这份提示词满是示例
JSON。要加变量时请走 render 的签名 + 拼字符串，**不要**改成模板。

⚠️ 与 memory_l1.md 里那份的已知差异
--------------------------------
原文里有几句是**调用方**的说明（怎么拼【待提取的新消息】、怎么把上一个
情境带进来），那些不属于系统提示词 —— 它们是**用户消息**那一半，住在
本文件的 `render_user_message` 里（原先散在 `MemoryAtomServiceImpl`
的 `_build_extract_request`，2026-10-11 搬过来）。除此之外逐字一致。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from app.agent.context.processor.messages import message_text
from app.agent.model import ModelRole

from ..contracts import PromptDefinition

__all__ = ["PROMPT", "TEXT", "MessageLike", "render_user_message"]


TEXT = """\
你是专业的"情境切分与记忆提取专家"。
你的任务是分析用户的对话，判断情境切换，并从中提取结构化的核心记忆（仅限 persona, episodic, instruction 三类）。

**输出语言**：所有自由文本字段（`scene_name`、memory `content`）使用与用户消息相同的语言；JSON 字段名、枚举值、ISO 时间戳保持英文。

### 任务一：情境切分（Scene Segmentation）
分析【待提取的新消息】，结合【上一个情境】，判断并输出当前对话的情境。
- 继承：无明显切换，沿用上一个情境。
- 切换条件：用户发出明确指令（如"换话题"）、意图转变、或提出独立新目标。
- 一段对话可能只有一个情境，也可能有多个情境（话题多次切换时）。
- 命名规则："我（AI）在和xxx（用户身份）做xxx（目标活动）"（**使用上述输出语言**，约 30-50 个字符或等价长度，单句，全局唯一）。

---

### 任务二：核心记忆提取（Memory Extraction）
结合背景和当前情境，仅从【待提取的新消息】中提取核心信息。

【通用提取原则】
1. 宁缺毋滥：过滤琐碎闲聊、临时性指令和一次性操作（如"这次、本单"）；剔除不可靠的边缘信息。
2. 独立完整：记忆必须"跳出当前对话依然成立"，无上下文也能看懂。提取主体必须以"用户（姓名）"或"AI"为核心。
3. 归纳合并：强关联或因果关系的多条消息，必须合并为一条完整记忆，不可碎片化。

【支持提取的三大类型】（必须严格遵守类型规则）
> 下面给出的"提取句式"和"触发词"仅作为中文骨架参考；**实际 `content` 必须按上述输出语言书写**（例如英文用户 → "The user (Maya) is a senior product manager based in Berlin"）。

1. 个性化记忆 (type: "persona")
   - 定义：用户的稳定属性、偏好、技能、价值观、习惯（如住所、职业、饮食禁忌）。
   - 提取句式："用户（[姓名]）喜欢/是/擅长..."
   - 打分 (priority)：80-100（健康/禁忌/核心特质）；50-70（一般喜好/技能）；<50（模糊次要，可丢弃）。
   - 触发词：喜欢、习惯、经常、我这个人...

2. 客观事件记忆 (type: "episodic")
   - 定义：客观发生的动作、决定、计划或达成结果。绝不包含纯主观感受。
   - 提取句式："用户（[姓名]）在 [最好是精确绝对时间] 于 [地点] [做了某事（可以包含起因、经过、结果）]"。
   - 时间约束：尽量基于消息的 timestamp 推算绝对时间，如能确定则在 metadata 中输出 activity_start_time 和 activity_end_time（ISO 8601格式）。无法确定时可省略。
   - 打分 (priority)：80-100（重要事件/计划）；60-70（一般完整活动）；<60（琐碎事项，直接丢弃）。

3. 全局指令记忆 (type: "instruction")
   - 定义：用户对 AI 提出的长期行为规则、格式偏好、语气控制。
   - 提取句式："用户要求/希望 AI 以后回答时..."
   - 触发词：以后都、从现在开始、记住、必须。
   - 打分 (priority)：-1（极其严格的全局死命令）；90-100（核心行为规则）；70-80（重要要求）；<70（临时要求，直接丢弃）。

---

### 不应该提取的内容
- 琐碎闲聊、问候；临时性的纯工具性请求（如"这次帮我翻译一下"）
- 一次性操作指令（如"这次、本单"相关）
- 重复的内容；AI助手自身的行为或输出
- 不属于以上3类的信息
- 纯主观感受（不带客观事件的情绪表达）

---

### 任务三：输出格式规范（JSON）
返回且仅返回一个合法的 JSON 数组。数组的每一项是一个情境，包含该情境及从中抽取到的记忆：

[
  {
    "scene_name": "当前生成或继承的情境名称",
    "memories": [
      {
        "content": "完整、独立的记忆陈述（按对应类型的句式要求）",
        "type": "persona|episodic|instruction",
        "priority": 80,
        "source_message_ids": ["消息ID_1", "消息ID_2"],
        "metadata": {}
      }
    ]
  }
]

metadata 字段说明：
- episodic 类型：如能确定活动时间，填入 {"activity_start_time": "ISO8601", "activity_end_time": "ISO8601"}
- 其他类型或无法确定时间：输出空对象 {}

如果整段对话无有意义的记忆，也要输出情境分割结果，memories 为空数组：
[
  {
    "scene_name": "情境名称",
    "memories": []
  }
]

请严格按上述 JSON 数组格式输出，不要输出任何额外的 Markdown 代码块修饰符（如 ```json）或解释文本。"""


def _render() -> str:
    """无参数 —— 这份提示词不随运行时数据变化（可变部分走 `render_user_message`）。"""

    return TEXT


# ============================================================
# 用户消息（HumanMessage）：上一个情境 + 待提取的新消息
# ============================================================
#
# ⚠️ 排版为什么归本文件
# -----------------
# 见 prompts/__init__.py 的约定：提示词文本与"这段运行时数据该怎么排版"
# 都是提示词的一部分。这一段原先在 `MemoryAtomServiceImpl._build_extract_request`
# 里，2026-10-11 搬过来 —— 那两行表头（"务必结合 timestamp 推算时间"、
# "只从这里提取记忆"）本来就是**提示词的正文**，只是位置在用户消息里；
# 放在服务里时，改提示词的人翻不到它们。
#
# ⚠️ 这里**刻意没有【背景对话】块**（2026-10-11 定）
# ------------------------------------------
# 原项目的 `formatExtractionPrompt` 会额外喂一批**游标之前**的消息，并标明
# 「仅供理解上下文推断关系/时间，严禁从中提取记忆」。本实现只喂增量，所以
# 系统提示词里那句「结合背景和当前情境」（任务二开头）当前是**无对象**的。
#
# 当时的取舍是**先不接**（要给 MessageRepository 加一个 `list_chat_before`，
# 而且要改这条**已经在跑**的链的输入）。代价不是零：一段增量从对话中间
# 开始时，"他 / 上次那个 / 那件事"没有先行词，而提示词要求记忆必须"跳出
# 当前对话依然成立"—— 模型于是要么丢掉这几条（漏抽），要么给主语猜一个
# （幻觉）。**两类都不报错**，只在窗口边界上零星发生。
#
# 要接的话落点在这里：多一个 `background` 参数，渲染成
# 「【背景对话】（仅供理解上下文推断关系/时间，严禁从中提取记忆）：…」，
# 没有历史时写 `无`（与参考一致）。下面那句"只从这里提取"的限制**必须
# 留着** —— 有了背景块之后它是防"同一段对话被抽两遍"的唯一一道。


class MessageLike(Protocol):
    """
    渲染【待提取的新消息】需要的**最小字段集**（duck-typed）。

    不 import `AgentMessage`：提示词的这一层不该认识领域实体（与
    `classification.py` 对候选清单用同一手法）。`content` 留给
    `message_text()` 去解析 —— 那是本仓库唯一一处"知道消息长什么样"的
    地方，langchain 的三种 content 形态都在它那里收口。
    """

    message_id: Any
    role: str
    created_at: datetime
    content: Any


def render_user_message(
        *,
        messages: Sequence[MessageLike],
        previous_scene: str,
) -> str:
    """
    上一个情境 + 这次要分析的消息 -> 一次调用的用户消息。

    没有可索引文本的消息（纯工具调用、空回复）**整条跳过** —— 与 L0 索引
    那条规则同源（`_assembly.message_text` 返回 None 即不进索引）。留一条
    空行反而会让模型以为"这里有一段空的对话"。
    """

    lines: list[str] = []

    for message in messages:

        text = (message_text(message) or "").strip()

        if not text:
            continue

        lines.append(
            # ⚠️ id 必须原样给出：模型要把它们填进 source_message_ids，
            # 那是溯源唯一的线索（出库之后没法再推）。
            f"[message_id={message.message_id}] "
            f"role={message.role} "
            f"time={message.created_at.isoformat()}\n{text}"
        )

    return "\n\n".join([
        "【上一个情境】\n"
        + (previous_scene or "（无 —— 这是本会话的第一次抽取）"),

        # ⚠️ 表头那两句是**提示词正文**，不是排版习惯：
        #
        #     "务必结合 timestamp 推算时间"   episodic 的 activity_start/end
        #                                   要靠它推出来，而抽取那一步是
        #                                   唯一看得到 timestamp 的地方
        #     "只从这里提取记忆！"            把提取范围钉死在增量上。它与
        #                                   系统提示词那句"仅从【待提取的
        #                                   新消息】中提取"是同一件事的两次
        #                                   表述 —— 后者在长提示词的中间，
        #                                   前者贴着数据，而贴着数据的那一
        #                                   句才是模型真正会照着做的
        "【待提取的新消息】"
        "（务必结合 timestamp 推算时间，只从这里提取记忆！）\n"
        + "\n\n".join(lines),
    ])


PROMPT = PromptDefinition(
    key=ModelRole.MEMORY_ATOM_EXTRACT,
    description=(
        "L1 记忆抽取：情境切分 + 从对话里提炼 persona / episodic / "
        "instruction 三类结构化记忆，输出 JSON。"
    ),
    render=_render,
)
