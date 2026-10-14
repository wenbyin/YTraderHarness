"""
L1（memory_atom）抽取产物的解析模型与解析函数。

它回答什么
--------
"模型那段 JSON 到底长什么样、怎么把它变成能落库的东西“。

⚠️ 与 `domain/schemas/memory.py` 不是一回事
--------------------------------------
那边是**对外契约**（memory.yaml 的请求/响应，同时也是工具入参 schema），
要跟着 OpenAPI 走。这里是**模型输出**的解析形状 —— 它不对任何调用方
负责，只对那一份提示词负责（见 app/agent/prompt/prompts/l1_extract.py）。
两边的字段名巧合地像，但改一边不必改另一边。

⚠️ extra="ignore" 而不是 "forbid"
--------------------------------
提示词规定了严格的 JSON，但模型偶尔会多带一两个字段（"confidence"、
"reason" 之类）。`forbid` 会让整次抽取因为一个多余字段而失败 —— 代价
（整批记忆没抽出来）与收益（发现模型多写了个字段）完全不对等。所以
忽略未知字段，只在**必需的字段缺失或类型不符**时报错。

⚠️ 解析失败要**抛**，不要吞
--------------------------
吞掉的后果是”这一轮什么都没抽到“，而它与”这一轮确实没有值得记的东西"
在数据上长得一模一样 —— 前者是 bug，后者是常态。抛出去让任务重试，
重试还不行就进 FAILED（看得见）。提示词里那句“不要输出 Markdown 代码
块修饰符”说明模型**会**这么干，所以还有一道去围栏的兜底 —— 那一步与
L2 的整合解析共用，住在 `_model_output.py`。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._model_output import load_array, validate_items

__all__ = [
    "ConflictDecision",
    "ExtractedMemory",
    "ExtractedScene",
    "ExtractionParseError",
    "parse_conflicts",
    "parse_extraction",
]


class ExtractionParseError(ValueError):
    """
    模型输出没法解析成情境列表。

    继承 ValueError（它是“输入不合法”那一类），因此既有的 `except
    ValueError` 接得住；任务层把它当**可重试**错误 —— 模型的这一次输出
    不合法不代表下一次也是。
    """


class ExtractedMemory(BaseModel):
    """一条抽出来的记忆 —— 提示词输出里 `memories[]` 的一项。"""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(
        min_length=1,
        description="完整的、跳出当前对话依然成立的记忆陈述。",
    )

    type: str = Field(
        default="",
        description=(
            "persona / episodic / instruction。**不校验取值域** —— 词表"
            "会长（work_* 四类还没做），而在这里挡掉一个没见过的值等于"
            "让整批抽取失败。落库时是什么就是什么（memory_atom.type 也"
            "没有 CHECK，理由见那个模型）。"
        ),
    )

    priority: int = Field(
        default=50,
        description=(
            "0-100，-1 = 全局死命令。**不校验范围** —— 它是给 LLM 自己用"
            "的阈值信号（不参与排序过滤），越界不是“数据坏了”而是“这次"
            "打分偏了”，为此让整批失败不划算。"
        ),
    )

    source_message_ids: list[str] = Field(
        default_factory=list,
        description=(
            "这条记忆来自哪几条消息（agent_message.id 的字符串形式）。\n\n"
            "⚠️ 字符串不是 UUID：它直接来自模型的输出，而模型可能给出"
            "一个不存在的 id。落库前由服务侧过滤掉解析不出 UUID 的那些"
            "（见 atom_service_impl），这里只保证“是字符串列表”。"
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "episodic 的 activity_start_time / activity_end_time。其它"
            "类型是空对象。落库进 memory_atom.meta。"
        ),
    )


class ExtractedScene(BaseModel):
    """模型输出数组里的一项 —— 一个情境及其记忆。"""

    model_config = ConfigDict(extra="ignore")

    scene_name: str = Field(
        default="",
        description=(
            "情境名。**由模型生成，没有枚举** —— 提示词只要求它单句、"
            "约 30-50 字、全局唯一。落库后不归一化（归一化会打断 L2 的"
            "归并依据）。空串是允许的（模型没给名字时）。"
        ),
    )

    message_ids: list[str] = Field(
        default_factory=list,
        description=(
            "属于这个情境的消息范围。\n\n"
            "⚠️ 提示词**已经不再要求输出它**（2026-10-10）：它不落库"
            "（memory_atom 里没有放它的列 —— 情境的权威定义在 L2），却是"
            "输出里最大的一块（一批 20 条消息 ≈ 800 字符的 UUID），把预算"
            "吃光之后输出会被截断。见 prompts/memory_atom_extract.py 的模块"
            " docstring。\n\n"
            "字段留着是**宽容**：模型偶尔仍会带上它，解析不报错、丢掉即可"
            "（extra=\"ignore\" 同理）。"
        ),
    )

    memories: list[ExtractedMemory] = Field(
        default_factory=list,
        description=(
            "该情境下抽出来的记忆。**空数组合法且常见** —— 提示词明写"
            "「如果整段对话无有意义的记忆，也要输出情境分割结果，"
            "memories 为空数组」。"
        ),
    )


class ConflictDecision(BaseModel):
    """
    一条新记忆的冲突判定 —— 提示词输出数组里的一项。

    ⚠️ `record_id` 是**字符串**，不是 UUID
    -----------------------------------
    它由模型回显（我们给它的新记忆 id）。解析不出来的在服务侧按“这条没有
    判定”处理 —— 与 `ExtractedMemory.source_message_ids` 同一条纪律：
    模型给的 id 只能当线索，不能当身份。
    """

    model_config = ConfigDict(extra="ignore")

    record_id: str = Field(
        default="",
        description="这条判定针对哪条**新**记忆。空串 = 模型没回显，服务侧会忽略它。",
    )

    action: str = Field(
        default="store",
        description=(
            "store / skip / update / merge。\n\n"
            "⚠️ **取值域外的动作按 store 处理**（服务侧），理由见那里的说明："
            "宁可多留一条记忆，也不要在看不懂的动作上删东西。"
        ),
    )

    target_ids: list[str] = Field(
        default_factory=list,
        description="要替换 / 合并掉的**已有**记忆 id。store / skip 时为空。",
    )

    merged_content: str = Field(default="", description="update / merge 时的最终内容。")

    merged_type: str = Field(default="", description="合并后归属的 type。空 = 沿用取定的那条。")

    merged_priority: int | None = Field(
        default=None,
        description=(
            "合并后的 priority。None = 沿用取定的那条 —— 用 None 而不是 50 当"
            "默认：50 是一个**有意义的取值**（一般喜好），拿它当“没给”会让"
            "合并悄悄把优先级降下来。"
        ),
    )

    merged_timestamps: list[str] = Field(
        default_factory=list,
        description=(
            "合并后的时间戳并集（提示词要求所有新旧时间戳去重排序）。落库进"
            " memory_atom.meta['timestamps']，另外用来算 timestamp_start/end。"
        ),
    )


# ============================================================
# 解析
# ============================================================
#
# 剥围栏、`json.loads`、"顶层是不是数组"、逐项 pydantic 校验这四步与 L2 的
# 整合解析**逐字相同**，所以住在 `_model_output.py`（见那个模块的 docstring：
# 这套判据只该有一份，复制三遍就会有三个"容忍到哪一步"的答案）。
#
# 下面两个函数因此只剩"要什么形状"这一件事。


def parse_extraction(text: str) -> list[ExtractedScene]:
    """
    抽取提示词的输出 -> 情境列表。

    Raises:
        ExtractionParseError: 不是合法 JSON、不是数组、或某一项不符合形状。
    """

    return validate_items(
        load_array(text, error=ExtractionParseError),
        ExtractedScene,
        error=ExtractionParseError,
    )


def parse_conflicts(text: str) -> list[ConflictDecision]:
    """
    冲突检测提示词的输出 -> 逐条判定。

    Raises:
        ExtractionParseError: 同上。
    """

    return validate_items(
        load_array(text, error=ExtractionParseError),
        ConflictDecision,
        error=ExtractionParseError,
    )
