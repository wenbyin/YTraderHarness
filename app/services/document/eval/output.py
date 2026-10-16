"""
模型输出的**形状**与解析。

这里只有两件事：pydantic 形状，以及"把模型吐出来的文本变成那个形状"。
业务判定（id 认不认识、答案是不是逐字子串）在 `generator.py` /
`adjudication.py` 里 —— 解析只管"它是不是一个合法 JSON 且字段对得上"。

⚠️ 这几步与 `app/services/memory/domain/schemas/_model_output.py` **是重复的**
------------------------------------------------------------------------
剥围栏、`json.loads`、顶层形状检查、报错时贴原文头尾 —— 那四步一模一样，
而那个模块的 docstring 明确警告过"复制三遍就会有三个『容忍到哪一步』的
答案"。

这里还是复制了一份，理由是**归属**：那份住在 memory 的 domain 包里，
让它被 document 服务 import 就等于 document 依赖 memory 的内部模块。
把它提到 `app/common/` 是对的做法，但那要动 memory 那三个调用点 ——
属于独立改动。

所以：**这份是已知的、有意的重复**。下次有人要动其中一份的容忍度时，
两份一起看。真正的解法是把 `_model_output.py` 提到公共位置。

⚠️ `chunk_id` 保持 `str`，不转 UUID
---------------------------------
转了就丢掉了"模型到底写了什么"。而"认不出的 id"这条拒绝恰恰需要原始
串才能给出可读的原因（"你写的 X 不在本窗口内"）—— 转 UUID 失败只会得到
一句 pydantic 的类型错误，看不出模型编的是哪个值。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "AdjudicationVerdict",
    "ModelOutputError",
    "QAGenerationItem",
    "WindowQuestions",
    "excerpt",
    "parse_adjudication_verdict",
    "parse_window_questions",
]


class ModelOutputError(ValueError):
    """
    模型输出不是可用的形状。

    ValueError 的子类：任务层一律当**可重试**错误 —— 这一次输出不合法
    不代表下一次也是。
    """


#: 模型偶尔会把 JSON 包在 Markdown 代码块里（提示词明令禁止，但它会）。
#:
#: 只在**整个输出**被围栏包住时剥掉 —— 不做"从任意文本里抠出 JSON"那种
#: 宽容解析：那会把"模型答了一堆解释 + 一小段 JSON"也当成合法输出，而
#: 那种情况该失败（提示词明确要求只输出 JSON）。
_FENCE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n(?P<body>.*?)\n?\s*```\s*$",
    re.DOTALL,
)

#: 报错时贴出来的原文片段长度（头尾各一段）。
_EXCERPT = 200


def excerpt(raw: str) -> str:
    """
    原文**头尾各一段** + 总长度。

    ⚠️ 头尾都要：只贴开头时，"模型写错了"与"输出被截断了"在日志里长得
    一模一样 —— 而这是两种不同的故障。尾部戛然而止在半个字符串里 =
    被截断（要加大预算或关思考模式）；尾部是完整 JSON 但格式错 = 模型
    写错了（要改提示词或换模型）。
    """

    if len(raw) <= _EXCERPT:
        return f"原文（共 {len(raw)} 字符）：{raw!r}"

    return (
        f"原文共 {len(raw)} 字符。"
        f"头 {_EXCERPT}：{raw[:_EXCERPT]!r}…"
        f"尾 {_EXCERPT}：…{raw[-_EXCERPT:]!r}"
    )


class QAGenerationItem(BaseModel):
    """
    模型对**一个源 chunk** 该产出的东西。

    `extra="ignore"`：模型偶尔会多给一个 "reason" 之类的字段。那种多给
    无害，不该让整个窗口作废 —— 而 `extra="forbid"` 会。
    """

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(
        description=(
            "这道题所依据的源 chunk 的 id。"
            "必须逐字复制自输入里给出的 id。"
        ),
    )

    question: str = Field(
        description="一个能用该 chunk 内原文直接回答的问题。",
    )

    answer_original: str = Field(
        description=(
            "该问题的答案，必须是从该 chunk 正文里逐字复制的连续片段。"
        ),
    )


class WindowQuestions(BaseModel):
    """
    一次出题调用的返回。

    `items` 为空是**合法**结果（窗口里确实没有可问的明确事实）。
    解析层不把它当错误 —— 把它当错误会让正常窗口反复重试。
    """

    model_config = ConfigDict(extra="ignore")

    items: list[QAGenerationItem] = Field(default_factory=list)

    insufficient_fact_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "没能凑够要求题数的源 chunk id。"
            "明确列出来而不是默默少出：服务端要能区分"
            "「这个 chunk 本来就没那么多明确事实」（正常）与"
            "「模型这次只出了一道就停了」（可能该重试）。"
        ),
    )


class AdjudicationVerdict(BaseModel):
    """跨 chunk 裁定的返回。"""

    model_config = ConfigDict(extra="ignore")

    answerable_chunk_ids: list[str] = Field(default_factory=list)

    evidence_excerpts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "chunk_id -> 支撑判定的**逐字原文片段**。"
            "服务端会拿它做精确匹配 —— 伪造的片段会让这条判定作废。"
        ),
    )


# ============================================================
# 解析
# ============================================================


def _load_object(text: str) -> dict[str, Any]:
    """模型输出 -> JSON 对象。不做字段校验。"""

    raw = (text or "").strip()

    if not raw:
        raise ModelOutputError(
            "模型返回了空输出（提示词要求一个 JSON 对象）。"
        )

    fenced = _FENCE.match(raw)

    if fenced is not None:
        raw = fenced.group("body").strip()

    try:
        payload = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise ModelOutputError(
            f"模型输出不是合法 JSON：{exc}。{excerpt(raw)}"
        ) from exc

    if not isinstance(payload, dict):
        # ⚠️ 数组**不算**合法（哪怕它只包了一层）。把它容忍成"外面少了个
        # 对象壳"会掩盖一次真的没按格式答的输出，而那种输出宁可失败重试。
        raise ModelOutputError(
            "模型输出的顶层不是对象（提示词要求一个 JSON 对象）。"
            f"实际类型：{type(payload).__name__}。{excerpt(raw)}"
        )

    return payload


def parse_window_questions(text: str) -> WindowQuestions:
    """
    出题提示词的输出 -> `WindowQuestions`。

    Raises:
        ModelOutputError: 不是合法 JSON 对象、或字段形状不符。
                     调用方一律当可重试错误。
    """

    payload = _load_object(text)

    try:
        return WindowQuestions.model_validate(payload)

    except ValidationError as exc:
        raise ModelOutputError(
            f"出题输出不符合形状：{exc}。{excerpt(text)}"
        ) from exc


def parse_adjudication_verdict(text: str) -> AdjudicationVerdict:
    """
    裁定提示词的输出 -> `AdjudicationVerdict`。

    Raises:
        ModelOutputError: 同上。
    """

    payload = _load_object(text)

    try:
        return AdjudicationVerdict.model_validate(payload)

    except ValidationError as exc:
        raise ModelOutputError(
            f"裁定输出不符合形状：{exc}。{excerpt(text)}"
        ) from exc
