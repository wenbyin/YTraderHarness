"""
模型输出 -> JSON 的**共用那几步**：剥围栏、解析、报错时贴原文头尾、逐项校验。

为什么抽出来
----------
本模块的三份提示词（L1 抽取 / L1 冲突检测 / L2 整合）的失败模式是**同一组**：

    模型把 JSON 包在 Markdown 围栏里     提示词明令禁止，但它会
    输出是空的                           一次截断 / 一次空回复
    顶层类型不对                         要数组的给了对象，要对象的给了数组
    某一项缺字段 / 类型不符               pydantic 挡住
    被截断                               尾部戛然而止在半个字符串里

这套判据只该有一份。复制三遍的话，"容忍到哪一步"就有了三个答案 ——
而它们的差异只会在出故障时显形（某一份剥了围栏、另一份没剥，
症状是"抽到一半的任务全 FAILED"）。

⚠️ 顶层形状由调用方挑
------------------
本模块只负责"得到一个 JSON **值**"：要数组走 `load_array`，要对象走
`load_object`。提示词要什么形状是各家自己的事。

⚠️ 错误类型由调用方给
------------------
三个契约各自抛自己的 `ValueError` 子类（`ExtractionParseError` /
`ConsolidationParseError`），而 `error` 参数就是那个类。它们都是
`ValueError` 的子类，因此任务层一律当**可重试**错误 —— 模型的这一次
输出不合法不代表下一次也是。

⚠️ 解析失败要**抛**，不要吞
--------------------------
吞掉的后果是"这一轮什么都没抽到"，而它与"这一轮确实没有值得记的东西"
在数据上长得一模一样 —— 前者是 bug，后者是常态。见 domain/schemas/atom.py
的模块 docstring。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

__all__ = [
    "excerpt",
    "load_array",
    "load_json",
    "load_object",
    "validate_items",
]


#: 模型偶尔会把 JSON 包在 Markdown 代码块里（提示词明令禁止，但它会）。
#:
#: 只在**整个输出**被围栏包住时剥掉 —— 不去做"从任意文本里抠出 JSON"
#: 那种宽容解析：那会把"模型答了一堆解释 + 一小段 JSON"也当成合法输出，
#: 而那种情况该失败（提示词明确要求只输出 JSON）。
_FENCE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n(?P<body>.*?)\n?\s*```\s*$",
    re.DOTALL,
)

#: 报错时贴出来的原文片段的长度（头尾各一段）。
_EXCERPT = 200


def excerpt(raw: str) -> str:
    """
    原文**头尾各一段** + 总长度。

    ⚠️ 为什么头尾都要
    -------------
    只贴开头时，"模型写错了"与"输出被截断了"在日志里长得**一模一样** ——
    而这是两种完全不同的故障：前者要改提示词/换模型，后者要加大预算或
    关掉思考模式（`max_tokens` 是思考与答案共用的）。

    头部永远是好的（错误往往在尾部），所以尾部和长度才是判据：
    尾部戛然而止在半个字符串里 = 被截断；尾部是完整的 JSON 但格式错 =
    模型写错了。
    """

    if len(raw) <= _EXCERPT:
        return f"原文（共 {len(raw)} 字符）：{raw!r}"

    return (
        f"原文共 {len(raw)} 字符。"
        f"头 {_EXCERPT}：{raw[:_EXCERPT]!r}…"
        f"尾 {_EXCERPT}：…{raw[-_EXCERPT:]!r}"
    )


def load_json(text: str, *, error: type[ValueError]) -> Any:
    """
    模型输出 -> 任意 JSON 值。顶层形状**不校验**（由 `load_array` /
    `load_object` 决定）。

    Raises:
        error: 空输出、不是合法 JSON。
    """

    raw = (text or "").strip()

    if not raw:
        raise error("模型返回了空输出。")

    fenced = _FENCE.match(raw)

    if fenced is not None:
        raw = fenced.group("body").strip()

    try:
        return json.loads(raw)

    except json.JSONDecodeError as exc:
        raise error(
            f"模型输出不是合法 JSON：{exc}。{excerpt(raw)}"
        ) from exc


def load_array(text: str, *, error: type[ValueError]) -> list[Any]:
    """模型输出 -> 一个 JSON 数组（未逐项校验）。"""

    payload = load_json(text, error=error)

    if not isinstance(payload, list):
        raise error(
            "模型输出的顶层不是数组（提示词要求一个数组）。"
            f"实际类型：{type(payload).__name__}。{excerpt(text or '')}"
        )

    return payload


def load_object(text: str, *, error: type[ValueError]) -> dict[str, Any]:
    """
    模型输出 -> 一个 JSON 对象（未逐字段校验）。

    ⚠️ 数组**不算**合法：提示词要的是对象。把它容忍成"包了一层"会掩盖
    一次真的没按格式答的输出，而那种输出我们宁可失败重试（见上面的
    `load_array` 同一条取舍）。
    """

    payload = load_json(text, error=error)

    if not isinstance(payload, dict):
        raise error(
            "模型输出的顶层不是对象（提示词要求一个 JSON 对象）。"
            f"实际类型：{type(payload).__name__}。{excerpt(text or '')}"
        )

    return payload


def validate_items(
        payload: list[Any],
        model: type[BaseModel],
        *,
        error: type[ValueError],
) -> list[Any]:
    """
    逐项校验。错误信息里带上该项的原文 —— 模型到底吐了什么只在这一次
    调用里存在，事后无从复现。
    """

    validated: list[Any] = []

    for index, item in enumerate(payload):

        if not isinstance(item, dict):
            raise error(
                f"数组第 {index} 项不是对象，而是 {type(item).__name__}。"
                f"该项原文：{item!r}"
            )

        try:
            validated.append(model.model_validate(item))

        except ValidationError as exc:
            raise error(
                f"数组第 {index} 项不符合形状：{exc}。"
                f"该项原文：{item!r}"
            ) from exc

    return validated
