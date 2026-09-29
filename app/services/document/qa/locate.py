"""
把 LLM 给出的**逐字摘录**定位回正文的字符区间。

为什么不直接要 LLM 报行号
-------------------------
行号是 LLM 最容易数错、且**错了不报错**的东西 —— 它不真的"看"行边界。
一个数错的行区间换算成 offset 之后，可能仍然落在某个合法 chunk 上，
只是那个 chunk 与问题无关。数据集于是带着错误的 ground truth 一路流到
评估，最后表现为"召回率莫名偏低"这种极难归因的症状。

摘录定位则相反：**精确、且可校验**。搜不到就是 LLM 编了原文，直接丢弃
该样本并记下原因，而不是把噪音写进数据集。

（产物里**仍然输出行号** —— 由 offset 反推，见 line_of。所以"要行号"
这个需求不受影响，只是行号不再是**定位依据**。）

搜索范围限定在窗口内
--------------------
摘录只在**它所属的那个窗口**里找，不全文搜。理由：同一份档案里
"2025年3月15日，公司披露「股东户数」事项"这类句子会出现很多次
（不同年份、不同公司），全文搜索会命中第一个，而那多半不是 LLM
当时看到的那一条。限定在窗口内，定位结果必然落在 LLM 实际读过的
那段正文里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Location",
    "Span",
    "line_of",
    "locate_quotes",
]


@dataclass(frozen=True)
class Span:
    """正文里的一个字符区间，左闭右开。"""

    start: int
    end: int


@dataclass(frozen=True)
class Location:
    """
    一次定位的结果。

    occurrences > 1 表示这条摘录在窗口内出现了多次，取的是**第一次**。
    调用方可以据此决定是否丢弃 —— 同一个串在一条事件里出现多次，
    说明它不足以唯一确定答案位置。
    """

    quote: str
    span: Span
    occurrences: int

    #: True = 精确匹配；False = 退化成空白折叠后匹配（LLM 的空白与
    #: 原文有出入，常见于换行/多空格）。退化匹配仍然可信 —— 匹配到的
    #: 是原文真实的字符区间，只是"相等"的判定放宽了。
    exact: bool


def locate_quotes(
        quotes: list[str],
        text: str,
        window: Span,
) -> tuple[list[Location], list[str]]:
    """
    在 `text[window.start:window.end]` 范围内定位每一条摘录。

    返回 `(定位成功的, 失败原因)`。

    失败不抛异常：一条摘录定位不到，只该让**这一条样本**作废，不该
    让整个数据集生成失败 —— LLM 偶尔编一句原文是常态。
    """

    located: list[Location] = []
    failures: list[str] = []

    for quote in quotes:

        cleaned = (quote or "").strip()

        if not cleaned:
            failures.append("空摘录")
            continue

        hit = _find(text, cleaned, window)

        if hit is None:
            failures.append(f"摘录在窗口内找不到：{cleaned[:40]!r}")
            continue

        start, exact = hit

        located.append(
            Location(
                quote=cleaned,
                span=Span(start=start, end=start + len(cleaned)),
                # count 只在**窗口范围内**数，且用同样的匹配方式，
                # 否则退化匹配下会出现"精确计数为 0 但仍算多次"的矛盾。
                occurrences=_count(
                    text, cleaned, window,
                    exact=exact,
                ),
                exact=exact,
            )
        )

    return located, failures


def line_of(text: str, offset: int) -> int:
    """
    offset 对应的**行号**（1 起）。

    用 `text.count("\\n", 0, offset) + 1` 而不是 splitlines：后者要
    复制整份正文（本项目实测 51 万字符），而这个函数每道题要调两次。
    """

    if offset <= 0:
        return 1

    return text.count("\n", 0, min(offset, len(text))) + 1


# ============================================================
# internals
# ============================================================


def _find(
        text: str,
        needle: str,
        window: Span,
) -> tuple[int, bool] | None:
    """
    先精确找，找不到再退化成空白折叠后再找。

    返回 `(offset, 是否精确匹配)`，或 None 表示两种都失败。
    """

    start = text.find(needle, window.start, window.end)

    if start != -1:
        return start, True

    return _find_loose(text, needle, window)


#: 连续的空白（含换行）。折叠成一个空格再比。
_WHITESPACE = re.compile(r"\s+")


def _find_loose(
        text: str,
        needle: str,
        window: Span,
) -> tuple[int, bool] | None:
    """
    空白折叠后匹配，并把偏移**映射回原文**。

    做法：把窗口内的正文折叠空白，同时记下每个折叠后字符对应的原文
    下标。折叠后的串里找到 needle（同样折叠过），再用映射还原起点。

    ⚠️ 还原出来的区间长度是**原文的**还是**折叠串的**？两者不同：
    折叠后 `"a   b"`（5 字符）变成 `"a b"`（3 字符）。这里返回的
    start 是原文偏移，长度用**原文**的 —— 即从 start 起、覆盖到
    匹配结束位置在原文中的下标。这样 `text[start:end]` 是真实存在
    的一段原文，只是它的空白与摘录不完全一致。
    """

    collapsed_needle = _WHITESPACE.sub(" ", needle).strip()

    if not collapsed_needle:
        return None

    chars: list[str] = []
    #: 折叠串第 i 个字符 -> 原文下标
    mapping: list[int] = []

    in_space = False

    for index in range(window.start, window.end):

        char = text[index]

        if char.isspace():
            if not in_space and chars:
                chars.append(" ")
                mapping.append(index)
            in_space = True
            continue

        in_space = False
        chars.append(char)
        mapping.append(index)

    collapsed = "".join(chars).rstrip()

    position = collapsed.find(collapsed_needle)

    if position == -1:
        return None

    return mapping[position], False


def _count(
        text: str,
        needle: str,
        window: Span,
        *,
        exact: bool,
) -> int:
    """
    在窗口内数这条摘录出现了几次。

    退化匹配下**不能**用 `text.count` —— 那是精确计数，会得到 0，
    与"我们刚刚确实用折叠匹配找到了它"自相矛盾。所以这里统一走
    同一个匹配函数，用它的副作用计数不划算，直接循环推进。
    """

    count = 0
    cursor = window.start

    while cursor < window.end:

        hit = _find(text, needle, Span(start=cursor, end=window.end))

        if hit is None:
            break

        count += 1
        cursor = hit[0] + 1

        # 防死循环：needle 非空时 cursor 必然前进。
        if cursor <= hit[0]:
            break

    return count
