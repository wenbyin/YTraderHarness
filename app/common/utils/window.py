"""
把 clean 正文按**标题小节**切成生成窗口。

为什么按小节切而不是按固定行数
------------------------------
本项目的数据是结构化事件档案，形如：

    ## 二、分类事件详情
    ### 1. 股东户数（81 条）
    记录公司股东户数的定期变化，户数下降通常意味着筹码趋于集中……
    #### 2026-08-22 ｜ 股东户数
    2026年8月22日，公司披露「股东户数」事项：截止2026-06-30，……
    #### 2026-04-25 ｜ 股东户数
    ……

按 `####` 切，一个窗口正好是**一条事件** —— 语义自洽，生成的问题天然
落在单条事件内，后面与 chunk 求交时目标也明确。

按固定行数切会让一个问题跨窗口、或答案被切在两半，边界处理很麻烦，
而且换一份结构不同的文档就失效。

为什么把 `###` / `##` 也当分界
-----------------------------
只按 `####` 切的话，每个分类的引言（`### 1. 股东户数（81 条）` 那两行）
会被并进**上一个分类的最后一条事件**里 —— 那一条窗口就混进了不属于它
的内容，生成的问题可能问的是引言而不是事件。

按 2~4 级标题切，引言自成一段，事件段保持干净。

一级标题（`# 东方财富…`）**不切**：它下面紧跟的是文档头部
（数据来源、生成时间等），单独成窗价值不大；把它并进第一个窗口即可。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "DEFAULT_BLOCK_CHARS",
    "DEFAULT_MAX_CHARS",
    "Section",
    "split_blocks",
    "split_sections",
]


#: 大块模式的默认块大小（字符）。
#:
#: 与 split_sections 的小节窗口是**两种用途**：
#:
#:     split_sections  一个窗口 = 一条事件（百字符量级）
#:                     适合"从这一条事件里出题"
#:
#:     split_blocks    一个块尽量大，装下**尽可能多的相邻事件**
#:                     适合"出跨事件的题"—— 模型能同时看到多条记录，
#:                     才可能问出"哪一年户数最高"这类需要横向比较的问题
#:
#: 为什么不是"整篇一次性塞进去"：实测某份档案正文 51.7 万字符，中文
#: 大致 1~1.5 字/token，折合 **35 万到 50 万 token** —— 超出主流模型的
#: 上下文窗口（Kimi k2 是 256k）。所以只能"尽量大"，不能"一定一次"。
#:
#: 3 万这个值：多数模型的窗口装得下，单次调用又不至于慢到不可用。
#: 真正的上限由调用方按所用模型定 —— 见请求参数 max_block_chars。
DEFAULT_BLOCK_CHARS = 30_000


#: 单个窗口的字符上限，超过就按行再切。
#:
#: 实测（比亚迪那份 51.7 万字符的档案）：1081 个窗口里 **中位数只有 100
#: 字符**，但最大的是 **20,568 字符** —— 全是「对外担保」，自带长表格。
#:
#: 这 38 个大窗口装着约 **73% 的正文**，所以不能一丢了之；而 2 万字符
#: 塞给 LLM 既慢又容易被它"抓大放小"（长上下文里的细节召回本来就差）。
#: 按行切成小窗口，两边的问题都解决。
#:
#: 2000 这个值：p90 是 286、p99 是 14995 —— 取 2000 意味着只有约 4%
#: 的窗口会被再切，绝大多数保持"一个窗口 = 一条事件"的语义。
DEFAULT_MAX_CHARS = 2000


#: 分界用的标题级别：2~4 级（`##` / `###` / `####`）。
#:
#: `re.M` 让 `^` 匹配行首 —— 正文里偶尔出现的 `####`（比如引用的示例）
#: 不该被当成标题，只有**行首**的才算。
_BOUNDARY = re.compile(r"^(#{2,4})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    """
    一个窗口。

    start/end 是相对**整份 clean 正文**的字符偏移，且满足
    `clean_text[start:end] == text` —— 与 `document_chunk` 的
    `start_index`/`end_index` 是同一个坐标系，后面才能直接求交。

    title 是标题行去掉 `#` 之后的文字（如 `2026-08-22 ｜ 股东户数`）；
    文档头部那段没有标题，title 为空串。
    """

    start: int
    end: int
    title: str
    text: str


def split_sections(
        text: str,
        *,
        max_chars: int | None = DEFAULT_MAX_CHARS,
) -> list[Section]:
    """
    按 2~4 级标题切分，再把过大的窗口按行切开。
    **不丢字符** —— 所有窗口首尾相接正好还原原文。

    没有任何标题时返回**单个**覆盖全文的窗口，而不是空列表：调用方
    可以无差别地遍历结果，不必处理"这份文档没标题"这种特例。

    `max_chars=None` 表示不做二次切分（调试或想看真实小节分布时用）。
    """

    if not text:
        return []

    boundaries = list(_BOUNDARY.finditer(text))

    if not boundaries:
        return [Section(start=0, end=len(text), title="", text=text)]

    sections: list[Section] = []

    # 第一个标题之前的头部（文档名、数据来源、生成时间……）。
    # 单独成窗而不是丢掉：里面有文档级的说明，丢掉会让这部分正文
    # 永远不参与生成，而它确实可能承载"这份档案是什么"这类问题。
    if boundaries[0].start() > 0:
        head_end = boundaries[0].start()
        sections.append(
            Section(
                start=0,
                end=head_end,
                title="",
                text=text[:head_end],
            )
        )

    for index, match in enumerate(boundaries):

        start = match.start()

        end = (
            boundaries[index + 1].start()
            if index + 1 < len(boundaries)
            else len(text)
        )

        sections.append(
            Section(
                start=start,
                end=end,
                title=match.group(2).strip(),
                text=text[start:end],
            )
        )

    return _split_oversized(sections, text, max_chars)


def split_blocks(
        text: str,
        *,
        max_chars: int = DEFAULT_BLOCK_CHARS,
) -> list[Section]:
    """
    切成**尽量大的块**，块内横跨多个小节。

    与 `split_sections` 的区别是切点密度：

        split_sections  每个标题都切 -> 一个窗口 = 一条事件
        split_blocks    只在**装不下时**才切 -> 一个块 = 尽量多的事件

    为什么需要它：只有让模型同时看到多条相邻记录，才可能问出
    **跨事件**的问题（"2024 年哪一次户数下降最多？"）—— 一个窗口只有
    一条事件时，这类问题根本无从提起。而这类问题恰恰是真实用户会问的，
    也是检索最容易失败的（答案分散在多处）。

    切点仍然落在标题边界上：块内多留几条事件没问题，但**不能从一条
    事件的中间切断** —— 那样两条相邻事件会被拆到两个块里，跨事件的
    问题两边都看不全。

    **不丢字符**，与 `split_sections` 同样的保证。

    ⚠️ `max_chars` 是**硬承诺**：返回的每一块都不超过它（除了单行本身
    就超长的极端情况）。

    这一点是踩过坑才写死的：早先的做法是"单条事件超预算时让它独占一块，
    不切"，理由是"别把一条事件切成两半"。但那让 `max_chars` 变成一个
    **不可靠的承诺** —— 调用方按 2000 估算耗时，实际却可能拿到一个
    20568 字符的块（对外担保那种长表格段就是这样），于是"为什么设了
    小块还是超时"变成了一个查不出来的问题。

    现在退回按行切分：长事件被切开，`max_chars` 说到做到。
    """

    if not text:
        return []

    # 用带二次切分的版本取候选单元：超长小节会被按行切开，
    # 这样下面按 max_chars 归组时不会有"单条就超预算"的漏网之鱼。
    units = split_sections(text, max_chars=max_chars)

    if not units:
        return []

    blocks: list[Section] = []

    start = units[0].start
    end = units[0].end
    title = units[0].title

    for unit in units[1:]:

        # 装得下就继续吃进来（这就是"跨 section"的来源）。
        if unit.end - start <= max_chars:
            end = unit.end
            # 标题记**首个**小节那条：块横跨多条事件时，标题只是给人
            # 看的溯源标签，取第一条最不容易误导。
            continue

        blocks.append(
            Section(
                start=start, end=end, title=title,
                text=text[start:end],
            )
        )
        start, end, title = unit.start, unit.end, unit.title

    blocks.append(
        Section(
            start=start, end=end, title=title, text=text[start:end],
        )
    )

    return blocks


def _split_oversized(
        sections: list[Section],
        text: str,
        max_chars: int | None,
) -> list[Section]:
    """
    把超过 max_chars 的窗口**按行**再切。

    切出来的子窗口**继承原标题**，并加上 `#1` `#2` 这样的序号后缀 ——
    否则一道题只能溯源到"某条 20K 字符的事件"，而不知道具体是哪一段，
    排查时会很痛苦。

    按行切而不是按字符切：表格行、列表项都是完整的语义单位，从中间
    截断会产出读不通的窗口，LLM 也就问不出好问题。
    """

    if max_chars is None or max_chars <= 0:
        return sections

    result: list[Section] = []

    for section in sections:

        if section.end - section.start <= max_chars:
            result.append(section)
            continue

        # 按行扫描，累积到接近 max_chars 就收一个窗口。
        # 单行本身就超长时（不会在这份数据里出现）单独成窗，
        # 而不是死循环。
        piece_start = section.start
        cursor = section.start

        while cursor < section.end:

            line_end = text.find("\n", cursor, section.end)

            if line_end == -1:
                line_end = section.end

            next_cursor = min(line_end + 1, section.end)

            if (
                    next_cursor - piece_start > max_chars
                    and cursor > piece_start
            ):
                result.append(
                    Section(
                        start=piece_start,
                        end=cursor,
                        title=section.title,
                        text=text[piece_start:cursor],
                    )
                )
                piece_start = cursor

            cursor = next_cursor

        if piece_start < section.end:
            result.append(
                Section(
                    start=piece_start,
                    end=section.end,
                    title=section.title,
                    text=text[piece_start:section.end],
                )
            )

    # 给同一标题下被切开的多个窗口加序号。只有真的切开了才加 ——
    # 没切开就加 `#1` 会让绝大多数窗口的标题平白多一个后缀。
    counts: dict[str, int] = {}

    for section in result:
        counts[section.title] = counts.get(section.title, 0) + 1

    numbered: list[Section] = []
    seen: dict[str, int] = {}

    for section in result:

        if counts[section.title] <= 1:
            numbered.append(section)
            continue

        index = seen.get(section.title, 0) + 1
        seen[section.title] = index

        numbered.append(
            Section(
                start=section.start,
                end=section.end,
                title=f"{section.title} #{index}",
                text=section.text,
            )
        )

    return numbered
