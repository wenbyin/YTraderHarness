"""
日期的**唯一**一份词汇表。

为什么要单列一个模块
--------------------
同一批语料里日期有两种写法，而系统的每一层都各需要它们的一种形态：

    正文散文      `2007年4月12日`
    小节标题      `#### 2007-04-12 ｜ 对外担保`

    索引侧（字段值）      存"这条块含哪些日期"（`content_dates`，精确匹配）
    查询侧（筛选/加权键） 与字段**同一种形态**，否则匹配静默失败
    查询侧（短语）        给出**在正文里找得到**的那两种写法

用途不同、形态不同，如果各写一份正则迟早会漂 —— 而漂的表现是"某些日期
的题莫名找不到"，看起来像检索质量问题。所以集中在这里。

    形态一  canonical_key   补零 `YYYYMMDD`  ← 字段值 / 查询键
    形态二  phrase_forms    补零短横线 + 中文 ← `match_phrase` 的串

⚠️ 形态一**必须补零**（不补零会撞键，见 `canonical_key`）；形态二的短横线
也必须补零（正文小标题就那么写的）。两者都补零，但不是同一个串 ——
前者没有分隔符，后者有。

⚠️ 曾经还有第三个形态：`index/es.py` 里一个 `cjk_date` 分析器把日期压成
原子词项、供 `content.dates` 子字段精确匹配。它已删除 —— 那个分析器用
char_filter 实现，**做不到条件补零**，于是只能产出不补零的形态，
继承了撞键问题；而字段那条路（本模块算的键）精确且无歧义，把它完全
覆盖了。所以现在**只有**本模块说了算。
"""

from __future__ import annotations

import re

__all__ = [
    "canonical_key",
    "canonical_keys",
    "phrase_forms",
    "scan",
]

#: 中文写法。前后各挡一位数字，免得从 `12007年4月12日` 里抠出半个日期。
_CN_DATE = re.compile(r"(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})日(?!\d)")

#: 短横线写法。同上挡数字。
_DASH_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")


def _in_range(year: int, month: int, day: int) -> bool:
    """
    粗筛：只挡明显不可能的（13 月、40 日）。

    不做日历校验（闰年 / 大小月）：目的只是"别拿显然不是日期的串去污染
    查询与字段"。真写错了日期，多一个健也筛不出东西，无害。
    """

    return 1 <= month <= 12 and 1 <= day <= 31


def scan(
        text: str,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    """
    扫一遍文本，返回 `(有效日期, 明显不是日期的串)`。

    两个要点：

    - 两种写法**合并后按出现位置**排序、并按 (年,月,日) 去重：不这么做
      的话 `2007-04-12 与 2007年4月12日` 会被当成两天。
    - 越界的**留下来报**，不是静默丢掉：调用方（改写器）要能把它写进
      `detail` —— 否则"这条题里的日期没被认出来"与"这条题没有日期"
      在数据上完全一样。
    """

    found: list[tuple[int, int, int]] = []
    skipped: list[str] = []

    matches = sorted(
        [
            (match.start(), match.groups())
            for match in _CN_DATE.finditer(text)
        ]
        + [
            (match.start(), match.groups())
            for match in _DASH_DATE.finditer(text)
        ],
        key=lambda item: item[0],
    )

    for _, groups in matches:

        year, month, day = (int(part) for part in groups)

        if not _in_range(year, month, day):
            skipped.append(f"{year}-{month}-{day}")
            continue

        triple = (year, month, day)

        if triple not in found:
            found.append(triple)

    return found, skipped


def canonical_key(year: int, month: int, day: int) -> str:
    """
    规范键：**补零**的 `YYYYMMDD`（`20070402`）。

    用于字段值（写入时算）与查询侧的筛选/加权键（查询时算）。

    ⚠️ **必须补零。** 不补零会撞键：

        2007-11-02  ->  "2007112"
        2007-01-12  ->  "2007112"    ← 两个不同的日子，同一个键

    而 `content_dates` 是 `terms` 精确匹配 —— 撞键的后果是"A 日的题被 B 日
    的块命中"，且**不报错**，只表现为排序莫名其妙。

    （第一版为了迁就 ES `pattern_replace` 做不到条件补零，把两边都统一到
    了不补零那一端 —— 那是拿正确性换实现方便。分析器那条路后来整体删掉了，
    形态由这里说了算。）
    """

    return f"{year:04d}{month:02d}{day:02d}"


def canonical_keys(text: str) -> list[str]:
    """一段文本里的全部日期键（去重、保序）。越界的那些直接不要。"""

    found, _ = scan(text)

    return [canonical_key(year, month, day) for year, month, day in found]


def phrase_forms(year: int, month: int, day: int) -> tuple[str, str]:
    """
    给人看、也给 `match_phrase` 用的两种写法：**短横线补零** + 中文不补零。

    短横线那条**必须补零**：它要在 `content` 里按位置相邻匹配，而正文的
    `#### 2007-04-12 ｜ 对外担保` 小标题写的就是补零形式 —— 不补零就
    对不上那一条。

    （与 `canonical_key` 不矛盾：两者都补零，但**面向的字符串**不同 ——
    那个没有分隔符（喂字段、当查询键），这个有（拿去在正文里找）。）
    """

    return (
        f"{year:04d}-{month:02d}-{day:02d}",
        f"{year}年{month}月{day}日",
    )
