"""
写入侧的日期与向量前缀（B6 / B7 / B8 / C9 / C10）。

这三条改的都是**写侧**——它们的产物是索引里的字段与向量，改完必须重灌
（或重算向量）才生效。所以用例都盯在"算出来的值对不对"，而不是"库里
有没有"：库里的东西由重灌决定，算错了才是我这边的责任。
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.document.dates import (
    canonical_key,
    canonical_keys,
    phrase_forms,
    scan,
)
from app.services.document.enricher.common import TxtCommonChunkEnricher
from app.services.document.index.es import _date_clauses, _phrase_clauses
from app.services.document.index.milvus import _encode_dates


# ============================================================
# 日期词汇表：一份形态，四个调用方
# ============================================================


def test_both_written_forms_give_the_same_key() -> None:
    """
    中文与短横线写法必须归到**同一个键**。

    这是 B6 成立的前提：索引期按这个键写字段，查询期按这个键查 ——
    "查询写中文、正文写短横线"能对上，全靠它俩是同一个串。
    （ES 侧那个负责同样事情的 char_filter 分析器已删除 —— 它做不到条件
    补零，会把两个不同的日子压成同一个词项。）
    """

    assert canonical_keys("于2015-05-20召开") == ["20150520"]
    assert canonical_keys("于2015年5月20日召开") == ["20150520"]
    assert canonical_key(2015, 5, 20) == "20150520"


def test_keys_are_padded_and_deduped_in_order() -> None:
    """
    补零；两种写法写同一个日期只算一次。

    ⚠️ 补零不是好看：不补零会**撞键**（见下一条）。
    """

    assert canonical_keys("2007年4月2日 与 2007-04-02") == ["20070402"]
    assert canonical_keys("2007年4月2日…2009年1月9日") == [
        "20070402", "20090109",
    ]


def test_single_digit_month_and_day_do_not_collide() -> None:
    """
    不补零时这两条会撞成同一个键（`2007112`）—— 而它是 `terms` 精确匹配的
    值，撞键的后果是"A 日的题被 B 日的块命中"，且**不报错**。

    这是被用户一句「为什么不统一为 YYYYMMDD」问出来的：第一版为了迁就
    ES char_filter 做不到条件补零，把两端都统一到了不补零那一头。
    """

    november = canonical_key(2007, 11, 2)
    january = canonical_key(2007, 1, 12)

    assert november == "20071102"
    assert january == "20070112"
    assert november != january


def test_phrase_forms_pad_the_dash_form_only() -> None:
    """
    拿去 `match_phrase` 的那两种写法：**短横线补零、中文不补**。

    补零是必须的：正文的小标题写 `#### 2007-04-02 ｜ …`，不补零就对不上
    那一条。中文不补是因为正文就那么写的。
    """

    assert phrase_forms(2007, 4, 2) == ("2007-04-02", "2007年4月2日")


def test_impossible_dates_are_reported_not_silently_dropped() -> None:
    """13 月 40 日不是日期：不给键，但要**报出来**（否则与"没写日期"同形）。"""

    found, skipped = scan("这家公司2007年13月40日干了什么？")

    assert found == []
    assert skipped == ["2007-13-40"]


def test_digits_adjacent_to_a_date_do_not_create_half_dates() -> None:
    """`12007年4月12日` 不能抠出 `2007年4月12日` —— 前后各挡一位数字。"""

    assert canonical_keys("编号12007年4月12日") == []


# ============================================================
# ES 侧：字段子句 / 短语子句
# ============================================================


def test_date_clauses_target_the_keyword_field() -> None:
    """
    `date_keys` 打的是 `content_dates` 字段（`terms`），去重、去空。

    空串不能进 `terms`：那会让 ES 去找一个空词项（永远不匹配），
    而且掩盖了"这里本该有个键"。
    """

    clauses = _date_clauses(["20070412", "", "20070412", "20150520"])

    assert len(clauses) == 1
    assert clauses[0]["terms"]["content_dates"] == ["20070412", "20150520"]
    assert clauses[0]["terms"]["boost"] > 0
    assert _date_clauses([]) == []


def test_phrase_clauses_and_date_clauses_are_independent_boosts() -> None:
    """
    两条通道**同时**存在：短语（在正文里找串）与字段（问"这块有没有这一天"）。

    实测里短语那一条把 keyword 名次从 140 推到 4；字段那一条是它的补充
    （查询与正文写法不同、短语匹配不上时仍然有效）。
    """

    assert _phrase_clauses(["2007-04-12"])[0]["match_phrase"]["content"]
    assert _date_clauses(["20070412"])[0]["terms"]["content_dates"]


# ============================================================
# Milvus 侧：只存不筛
# ============================================================


def test_milvus_encodes_dates_with_single_delimiters() -> None:
    """
    `|a|b|` 而不是 `|a||b|`：每个键都被**单个** `|` 包住。

    凑成双分隔符时，`like '%|a|%'` 与 `like '%|b|%'` 看起来都能命中，
    但按"每个键被 | 包住"去解析就会漏掉中间那些 —— 而这一列将来是要
    被筛的（Milvus 的标量字段只能当 filter；当前只写不读，理由见
    `index/milvus.py` 的 `_SCALAR_FIELDS`）。
    """

    assert _encode_dates(["20070412", "20150520"]) == "|20070412|20150520|"
    assert _encode_dates(["20070412"]) == "|20070412|"
    assert _encode_dates([]) == ""

    encoded = _encode_dates(["a", "b", "c"])

    assert encoded.strip("|").split("|") == ["a", "b", "c"]


# ============================================================
# 前缀与 section（C10 / B8）
# ============================================================


_DOC_METADATA = {
    "title": "荣安地产·风险、法律与上市状态",
    "file_name": "07_风险、法律与上市状态.md",
    "document_key": "000517.SZ",
    "dtd_name": "风险、法律与上市状态(COMPANY_EVENT_RISK_COMPLIANCE)",
}


def _enrich_sequence(contents: list[str]):
    """
    跑一遍真增强（不碰库、不碰模型），按顺序返回那几条 chunk。

    取**一批**而不是一条：section 是靠游标顺序累积的，单条测不出"块首的
    标题在上一块"这件事 —— 而那正是这批文档的形状。
    """

    import asyncio

    chunks = [
        SimpleNamespace(
            chunk_index=index,
            content=content,
            metadata={},
            locator={},
        )
        for index, content in enumerate(contents)
    ]

    asyncio.run(
        TxtCommonChunkEnricher().enrich(
            chunks, config={}, document_metadata=dict(_DOC_METADATA),
        )
    )

    return chunks


def _enrich_one(content: str):
    """单条版（够用的地方就别铺开）。"""

    return _enrich_sequence([content])[0]


def _prefix_of(chunk) -> str:
    """
    `embed_text` 的**前缀**那段（向量那一路真正补的上下文）。

    断言要打在它上面而不是整个 `embed_text`：后者 = 前缀 + 正文，而正文里
    本来就可能出现别的日期（`#### 2006-08-29 ｜ …` 这种小标题）——
    拿整个串去断言"不含某日期"，测的其实是正文，不是前缀。
    """

    marker = chunk.content

    assert marker in chunk.embed_text, "embed_text 应当是 前缀 + 正文"

    return chunk.embed_text[: chunk.embed_text.index(marker)]


def test_embed_prefix_carries_the_company_name() -> None:
    """
    C10：前缀里必须出现**公司名**（它住在 title 里，此前不在前缀的键表里）。

    实测后果：没有它时，问句里的公司名在 dense 那一侧**无处可比** ——
    那条正文不写公司名的金标连 top-200 都进不去。
    """

    chunk = _enrich_one("#### 2007-04-12 ｜ 对外担保\n\n公司披露对外担保事项。")

    assert "荣安地产" in chunk.embed_text
    assert chunk.embed_text.startswith("荣安地产")


def test_embed_prefix_drops_the_constant_update_field() -> None:
    """上传时间对所有块都一样，进前缀只是占位（C10 顺手去掉的）。"""

    chunk = _enrich_one("正文内容")

    assert "2026-10-16" not in chunk.embed_text


def test_section_of_a_body_first_chunk_comes_from_the_previous_chunk() -> None:
    """
    B8：块首那段**正文**属于上一块末尾那个标题 —— 不是块内的第一个标题。

    这批文档的事件按时间倒序排，所以"块首那件事"的标题常常被切在上一块
    （实测 chunk#49 就是这样：正文从 `2007年4月12日…` 开始，而
    `#### 2007-04-12 ｜ 对外担保` 在 #48 的末尾）。

    改之前取的是块内**最后**一个标题 → 前缀里是 `2006-08-29`，与问句的
    日期（2007-04-12）**正好相反**；只改成"块内第一个"也还差一步
    （那是 `2007-04-04`，第二件事）。所以要两个值：本块的 section 看块首，
    游标推进到块尾。
    """

    headings = _enrich_sequence([
        # 上一块：正文 + 下一个事件的标题落在它的末尾
        "2007年4月30日，公司披露「对外担保」事项。\n\n"
        "#### 2007-04-12 ｜ 对外担保",
        # 本块：以 2007-04-12 那件事的**正文**开头
        "2007年4月12日，公司披露「对外担保」事项：对深圳市成功通信技术"
        "有限公司进行担保。\n\n"
        "#### 2007-04-04 ｜ 对外担保\n\n2007年4月4日……\n\n"
        "#### 2006-08-29 ｜ 对外担保\n\n2006年8月29日……",
    ])

    second = headings[1]

    assert second.metadata["section"] == "2007-04-12 ｜ 对外担保"

    prefix = _prefix_of(second)

    assert "2007-04-12" in prefix, "前缀里该是覆盖块首那件事的标题"
    assert "2006-08-29" not in prefix, "不该是块内最旧那个事件"


def test_section_of_a_heading_first_chunk_is_its_own_heading() -> None:
    """块首就是标题时用它自己那个（这条行为与改前一致）。"""

    headings = _enrich_sequence([
        "#### 2007-08-29 ｜ 对外担保\n\n2007年8月29日，公司披露……\n\n"
        "#### 2007-04-12 ｜ 对外担保\n\n2007年4月12日……",
    ])

    assert headings[0].metadata["section"] == "2007-08-29 ｜ 对外担保"


def test_section_falls_back_to_the_previous_chunk_heading() -> None:
    """块内没有标题时沿用上一条的（这条行为没变）。"""

    chunk = SimpleNamespace(
        chunk_index=1, content="没有标题的续块。", metadata={}, locator={},
    )

    import asyncio

    asyncio.run(
        TxtCommonChunkEnricher().enrich(
            [chunk], config={}, document_metadata={},
        )
    )

    assert "section" not in (chunk.locator or {})
