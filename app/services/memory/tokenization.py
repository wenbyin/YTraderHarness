"""
L0 记忆检索用的分词器 —— 对齐 ES 的 `cjk` 分析器。

为什么需要它
-----------
`agent_message_fts` 是 Postgres 的 `tsvector` + GIN，而 Postgres 侧
**没有**中文分词配置：本实例的 `pg_ts_config` 里一个中文配置都没有
（只有 arabic…yiddish 那一串），也没装 zhparser / pg_jieba。
于是 `to_tsvector('simple', '今天天气不错')` 会把整句当成**一个**词元
—— 检索等于废的。

所以分词放在**应用侧**：先切好词、空格连起来，再交给 `simple` 配置
（它按空格切，不做词干还原）。

这不是绕路 —— L0 的设计本来就是这样。原规格里 FTS5 那一列存的是
"jieba 分词后的文本"、原文另存一列 UNINDEXED，落库的是**分好词的串**
而不是原文，检索时按同一套规则切 query。移植到 Postgres 时把 jieba
换成 ES 那套分析器即可，`simple` 配置就是 `standard` 在那边扮演的角色。

⚠️ 对齐的是**切词语义**，不是分数
-----------------------------
`app/services/document/index/es.py` 的内容字段用 ES 内置 `cjk`，逐项展开是

    standard + cjk_width + lowercase + cjk_bigram

本模块复现其中三件事：

    ① NFKC 归一化 + 小写        ≈ cjk_width + lowercase
    ② 非 CJK 段按空白/标点切开   ≈ standard（UAX#29 的近似）
    ③ CJK 连续串做窗口 2 的滑窗  ≈ cjk_bigram

同一段文本因此在 ES 与 PG 两侧切出**同一批词元**，召回集合一致。
但**分数不可比**：ES 那边是 BM25，这边是 `ts_rank` —— 两套不同的打分
函数。要跨源融合只能按排名（RRF 那类），不能按分数相加。

⚠️ 归一化的方向不必与 ES 逐字相同
-----------------------------
NFKC 是把全角压成半角，而 ES 的 `cjk_width` 还会把半角片假名拉成全角。
两者方向不一致，但**不影响一致性**：写索引与写 query 都走本模块，
只要两侧用同一套归一化，词元就对得上。真正会出错的是"索引侧归一化了、
查询侧没有"—— 所以本模块是唯一的入口，两边都别自己再处理一遍。

⚠️ cjk_bigram 的 unigram 规则必须照抄
--------------------------------
`output_unigrams` 默认 false，ES 的语义是"**只有不构成 bigram 的才输出
unigram**"：

    东方财富   ->  东方 方财 财富      （4 字 -> 3 个 bigram，**没有**单字）
    东         ->  东                 （长度 1 的串没有 bigram，保留单字）

自己实现时顺手把单字也留下，会让"股""东"这类单字变成词元 —— 而一份
财报里几乎每条消息都含它们，召回于是退化成近似随机。es.py 那段实测
注释（"标准分析器按单字切"）讲的正是这个症状，这里不能再犯一次。

⚠️ bigram 不跨非 CJK 段
-------------------
`股abc东` 里 `股` 与 `东` 是两个**独立的长度 1 串**，各自保留单字，
不会跨过中间的 Latin token 合成 bigram。ES 的 cjk_bigram 也是按
token 相邻性组合的，中间隔开就断开。

用法
----
写索引   `index_text(text)`  -> 空格连接的词元串（存进 agent_message_fts.tokens）
检索     `tsquery_text(text)` -> `to_tsquery('simple', ...)` 的入参

两者**必须**用同一个分词器。不一致时的症状是"明明有这条却搜不到"，
而且**不报错** —— 索引里是 A 切法、查询是 B 切法，交集空而已。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "MAX_QUERY_TOKENS",
    "index_text",
    "tokenize",
    "tsquery_text",
]


#: 检索时最多取多少个词元（OR 起来）。
#:
#: `to_tsquery` 的入参长度直接决定查询计划的规模，而用户/模型给的 query
#: 长度没有上界 —— 一段 500 字的输入会切出近千个 bigram，拼成一条巨大
#: 的 OR 表达式。截断是**按顺序**取的：CJK bigram 是有序滑窗，前面的
#: 词元对应 query 的开头，与"取前 N 个"在语义上是一致的。
#:
#: 代价说清楚：超长的 query 会被静默地只看前半段。之所以接受，是因为
#: 这条路上真实出现的 query 是检索式（几个词到一句话），不是整篇文档；
#: 真出现超长输入时，关键词路本来也不该是唯一的一路 —— 向量路不受影响。
MAX_QUERY_TOKENS = 64


#: `standard` 分析器里"一个词"的近似：\w 去掉下划线。
#:
#: 用 `[^\W_]` 而不是 `[a-zA-Z0-9]`：**只作用于非 CJK 段**（见
#: `tokenize` 的分段），所以这里放 `\w` 是为了让西里尔、重音拉丁这些
#: 也各自成为词元 —— 与 standard 的 UAX#29 行为一致。下划线被排除，
#: 因为 standard 也会把它当分隔符。
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _is_cjk(ch: str) -> bool:
    """
    这个字符是不是 CJK —— 也就是 `cjk_bigram` 会做滑窗的那几类文字。

    覆盖 ES `cjk_bigram` 认的四类：**Han / Hiragana / Katakana / Hangul**
    （含各自的扩展区与兼容区）。韩文音节单列区间，因为它在 Unicode 里
    不与汉字相邻。

    不覆盖的：中日韩的**标点**（`、`「」` 等）与全角符号 —— 它们由
    `standard` 当分隔符处理，正则是按"非 CJK 段"走 `_WORD` 的，
    不会成为词元。这是刻意的：标点做词元只会污染倒排。
    """

    code = ord(ch)

    return (
        0x3040 <= code <= 0x30FF      # Hiragana + Katakana
        or 0x31F0 <= code <= 0x31FF   # Katakana phonetic extensions
        or 0x3400 <= code <= 0x4DBF   # CJK unified ext A
        or 0x4E00 <= code <= 0x9FFF   # CJK unified
        or 0xF900 <= code <= 0xFAFF   # CJK compatibility ideographs
        or 0x1100 <= code <= 0x11FF   # Hangul Jamo
        or 0x3130 <= code <= 0x318F   # Hangul compatibility Jamo
        or 0xAC00 <= code <= 0xD7AF   # Hangul syllables
    )


def tokenize(text: str) -> list[str]:
    """
    文本 -> 词元列表。写索引与写 query 都走这里。

    空串、纯标点、纯空白都返回**空列表**（不是 None、不抛）——
    调用方据此决定"没有可用词元、这一路别跑"，见 `tsquery_text`。

    ⚠️ 顺序即语义
    ----------
    返回的顺序有意义：CJK bigram 是有序滑窗，`tsquery_text` 的截断、
    以及将来要做的高亮定位都依赖它。不要把它当集合用（`set()` 会打乱
    顺序，也会掩盖重复 —— 而重复恰恰是判断"同一个词在 query 里出现两次"
    的唯一线索）。
    """

    if not text:
        return []

    # ① width + case。NFKC 一步做完半角化与兼容字折叠，再统一小写。
    #    顺序不能反：先小写再 NFKC 会让某些全角大写字母漏掉。
    normalized = unicodedata.normalize("NFKC", text).lower()

    tokens: list[str] = []

    # ② 切成"连续的 CJK"与"非 CJK"两类段。
    #    逐字符扫描而不是用一条大正则：分段规则会随 cjk_bigram 的覆盖面
    #    变化（将来要支持泰文那类无空格文字时只改 `_is_cjk`），分成两条
    #    正则的话规则就分散在四处了。
    segment: list[str] = []
    segment_is_cjk = False

    def flush() -> None:
        if not segment:
            return

        run = "".join(segment)

        if segment_is_cjk:
            # ③ 滑窗。长度 1 的串没有 bigram，按 ES 的语义保留单字。
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(
                    run[index:index + 2]
                    for index in range(len(run) - 1)
                )
        else:
            tokens.extend(_WORD.findall(run))

        segment.clear()

    for ch in normalized:

        is_cjk = _is_cjk(ch)

        if segment and is_cjk != segment_is_cjk:
            flush()

        segment.append(ch)
        segment_is_cjk = is_cjk

    flush()

    return tokens


def index_text(text: str) -> str:
    """
    文本 -> 存进 `agent_message_fts.tokens` 的那一列。

    空格连接的词元串。**不是给人读的** —— 原文另外存一列（`original`），
    展示与高亮都从那一列出。

    ⚠️ 为什么存"分好词的串"而不是让 Postgres 现算
    ---------------------------------------
    因为 Postgres 算不出来：`to_tsvector('simple', 原文)` 没有中文分词
    配置，整句会变成一个词元。这一列的存在就是把"切词"这件事固定下来 ——
    而它也因此**必须**与 `tsquery_text` 用同一个分词器版本：改了分词规则
    就要重建索引，否则新旧两种切法在同一张倒排里互不可见（这条写在
    models.sql 里，真需要重建时按那个脚本走）。
    """

    return " ".join(tokenize(text))


def tsquery_text(text: str) -> str | None:
    """
    文本 -> `to_tsquery('simple', ?)` 的入参。没有可用词元时返回 None。

    ⚠️ 用 OR(`|`) 而不是空格
    --------------------
    `plainto_tsquery` / `websearch_to_tsquery` 把多个词元按 **AND** 连，
    而我们要的是召回：query "东方财富 股东户数" 切出来是
    `东方 方财 财富 股东 东户 户数`，AND 起来等于要求一条消息**同时**
    含这六个 bigram —— 那是精确串匹配，会把只讲股东户数的消息全部漏掉。
    OR 之后由 `ts_rank` 排序，命中多的排前面，语义才对得上。

    ⚠️ 返回 None 而不是空串
    -------------------
    空串会让 `to_tsquery` 抛错，而"query 全是标点"是输入问题不是系统
    故障。返回 None 让调用方走"这一路没跑"的分支（并如实写进
    degraded_routes），而不是把一个语法错误变成 500。

    ⚠️ 词元不需要转义
    -------------
    `tokenize` 只产出 `\\w` 字符与 CJK，不带 `&` `|` `!` `:` `*` `(`
    这些 tsquery 的元字符，所以直接拼接是安全的。将来若放宽分词规则
    （比如允许带连字符的标识符），**这里必须补转义** —— 拼错的表现是
    语法错误，而它离原因（改了分词器）很远。
    """

    tokens = tokenize(text)

    if not tokens:
        return None

    return " | ".join(tokens[:MAX_QUERY_TOKENS])
