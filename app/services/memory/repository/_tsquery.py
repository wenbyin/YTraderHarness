"""
Postgres 全文检索的三个 SQL 表达式构造 —— L0 与 L1 共用。

⚠️ 为什么必须只有一份
------------------
它们把 `simple` 这个 regconfig 写死在里面，而那个值**必须与两张表的生成
列逐字一致**：

    agent_message_fts.tsv / memory_atom_fts.tsv
        GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, <tokens>))

两边不一致的症状是"查询永远匹配不上"—— 不报错，只是什么都搜不到，或者
搜出一堆无关的。配置各写一份，就多一次"改一处忘另一处"的机会。

⚠️ `to_tsquery` 的第一个参数必须显式 **cast**
----------------------------------------
Postgres 有两个重载：`to_tsquery(regconfig, text)` 与 `to_tsquery(text)`。
绑定参数进来时类型是 text，于是解析器找不到 `to_tsquery(text, text)`，
报的是"函数不存在" —— 报错指向函数名，离真正的原因（少了一次转型）
有点距离。所以写 `CAST(:config AS regconfig)`。

⚠️ `ts_rank` / `ts_match` 写成"吃 model"而不是吃列
--------------------------------------------
两张表的列名不同（`tokens`/`tsv` 是 L0，`content_tokens`/`tsv` 是 L1），
但**都叫 `tsv`** —— 传 model 进来取 `model.tsv` 是两者唯一需要的共同点。
传列的话调用方要自己写 `model.tsv`，那这个函数就没剩下什么了。
"""

from __future__ import annotations

from sqlalchemy import ColumnElement, cast, func, literal
from sqlalchemy.dialects.postgresql import REGCONFIG

__all__ = [
    "TEXT_SEARCH_CONFIG",
    "ts_match",
    "ts_rank",
    "tsquery_expr",
]


#: 全文检索配置。必须与两张表的生成列逐字一致 —— 见模块 docstring。
TEXT_SEARCH_CONFIG = "simple"


def tsquery_expr(tsquery: str):
    """`to_tsquery('simple'::regconfig, :q)`。"""

    return func.to_tsquery(
        cast(literal(TEXT_SEARCH_CONFIG), REGCONFIG),
        tsquery,
    )


def ts_match(model, query) -> ColumnElement[bool]:
    """`tsv @@ query`。"""

    return model.tsv.op("@@")(query)


def ts_rank(model, query):
    """`ts_rank(tsv, query)`。"""

    return func.ts_rank(model.tsv, query)
