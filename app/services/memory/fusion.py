"""
RRF 融合 —— L0 的检索与 L1 的候选召回共用的一份实现。

    score(d) = Σ_routes  1 / (K + rank_in_route(d))

⚠️ 为什么提出来，而不是各写一份
--------------------------
两处都要"按名次融合"（L0 的对话检索、L1 去重的候选池），而 `K` 是个**有
语义的选择**：它决定"第 1 名与第 2 名的差距"有多陡。两份 60 迟早会在某
个下午变成 60 与 45，而"同一份索引两次查询给出不同排序"是最难查的那类
bug —— 它没有任何报错，只是结果不一样。

⚠️ 融合按**排名**，不按分数相加
---------------------------
两路的分数不同量纲：关键词是 `ts_rank`，向量是余弦相似度。直接相加的话，
谁的量级大谁说了算 —— 而那是实现细节决定的，不是相关性。RRF 只用"在第
几名"，天然免疫量纲。

⚠️ 排序必须是**确定**的
-------------------
融合分相同时（两路都排第 1、或只被一路召回到同样名次）需要稳定的次序，
否则同一份索引两次查询可能给出不同顺序。次键取主键的字符串形式。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["RRF_K", "fuse_by_rank"]


#: RRF 的常数。见模块 docstring —— 它只影响排序的陡峭程度，不影响召回集合。
#:
#: 60 是 RRF 那篇论文的常用值：K 小则头部龙头的权重更大，K 大则两路的
#: 名次更平等。60 是既不极端、也不需要按语料调的中间值。
RRF_K = 60


def fuse_by_rank(
        rankings: dict[Any, list[Any]],
        *,
        key: Callable[[Any], Any],
) -> list[tuple[Any, set[Any], float]]:
    """
    多路排序 -> 按融合分降序的 `(任一路的一条召回, 它被哪几路召回, 融合分)`。

    Parameters
    ----------
    rankings
        `路名 -> 该路按相关性降序的召回列表`。路名只要可哈希（L0 用 route
        枚举、L1 用 "keyword"/"vector" 字符串）。
    key
        从一条召回里取出身份（主键）的函数。身份用来做跨路合并，也用来
        做同分时的确定性次键。

    ⚠️ 融合分**必须原样带出来**，不能回去用"被哪几路召回"反推
    ----------------------------------------------------
    那个分是逐路按**名次**累加的（`1/(K+rank)`），而路集合只记得"被哪几路
    找到" —— 同样的两路命中，第 1 名与第 9 名贡献差好几倍。用路集合反推
    出来的分数与排序对不上，而那正是"分数"这个字段最不该出现的样子。

    ⚠️ 保留的是"任意一路的那个对象"
    ---------------------------
    同一条记录在两路里各有一个对象（分数不同、量纲不同）。融合后分数与那
    两个都无关，所以留哪一个都行 —— 留先遇到的那个，因为主键才是身份，
    对象上的 `score` 在这之后不再被读。
    """

    scores: dict[Any, float] = {}
    routes: dict[Any, set[Any]] = {}
    sample: dict[Any, Any] = {}

    for route, recalls in rankings.items():

        for rank, recall in enumerate(recalls):

            identity = key(recall)

            scores[identity] = (
                scores.get(identity, 0.0) + 1.0 / (RRF_K + rank)
            )

            routes.setdefault(identity, set()).add(route)
            sample.setdefault(identity, recall)

    ordered = sorted(
        scores,
        key=lambda identity: (-scores[identity], str(identity)),
    )

    return [
        (sample[identity], routes[identity], scores[identity])
        for identity in ordered
    ]
