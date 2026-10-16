"""
从候选 chunk 池里抽 N 个。

**确定性是这里唯一真正重要的性质。**

固定种子 + 不变的候选集必须抽出同一批 chunk。否则调参时无法回答一个
最基本的问题：这次指标变了，是因为我改了检索配置，还是因为抽到了
另一批题？—— 两个变量同时动，结论就没有意义了。

所以：
    1. 抽样前先把候选**排序**（顺序不能依赖数据库返回顺序）
    2. 只用 `random.Random(seed)`，不碰全局 `random`
    3. 每个 DTD 用**种子派生出的**独立随机源，于是"给某个 DTD 增减
       候选"不会改变其它 DTD 抽到什么

第 3 条容易被忽略但很要紧：如果所有 DTD 共用一个随机源、顺序又是
先排 A 再排 B，那么在 A 里插入一个 chunk 会让 B 抽中的整批平移 ——
一次无关紧要的语料变更，让全部 DTD 的样本都变了。

按 DTD 分名额（balanced_by_dtd）
--------------------------------
样本数悬殊时，uniform 会让大的那个 DTD 吃掉绝大部分题目。而数据集要
回答的是"检索在这几类文档上表现如何"，不是"在语料最多的那类上表现
如何"。

名额用**最大余数法**分配（先各取整数部分，余数按小数部分从大到小
补），因为它满足"名额之和恰好等于 N"且"增减一个名额的扰动最小"——
后者正是确定性所依赖的。
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Any

from ..domain.schemas.enum import EvalSamplingStrategy

__all__ = [
    "sample_chunks",
]


def sample_chunks(
        candidates: Sequence[dict[str, Any]],
        *,
        count: int,
        seed: int,
        strategy: EvalSamplingStrategy = (
            EvalSamplingStrategy.BALANCED_BY_DTD
        ),
) -> list[dict[str, Any]]:
    """
    抽 `count` 个 chunk。返回的顺序是 **DTD、chunk_id 稳定排序后的顺序**。

    返回顺序固定而不是"再打乱一次"：窗口切分是贪心按序装的，顺序固定
    才让窗口划分固定 —— 而重试一个失败窗口要靠 `window_index` 定位，
    划分变了那个序号就指向别的东西了。

    `candidates` 的元素至少要有 `chunk_id` 与
    `document_type_definition_id`。
    """

    if count < 1:
        raise ValueError("sampled_chunk_count 至少为 1。")

    if not candidates:
        return []

    # 先用一个**稳定**的序把所有候选排定 —— 后面所有随机都建立在这个
    # 序之上。不排的话，同样的种子在不同次数据库返回顺序下会抽出不同
    # 的结果，而那种不一致极难被发现（它只在翻页、或者加了索引之后
    # 才显形）。
    ordered = sorted(
        candidates,
        key=lambda item: (
            str(item.get("document_type_definition_id")),
            str(item.get("chunk_id")),
        ),
    )

    if count >= len(ordered):
        return ordered

    if strategy is EvalSamplingStrategy.UNIFORM_ALL:

        return _sample_from(ordered, count, seed)

    # ---- balanced_by_dtd：先分组，再分名额 ----

    groups: dict[str, list[dict[str, Any]]] = {}

    for item in ordered:

        key = str(item.get("document_type_definition_id"))
        groups.setdefault(key, []).append(item)

    quotas = _allocate_quotas(
        {key: len(items) for key, items in groups.items()},
        count,
    )

    picked: list[dict[str, Any]] = []

    for key in sorted(groups):

        picked.extend(
            _sample_from(
                groups[key],
                quotas.get(key, 0),
                _derive_seed(seed, key),
            )
        )

    # 名额分配是精确的，所以这里恰好 count 个。
    #
    # 用 raise 而不是 assert：`python -O` 会把 assert 整个去掉，而这条
    # 检查失败时的表现（抽多了或抽少了）只会悄悄改变数据集的规模
    # （case_count 与预期对不上），事后极难归因。要拦就拦得住的拦。
    if len(picked) != count:
        raise RuntimeError(
            f"名额分配有误：期望抽 {count} 个，实得 {len(picked)} 个。"
            "这是抽样算法的缺陷，不该继续 —— 数据集规模会与配置不符。"
        )

    return sorted(
        picked,
        key=lambda item: (
            str(item.get("document_type_definition_id")),
            str(item.get("chunk_id")),
        ),
    )


# ============================================================
# internals
# ============================================================


def _sample_from(
        pool: list[dict[str, Any]],
        count: int,
        seed: int,
) -> list[dict[str, Any]]:
    """从已排序的池子里抽 count 个。count >= len(pool) 时全取。"""

    if count <= 0:
        return []

    if count >= len(pool):
        return list(pool)

    picked = random.Random(seed).sample(pool, count)

    # 抽样结果再排一次序：Random.sample 保留的是"池子里的相对顺序"
    # 吗？不保证。输出的顺序必须只由排序决定，不能掺进随机性。
    return sorted(
        picked,
        key=lambda item: str(item.get("chunk_id")),
    )


def _derive_seed(seed: int, key: str) -> int:
    """
    由一个 DTD 键派生它自己的种子。

    用哈希而不是 `seed + 序号`：序号会随"哪些 DTD 有候选"变化，于是
    新增一个 DTD 会把它后面所有 DTD 的种子全部平移 —— 又一次"无关
    变更改动了全部样本"。
    """

    digest = hashlib.sha256(
        f"{seed}:{key}".encode("utf-8")
    ).digest()

    return int.from_bytes(digest[:8], "big")


def _allocate_quotas(
        sizes: dict[str, int],
        total: int,
) -> dict[str, int]:
    """
    最大余数法分配名额。

    先按比例取整数部分，剩下的名额按小数部分从大到小分给各家。
    小数部分相同时**按键名排序**决定先后 —— 平局必须有个确定的
    裁决，否则同一份输入可能分配出不同结果。
    """

    population = sum(sizes.values())

    if population <= 0:
        return {key: 0 for key in sizes}

    exact = {
        key: total * size / population
        for key, size in sizes.items()
    }

    quotas = {
        key: min(int(value), sizes[key])
        for key, value in exact.items()
    }

    # 名额不能超过该组的候选数（小 DTD 会吃不下自己的份额）。
    # 吃不下时多出来的名额要**退回**再分给还有余量的组 ——
    # 直接丢掉会让最终总数少于 count。
    assigned = sum(quotas.values())

    while assigned < total:

        # 有余量、且（余数最大 / 键名最小）的组优先。
        eligible = [
            key for key in sorted(sizes)
            if quotas[key] < sizes[key]
        ]

        if not eligible:
            # 全部组的候选都被取空了。正常不会发生（count < population
            # 已在上层保证），保守地停下比死循环好。
            break

        winner = max(
            eligible,
            key=lambda key: (
                exact[key] - int(exact[key]),
                # 取负是为了让**键名小的**在余数相同时胜出：
                # max 会选最大项，而我们要最小键名。
                _negated_rank(key, sorted(sizes)),
            ),
        )

        quotas[winner] += 1
        assigned += 1

    return quotas


def _negated_rank(key: str, ordered_keys: list[str]) -> int:
    """键名在给定序里的排名取负 —— 让 `max` 偏向靠前的键。"""

    return -ordered_keys.index(key)
