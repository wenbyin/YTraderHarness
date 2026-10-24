"""
Runner 侧指标的样本收集与分位数。

Phase A 用**有界精确样本**：每类分布最多保留 ``limit`` 个样本，分位数
在快照时对已保存的样本精确计算。超过上限的样本只计入 ``count``，不参与
分位数，并在快照里用 ``truncated`` 标记；``aggregation_algorithm`` 记录
``runner-exact-capped-v1``。这比“平均多个批次的 P95”可靠，也比引入
HDR 依赖简单。

高基数 id 不作为分布分组键；``turn_index`` 是 1~30 的有限枚举，可以分组。
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "DistributionBuilder",
    "distribution",
    "empty_distribution",
]

DEFAULT_SAMPLE_LIMIT = 200_000


def empty_distribution() -> dict[str, Any]:
    return {
        "count": 0,
        "mean": None,
        "min": None,
        "max": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    """线性插值分位数；``ordered`` 必须已排序。"""

    count = len(ordered)

    if count == 1:
        return ordered[0]

    position = (count - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * weight
    )


def distribution(
        values: list[float],
        *,
        seen: int | None = None,
) -> dict[str, Any]:
    """
    样本列表 -> Distribution JSON。

    ``seen`` 是超过样本上限前的总样本数；缺省等于 ``len(values)``。
    """

    total = len(values) if seen is None else seen

    if not values:
        result = empty_distribution()
        result["count"] = total
        return result

    ordered = sorted(float(value) for value in values)

    result = {
        "count": total,
        "mean": round(sum(ordered) / len(ordered), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p90": round(_percentile(ordered, 0.90), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
    }

    if total > len(values):
        result["truncated"] = True
        result["samples_used"] = len(values)

    return result


class DistributionBuilder:
    """有界样本收集器。单 event loop 使用，不需要锁。"""

    def __init__(
            self,
            *,
            limit: int = DEFAULT_SAMPLE_LIMIT,
    ) -> None:

        self._limit = max(1, limit)
        self._values: list[float] = []
        self._seen = 0

    def add(self, value: float) -> None:

        self._seen += 1

        if len(self._values) < self._limit:
            self._values.append(float(value))

    def snapshot(self) -> dict[str, Any]:

        return distribution(self._values, seen=self._seen)

    @property
    def count(self) -> int:
        return self._seen
