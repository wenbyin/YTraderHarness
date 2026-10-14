"""
同一个**批次**里撞了同一个主键时怎么办 —— 两个仓储共用的那一条规则。

为什么单独一个文件
--------------
`memory_atom`（本体 + 两张挂载表）与 `memory_scene` 都是"一次写一批行、
按主键 upsert"。而 Postgres 的 `ON CONFLICT DO UPDATE` 在**同一条语句**里
撞两次同一个键会直接报

    ON CONFLICT DO UPDATE command cannot affect row a second time

这不是理论风险：L1 的抽取提示词允许两条记忆落进同一个情境；L2 的整合
清单里同一个文件名完全可以出现两次（模型写重了）。在应用侧先收一次比
让它炸简单，而"留第一条还是最后一条"这个决定只该有一份 —— 两份分叉时，
症状是"某一边偶尔丢掉一条刚写的内容"，且不报错。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

__all__ = ["dedupe_by"]

Row = TypeVar("Row")


def dedupe_by(
        rows: Iterable[Row],
        *,
        key: Callable[[Row], Any],
) -> list[Row]:
    """
    同一次写入里出现重复主键时**留最后一条**。

    留最后一条而不是第一条：调用方按顺序构造，后面的通常是更完整的那个
    （与 dict 覆写的直觉一致）—— L2 那边尤其如此：模型对同一个文件写两次
    时，后一次是它修正过的版本。
    """

    deduped: dict[Any, Row] = {}

    for row in rows:
        deduped[key(row)] = row

    return list(deduped.values())
