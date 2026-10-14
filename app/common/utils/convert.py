"""
类型转换的小工具。

本项目把 UUID 一路当字符串传（state、payload、subject_ref 都是
JSON 友好的结构），边界上再转回来。这个"转回来"被抄了四遍：

    app/services/document/pipeline/runtime.py
    app/services/document/repository/pipeline_run_impl.py

四份里核心那句逐字相同，区别只在"取不到值时怎么办"：

    必填    抛 ValueError
    可选    返回 None

两种语义都是对的，所以这里是**两个函数**而不是一个带开关的 ——
一个叫 `to_uuid` 一个叫 `to_uuid_or_none`，看名字就知道缺值会怎样，
不必去读调用点传的那个布尔参数。

为什么不做成"吞异常返回 None"
-----------------------------
看着更省事，但会把"payload 里是个拼错的字符串"和"这个键本来就没
填"变成同一个结果。前者是上游写错了，后者是正常缺省 —— 混在一起
之后，一次拼写错误会静默变成"这个字段没值"。
（确实有一处故意要"转不动就 None"：
`version_chunk_service_impl._as_uuid`。那是它自己的策略，不并进来。）
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

__all__ = ["to_uuid", "to_uuid_or_none"]


def to_uuid(value: Any) -> UUID:
    """
    转成 UUID。已经是 UUID 的原样返回。

    `None` 与"转不动"都抛 —— 见模块 docstring：区分不了"没填"与
    "填错了"的调用方应该用 `to_uuid_or_none`。
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def to_uuid_or_none(value: Any) -> UUID | None:
    """
    转成 UUID；`None` 原样返回 None。

    ⚠️ 只对 `None` 宽容 —— "转不动的字符串"照样抛。理由同上：
    把拼错的值悄悄变成"没有值"是最难查的一类问题。
    """

    if value is None:
        return None

    return to_uuid(value)
