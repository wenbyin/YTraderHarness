"""
不透明游标：把「上一页最后一条的排序键」编成一个调用方看不懂的串。

为什么是游标而不是 offset
--------------------
三个列表端点的排序键**都是可变或非唯一的**：

    L0   created_at（同一事务里写的多条消息时间戳完全相同）、seq（会话内唯一）
    L1   updated_time（合并会原地改写、把它往前推）、priority（大量重复）
    L2   updated_at（每次写入都变）、filename

offset 分页在这些键上翻页会出错：翻页期间有人写一条（或合并一次），后面
所有页的窗口整体平移 —— 表现为**漏掉一整屏或重复一整屏**，而且不报错。
keyset 的判据是"上一条的值"，与已写入的数据无关。

⚠️ 载荷里带 `order`，而且**必须校验它**
------------------------------
游标与排序键是一对：`created_at_desc` 的游标解出来的键拿去配
`session_seq_asc` 用，条件就是错的 —— 而它**不会报错**，只会返回一屏
看似正常的结果。所以 `decode_cursor` 要求调用方声明 order，对不上直接
400（"这个游标不是给这个排序用的"）。

⚠️ 没有签名、不加密
------------
游标不是凭证：它只装"从哪继续"，而**能不能看**由 `target` 与归属校验
决定（每一页都重新校验）。所以这里不做 HMAC —— 伪造一个游标最多让人
从别的位置开始读同一批他有权限读的数据。篡改结构则会在解析阶段被拒。

⚠️ 客户端不该解析它
------------
载荷格式（字段名、版本号）是实现细节，随时会变。`v` 字段存在的意义就是
"将来变了能让旧游标明确地失效"，而不是让调用方按版本号分支。
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

__all__ = [
    "CURSOR_VERSION",
    "decode_cursor",
    "encode_cursor",
]


#: 载荷版本。改动载荷结构时 +1（旧游标于是明确失效，而不是被误解）。
CURSOR_VERSION = 1


def _dump(value: Any) -> Any:
    """排序键 -> JSON 可序列化的形式。"""

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


def encode_cursor(
        order: str,
        key: Sequence[Any],
) -> str:
    """
    `(order, 排序键...)` -> 不透明串。

    键的顺序由调用方与 `decode_cursor` 的 `kinds` 约定，两边必须一致
    （它们在同一个方法里成对出现）。
    """

    payload = {
        "v": CURSOR_VERSION,
        "o": order,
        "k": [_dump(value) for value in key],
    }

    raw = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    # `strip("=")`：padding 只是 base64 的要求，带着它会让同一份载荷
    # 出现两种写法（有些客户端会自行去掉），而游标是"照原样回传"的东西。
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(
        cursor: str,
        *,
        order: str,
        kinds: Sequence[str],
) -> list[Any]:
    """
    不透明串 -> 排序键列表。**任何解析问题都抛 `ValueError`（HTTP 400）**。

    Parameters
    ----------
    order
        本次请求的排序键。与载荷里那个不一致时拒绝。
    kinds
        键的类型，按位置给：`"datetime"` / `"uuid"` / `"int"` / `"str"`。
        类型在这里校验而不是留给 SQL —— 一个 `int` 位置上的字符串到了
        SQLAlchemy 手里会以"类型不匹配"炸开，那时离"游标坏了"这个原因
        已经很远。

    ⚠️ 为什么不用 `TypeAdapter` / 直接 dump 类型名
    ----------------------------------------
    载荷里存的是**值**不是类型：类型由调用方声明。这样载荷更小，而且
    "这一页按什么排"这件事只有一个来源（端点代码），不会出现"游标里说
    它是 int、调用方按 datetime 解"这种两边打架的情况。
    """

    if not cursor:
        raise ValueError("游标为空。第一页不要传 cursor。")

    padded = cursor + "=" * (-len(cursor) % 4)

    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))

    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("游标不是合法的 base64 串。") from exc

    try:
        payload = json.loads(raw)

    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("游标的内容不是一个 JSON 对象。") from exc

    if not isinstance(payload, dict):
        raise ValueError("游标的内容不是一个 JSON 对象。")

    if payload.get("v") != CURSOR_VERSION:
        raise ValueError(
            f"游标版本不受支持（收到 {payload.get('v')!r}，"
            f"当前是 {CURSOR_VERSION}）。重新从第一页开始。"
        )

    if payload.get("o") != order:
        raise ValueError(
            f"游标不是给 order={order!r} 用的（它属于 "
            f"{payload.get('o')!r}）。换排序时必须重新从第一页开始 —— "
            "把两种排序的键混起来用不会报错，只会返回错的一屏。"
        )

    values = payload.get("k")

    if not isinstance(values, list) or len(values) != len(kinds):
        raise ValueError(
            f"游标的排序键个数不对（期望 {len(kinds)} 个）。"
        )

    return [
        _load(value, kind)
        for value, kind in zip(values, kinds)
    ]


def _load(value: Any, kind: str) -> Any:
    """载荷里的一项 -> 该类型。转不了就 400。"""

    if value is None:
        # 排序键为 NULL 的行：`updated_at` / `created_at` 在库里有 NOT NULL，
        # 所以这里只可能来自伪造的游标。当成坏游标拒掉，别让 None 进 SQL
        # 去参与比较（`NULL > x` 恒为 NULL，结果是静默地少一屏）。
        raise ValueError("游标的排序键里有 null。")

    try:
        if kind == "datetime":
            return datetime.fromisoformat(value)

        if kind == "uuid":
            return UUID(value)

        if kind == "int":
            return int(value)

        return str(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"游标的排序键与类型 {kind!r} 不符。"
        ) from exc
