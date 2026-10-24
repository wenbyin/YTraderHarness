"""
列表页游标的编码 / 解码。

游标对客户端是不透明的字符串，内部是 (created_at, run_id) 两列。
用 base64url(JSON) 而不是把时间戳直接拼进查询串：

    - URL 安全，不必再处理日志里的 + / =
    - 以后加列（例如按 status 分组分页）不必改客户端的解析
    - 客户端**看不出**分页是按时间倒序的 —— 那正是服务端可以改的实现细节

⚠️ 解码失败一律抛 ValueError（路由映射成 400）。不要静默忽略坏游标：
那会让「翻到第二页突然从头开始」看起来像正常返回。
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from uuid import UUID

from ..domain.entities import StressRunCursor

__all__ = [
    "decode_cursor",
    "encode_cursor",
]


def encode_cursor(
        cursor: StressRunCursor,
) -> str:
    """(created_at, run_id) -> 不透明字符串。"""

    payload = {
        "created_at": cursor.created_at.isoformat(),
        "run_id": str(cursor.run_id),
    }

    raw = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(
        raw: str,
) -> StressRunCursor:
    """
    不透明字符串 -> (created_at, run_id)。

    Raises:
        ValueError: 不是本服务签发过的游标形状。
    """

    try:
        # 编码时去掉了尾部的 =，解码前补回来。
        padded = raw + "=" * (-len(raw) % 4)

        payload = json.loads(
            base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        )

        return StressRunCursor(
            created_at=datetime.fromisoformat(payload["created_at"]),
            run_id=UUID(payload["run_id"]),
        )

    except (
        binascii.Error,
        UnicodeDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "cursor 不是合法的分页游标；请原样传回上一页的 next_cursor。"
        ) from exc
