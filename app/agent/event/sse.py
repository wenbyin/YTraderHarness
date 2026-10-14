"""
把一条流记录编码成 **SSE 帧**。

SSE 的帧格式（`text/event-stream`，见 session.yaml 的 streamRunEvents）：

    id: 1759412273324-0
    event: message.delta
    data: {"event": "message.delta", ...}
    <空行>

空行是**帧的结束**，不是排版 —— 少一个空行，客户端就会把后面那帧的
第一行当成这帧的续行，症状是"事件时有时无"。

为什么单独一个模块
-----------------
编码规则属于**传输层**，不是事件本身（app/common/event）也不是存储
（reader.py）。将来换成 WebSocket 或别的 SSE 消费端，改的是这里，
事件与读取端都不动。
"""

from __future__ import annotations

import json
from typing import Any

from .reader import StreamEntry

__all__ = [
    "HEARTBEAT",
    "sse_frame",
]


#: SSE 的行分隔符固定是 `\n`（**不是** `\r\n`）。
_LINE = "\n"


def _envelope(entry: StreamEntry) -> dict[str, Any]:
    """
    帧里要发的那个对象：信封字段 + 解开嵌套的 `data`。

    ⚠️ 为什么要把 `data` 解回对象
    ---------------------------
    存储里 `data` **必须**是 JSON 字符串（Redis Stream 的字段值不能嵌套，
    见 app/common/event/event.py 的 to_fields）。但 SSE 没有这条限制 ——
    照搬那个形状发出去，客户端拿到的是：

        {"event": "...", "data": "{\\"delta\\": \\"你\\"}"}

    也就是**两层 JSON**：先 parse 一次拿信封，再 parse 一次拿正文。忘掉
    第二层不会报错，只会让业务字段变成一坨字符串，然后在很远的地方
    才炸。

    所以这里解开一层，客户端一次 `JSON.parse` 拿到全部：

        {"event": "...", "data": {"delta": "你"}}

    这是**有意的差异**（SSE 比存储更宽松），不是不一致：两边各自用自己
    最自然的形状，转换只在这一个函数里发生。
    """

    payload: dict[str, Any] = dict(entry.fields)

    raw = payload.get("data")

    if isinstance(raw, str):
        try:
            payload["data"] = json.loads(raw)
        except ValueError:
            # 解不开就原样留着（仍是字符串）。一条坏记录不该让整帧发不
            # 出去 —— 订阅端至少还能看到事件名与 id，"少一段"比"断流"好。
            pass

    return payload


def sse_frame(entry: StreamEntry) -> str:
    """
    一条记录 -> 一个完整的 SSE 帧。

    `id:` 用 Redis 的 entry id（`<毫秒>-<序号>`），客户端断线重连时
    按 SSE 规范把它回填到 `Last-Event-ID` 头里，就能**不重不漏**地
    续上（路由会把它当作 XREAD 的起点）。

    `event:` 用信封里的 `event` 字段，于是浏览器可以直接

        source.addEventListener("message.delta", ...)

    而不必先解 JSON 再自己分派。

    `data:` 放**整个信封**（不止 data 子字段），且 `data` 子字段会被
    解开成对象（见 `_envelope`）—— 订阅端一次 JSON.parse 就能拿到全部。
    """

    payload = json.dumps(_envelope(entry), ensure_ascii=False)

    lines = [f"id: {entry.id}"]

    if entry.event_type:
        lines.append(f"event: {entry.event_type}")

    # ⚠️ 逐行加 `data: ` 前缀。
    #
    # json.dumps 会把字符串里的换行转义成 `\n`（两个字符），因此正常
    # 情况下 payload 只有一行。但字段是**外部数据**，一旦真的出现裸
    # 换行，只写一次 `data: ` 就会让后半段被客户端当成另一个 SSE 字段
    # （"data: {"event": ...
    #   "hacked": ...}"）—— 那是能被注入的格式，且不会报错。
    for line in payload.split(_LINE):
        lines.append(f"data: {line}")

    return _LINE.join(lines) + _LINE + _LINE


#: 心跳：SSE 的**注释行**（以 `:` 开头，客户端一律忽略）。
#:
#: 两个作用：让中间的代理/负载均衡看见连接是活的（否则长时间没有正文
#: 的连接会被掐断），以及让服务端有机会发现客户端已经断开。
HEARTBEAT = f": ping{_LINE}{_LINE}"
