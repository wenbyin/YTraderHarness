"""
Checkpoint I/O 的并发化。

背景（压测实测）
----------------
`AsyncPostgresSaver._cursor`（langgraph/checkpoint/postgres/aio.py:374）
用 **一个 `asyncio.Lock`** 把 saver 的**所有** cursor 操作串起来：

    async with self.lock, _ainternal.get_connection(self.conn) as conn:

单连接（`AsyncConnection`）时这是必需的 —— 一条连接不能同时开两个 cursor。
但本仓库用的是 **连接池**（`AsyncConnectionPool`），每次操作各借各的连接，
这把锁只会把**整个进程**的 checkpoint 读写排成一条队。

实测（322 个并发 run）：`run.started -> 第一个 node.started` 的空档
p50 = 6.3s，其中两次 checkpoint 读分别 p50 1.6s / 4.7s，而这些 run 的
checkpoint **全是空的**（新 thread，prior_count=0）——那几秒不是数据量，
是排队。关锁之后同一压测口径的 pre-node 空档 p50 降到 0.6s。

安全性
------
每次 `_cursor` 自己从池里借连接；`pipe` 在池模式下恒为 None（构造函数
会拒绝池 + pipeline 的组合）；serde 无状态；同一 thread 的并发由会话级
run 锁排除。`aput` 的跨 run 并发不改变任何单 run 内的顺序 —— 那个顺序
由 LangGraph 的 `_checkpointer_put_after_previous` / `durability` 保证。
"""

from __future__ import annotations

from typing import Any

__all__ = ["disable_saver_lock"]


class _NoopAsyncLock:
    """
    空锁。

    存在的唯一目的：`_cursor` 里那句 `async with self.lock` 必须能进。
    换成空锁之后每次操作只受连接池约束，不再被进程级串行化。
    """

    async def __aenter__(self) -> "_NoopAsyncLock":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def disable_saver_lock(saver: Any) -> None:
    """
    把 saver 的全局串行锁换成空锁。

    ⚠️ 只在**连接池**模式下调用。单连接模式下这么换会让两个 cursor
    同时用一条连接 —— psycopg 会报 "another command is already in progress"。
    调用点（GraphCheckpointerProvider）永远用池，所以前提成立。
    """

    saver.lock = _NoopAsyncLock()
