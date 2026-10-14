"""
进程内的 thread 运行锁：同一个 thread_id 同时只允许一次运行。

它挡的是什么
------------
thread_id 是 LangGraph 的 checkpoint 键（见 ids.py）。同一个 thread_id 上
并发两次 `ainvoke`，两次会同时读写同一份 checkpoint —— 那不是"多跑一次"
的问题，是互相覆盖：后一个 super-step 的写入建立在前一个读到一半的状态上。
这层保护与 checkpointer / 埋点同属**图运行时**的横切设施，所以放 infra。

锁的持有者是 run_id
------------------
每次成功取锁都记下持有它的 run_id，释放时要求对得上。这不是洁癖：
thread_id 现在**按会话稳定**，所以一次迟到的释放很容易落在"另一个 run
刚拿到"的锁上，把别人的锁放掉。带 owner 校验后，这种释放是 no-op。

⚠️ 只在本进程有效 —— 部署必须 workers=1
--------------------------------------
每个进程各有一份登记表。多 worker 下同一个 thread_id 的两次运行可能落到
不同进程，护栏形同虚设 —— 而 checkpointer 是**共享的 Postgres**，所以
后果不只是"没拦住"，而是并发写同一份 checkpoint。单进程是这套方案的前提，
不是巧合。

将来要多 worker，把本模块换成 Redis 实现即可（`SET NX` + 持有者校验）：
调用方只依赖 try_acquire / release / holder 三个方法。数据源已经就绪
（app/config/datasource/redis.yaml）。

为什么不加超时 / 租约 / 自愈
--------------------------
"这次运行还在途吗"只有持有者自己知道。租约把这个问题重新引进来
（过期了但它其实还在跑，怎么办），而它并不能覆盖真正的卡死场景
—— 调度被拒那条路（见 agent_run_service_impl._schedule），那里的 run
停在 queued，看起来永远"非终态"。那条路是在源头释放锁解决的，
不是靠过期。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

__all__ = [
    "InProcessThreadRunLock",
    "ThreadLockHolder",
    "get_thread_run_lock",
    "reset_thread_run_lock",
]


@dataclass(frozen=True, slots=True)
class ThreadLockHolder:
    """当前持有某个 thread 的那次运行。"""

    run_id: UUID

    #: 取锁时刻（`time.monotonic()`）。**只用于日志与排查** ——
    #: 没有任何过期逻辑读它，理由见模块 docstring。
    acquired_at: float


class InProcessThreadRunLock:
    """
    `thread_id -> 在途运行` 的进程内登记表。

    ⚠️ 为什么不需要 `threading.Lock` / `asyncio.Lock`
    ----------------------------------------------
    `try_acquire` 与 `release` 里**没有 await**，因此在单线程事件循环里，
    "读字典 + 写字典"是不可分割的一个回合 —— 两个请求无论怎么交错，
    都不可能在同一个 thread_id 上双双成功。

    加一把 `asyncio.Lock` 不但没用，还有害：它会绑到当前事件循环上，
    让这个进程级单例在"测试里新建循环"时变得不可用；而且等待锁正是
    我们要避免的行为（要求是**立即 409**，不是排队）。

    同理，`release` 刻意是**同步**的：它会在 handler 的 `finally` 里被调用，
    而那一刻任务可能正在被取消 —— 后面任何 await 都可能立刻再抛一次，
    finally 里必须放一个不会被打断的动作。
    """

    def __init__(self) -> None:
        self._held: dict[str, ThreadLockHolder] = {}

    # ========================================================
    # 取 / 放
    # ========================================================

    def try_acquire(
            self,
            thread_id: str,
            *,
            run_id: UUID,
    ) -> ThreadLockHolder | None:
        """
        试取一次锁。**不等待。**

        Returns:
            成功时 None；已被占用时返回**当前持有者** —— 调用方要的
            不是"失败了"，而是"谁在占着"，那才是能给用户看的 409 信息。
        """

        existing = self._held.get(thread_id)

        if existing is not None:
            return existing

        self._held[thread_id] = ThreadLockHolder(
            run_id=run_id,
            acquired_at=time.monotonic(),
        )

        return None

    def release(
            self,
            thread_id: str,
            *,
            run_id: UUID,
    ) -> bool:
        """
        释放锁。**只有持有者放得掉**，其余情况是 no-op 并返回 False。

        幂等且**永不抛异常** —— 它会在 `finally` 里被调用，那里抛出去
        会把真正该被看到的业务异常顶掉，也正是"补偿钩子不该失败"
        的同一条理由。对"不是自己持有的键"返回 False 而不是报错：
        迟到的释放是预期内的（见模块 docstring），不是错误。
        """

        holder = self._held.get(thread_id)

        if holder is None or holder.run_id != run_id:
            return False

        del self._held[thread_id]

        return True

    # ========================================================
    # 只读
    # ========================================================

    def holder(
            self,
            thread_id: str,
    ) -> ThreadLockHolder | None:
        """当前持有者；空闲时 None。给观测与测试用。"""

        return self._held.get(thread_id)

    def held_threads(self) -> frozenset[str]:
        """当前被占用的全部 thread_id。给自检脚本 / 排查用。"""

        return frozenset(self._held)


#: 进程级单例。锁本来就该是"全进程一份" —— 每个服务实例各存一份
#: 等于没有锁。
_lock: InProcessThreadRunLock | None = None


def get_thread_run_lock() -> InProcessThreadRunLock:
    """
    取进程级 thread 锁。

    ⚠️ 为什么是模块级单例，而不是注入 ServiceFactory 的 pool
    ----------------------------------------------------
    取锁的是 `AgentRunService`（经 ServiceFactory 反射构造），放锁的是
    `build_dispatch_handler` 产出的 handler（在 main.py 里直接构造）——
    两个**互不相干的装配路径**。走池子注入要两处都记得接，而漏一处的
    后果不对称：服务侧漏接会退回参数的默认值 None，于是护栏静默失效
    —— 不报错、不警告，只是永远不冲突。

    模块级单例从构造上排除了这种可能。先例：`get_tool_cache()`
    （agent/tools/processor/runtime/cache.py）。
    """

    global _lock

    if _lock is None:
        _lock = InProcessThreadRunLock()

    return _lock


def reset_thread_run_lock() -> None:
    """
    丢掉当前单例，下次取时重建。**只给测试用**
    —— 与 `reset_factories()`（common/factories.py）同一条理由：
    进程级单例必须有一个显式的复位口子，否则测试之间会互相污染。
    """

    global _lock

    _lock = None
