"""
`run_id -> 已请求取消` 的进程内登记表。

为什么需要它（而不是直接 cancel 那个 asyncio 任务就够了）
----------------------------------------------------
"任务被取消"这件事要**穿过**好几层才落成一次业务事实：

    POST /runs/{id}/cancel
      -> dispatcher.cancel(run_id)       取消 asyncio 任务（或标记排队中的）
      -> handler 捕获 CancelledError      它要把状态记成 cancelled
      -> 以及 rollback                   丢掉这一轮写进 checkpoint 的东西

中间那一步的困难在于：handler 拿到的只有 `DispatchEnvelope`，而
`asyncio.CancelledError` 本身**不携带"是谁取消的"**。应用关闭时
任务也会被取消，而那是**另一回事**（那时应当走 finally 收尾、按失败或
中断处理，不该记成"用户取消了这次运行"）。

所以"被取消"必须是一个**显式的事实**，而不是从异常类型反推出来的：
dispatcher 取消之前先在这里登记，handler 拿到 CancelledError 之后来这里问
"这次是我被取消了吗"。

⚠️ 与 thread_lock 同一条立场：**进程内**，因此部署必须 workers=1
----------------------------------------------------------------
多进程下发起取消的那个进程与正在跑图的进程可能不是同一个 —— 请求打到 A，
图跑在 B。A 的登记表里没有那个 run，于是它以为"无事可做"，而 B 那边
照跑不误。这与 `InProcessThreadRunLock` 是同一个前提（见 thread_lock.py），
不是本模块自己引入的限制。换成真队列时，这件事变成 outbox 表上的一行。

⚠️ 为什么要**主动清理**
--------------------
登记表按 run_id 攒，而 run 的数量没有上界。不清理就是一条内存泄漏，且
症状是"跑得越久占用越大"这种最难归因的那类。因此 handler 在收尾路径上
**必须**调 `discard()` —— 那里的 `finally` 正是唯一确定会跑到的地方。
"""

from __future__ import annotations

import logging
import threading
from uuid import UUID

logger = logging.getLogger(__name__)

__all__ = [
    "RunCancellationRegistry",
    "get_run_cancellation_registry",
    "reset_run_cancellation_registry",
]


class RunCancellationRegistry:
    """
    哪些 run 被请求过取消。

    线程安全：dispatcher 的取消路径可能来自请求线程（HTTP），而 handler
    在事件循环的另一个任务里读它 —— 两者不在同一个线程上，
    所以这里用锁而不是裸 set。
    """

    def __init__(self) -> None:
        self._requested: set[UUID] = set()
        self._lock = threading.Lock()

    def request(self, run_id: UUID) -> None:
        """登记一次取消请求。重复调用是幂等的。"""

        with self._lock:
            self._requested.add(run_id)

    def is_requested(self, run_id: UUID) -> bool:
        """这个 run 被请求过取消吗。**只读，不改状态** —— handler 可能要
        问好几次（每个收尾分支各一次），问一次就清掉会让后面的分支看不到。"""

        with self._lock:
            return run_id in self._requested

    def discard(self, run_id: UUID) -> None:
        """
        handler 收尾时把它拿掉。

        ⚠️ 必须与 `is_requested` 分开，且由 handler 在**终结之后**调：
        读的时候清掉的话，同一个 run 上并发的两个收尾分支会有一个看不到
        这个事实（而它恰恰是决定"记 completed 还是 cancelled"的那一个）。
        """

        with self._lock:
            self._requested.discard(run_id)

    def pending(self) -> list[UUID]:
        """当前登记着的 run_id。给运维/冒烟用，不在执行路径上。"""

        with self._lock:
            return sorted(self._requested)

    def __len__(self) -> int:
        with self._lock:
            return len(self._requested)


#: 进程级单例。与 `get_thread_run_lock()` 同一形状 —— 进程级事实只有一个。
_registry: RunCancellationRegistry | None = None


def get_run_cancellation_registry() -> RunCancellationRegistry:
    global _registry

    if _registry is None:
        _registry = RunCancellationRegistry()

    return _registry


def reset_run_cancellation_registry() -> None:
    """
    丢掉当前登记表（下一次 `get_run_cancellation_registry()` 会重建）。

    存在只是为了冒烟脚本能在用例之间回到干净状态 —— 生产没有任何调用点，
    因为"取消登记"本来就是一次性的进程内状态。
    """

    global _registry
    _registry = None
