"""
事件循环卡顿哨兵。

为什么需要它
------------
本仓库真实发生过：启动后第一次请求时报出一条

    redis.exceptions.TimeoutError: Timeout reading from localhost:6379

而实测（探针复现）的结论是：某段**同步**代码把事件循环按住 ≥5 秒，
asyncio 的读超时定时器在循环恢复的**瞬间**先到期，把"回复早就到了"
谎报成了网络故障。详见 app/agent/event/reader.py 的模块 docstring。

装上这个哨兵之后，第一次跑就抓到了元凶（那次是 Milvus 数据源在
`__init__` 里同步做 gRPC 建连，按住循环 3.3 秒）：

    ERROR  事件循环被卡住 3.3 秒（阈值 2.0 秒）—— 它正卡在下面这段
           同步代码里 … [事件循环线程] File "app/infra/datasource/milvus.py",
           line 44 in __init__ …

怎么读这条日志
--------------
栈里**事件循环线程**那一段，最上面那个"不是 asyncio 内部"的帧就是
元凶所在的函数（上面例子里是 milvus.py 的 `__init__`）。把它挪进
`asyncio.to_thread`，或挪到启动阶段预热，就完了。

⚠️ 与 asyncio 自带的调试模式是什么关系
------------------------------------
`PYTHONASYNCIODEBUG=1` 也能报（"Executing <Task ...> took N seconds"），
而且**零代码**。区别：

    asyncio 调试模式   报"哪个 Task 卡了多久"；要开环境变量、重启才生效，
                       阈值是每个回调 100ms，日志很吵
    本模块             报"当时每条线程卡在哪个函数"；常驻、阈值按需、
                       不依赖环境变量（生产也能一直开着）

两者不冲突，排查时可以同时开。

⚠️ 已知的盲区（说清楚，免得下次误以为"没报就是没卡"）
--------------------------------------------------
检测靠的是"事件循环之外的**一条 Python 线程**发现时间戳变旧了"。如果
卡住的那段 C 扩展**一直不释放 GIL**（少数情况：某些原生库的长调用），
这条 Python 线程自己也调度不上，就报不出来。那种情况下用
`faulthandler`（`PYTHONFAULTHANDLER=1`）——它从信号和 C 层看门狗计时，
不经过 Python 调度器。本模块用 `faulthandler.dump_traceback` 采集栈，
所以两条路看到的是同一份现场。
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import sys
import threading
import time
import traceback
from typing import Any

__all__ = [
    "DEFAULT_CHECK_INTERVAL_SECONDS",
    "DEFAULT_REPORT_INTERVAL_SECONDS",
    "DEFAULT_THRESHOLD_SECONDS",
    "LoopWatchdog",
]


logger = logging.getLogger(__name__)


#: 卡多久算"卡住"。刻意小于 redis.yaml 的 socket_timeout（5 秒）——
#: 我们要在它把读超时引爆**之前**就抓到现场，而不是事后解释那条报错。
DEFAULT_THRESHOLD_SECONDS = 2.0

#: 心跳间隔与检测间隔。检测粒度就是这个量级：卡住的时长按心跳间隔
#: 取整上报（默认 0.5 秒，够用）。
DEFAULT_CHECK_INTERVAL_SECONDS = 0.5

#: 一直卡着的话，每隔这么久再报一次（第一份现场可能是误导性的：
#: 卡住的入口往往只是"发起者"，多打几份才看得出真正的堵点）。
DEFAULT_REPORT_INTERVAL_SECONDS = 5.0


class LoopWatchdog:
    """
    监控事件循环有没有被同步代码按住。

    形状与 dispatcher / model_factory 一致：组合根（app/main.py 的
    lifespan）持有它、start() 起、aclose() 停。

    机制：一条心跳协程在循环上每 check_interval 秒打一次时间戳；一条
    **普通线程**（不在循环上，所以循环卡住时它照跑）盯着那个时间戳，
    发现变旧超过 threshold 就采一份现场。
    """

    def __init__(
            self,
            *,
            threshold: float = DEFAULT_THRESHOLD_SECONDS,
            check_interval: float = DEFAULT_CHECK_INTERVAL_SECONDS,
            report_interval: float = DEFAULT_REPORT_INTERVAL_SECONDS,
            file: Any | None = None,
    ) -> None:

        self._threshold = threshold
        self._check_interval = check_interval
        self._report_interval = report_interval

        #: 栈往哪打。默认 stderr（uvicorn 那个控制台）。
        #:
        #: ⚠️ 不能直接交给 logging 的 handler：faulthandler 只能往"有
        #: fileno 的文件"写。所以那条带时间戳、能被日志系统收走的报告走
        #: logging，随后由 faulthandler 把原始栈追在它后面 —— 两者靠先后
        #: 顺序对上，读日志的人先看到"发生了什么"，再看到"卡在哪"。
        self._file = file if file is not None else sys.stderr

        #: 循环最后一次证明自己活着的时间（单调时钟）。
        self._last_beat = time.monotonic()

        #: 事件循环所在线程 —— 报告里标出来，省得读的人自己在几十条
        #: 线程栈里找。
        self._loop_thread_id: int | None = None

        self._beat: asyncio.Task[None] | None = None
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._dumping = False

    # ========================================================

    def start(self) -> None:
        """起监控（幂等）。必须在事件循环里调用。"""

        if self._watcher is not None:
            return

        self._loop_thread_id = threading.get_ident()
        self._last_beat = time.monotonic()
        self._stop.clear()

        self._beat = asyncio.create_task(
            self._beat_forever(),
            name="loop-watchdog-beat",
        )

        self._watcher = threading.Thread(
            target=self._watch,
            name="loop-watchdog",
            daemon=True,
        )

        self._watcher.start()

    async def aclose(self) -> None:
        """停监控（照 model_factory.aclose 的形状）。"""

        beat, self._beat = self._beat, None
        watcher, self._watcher = self._watcher, None

        self._stop.set()

        if beat is not None:
            beat.cancel()

            try:
                await beat

            except asyncio.CancelledError:
                pass

        if watcher is not None:
            # join 是阻塞调用，包进 to_thread —— 它不该占住事件循环。
            await asyncio.to_thread(watcher.join, 2.0)

    @property
    def running(self) -> bool:
        """探活用：监控是否在跑。"""

        return self._watcher is not None and self._watcher.is_alive()

    # ========================================================

    async def _beat_forever(self) -> None:
        """
        心跳：证明"循环还能跑"。

        注意它**只打时间戳，不做任何判断** —— 判断必须由循环之外的线程
        来做，因为循环被卡住时，它自己没法上报自己。
        """

        while True:

            self._last_beat = time.monotonic()

            await asyncio.sleep(self._check_interval)

    def _watch(self) -> None:
        """看门狗线程：时间戳旧了就采现场。"""

        while not self._stop.is_set():

            stalled = time.monotonic() - self._last_beat

            if stalled < self._threshold:
                # wait() 而不是 sleep()：停的时候能立刻醒，不用等满一轮。
                self._stop.wait(self._check_interval)
                continue

            self._report(stalled)

            # 还卡着就只等一小会儿再采一份，但别刷屏。
            self._stop.wait(self._report_interval)

    def _report(self, stalled: float) -> None:
        """
        报一次现场。

        分两段，因为读者要的东西不一样：

            日志正文里  **事件循环线程**当时的栈 —— 元凶就在这一段里，
                        而且它带时间戳、能被日志系统收走、能 grep
            紧随其后    faulthandler 的**全部**线程原始栈 —— 用于对照，
                        比如"循环在等一把别人持有的锁"这种情况

        ⚠️ 不能只靠 faulthandler 那一段：它对**主线程**不打印线程名
        （实测 CPython 3.14：pydevd 那些线程有 `[名字]`，主线程没有），
        而 uvicorn 的事件循环恰好跑在主线程上 —— 于是"哪一段是循环的栈"
        就没人告诉读者了。所以循环那一段必须由我们自己取出来写进日志。
        """

        logger.error(
            "事件循环被卡住 %.1f 秒（阈值 %.1f 秒）—— 它正卡在下面这段"
            "**同步**代码里。这段时间内所有请求、SSE 心跳、以及 Redis 的"
            "读超时定时器都在停摆（后者会伪装成一条网络故障报错）。\n"
            "事件循环线程当时卡在（最上面那个不是 asyncio 内部的帧就是"
            "元凶 —— 把它挪进 asyncio.to_thread，或挪到启动阶段预热）：\n"
            "%s"
            "（其余线程的原始栈紧随其后，仅供对照）",
            stalled,
            self._threshold,
            self._loop_stack(),
        )

        self._dump()

    def _loop_stack(self) -> str:
        """取事件循环线程当时的调用栈。"""

        frames = sys._current_frames()

        frame = (
            frames.get(self._loop_thread_id)
            if self._loop_thread_id is not None
            else None
        )

        if frame is None:       # pragma: no cover - 兜底
            return "  （读不到事件循环线程的帧）\n"

        return "".join(traceback.format_stack(frame))

    def _dump(self) -> None:
        """
        采一份线程栈。

        `_dumping` 是个再入保护：某条线程栈里如果正好压在 stderr 上
        （写日志也可能被卡住），再进来一次只会把日志搅成一团。
        """

        if self._dumping:
            return

        self._dumping = True

        try:
            faulthandler.dump_traceback(file=self._file)

        except (ValueError, RuntimeError, OSError) as exc:

            logger.warning(
                "卡顿哨兵采不到线程栈（%s: %s）—— 说明里那条报错仍然有效，"
                "原始栈可以改用 PYTHONFAULTHANDLER=1 重启来看。",
                type(exc).__name__,
                exc,
            )

        finally:
            self._dumping = False

