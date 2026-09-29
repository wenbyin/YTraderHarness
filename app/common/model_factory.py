"""
进程级模型工厂：启动时**异步预热**，使用时直接取。

为什么要把加载从"首次用到时"挪到"启动时"
----------------------------------------
原来的做法是懒加载 + 进程内单例（见 retrieve/rerank.py 与
embedder/bge_small_zh_local.py 的历史实现）。它解决了"每个请求加载
一次"，但没解决**第一个请求**：那一个请求要独自承担几秒到十几秒的
加载，表现为"服务刚起来时特别慢"，而且第一个用户的超时设置很可能
撑不住。

挪到启动阶段之后，加载与请求解耦：服务起来的那一刻就开始加载，等
真正有请求时通常已经好了。

没有加载完就阻塞，而不是失败
----------------------------
这是刻意的：模型**正在加载**和模型**加载失败**是两回事。

    还在加载  -> 等一下就好了，阻塞 + 打日志（本模块的 get）
    加载失败  -> 等多久都没用，立刻抛（见 _Slot.error）

把它们混成一种处理（比如"没就绪就当失败"）会让服务刚启动的那几秒
无谓地报错。

与请求的关系
------------
预热是**后台任务**，不阻塞应用启动 —— 数据源、路由、调度器照常就绪，
只是第一个用到模型的请求会在这里等一会儿。等待有超时（见
DEFAULT_WAIT_TIMEOUT），因为加载卡住（磁盘挂死、网络盘掉线）时无限
等待比明确报错更糟。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "DEFAULT_WAIT_TIMEOUT",
    "ModelFactory",
    "ModelNotReadyError",
    "get_model_factory",
]

logger = logging.getLogger(__name__)

#: get() 等待模型加载完成的默认上限（秒）。
#:
#: 取一个明显大于最慢模型加载时间的值：本项目的精排模型约 2.3GB，
#: 冷启动从磁盘读要十几秒。设得比正常加载还短会让每个冷启动请求都
#: 白等一次超时。
#:
#: 但它必须存在：加载卡住时（G 盘掉线、磁盘 IO 挂死）无限等待会把
#: 请求连接一直挂住，而明确报错至少让调用方知道发生了什么。
DEFAULT_WAIT_TIMEOUT = 300.0


class ModelNotReadyError(RuntimeError):
    """模型没能就绪：加载失败，或等待加载完成超时。"""


@dataclass
class _Slot:
    """一个模型的加载状态。"""

    key: str
    loader: Callable[[], Any]

    model: Any = None
    error: BaseException | None = None

    #: 加载结束（成功或失败）时置位。等待方等它。
    done: asyncio.Event = field(default_factory=asyncio.Event)

    #: 加载任务。None = 还没开始。
    task: asyncio.Task | None = None


class ModelFactory:
    """
    按 key 登记与获取进程级模型。

    key 用什么由调用方定 —— 本项目用的是模型的**路径/名字**，因为
    加载哪个模型取决于配置，而不只是类型。同一个 key 只会加载一次。
    """

    def __init__(self) -> None:
        self._slots: dict[str, _Slot] = {}

    # ========================================================
    # 登记与预热
    # ========================================================

    def register(
            self,
            key: str,
            loader: Callable[[], Any],
    ) -> None:
        """
        登记一个模型。**只登记，不加载** —— 加载由 preload_all 或
        首次 get 触发。

        loader 是**同步**可调用对象：真正加载时会被丢进线程池
        （加载是阻塞 IO + CPU，不能占着事件循环）。
        """

        existing = self._slots.get(key)

        if existing is not None:
            # 重复登记同一个 key 用**同一个** loader 是允许的：模块
            # 可能被不同路径 import，登记多次不该报错。但换了 loader
            # 就是真冲突 —— 那意味着两个地方想用同一个 key 加载不同
            # 的东西，静默取其一比报错危险得多。
            if existing.loader is not loader:
                raise ValueError(
                    f"模型 key {key!r} 已经被另一个 loader 登记过。"
                    "同一个 key 必须指向同一个模型。"
                )
            return

        self._slots[key] = _Slot(key=key, loader=loader)

    def preload_all(self) -> None:
        """
        启动全部已登记模型的加载（**不等待**）。

        不 await 是刻意的：调用方是应用启动流程，让服务先就绪、模型
        在后台加载，比让整个启动卡十几秒好。需要就绪保证的请求会在
        get() 里等。
        """

        for slot in self._slots.values():
            self._start(slot)

    async def aclose(self) -> None:
        """
        取消尚未完成的加载任务。

        只在关闭阶段调用。不这么做的话进程退出时 asyncio 会警告
        "Task was destroyed but it is pending"。
        """

        pending = [
            slot.task
            for slot in self._slots.values()
            if slot.task is not None and not slot.task.done()
        ]

        for task in pending:
            task.cancel()

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # ========================================================
    # 取用
    # ========================================================

    async def get(
            self,
            key: str,
            *,
            loader: Callable[[], Any] | None = None,
            timeout: float | None = DEFAULT_WAIT_TIMEOUT,
    ) -> Any:
        """
        取模型。没就绪就**等**，等到或超时。

        loader 参数用于"没登记过就现场登记"：让不走启动流程的场景
        （tools/ 下的脚本、冒烟测试）仍然可用，而不必先跑一遍
        preload_all。服务进程里模型早已登记，这个参数用不上。
        """

        slot = self._slots.get(key)

        if slot is None:
            if loader is None:
                raise ModelNotReadyError(
                    f"模型 {key!r} 没有登记过，也没有给 loader。"
                    "要么在启动阶段 register，要么调用时传 loader。"
                )
            self.register(key, loader)
            slot = self._slots[key]

        self._start(slot)

        if not slot.done.is_set():
            # 只在**确实要等**的时候打这一条 —— 预热完成的正常路径
            # 不该刷日志。
            logger.info(
                "模型 %s 尚未加载完成，本次请求等待中 ……", key,
            )

            try:
                await asyncio.wait_for(
                    slot.done.wait(), timeout=timeout,
                )

            except TimeoutError as exc:
                raise ModelNotReadyError(
                    f"等待模型 {key!r} 加载完成超时"
                    f"（{timeout} 秒）。加载可能卡住了 —— "
                    "检查模型文件所在磁盘是否可读。"
                ) from exc

        if slot.error is not None:
            raise ModelNotReadyError(
                f"模型 {key!r} 加载失败："
                f"{type(slot.error).__name__}: {slot.error}"
            ) from slot.error

        return slot.model

    # ========================================================
    # internals
    # ========================================================

    def _start(self, slot: _Slot) -> None:
        """确保加载任务已启动。幂等 —— 重复调用不会重复加载。"""

        if slot.task is None:
            slot.task = asyncio.create_task(self._load(slot))

    async def _load(self, slot: _Slot) -> None:
        started = time.monotonic()

        logger.info("开始加载模型 %s ……", slot.key)

        try:
            # 加载是阻塞的（读盘 + 建图 + 占内存）：丢线程池，
            # 别占着事件循环 —— 预热是后台任务，但它跑在同一个循环里。
            slot.model = await asyncio.to_thread(slot.loader)

        except asyncio.CancelledError as exc:
            # 关闭时取消（见 aclose）。这是**正常**的收尾动作，不能按
            # 加载失败记 —— 打一条 error 加整条 traceback 会让每次
            # 关闭都看起来像出了故障。
            #
            # 但仍然要让 done 置位、仍然要记 error：万一有请求正等在这
            # 个模型上，它得立刻拿到"被取消了"而不是干等到超时。
            slot.error = exc
            logger.info("模型 %s 的加载被取消（通常是服务正在关闭）。", slot.key)

        except BaseException as exc:
            slot.error = exc
            logger.error(
                "模型 %s 加载失败（耗时 %.1fs）：%s: %s",
                slot.key,
                time.monotonic() - started,
                type(exc).__name__,
                exc,
                exc_info=True,
            )

        else:
            logger.info(
                "模型 %s 加载完成，耗时 %.1fs。",
                slot.key,
                time.monotonic() - started,
            )

        finally:
            slot.done.set()


#: 进程级单例。
#:
#: 为什么用模块级单例而不是依赖注入：消费方（reranker / embedder）由
#: 能力注册表按类型反射实例化，**构造函数拿不到任何参数**，没有注入
#: 的入口。见 pipeline/registry.py 的 found_capabilities。
_factory = ModelFactory()


def get_model_factory() -> ModelFactory:
    """取进程级工厂。"""

    return _factory
