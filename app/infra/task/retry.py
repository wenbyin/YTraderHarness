"""
重试策略：backoff 与"该不该再试"。

集中放在这里，是因为 spec §10 明确要求 retry policy **不得散落到业务
Service 里**。散落的具体后果是：每个业务各自决定"退避几秒""试几次"，
于是"这条任务为什么等了 40 秒才重试"要去翻那个业务的代码才知道，
而任务本身的 attempt / max_attempts 反而没人看。

attempt 与 max_attempts 存在 **Task 上**（DB 列），不在这里 ——
本模块只回答两个纯粹的问题：这次该不该再试、该等多久。
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from pydantic import ValidationError

from app.common.task.exceptions import (
    TaskParamsError,
    TaskPermanentError,
    TaskStateError,
    UnknownHandlerError,
)

__all__ = ["RetryPolicy"]


#: **不该重试**的异常。分两类：
#:
#: 1. 配置/调用错误 —— 重试多少次结果都一样
#:        UnknownHandlerError   handler 没注册
#:        TaskParamsError       参数与注册的类型对不上
#:        TaskStateError        状态不允许（重试还会再撞一次）
#:        ValidationError       参数没通过 pydantic 校验
#:
#: 2. 取消 —— 它是**被要求停下来**，不是失败
#:        CancelledError
#:
#: ⚠️ CancelledError 其实轮不到这里判断：Worker 会先单独捕获它、把任务
#: 记成 cancelled（见 infra/task/worker.py 的说明）。列在这里只是防御 ——
#: 万一将来有人改了 Worker 的捕获顺序，"取消被重试"这种最糟的结果
#: （用户点了停止，任务却自己又跑了一遍）也不会发生。
_FATAL: tuple[type[BaseException], ...] = (
    UnknownHandlerError,
    TaskParamsError,
    TaskStateError,
    ValidationError,
    asyncio.CancelledError,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """
    指数退避 + 上限，外加可选抖动。

    默认值偏保守（5s 起、翻倍、封顶 5 分钟）：延迟任务本来就不赶时间，
    而重试太急会把一个正在恢复的上游压垮 —— 那正是"重试风暴"的成因。
    """

    #: 第一次失败的退避时长（秒）。之后的每次翻 multiplier 倍。
    base_delay: float = 5.0

    #: 退避倍数。
    multiplier: float = 2.0

    #: 退避上限（秒）。封顶是必须的：不限的话第 10 次就是 5 分钟 × 2^9。
    max_delay: float = 300.0

    #: 抖动比例。计算出的间隔会被乘以 [1 - jitter, 1] 之间的一个随机数。
    #:
    #: 只**往下**抖，不往上：往上抖会让实际间隔超过 max_delay，封顶就
    #: 白设了。
    #:
    #: ⚠️ 非 0 时 delay_for 不再确定。要精确断言退避值的测试请显式传
    #: jitter=0.0（见 task_worker_smoke.py）。
    #:
    #: 默认给 10% 而不是 0：一批任务同时失败时（上游刚挂），没有抖动
    #: 它们会在同一毫秒一起回来，把刚恢复的上游再压垮一次。
    jitter: float = 0.1

    def delay_for(
            self,
            attempt: int,
    ) -> float:
        """
        第 attempt 次尝试失败之后，该等多少秒再试。

        attempt 从 1 开始数（等于 Task.attempt，也就是"已经试过几次"）。
        因此第一次失败的等待是 base_delay 本身。
        """

        # 防御：attempt 小于 1 时不该出现负数次幂（那会算出比 base 还小的
        # 间隔，让第一次重试反而更快）。
        exponent = max(0, attempt - 1)

        raw = self.base_delay * (self.multiplier ** exponent)

        capped = min(raw, self.max_delay)

        if self.jitter <= 0.0:
            return capped

        return capped * (1.0 - self.jitter * random.random())

    def is_retryable(
            self,
            exc: BaseException,
            *,
            attempt: int,
            max_attempts: int,
    ) -> bool:
        """
        这次失败还要不要再试。

        ⚠️ 默认**重试**，而不是默认不试
        -----------------------------
        默认不试的话，"忘了考虑重试"与"故意不重试"长得一模一样，而
        前者的后果（一次网络抖动就永久失败）远比后者严重。所以默认给
        几次机会，由 Handler 用 TaskPermanentError 明确说"别试了"。

        上限优先于一切：attempt 已经到 max_attempts，无论什么异常都不再试
        —— 否则"重试上限"就名不副实了。
        """

        if attempt >= max_attempts:
            return False

        if isinstance(exc, TaskPermanentError):
            return False

        return not isinstance(exc, _FATAL)
